import argparse
import gc
import json
import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np
import zarr


RETURN_GAMMA = 0.99
ZARR_CHUNK_LEAD = 100
IMAGENET_MEAN_UINT8 = np.array([123, 116, 104], dtype=np.uint8)


def find_motion_onset(
    state: np.ndarray,
    threshold_rad: float,
    window: int,
    min_moving_frames: int,
) -> int | None:
    """返回可靠运动窗口的起始帧；只使用六个关节，不使用恒定夹爪维度。

    单次反馈抖动不应被当成示教开始，因此要求连续 ``window`` 个 transition
    中至少有 ``min_moving_frames`` 个的最大关节变化超过阈值。
    """

    if state.ndim != 2 or state.shape[1] < 6:
        raise ValueError(f"state shape must be [T,>=6], got {state.shape}")
    if len(state) < window + 1:
        return None

    joint_step = np.max(np.abs(np.diff(state[:, :6], axis=0)), axis=1)
    moving = (joint_step > threshold_rad).astype(np.int32)
    moving_count = np.convolve(
        moving,
        np.ones(window, dtype=np.int32),
        mode="valid",
    )
    candidates = np.flatnonzero(moving_count >= min_moving_frames)
    if len(candidates) == 0:
        return None

    # 返回这个可靠窗口里的第一条真实运动 transition，而不是窗口左边界；
    # transition i 描述 frame i -> frame i+1，因此运动帧从 i+1 开始。
    window_start = int(candidates[0])
    first_moving_offset = int(
        np.flatnonzero(moving[window_start : window_start + window])[0]
    )
    return window_start + first_moving_offset + 1


def static_prefix_trim_start(state: np.ndarray, args) -> tuple[int, int | None]:
    """计算静止前缀裁剪位置，同时保留策略观察所需的动作前帧。"""

    if not args.trim_static_prefix:
        return 0, None
    onset = find_motion_onset(
        state=state,
        threshold_rad=args.motion_threshold_rad,
        window=args.motion_window,
        min_moving_frames=args.motion_min_frames,
    )
    if onset is None:
        if args.keep_no_motion_episode:
            return 0, None
        raise RuntimeError(
            "no reliable joint motion detected; use --keep-no-motion-episode "
            "to keep this episode for inspection"
        )
    return max(0, onset - args.motion_preroll_frames), onset


def compute_return(reward: np.ndarray, not_done: np.ndarray, gamma: float = RETURN_GAMMA) -> np.ndarray:
    out = np.zeros_like(reward, dtype=np.float32)
    running = 0.0
    for i in reversed(range(len(reward))):
        running = float(reward[i, 0]) + gamma * running * float(not_done[i, 0])
        out[i] = running
    return out


def create_array(group, name, array, chunks=None, dtype=None):
    try:
        from numcodecs import Blosc

        compressor = Blosc(cname="zstd", clevel=3, shuffle=1)
    except Exception:
        compressor = None

    if hasattr(group, "create_dataset"):
        kwargs = {"data": array, "overwrite": True}
        if dtype is not None:
            kwargs["dtype"] = dtype
        if chunks is not None:
            kwargs["chunks"] = chunks
        if compressor is not None:
            kwargs["compressor"] = compressor
        return group.create_dataset(name, **kwargs)

    kwargs = {"data": array, "overwrite": True}
    if chunks is not None:
        kwargs["chunks"] = chunks
    return group.create_array(name, **kwargs)


def resize_bgr_to_chw_rgb(
    image_bgr: np.ndarray,
    size: int,
    resize_mode: str = "stretch",
) -> np.ndarray:
    """BGR→RGB，并以拉伸或等比例 letterbox 方式生成正方形 CHW 图像。"""

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    if resize_mode == "stretch":
        output = cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_AREA)
    elif resize_mode == "letterbox":
        height, width = image_rgb.shape[:2]
        scale = min(size / width, size / height)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(
            image_rgb,
            (resized_width, resized_height),
            interpolation=interpolation,
        )
        output = np.empty((size, size, 3), dtype=np.uint8)
        output[...] = IMAGENET_MEAN_UINT8
        top = (size - resized_height) // 2
        left = (size - resized_width) // 2
        output[top : top + resized_height, left : left + resized_width] = resized
    else:
        raise ValueError(f"unknown image resize mode: {resize_mode}")
    return np.transpose(output, (2, 0, 1))


def load_camera_meta(f: h5py.File) -> dict:
    raw = f.attrs.get("camera_meta_json")
    if raw is None:
        return {}
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def build_point_cloud_from_depth(
    depth: np.ndarray,
    depth_scale: np.ndarray,
    camera_meta: dict,
    num_points: int,
    allow_default_intrinsics: bool,
) -> np.ndarray:
    from realsense import camera_intrinsics as default_intrinsics
    from realsense import depth2pc, point_cloud_downsample

    if "depth_intrinsics" in camera_meta:
        intrinsics = tuple(camera_meta["depth_intrinsics"])
    elif allow_default_intrinsics:
        intrinsics = default_intrinsics
    else:
        raise RuntimeError(
            "raw file has no camera_meta_json/depth_intrinsics. "
            "Use raw episodes collected after intrinsics saving was added, "
            "or pass --allow-default-intrinsics for a rough smoke-test conversion."
        )

    pcs = []
    for i in range(depth.shape[0]):
        pc = depth2pc(depth[i] * float(depth_scale[i]), intrinsics)
        pcs.append(point_cloud_downsample(pc, num_points).astype(np.float32))
    return np.stack(pcs, axis=0)


def load_raw_episode(path: Path, args) -> dict:
    with h5py.File(path, "r") as f:
        state_full = f["robot/state_7d"][:].astype(np.float32)
        original_len = len(state_full)
        trim_start, motion_onset = static_prefix_trim_start(state_full, args)

        state = state_full[trim_start:]
        action = f["robot/action_7d"][trim_start:].astype(np.float32)
        rgb = f["camera/rgb"][trim_start:] if "camera/rgb" in f else None
        depth = f["camera/depth"][trim_start:] if "camera/depth" in f else None
        depth_scale = (
            f["camera/depth_scale"][trim_start:]
            if "camera/depth_scale" in f
            else None
        )
        camera_meta = load_camera_meta(f)

        if args.point_cloud_source in ("auto", "stored") and "camera/point_cloud" in f:
            point_cloud = f["camera/point_cloud"][trim_start:].astype(np.float32)
        elif args.point_cloud_source == "stored":
            raise RuntimeError(f"{path} has no camera/point_cloud")
        else:
            if depth is None or depth_scale is None:
                raise RuntimeError(f"{path} has no camera/depth or camera/depth_scale")
            point_cloud = build_point_cloud_from_depth(
                depth=depth,
                depth_scale=depth_scale,
                camera_meta=camera_meta,
                num_points=args.num_points,
                allow_default_intrinsics=args.allow_default_intrinsics,
            )

        if point_cloud.shape[1] != args.num_points:
            if point_cloud.shape[1] < args.num_points:
                reps = args.num_points // point_cloud.shape[1] + 1
                point_cloud = np.concatenate([point_cloud] * reps, axis=1)
            point_cloud = point_cloud[:, : args.num_points, :]

        if rgb is None:
            img = np.zeros((len(state), 3, args.image_size, args.image_size), dtype=np.uint8)
        else:
            img = np.stack(
                [
                    resize_bgr_to_chw_rgb(
                        rgb[i],
                        args.image_size,
                        resize_mode=args.image_resize_mode,
                    )
                    for i in range(rgb.shape[0])
                ],
                axis=0,
            ).astype(np.uint8)

    n = len(state)
    for name, array in {
        "action": action,
        "point_cloud": point_cloud,
        "img": img,
    }.items():
        if len(array) != n:
            raise RuntimeError(
                f"{path} {name} length mismatch after trimming: {len(array)} != {n}"
            )
    if n < args.min_episode_len:
        raise RuntimeError(
            f"{path} too short after trimming: {n} < "
            f"min_episode_len={args.min_episode_len}"
        )

    reward = np.zeros((n, 1), dtype=np.float32)
    reward[-1, 0] = args.terminal_reward
    done = np.zeros((n, 1), dtype=bool)
    timeout = np.zeros((n, 1), dtype=bool)
    done[-1, 0] = True
    timeout[-1, 0] = True

    next_state = np.empty_like(state)
    next_action = np.empty_like(action)
    next_point_cloud = np.empty_like(point_cloud)
    next_img = np.empty_like(img)
    next_state[:-1] = state[1:]
    next_action[:-1] = action[1:]
    next_point_cloud[:-1] = point_cloud[1:]
    next_img[:-1] = img[1:]
    next_state[-1] = state[-1]
    next_action[-1] = action[-1]
    next_point_cloud[-1] = point_cloud[-1]
    next_img[-1] = img[-1]

    return {
        "path": str(path),
        "original_len": original_len,
        "trim_start": trim_start,
        "motion_onset": motion_onset,
        "state": state,
        "next_state": next_state,
        "action": action,
        "next_action": next_action,
        "point_cloud": point_cloud.astype(np.float32),
        "next_point_cloud": next_point_cloud.astype(np.float32),
        "img": img,
        "next_img": next_img,
        "reward": reward,
        "done": done,
        "timeout": timeout,
    }


def find_raw_files(inputs: list[Path]) -> list[Path]:
    files = []
    for item in inputs:
        if item.is_dir():
            files.extend(sorted(item.glob("*.hdf5")))
            files.extend(sorted(item.glob("*.h5")))
        elif item.is_file():
            files.append(item)
        else:
            raise FileNotFoundError(item)
    return sorted(set(files))


def write_zarr(
    episodes: list[dict],
    output: Path,
    overwrite: bool,
    image_preprocessing: dict | None = None,
) -> None:
    if output.exists():
        if not overwrite:
            raise FileExistsError(output)
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    arrays = {}
    for key in [
        "state",
        "next_state",
        "action",
        "next_action",
        "point_cloud",
        "next_point_cloud",
        "img",
        "next_img",
        "reward",
        "done",
        "timeout",
    ]:
        arrays[key] = np.concatenate([ep[key] for ep in episodes], axis=0)

    not_done = 1.0 - (arrays["done"] | arrays["timeout"]).astype(np.float32)
    arrays["return"] = compute_return(arrays["reward"], not_done)

    episode_ends = []
    total = 0
    for ep in episodes:
        total += len(ep["state"])
        episode_ends.append(total)
    episode_ends = np.asarray(episode_ends, dtype=np.int64)

    root = zarr.group(str(output))
    data = root.create_group("data")
    meta = root.create_group("meta")
    root.attrs["source_manifest"] = [
        {
            "kind": "piper_raw_hdf5",
            "path": ep["path"],
            "original_len": ep["original_len"],
            "trim_start": ep["trim_start"],
            "motion_onset": ep["motion_onset"],
            "converted_len": len(ep["state"]),
        }
        for ep in episodes
    ]
    root.attrs["schema"] = "rl100_replay_zarr_from_piper_raw_v1"
    if image_preprocessing is not None:
        root.attrs["image_preprocessing"] = image_preprocessing

    for key, arr in arrays.items():
        dtype = str(arr.dtype)
        if arr.ndim == 2:
            chunks = (ZARR_CHUNK_LEAD, arr.shape[1])
        elif arr.ndim == 3:
            chunks = (ZARR_CHUNK_LEAD, arr.shape[1], arr.shape[2])
        elif arr.ndim == 4:
            chunks = (ZARR_CHUNK_LEAD, arr.shape[1], arr.shape[2], arr.shape[3])
        else:
            chunks = None
        create_array(data, key, arr, chunks=chunks, dtype=dtype)
        print(f"[zarr] data/{key}: {arr.shape} {arr.dtype}", flush=True)
        gc.collect()

    create_array(meta, "episode_ends", episode_ends, dtype="int64")
    print(f"[zarr] meta/episode_ends: {episode_ends.shape} {episode_ends.dtype}", flush=True)
    print(f"[zarr] saved to {output}", flush=True)


def validate_zarr(path: Path) -> None:
    root = zarr.open(str(path), mode="r")
    required = [
        "state",
        "next_state",
        "action",
        "next_action",
        "point_cloud",
        "next_point_cloud",
        "img",
        "next_img",
        "reward",
        "return",
        "done",
        "timeout",
    ]
    for key in required:
        if key not in root["data"]:
            raise RuntimeError(f"missing data/{key}")
    if "episode_ends" not in root["meta"]:
        raise RuntimeError("missing meta/episode_ends")

    n = root["meta/episode_ends"][-1]
    for key in required:
        if root["data"][key].shape[0] != n:
            raise RuntimeError(f"data/{key} length mismatch: {root['data'][key].shape[0]} != {n}")
    print("[validate] ok", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True, help="raw .hdf5/.h5 files or dirs")
    parser.add_argument("--output", type=Path, required=True, help="output .zarr path")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num-points", type=int, default=512)
    parser.add_argument("--image-size", type=int, default=84)
    parser.add_argument(
        "--image-resize-mode",
        choices=["stretch", "letterbox"],
        default="stretch",
        help="letterbox preserves aspect ratio and pads with ImageNet mean RGB",
    )
    parser.add_argument("--min-episode-len", type=int, default=2)
    parser.add_argument("--terminal-reward", type=float, default=1.0)
    parser.add_argument(
        "--trim-static-prefix",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="detect and remove the initial stationary segment (enabled by default)",
    )
    parser.add_argument(
        "--motion-threshold-rad",
        type=float,
        default=0.001,
        help="a transition is moving when any joint changes more than this (default: 0.001 rad)",
    )
    parser.add_argument(
        "--motion-window",
        type=int,
        default=5,
        help="number of transitions in the robust motion-detection window",
    )
    parser.add_argument(
        "--motion-min-frames",
        type=int,
        default=3,
        help="minimum moving transitions required inside the detection window",
    )
    parser.add_argument(
        "--motion-preroll-frames",
        type=int,
        default=5,
        help="frames retained immediately before detected motion onset",
    )
    parser.add_argument(
        "--keep-no-motion-episode",
        action="store_true",
        help="keep an episode when no reliable motion is detected; default is to fail",
    )
    parser.add_argument(
        "--point-cloud-source",
        choices=["auto", "stored", "depth"],
        default="auto",
        help="auto uses camera/point_cloud if present, otherwise rebuilds from depth",
    )
    parser.add_argument(
        "--allow-default-intrinsics",
        action="store_true",
        help="allow fallback to realsense.py default intrinsics when raw file lacks saved intrinsics",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.image_size <= 0:
        raise ValueError("--image-size must be positive")
    if args.trim_static_prefix:
        if args.motion_threshold_rad <= 0:
            raise ValueError("--motion-threshold-rad must be positive")
        if args.motion_window <= 0:
            raise ValueError("--motion-window must be positive")
        if not 1 <= args.motion_min_frames <= args.motion_window:
            raise ValueError("--motion-min-frames must be in [1, --motion-window]")
        if args.motion_preroll_frames < 2:
            raise ValueError(
                "--motion-preroll-frames must be >=2 for the current n_obs_steps=3 policy"
            )
    files = find_raw_files(args.input)
    if not files:
        raise RuntimeError("no raw hdf5 files found")
    print(f"[input] {len(files)} raw episode file(s)", flush=True)
    episodes = []
    for path in files:
        print(f"[load] {path}", flush=True)
        ep = load_raw_episode(path, args)
        print(
            f"[trim] removed={ep['trim_start']} onset={ep['motion_onset']} "
            f"original={ep['original_len']} converted={len(ep['state'])}\n"
            f"[load] state={ep['state'].shape} "
            f"pc={ep['point_cloud'].shape} img={ep['img'].shape}",
            flush=True,
        )
        episodes.append(ep)
    total_original = sum(ep["original_len"] for ep in episodes)
    total_removed = sum(ep["trim_start"] for ep in episodes)
    print(
        f"[trim-summary] removed={total_removed}/{total_original} "
        f"({total_removed / total_original:.1%}), "
        f"remaining={total_original - total_removed}",
        flush=True,
    )
    write_zarr(
        episodes,
        args.output,
        overwrite=args.overwrite,
        image_preprocessing={
            "color_conversion": "BGR_to_RGB",
            "output_layout": "CHW",
            "output_size": [args.image_size, args.image_size],
            "resize_mode": args.image_resize_mode,
            "letterbox_pad_rgb": IMAGENET_MEAN_UINT8.tolist(),
        },
    )
    validate_zarr(args.output)


if __name__ == "__main__":
    main()
