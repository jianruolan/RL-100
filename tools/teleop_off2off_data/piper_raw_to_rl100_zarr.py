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


def resize_bgr_to_chw_rgb(image_bgr: np.ndarray, size: int) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_AREA)
    return np.transpose(image_rgb, (2, 0, 1))


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
        state = f["robot/state_7d"][:].astype(np.float32)
        action = f["robot/action_7d"][:].astype(np.float32)
        rgb = f["camera/rgb"][:] if "camera/rgb" in f else None
        depth = f["camera/depth"][:] if "camera/depth" in f else None
        depth_scale = f["camera/depth_scale"][:] if "camera/depth_scale" in f else None
        camera_meta = load_camera_meta(f)

        if args.point_cloud_source in ("auto", "stored") and "camera/point_cloud" in f:
            point_cloud = f["camera/point_cloud"][:].astype(np.float32)
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
                [resize_bgr_to_chw_rgb(rgb[i], args.image_size) for i in range(rgb.shape[0])],
                axis=0,
            ).astype(np.uint8)

    n = len(state)
    if n < args.min_episode_len:
        raise RuntimeError(f"{path} too short: {n} < min_episode_len={args.min_episode_len}")

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


def write_zarr(episodes: list[dict], output: Path, overwrite: bool) -> None:
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
        {"kind": "piper_raw_hdf5", "path": ep["path"]} for ep in episodes
    ]
    root.attrs["schema"] = "rl100_replay_zarr_from_piper_raw_v0"

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
    parser.add_argument("--min-episode-len", type=int, default=2)
    parser.add_argument("--terminal-reward", type=float, default=1.0)
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
    files = find_raw_files(args.input)
    if not files:
        raise RuntimeError("no raw hdf5 files found")
    print(f"[input] {len(files)} raw episode file(s)", flush=True)
    episodes = []
    for path in files:
        print(f"[load] {path}", flush=True)
        ep = load_raw_episode(path, args)
        print(
            f"[load] len={len(ep['state'])} state={ep['state'].shape} "
            f"pc={ep['point_cloud'].shape} img={ep['img'].shape}",
            flush=True,
        )
        episodes.append(ep)
    write_zarr(episodes, args.output, overwrite=args.overwrite)
    validate_zarr(args.output)


if __name__ == "__main__":
    main()
