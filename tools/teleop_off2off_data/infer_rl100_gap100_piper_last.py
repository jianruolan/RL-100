#!/usr/bin/env python3
"""Deploy the RL100 Gap-100 Piper ``last.pt`` checkpoint.

This implementation follows RL100_GAP100_PIPER_INFERENCE.md: EMA weights,
checkpoint normalizer, two genuinely new RGB/state frames, a 100 ms gap gate,
the full 16-step ``action_pred`` chunk, and exponential temporal ensembling.
Shadow mode is the default. Real-robot commands require ``--execute`` and an
interactive confirmation phrase.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rl100-matplotlib")

SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_REPO_ROOT = Path(
    os.environ.get("RL100_REPO_ROOT", "/home/mtarch/Desktop/zyf/RL-100")
).expanduser().resolve()
for import_path in (DEFAULT_REPO_ROOT, DEFAULT_REPO_ROOT / "RL-100"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import cv2
import numpy as np
import torch

from tools.teleop_off2off_data import infer_piper_pick_and_place_policy as piper_base
from tools.teleop_off2off_data import infer_piper_contact_policy as contact
from tools.teleop_off2off_data.realsense import RealSense
from rl_100.model.common.normalizer import LinearNormalizer
from rl_100.model.vision.model_getter import get_resnet
from rl_100.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from rl_100.policy.rl100_2d import RL1002D
from rl_100.unidpg.diffusion_policy.diffusers_patch.ddim_with_logprob_dpok import (
    DDIMSchedulerExtended,
)
from rl_100.unidpg.diffusion_policy.diffusers_patch.lcm_scheduler import (
    LCMSchedulerExtended,
)


OBS_STEPS = 2
ACTION_STEPS = 16
MAX_GAP_NS = 100_000_000
ENSEMBLE_DECAY = 0.01
RGB_HEIGHT = 192
RGB_WIDTH = 256
CHECKPOINT_KEYS = {
    "model",
    "ema_model",
    "epoch",
    "steps",
    "train_args",
    "episode_ids",
    "shape_meta",
    "action_semantics",
    "upstream",
}


def load_checkpoint(
    path: Path,
    device: torch.device,
    normalizer_path: Path | None = None,
) -> tuple[RL1002D, dict]:
    """Rebuild the upstream RL1002D architecture and strictly load EMA."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"checkpoint应为dict，实际为{type(checkpoint)!r}")
    missing = CHECKPOINT_KEYS - checkpoint.keys()
    if missing:
        raise KeyError(f"checkpoint缺少字段: {sorted(missing)}")
    train_args = checkpoint["train_args"]
    required_identity = {
        "horizon": ACTION_STEPS,
        "obs_steps": OBS_STEPS,
    }
    for key, expected in required_identity.items():
        if int(train_args.get(key, -1)) != expected:
            raise RuntimeError(
                f"checkpoint训练身份不匹配: {key}={train_args.get(key)!r}, "
                f"期望{expected}"
            )
    if checkpoint["action_semantics"] != "absolute_joint_target":
        raise RuntimeError(
            f"不支持的动作语义: {checkpoint['action_semantics']!r}"
        )

    shape_meta = checkpoint["shape_meta"]
    image_shape = tuple(shape_meta["obs"]["image"]["shape"])
    state_shape = tuple(shape_meta["obs"]["agent_pos"]["shape"])
    action_shape = tuple(shape_meta["action"]["shape"])
    if image_shape != (3, RGB_HEIGHT, RGB_WIDTH):
        raise RuntimeError(f"RGB shape不匹配: {image_shape}")
    if state_shape != (7,) or action_shape != (7,):
        raise RuntimeError(
            f"state/action shape不匹配: {state_shape}/{action_shape}"
        )

    obs_encoder = MultiImageObsEncoder(
        shape_meta=shape_meta,
        rgb_model=get_resnet("resnet18", weights=None),
        resize_shape=(84, 112),
        crop_shape=None,
        random_crop=False,
        use_group_norm=True,
        share_rgb_model=False,
        imagenet_norm=True,
        use_agent_pos=True,
        use_vib=False,
        use_recon=False,
    )
    ddim_scheduler = DDIMSchedulerExtended(
        num_train_timesteps=100,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        set_alpha_to_one=True,
        steps_offset=1,
        prediction_type="epsilon",
        clip_std_min=0.0067,
        clip_std_max=None,
    )
    cm_scheduler = LCMSchedulerExtended(
        num_train_timesteps=100,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        set_alpha_to_one=True,
        steps_offset=1,
        prediction_type="sample",
        original_inference_steps=10,
        clip_std_min=0.0067,
        clip_std_max=None,
    )
    policy = RL1002D(
        shape_meta=shape_meta,
        obs_encoder=obs_encoder,
        cm_noise_scheduler=cm_scheduler,
        ddim_noise_scheduler=ddim_scheduler,
        scheduler_type="ddim",
        horizon=ACTION_STEPS,
        n_action_steps=ACTION_STEPS,
        n_obs_steps=OBS_STEPS,
        num_inference_steps=10,
        obs_as_global_cond=True,
        diffusion_step_embed_dim=128,
        down_dims=(128, 256, 512),
        kernel_size=5,
        n_groups=8,
        condition_type="film",
        use_down_condition=True,
        use_mid_condition=True,
        use_up_condition=True,
        model="sample_dp3",
        w_pc=False,
        use_agent_pos=True,
        action_norm=True,
        chunk_as_single_action=True,
        encoder_type="resnet",
        use_aug=False,
        ddim_inference_steps=10,
    )
    load_result = policy.load_state_dict(checkpoint["ema_model"], strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(f"EMA严格加载失败: {load_result}")
    # Prefer the explicitly supplied normalizer artifact.  This supports
    # checkpoints whose model and normalization statistics were saved
    # separately, while retaining checkpoint-embedded normalization as a
    # backward-compatible fallback.
    normalizer_state = checkpoint.get("normalizer")
    if normalizer_path is not None:
        normalizer_path = normalizer_path.expanduser().resolve()
        if not normalizer_path.is_file():
            raise FileNotFoundError(f"normalizer文件不存在: {normalizer_path}")
        external_state = torch.load(
            normalizer_path, map_location="cpu", weights_only=False
        )
        if not isinstance(external_state, dict):
            raise TypeError(
                f"normalizer文件应为state_dict，实际为{type(external_state)!r}"
            )
        if normalizer_state is not None:
            if set(external_state) != set(normalizer_state):
                raise RuntimeError("外部normalizer与checkpoint的key集合不一致")
            for key in external_state:
                if tuple(external_state[key].shape) != tuple(normalizer_state[key].shape):
                    raise RuntimeError(f"normalizer shape不一致: {key}")
        normalizer_state = external_state
        checkpoint["normalizer"] = external_state
        print(f"[归一化] 使用外部normalizer: {normalizer_path}", flush=True)
    if normalizer_state is None:
        raise KeyError("checkpoint和命令行都没有提供normalizer")
    normalizer = LinearNormalizer()
    normalizer.load_state_dict(normalizer_state)
    policy.set_normalizer(normalizer)
    # The package contract requires action_pred[:, :16]. Keeping the upstream
    # default no_pre_action=True would allocate only horizon-(obs_steps-1)=15.
    policy.no_pre_action = False
    policy.to(device).eval()
    print(
        f"[模型] EMA加载成功: epoch={checkpoint['epoch']} steps={checkpoint['steps']} "
        f"episodes={len(checkpoint['episode_ids'])}",
        flush=True,
    )
    return policy, checkpoint


def policy_stats(checkpoint: dict) -> dict[str, np.ndarray | str]:
    state = checkpoint["normalizer"]
    state_min = state["params_dict.agent_pos.input_stats.min"].cpu().numpy().copy()
    state_max = state["params_dict.agent_pos.input_stats.max"].cpu().numpy().copy()
    action_min = state["params_dict.action.input_stats.min"].cpu().numpy().copy()
    action_max = state["params_dict.action.input_stats.max"].cpu().numpy().copy()
    # Checkpoint agent_pos[6] is normalized feedback, while Piper safety uses m.
    state_min[6], state_max[6] = 0.05, 0.07
    return {
        "state_min": state_min,
        "state_max": state_max,
        "action_min": action_min,
        "action_max": action_max,
        "gripper_action_mode": "command",
    }


def make_agent_pos(state: np.ndarray) -> np.ndarray:
    result = np.asarray(state, dtype=np.float32).copy()
    result[6] = np.clip((result[6] - 0.05) / 0.02, 0.0, 1.0)
    return result


def camera_rgb(frame: dict) -> np.ndarray:
    color_bgr = np.asarray(frame["color"])
    if color_bgr.shape != (480, 640, 3):
        raise RuntimeError(f"RealSense RGB shape异常: {color_bgr.shape}")
    rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    # Camera acquisition produces the checkpoint contract's native 192x256
    # RGB tensor. The policy itself then performs its trained 84x112 resize.
    rgb = cv2.resize(rgb, (RGB_WIDTH, RGB_HEIGHT), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(rgb, dtype=np.uint8)


def predict_chunk(
    policy: RL1002D,
    history: deque[dict[str, np.ndarray]],
    device: torch.device,
) -> tuple[np.ndarray, float]:
    if len(history) != OBS_STEPS:
        raise RuntimeError(f"推理历史必须为{OBS_STEPS}帧，实际为{len(history)}")
    # The public deployment contract accepts RGB uint8.  RL1002D itself does
    # not cast image tensors before the torchvision ImageNet normalization,
    # so the reference helper's input boundary conversion is required here.
    # This is preprocessing, not reduced-precision inference: the policy and
    # all model parameters remain float32.
    image = np.stack([item["image"] for item in history])
    obs = {
        "image": torch.from_numpy(image).unsqueeze(0).to(
            device=device, dtype=torch.float32
        ).div_(255.0),
        "agent_pos": torch.from_numpy(
            np.stack([item["agent_pos"] for item in history])
        ).unsqueeze(0).to(device),
    }
    start = time.monotonic()
    with torch.inference_mode():
        result = policy.predict_action(obs, deterministic=True, use_cm=False)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.monotonic() - start
    if "action_pred" not in result:
        raise RuntimeError(f"模型输出缺少action_pred: {list(result)}")
    raw = result["action_pred"].detach().cpu().numpy()
    if raw.shape != (1, ACTION_STEPS, 7):
        raise RuntimeError(
            f"action_pred必须为[1,{ACTION_STEPS},7]，实际为{raw.shape}"
        )
    chunk = raw[0, :ACTION_STEPS].astype(np.float32)
    if not np.all(np.isfinite(chunk)):
        raise RuntimeError("模型chunk包含NaN/Inf")
    return chunk, elapsed


class AsyncGap100PolicyWorker:
    """Run FP32 diffusion in the background and keep only the newest input."""

    def __init__(self, policy: RL1002D, device: torch.device):
        self.policy = policy
        self.device = device
        self.condition = threading.Condition()
        self.pending = None
        self.latest_result = None
        self.next_id = 0
        self.stop_requested = False
        self.error: BaseException | None = None
        self.thread = threading.Thread(
            target=self._run, name="gap100-policy-worker", daemon=True
        )

    def start(self) -> None:
        self.thread.start()

    def submit(self, history: deque[dict[str, np.ndarray]], origin_time: float) -> int:
        # Copy the two small history records. Camera buffers must never be
        # mutated/reused while CUDA is consuming them.
        snapshot = deque(
            (
                {
                    "image": item["image"].copy(),
                    "agent_pos": item["agent_pos"].copy(),
                }
                for item in history
            ),
            maxlen=OBS_STEPS,
        )
        with self.condition:
            request_id = self.next_id
            self.next_id += 1
            # A request which has not started is replaced by the freshest pair.
            self.pending = (request_id, float(origin_time), snapshot)
            self.condition.notify()
            return request_id

    def latest(self):
        with self.condition:
            if self.error is not None:
                raise RuntimeError("异步Gap100策略推理失败") from self.error
            return self.latest_result

    def _run(self) -> None:
        try:
            while True:
                with self.condition:
                    while self.pending is None and not self.stop_requested:
                        self.condition.wait()
                    if self.stop_requested:
                        return
                    request_id, origin_time, history = self.pending
                    self.pending = None
                chunk, elapsed = predict_chunk(self.policy, history, self.device)
                with self.condition:
                    self.latest_result = (
                        request_id,
                        origin_time,
                        chunk,
                        elapsed,
                        time.monotonic(),
                    )
        except BaseException as exc:
            with self.condition:
                self.error = exc

    def stop(self) -> None:
        with self.condition:
            self.stop_requested = True
            self.condition.notify_all()
        self.thread.join(timeout=2.0)


class JointTrajectoryPlanner:
    """Acceleration-limited joint interpolation below the learned policy."""

    def __init__(
        self,
        initial: np.ndarray,
        joint_speed: float,
        joint_accel: float,
        gripper_speed: float,
        tracking_error: float | None,
    ):
        self.position = np.asarray(initial, dtype=np.float64).copy()
        self.velocity = np.zeros(6, dtype=np.float64)
        self.joint_speed = float(joint_speed)
        self.joint_accel = float(joint_accel)
        self.gripper_speed = float(gripper_speed)
        self.tracking_error = (
            None if tracking_error is None else float(tracking_error)
        )

    def hold(self, measured: np.ndarray, dt: float) -> np.ndarray:
        target = self.position.copy()
        target[:6] = np.asarray(measured, dtype=np.float64)[:6]
        target[6] = float(measured[6])
        return self.step(target, measured, dt)

    def step(
        self, target: np.ndarray, measured: np.ndarray, dt: float
    ) -> np.ndarray:
        dt = float(np.clip(dt, 1e-4, 0.1))
        target = np.asarray(target, dtype=np.float64)
        measured = np.asarray(measured, dtype=np.float64)

        # Never let the planned command run far ahead of feedback. This also
        # provides a bumpless recovery after a missed control deadline.
        if self.tracking_error is not None:
            tracking_delta = self.position[:6] - measured[:6]
            bad = np.abs(tracking_delta) > self.tracking_error
            if np.any(bad):
                self.position[:6][bad] = measured[:6][bad] + np.clip(
                    tracking_delta[bad], -self.tracking_error, self.tracking_error
                )
                self.velocity[bad] = 0.0

        error = target[:6] - self.position[:6]
        # The braking bound makes velocity approach zero before the target;
        # acceleration limiting prevents abrupt changes when policy targets jump.
        braking_speed = np.sqrt(2.0 * self.joint_accel * np.abs(error))
        desired_velocity = np.sign(error) * np.minimum(
            self.joint_speed, braking_speed
        )
        max_dv = self.joint_accel * dt
        self.velocity += np.clip(
            desired_velocity - self.velocity, -max_dv, max_dv
        )
        self.velocity = np.clip(
            self.velocity, -self.joint_speed, self.joint_speed
        )
        delta = self.velocity * dt
        overshoot = (np.sign(delta) == np.sign(error)) & (
            np.abs(delta) >= np.abs(error)
        )
        delta[overshoot] = error[overshoot]
        self.velocity[overshoot] = 0.0
        self.position[:6] += delta

        grip_delta = float(target[6] - self.position[6])
        grip_step = float(np.clip(
            grip_delta, -self.gripper_speed * dt, self.gripper_speed * dt
        ))
        self.position[6] += grip_step
        return self.position.astype(np.float32).copy()


def temporal_ensemble(
    chunks: deque[tuple[int, np.ndarray]], frame_index: int
) -> tuple[np.ndarray, int]:
    candidates, weights = [], []
    for origin, chunk in chunks:
        age = frame_index - origin
        if 0 <= age < ACTION_STEPS:
            candidates.append(chunk[age])
            weights.append(math.exp(-ENSEMBLE_DECAY * age))
    if not candidates:
        raise RuntimeError("temporal ensemble没有可用候选")
    weight_array = np.asarray(weights, dtype=np.float64)
    fused = np.average(np.stack(candidates), axis=0, weights=weight_array)
    return fused.astype(np.float32), len(candidates)


def temporal_ensemble_at_time(
    chunks: deque[tuple[float, np.ndarray]], now: float, action_rate: float = 15.0
) -> tuple[np.ndarray, int, int]:
    """Fuse chunks using their trained 15 Hz time axis, not control cycles."""

    candidates, weights, ages = [], [], []
    for origin_time, chunk in chunks:
        age = max(0, int(math.floor((now - origin_time) * action_rate)))
        if age < ACTION_STEPS:
            candidates.append(chunk[age])
            weights.append(math.exp(-ENSEMBLE_DECAY * age))
            ages.append(age)
    if not candidates:
        raise RuntimeError("temporal ensemble没有未过期候选")
    fused = np.average(
        np.stack(candidates),
        axis=0,
        weights=np.asarray(weights, dtype=np.float64),
    )
    return fused.astype(np.float32), len(candidates), max(ages)


def cadence_summary(timestamps: list[float]) -> dict:
    intervals = np.diff(np.asarray(timestamps, dtype=np.float64))
    intervals = intervals[np.isfinite(intervals) & (intervals > 0)]
    if len(intervals) == 0:
        return {"count": 0, "mean_hz": None, "median_hz": None}
    return {
        "count": int(len(intervals)),
        "mean_hz": float(1.0 / intervals.mean()),
        "median_hz": float(1.0 / np.median(intervals)),
        "p05_interval_s": float(np.percentile(intervals, 5)),
        "p95_interval_s": float(np.percentile(intervals, 95)),
    }


def offline_smoke(args) -> None:
    device = torch.device(args.device)
    policy, _ = load_checkpoint(args.checkpoint, device, args.normalizer)
    if args.rgb_npy is None:
        rgb = np.zeros((RGB_HEIGHT, RGB_WIDTH, 3), dtype=np.uint8)
        print("[offline-smoke] 未提供--rgb-npy，使用全零RGB做shape检查", flush=True)
    else:
        rgb = np.load(args.rgb_npy)
    if rgb.shape != (RGB_HEIGHT, RGB_WIDTH, 3) or rgb.dtype != np.uint8:
        raise RuntimeError(
            f"rgb-npy必须为uint8 [{RGB_HEIGHT},{RGB_WIDTH},3]，"
            f"实际为{rgb.shape}/{rgb.dtype}"
        )
    state = np.r_[np.asarray(args.joint, dtype=np.float32), args.gripper_width_m]
    item = {"image": rgb, "agent_pos": make_agent_pos(state)}
    history = deque(
        [
            {
                "image": item["image"].copy(),
                "agent_pos": item["agent_pos"].copy(),
            }
            for _ in range(OBS_STEPS)
        ],
        maxlen=OBS_STEPS,
    )
    chunk, elapsed = predict_chunk(policy, history, device)
    print(f"joint_target_rad {chunk[:, :6]}")
    print(f"gripper_state {chunk[:, 6]}")
    print(f"raw_chunk {chunk}")
    print(f"[offline-smoke] shape={chunk.shape} inference={elapsed*1000:.1f}ms")


def parse_args():
    default_checkpoint = SCRIPT_PATH.with_name("last.pt")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint)
    parser.add_argument(
        "--normalizer",
        type=Path,
        default=None,
        help="可选的外部LinearNormalizer state_dict；不传时使用checkpoint内嵌normalizer",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--offline-smoke", action="store_true")
    parser.add_argument("--rgb-npy", type=Path)
    parser.add_argument(
        "--joint", nargs=6, type=float,
        default=[-0.1, 1.2, -0.8, 0.1, 0.6, -0.2],
    )
    parser.add_argument("--gripper-width-m", type=float, default=0.068)
    parser.add_argument("--can", default="can_piper")
    parser.add_argument("--piper-sdk-root", type=Path, default=None)
    parser.add_argument("--rate", type=float, default=15.0)
    parser.add_argument("--camera-fps", type=int, default=15)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--max-camera-gap-ms", type=float, default=100.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--speed-percent", type=int, default=10)
    parser.add_argument("--max-joint-speed-rad-s", type=float, default=0.15)
    parser.add_argument(
        "--max-joint-accel-rad-s2", type=float, default=0.20,
        help="底层轨迹规划的关节加速度上限",
    )
    parser.add_argument(
        "--planner-tracking-error-rad", type=float, default=0.08,
        help="规划指令允许领先关节反馈的最大距离",
    )
    parser.add_argument(
        "--disable-planner-tracking-error",
        action="store_true",
        help="完全关闭规划位置相对反馈的回拉与速度清零逻辑。",
    )
    parser.add_argument("--max-gripper-speed-m-s", type=float, default=0.02)
    parser.add_argument(
        "--temporal-ensemble",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "融合多个时间对齐的历史action chunk；使用"
            "--no-temporal-ensemble时仅采用最新chunk。"
        ),
    )
    parser.add_argument("--clip-actions", action="store_true")
    parser.add_argument("--skip-current-joint-range-check", action="store_true")
    parser.add_argument("--skip-gripper-safety-check", action="store_true")
    parser.add_argument("--dataset-margin-rad", type=float, default=0.03)
    parser.add_argument("--gripper-margin-m", type=float, default=0.005)
    parser.add_argument("--gripper-open-threshold", type=float, default=0.55)
    parser.add_argument("--gripper-close-threshold", type=float, default=0.45)
    parser.add_argument("--state-stale-seconds", type=float, default=0.5)
    parser.add_argument("--enable-timeout", type=float, default=5.0)
    parser.add_argument(
        "--log", type=Path,
        default=DEFAULT_REPO_ROOT / "data/piper_inference/gap100_last.json",
    )
    return parser.parse_args()


def validate_args(args) -> None:
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if args.normalizer is not None:
        args.normalizer = args.normalizer.expanduser().resolve()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.normalizer is not None and not args.normalizer.is_file():
        raise FileNotFoundError(f"normalizer文件不存在: {args.normalizer}")
    if args.rate <= 0 or args.camera_fps <= 0:
        raise ValueError("--rate和--camera-fps必须为正数")
    if args.camera_fps != 15:
        print(
            f"[频率警告] checkpoint在15Hz图像轨迹上训练，当前camera="
            f"{args.camera_fps}Hz，视觉时间尺度与训练不一致。",
            flush=True,
        )
    if args.max_camera_gap_ms <= 0:
        raise ValueError("--max-camera-gap-ms必须为正数")
    if args.max_steps <= 0:
        raise ValueError("--max-steps必须为正")
    if not 1 <= args.speed_percent <= 100:
        raise ValueError("--speed-percent必须在[1,100]")
    if args.max_joint_speed_rad_s <= 0 or args.max_joint_accel_rad_s2 <= 0:
        raise ValueError("关节速度和加速度上限必须为正数")
    if args.max_gripper_speed_m_s <= 0:
        raise ValueError("夹爪速度上限必须为正数")
    if args.planner_tracking_error_rad <= 0:
        raise ValueError("--planner-tracking-error-rad必须为正数")
    if not 0 < args.gripper_close_threshold < args.gripper_open_threshold < 1:
        raise ValueError("夹爪迟滞阈值必须满足0<close<open<1")


def run_robot(args) -> None:
    device = torch.device(args.device)
    policy, checkpoint = load_checkpoint(args.checkpoint, device, args.normalizer)
    trained_gap_ms = float(checkpoint["train_args"].get("max_gap_ms", 0.0))
    if trained_gap_ms <= 0:
        raise RuntimeError("checkpoint缺少有效的train_args.max_gap_ms")
    if args.max_camera_gap_ms > trained_gap_ms:
        raise ValueError(
            f"--max-camera-gap-ms={args.max_camera_gap_ms:g}超过训练门限"
            f"{trained_gap_ms:g}ms"
        )
    print(
        f"[时序] 控制/下发={args.rate:g}Hz，相机/策略输入={args.camera_fps}Hz，"
        f"训练gap门限={trained_gap_ms:g}ms；仅新相机帧触发异步推理",
        flush=True,
    )
    stats = policy_stats(checkpoint)
    # Reuse the established safety function; it expects this threshold name.
    args.gripper_command_threshold = 0.5

    raw_camera = RealSense(
        fps=args.camera_fps,
        color_width=640,
        color_height=480,
        depth_width=640,
        depth_height=480,
        num_points=512,
        point_cloud_frame="camera",
        align_depth_to_color=False,
    )
    camera = piper_base.LatestFrameCamera(raw_camera)
    camera.start()
    piper = None
    policy_chunks: deque[tuple[float, np.ndarray]] = deque(maxlen=ACTION_STEPS)
    history: deque[dict[str, np.ndarray]] = deque(maxlen=OBS_STEPS)
    records = []
    control_times: list[float] = []
    send_times: list[float] = []
    inference_completion_times: list[float] = []
    inference_durations: list[float] = []
    operator = contact.OperatorConsole()
    robot_enabled = False
    hard_emergency = False
    last_state = None
    worker = None
    stop_reason = "max_steps"
    try:
        piper = piper_base.PiperProcessProxy(args.can, args.piper_sdk_root)
        piper.ConnectPort()
        reader = piper_base.PiperPickPlaceStateReader(piper)

        # Warm up with two genuinely new frames before enabling the robot.
        previous_timestamp_ns = None
        while len(history) < OBS_STEPS:
            state, _, _ = reader.read()
            frame = camera.get_frame(require_pc=False)
            timestamp_ns = int(round(float(frame["timestamp"]) * 1e9))
            if previous_timestamp_ns is not None and timestamp_ns <= previous_timestamp_ns:
                time.sleep(0.001)
                continue
            previous_timestamp_ns = timestamp_ns
            history.append({
                "timestamp_ns": np.int64(timestamp_ns),
                "image": camera_rgb(frame),
                "agent_pos": make_agent_pos(state),
            })
        _, warmup_elapsed = predict_chunk(policy, history, device)
        print(f"[CUDA] 预热完成: {warmup_elapsed*1000:.1f}ms", flush=True)

        if args.execute:
            phrase = input("确认工作区安全后输入 EXECUTE VISUAL PIPER 继续：").strip()
            if phrase not in {"EXECUTE VISUAL PIPER", "EXECUTE GAP100 PIPER"}:
                raise RuntimeError("确认短语不匹配，取消执行")
            contact.enable_robot_for_position_control(
                piper, args.speed_percent, args.enable_timeout
            )
            robot_enabled = True
            piper_base.enable_gripper(
                piper,
                reader,
                float(state[6]),
                require_homing=not args.skip_gripper_safety_check,
            )

        # Confirmation can make the warmup history stale. Policy inference is
        # asynchronous so the low-level planner can continue at --rate.
        history.clear()
        policy_chunks.clear()
        previous_timestamp_ns = None
        last_new_frame_host = time.monotonic()
        valid_result_after = last_new_frame_host
        camera_stale = False
        gripper_open = bool(float(state[6]) >= 0.06)
        last_joint_ts = last_grip_ts = None
        last_joint_fresh = last_grip_fresh = time.monotonic()
        last_result_id = -1
        latest_inference_elapsed = None
        latest_raw_chunk = None
        last_gap_ms = None
        safety_args = copy.copy(args)
        # Range checking stays enabled, but velocity shaping belongs exclusively
        # to the stateful planner below instead of safe_action's one-step clip.
        safety_args.max_joint_speed_rad_s = 1e6
        safety_args.max_gripper_speed_m_s = 1e6
        planner = JointTrajectoryPlanner(
            state,
            joint_speed=args.max_joint_speed_rad_s,
            joint_accel=args.max_joint_accel_rad_s2,
            gripper_speed=args.max_gripper_speed_m_s,
            tracking_error=(
                None
                if args.disable_planner_tracking_error
                else args.planner_tracking_error_rad
            ),
        )
        worker = AsyncGap100PolicyWorker(policy, device)
        worker.start()
        operator.start()
        period = 1.0 / args.rate
        previous_loop_time = time.monotonic()
        next_tick = previous_loop_time
        print(
            f"[模式] {'真机执行' if robot_enabled else 'shadow'} | "
            f"异步EMA | 底层{args.rate:g}Hz规划 | "
            f"temporal_ensemble={args.temporal_ensemble} | "
            f"v<={args.max_joint_speed_rad_s:g}rad/s | "
            f"a<={args.max_joint_accel_rad_s2:g}rad/s²",
            flush=True,
        )

        for step in range(args.max_steps):
            loop_start = time.monotonic()
            control_times.append(loop_start)
            dt = loop_start - previous_loop_time
            previous_loop_time = loop_start
            command = operator.poll()
            if command in {"estop", "e", "emergency"}:
                if robot_enabled:
                    contact.quick_stop(piper)
                    hard_emergency = True
                stop_reason = "hard_emergency"
                break
            if command in {"stop", "s", "q", "quit"}:
                stop_reason = "operator_hold"
                break

            state, joint_ts, grip_ts = reader.read()
            if last_joint_ts != joint_ts:
                last_joint_fresh = loop_start
            elif loop_start - last_joint_fresh > args.state_stale_seconds:
                raise RuntimeError("Piper关节反馈超时")
            if last_grip_ts != grip_ts:
                last_grip_fresh = loop_start
            elif loop_start - last_grip_fresh > args.state_stale_seconds:
                raise RuntimeError("Piper夹爪反馈超时")
            last_joint_ts, last_grip_ts = joint_ts, grip_ts
            last_state = state.copy()

            frame = camera.get_frame(require_pc=False)
            timestamp_ns = int(round(float(frame["timestamp"]) * 1e9))
            new_frame = (
                previous_timestamp_ns is None
                or timestamp_ns > previous_timestamp_ns
            )
            if new_frame:
                gap_ns = (
                    None if previous_timestamp_ns is None
                    else timestamp_ns - previous_timestamp_ns
                )
                last_gap_ms = None if gap_ns is None else gap_ns / 1e6
                previous_timestamp_ns = timestamp_ns
                last_new_frame_host = loop_start
                camera_stale = False
                if gap_ns is not None and gap_ns > int(args.max_camera_gap_ms * 1e6):
                    history.clear()
                    policy_chunks.clear()
                    valid_result_after = loop_start
                    print(
                        f"[时间门控] gap={gap_ns/1e6:.1f}ms>"
                        f"{args.max_camera_gap_ms:g}ms，清空history/chunks并减速保持",
                        flush=True,
                    )
                history.append({
                    "timestamp_ns": np.int64(timestamp_ns),
                    "image": camera_rgb(frame),
                    "agent_pos": make_agent_pos(state),
                })
                if len(history) == OBS_STEPS:
                    worker.submit(history, loop_start)
            elif loop_start - last_new_frame_host > args.max_camera_gap_ms / 1000.0:
                if not camera_stale:
                    history.clear()
                    policy_chunks.clear()
                    valid_result_after = loop_start
                    camera_stale = True
                    print(
                        "[时间门控] 超过上限未收到新相机帧，"
                        "清空策略并减速保持",
                        flush=True,
                    )

            result = worker.latest()
            if result is not None and result[0] != last_result_id:
                request_id, origin_time, chunk, infer_elapsed, completed_at = result
                last_result_id = request_id
                inference_completion_times.append(float(completed_at))
                inference_durations.append(float(infer_elapsed))
                if origin_time >= valid_result_after:
                    if not args.temporal_ensemble:
                        # Preserve latency compensation within the newest
                        # chunk, but never average it with older predictions.
                        policy_chunks.clear()
                    policy_chunks.append((origin_time, chunk))
                    latest_raw_chunk = chunk
                    latest_inference_elapsed = infer_elapsed

            warnings = []
            fused = None
            candidate_count = 0
            max_chunk_age = None
            safe_target = None
            if policy_chunks and not camera_stale:
                try:
                    fused, candidate_count, max_chunk_age = temporal_ensemble_at_time(
                        policy_chunks, loop_start
                    )
                except RuntimeError:
                    policy_chunks.clear()
            if fused is not None:
                continuous_gripper = float(fused[6])
                if continuous_gripper >= args.gripper_open_threshold:
                    gripper_open = True
                elif continuous_gripper <= args.gripper_close_threshold:
                    gripper_open = False
                fused_for_safety = fused.copy()
                fused_for_safety[6] = 1.0 if gripper_open else 0.0
                safe_target, warnings = piper_base.safe_action(
                    fused_for_safety, state, stats, safety_args
                )
                planned_action = planner.step(safe_target, state, dt)
            else:
                planned_action = planner.hold(state, dt)

            if robot_enabled:
                send_times.append(time.monotonic())
                piper_base.send_action(piper, planned_action, args.speed_percent)

            cycle_elapsed = time.monotonic() - loop_start
            record = {
                "step": step,
                "host_time": time.time(),
                "camera_timestamp_ns": timestamp_ns,
                "camera_new": new_frame,
                "camera_gap_ms": last_gap_ms,
                "camera_stale": camera_stale,
                "joint_gripper_state": state.tolist(),
                "new_raw_chunk": (
                    None if latest_raw_chunk is None else latest_raw_chunk.tolist()
                ),
                "ensemble_candidates": candidate_count,
                "max_chunk_age_15hz_steps": max_chunk_age,
                "fused_target_continuous": (
                    None if fused is None else fused.tolist()
                ),
                "safe_policy_target": (
                    None if safe_target is None else safe_target.tolist()
                ),
                "planned_action": planned_action.tolist(),
                "planned_joint_velocity": planner.velocity.tolist(),
                "gripper_open": gripper_open,
                "latest_inference_elapsed_s": latest_inference_elapsed,
                "cycle_elapsed_s": cycle_elapsed,
                "warnings": warnings,
                "executed": robot_enabled,
            }
            records.append(record)
            latest_raw_chunk = None
            if step % max(1, int(round(args.rate / 5.0))) == 0:
                infer_text = (
                    "pending" if latest_inference_elapsed is None
                    else f"{latest_inference_elapsed*1000:.1f}ms"
                )
                print(
                    f"[step {step:04d}] state={np.round(state[:6],3)} "
                    f"cmd={np.round(planned_action[:6],3)} "
                    f"vel={np.round(planner.velocity,3)} "
                    f"candidates={candidate_count} infer={infer_text} "
                    f"cycle={cycle_elapsed*1000:.1f}ms "
                    f"{'EXEC' if robot_enabled else 'SHADOW'}",
                    flush=True,
                )

            next_tick += period
            remaining = next_tick - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            elif remaining < -period:
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt_hold"
    except Exception:
        stop_reason = "exception_hold"
        raise
    finally:
        if worker is not None:
            try:
                worker.stop()
            except Exception as exc:
                print(f"[推理线程] 停止失败: {exc}", file=sys.stderr, flush=True)
        if robot_enabled and not hard_emergency and last_state is not None:
            try:
                for _ in range(10):
                    piper_base.send_action(piper, last_state, args.speed_percent)
                    time.sleep(0.05)
            except Exception as exc:
                print(f"[保持] 失败: {exc}", file=sys.stderr, flush=True)
        try:
            camera.stop()
        except Exception as exc:
            print(f"[相机] 停止失败: {exc}", file=sys.stderr, flush=True)
        if piper is not None:
            try:
                piper.close()
            except Exception as exc:
                print(f"[Piper] 停止失败: {exc}", file=sys.stderr, flush=True)
        log_path = args.log.expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": {
                "checkpoint": str(args.checkpoint),
                "ema": True,
                "rate_hz": args.rate,
                "obs_steps": OBS_STEPS,
                "action_steps": ACTION_STEPS,
                "max_gap_ms": args.max_camera_gap_ms,
                "ensemble_decay": ENSEMBLE_DECAY,
                "temporal_ensemble": args.temporal_ensemble,
                "async_policy": True,
                "planner_max_joint_speed_rad_s": args.max_joint_speed_rad_s,
                "planner_max_joint_accel_rad_s2": args.max_joint_accel_rad_s2,
                "planner_tracking_error_rad": args.planner_tracking_error_rad,
                "planner_tracking_error_enabled": (
                    not args.disable_planner_tracking_error
                ),
                "stop_reason": stop_reason,
                "cadence_monitor": {
                    "requested_control_rate_hz": args.rate,
                    "camera_fps": args.camera_fps,
                    "control_loop": cadence_summary(control_times),
                    "send_call": cadence_summary(send_times),
                    "inference_completion": cadence_summary(
                        inference_completion_times
                    ),
                    "inference_duration_s": {
                        "count": len(inference_durations),
                        "mean_s": (
                            float(np.mean(inference_durations))
                            if inference_durations else None
                        ),
                        "p95_s": (
                            float(np.percentile(inference_durations, 95))
                            if inference_durations else None
                        ),
                    },
                },
            },
            "records": records,
        }
        log_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[日志] 已保存: {log_path}", flush=True)
        cadence = payload["meta"]["cadence_monitor"]
        print(
            "[频率汇总] "
            f"control={cadence['control_loop']['mean_hz']}Hz "
            f"send={cadence['send_call']['mean_hz']}Hz "
            f"infer={cadence['inference_completion']['mean_hz']}Hz",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.offline_smoke:
        offline_smoke(args)
        return
    run_robot(args)


if __name__ == "__main__":
    main()
