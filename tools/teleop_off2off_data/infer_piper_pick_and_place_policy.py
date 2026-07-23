#!/usr/bin/env python3
"""Piper pick-and-place 7D policy 的安全推理脚本。

该脚本与 ``piper_rosbag_pick_place_to_rl100_zarr.py`` 使用同一套观测/动作约定：

* observation agent_pos: ``[joint_feedback_rad(6), gripper_width_m]``；
* action: ``[joint_target_rad(6), gripper_width_m]``，是绝对目标，不是增量；
* 从训练输出目录的 Hydra 配置自动读取 ``n_obs_steps``、``horizon``
  和 ``n_action_steps``，兼容不同长度的 action chunk。默认使用
  receding-horizon：每个控制周期重新推理，只执行当前 chunk 的第一步。

默认是 shadow 模式，只读机械臂/相机和打印动作；只有传入 ``--execute`` 并
输入确认短语才会使能 Piper。第一次上真机建议先运行 ``--offline-smoke``，
再运行 shadow，最后使用极低速、短时长的 execute。
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
TRAIN_ROOT = REPO_ROOT / "RL-100"
PYTORCH3D_ROOT = REPO_ROOT / "third_party/pytorch3d_simplified"
for import_path in (REPO_ROOT, TRAIN_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from tools.teleop_off2off_data import infer_piper_contact_policy as contact

# RL1003D 依赖这个仓库内的 ARM64 PyTorch3D 简化实现。contact 模块导入时
# 已将 PYTORCH3D_ROOT 加入 sys.path，这里不能再删除，否则 Hydra
# 实例化 rl_100.policy.rl100_3d 时会报 ModuleNotFoundError: pytorch3d。


DEFAULT_OUTPUT_DIR = TRAIN_ROOT / "data/outputs/piper_pick_and_place_chunk4_bc_cm_offline_seed42"
GRIPPER_LOWER_M = 0.0
GRIPPER_UPPER_M = 0.07
GRIPPER_RAW_PER_M = 1_000_000.0  # Piper GripperCtrl 单位是 0.001 mm
GRIPPER_EFFORT = 1000


class PiperPickPlaceStateReader:
    """读取六关节反馈、夹爪宽度和各自 SDK 时间戳。"""

    def __init__(self, piper: Any):
        self.piper = piper
        self.last_gripper_status: dict[str, bool] = {}

    def read(self) -> tuple[np.ndarray, float, float]:
        arm_msg = self.piper.GetArmJointMsgs()
        joint_state = arm_msg.joint_state
        raw = np.array(
            [
                joint_state.joint_1,
                joint_state.joint_2,
                joint_state.joint_3,
                joint_state.joint_4,
                joint_state.joint_5,
                joint_state.joint_6,
            ],
            dtype=np.float64,
        )
        joint_rad = (raw * contact.RAW_TO_RAD).astype(np.float32)
        if not np.all(np.isfinite(joint_rad)):
            raise RuntimeError(f"关节反馈包含 NaN/Inf: {joint_rad}")

        gripper_msg = self.piper.GetArmGripperMsgs()
        gripper_state = gripper_msg.gripper_state
        # SDK feedback 是 0.001 mm；训练 Zarr 第 7 维统一为总开口宽度（m）。
        gripper_width_m = float(gripper_state.grippers_angle) / GRIPPER_RAW_PER_M
        if not math.isfinite(gripper_width_m):
            raise RuntimeError(f"夹爪反馈包含 NaN/Inf: {gripper_width_m}")
        foc = gripper_state.foc_status
        self.last_gripper_status = {
            "voltage_too_low": bool(foc.voltage_too_low),
            "motor_overheating": bool(foc.motor_overheating),
            "driver_overcurrent": bool(foc.driver_overcurrent),
            "driver_overheating": bool(foc.driver_overheating),
            "sensor_status": bool(foc.sensor_status),
            "driver_error_status": bool(foc.driver_error_status),
            "driver_enable_status": bool(foc.driver_enable_status),
            "homing_status": bool(foc.homing_status),
        }
        fault_names = [
            name
            for name in (
                "voltage_too_low",
                "motor_overheating",
                "driver_overcurrent",
                "driver_overheating",
                "sensor_status",
                "driver_error_status",
            )
            if self.last_gripper_status[name]
        ]
        if fault_names:
            raise RuntimeError(f"Piper 夹爪反馈报告故障: {fault_names}")
        return (
            np.r_[joint_rad, np.float32(gripper_width_m)],
            float(getattr(arm_msg, "time_stamp", 0.0)),
            float(getattr(gripper_msg, "time_stamp", 0.0)),
        )


def resize_rgb_to_chw(rgb: np.ndarray, size: int = 84) -> np.ndarray:
    if rgb.shape != (480, 640, 3):
        raise RuntimeError(f"D435i RGB shape 异常: {rgb.shape}，预期 (480,640,3)")
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    return np.transpose(resized, (2, 0, 1)).astype(np.float32)


def depth_to_point_cloud(depth_raw: np.ndarray, intrinsics: tuple[float, ...], num_points: int = 512) -> np.ndarray:
    """与转换脚本一致：原始 16UC1 深度、米制换算、确定性均匀采样。"""
    if depth_raw.shape != (480, 640):
        raise RuntimeError(f"D435i depth shape 异常: {depth_raw.shape}，预期 (480,640)")
    depth_m = depth_raw.astype(np.float32) * 0.001
    valid = (
        (depth_raw > 0)
        & (depth_raw < 65535)
        & (depth_m >= 0.1)
        & (depth_m <= 2.0)
    )
    v, u = np.nonzero(valid)
    if len(u) == 0:
        raise RuntimeError("实时深度没有 0.1～2.0m 内的有效点")
    fx, fy, cx, cy = map(float, intrinsics)
    z = depth_m[v, u]
    points = np.stack(
        ((u.astype(np.float32) - cx) * z / fx,
         (v.astype(np.float32) - cy) * z / fy,
         z),
        axis=1,
    )
    if len(points) >= num_points:
        return points[np.linspace(0, len(points) - 1, num_points, dtype=np.int64)].astype(np.float32)
    repeats = math.ceil(num_points / len(points))
    return np.concatenate([points] * repeats, axis=0)[:num_points].astype(np.float32)


def preprocess_camera_frame(frame: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """D435i 返回 BGR，转换为训练时 RGB CHW；点云使用未对齐原始 depth。"""
    color_bgr = np.asarray(frame["color"])
    rgb = cv2.cvtColor(color_bgr[..., :3], cv2.COLOR_BGR2RGB)
    image = resize_rgb_to_chw(rgb)
    if frame.get("depth_aligned_to_color", True):
        raise RuntimeError("pick-and-place 必须使用未对齐 depth；请确认 align_depth_to_color=False")
    point_cloud = depth_to_point_cloud(
        np.asarray(frame["depth"]),
        tuple(frame["depth_intrinsics"]),
        num_points=512,
    )
    if not np.all(np.isfinite(point_cloud)):
        raise RuntimeError("实时点云包含 NaN/Inf")
    return image, point_cloud


def build_obs(history: deque[dict[str, np.ndarray]], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "agent_pos": torch.from_numpy(np.stack([x["agent_pos"] for x in history])).unsqueeze(0).to(device),
        "point_cloud": torch.from_numpy(np.stack([x["point_cloud"] for x in history])).unsqueeze(0).to(device),
        "image": torch.from_numpy(np.stack([x["image"] for x in history])).unsqueeze(0).to(device),
    }


def extract_action_chunk(
    action_dict: dict[str, torch.Tensor],
    expected_steps: int = 4,
    expected_dims: int = 7,
) -> np.ndarray:
    if "action" not in action_dict:
        raise RuntimeError(f"predict_action() 没有 action，keys={list(action_dict)}")
    action = action_dict["action"].detach().cpu().numpy()
    if action.ndim == 2:
        action = action[None]
    if action.ndim != 3 or action.shape[0] != 1 or action.shape[2] not in (6, 7):
        raise RuntimeError(
            f"策略 action 应为 [1,{expected_steps},{expected_dims}]，实际为 {action.shape}"
        )
    if action.shape[2] != expected_dims:
        raise RuntimeError(
            f"策略 action 维度应为{expected_dims}，实际为 {action.shape[2]}"
        )
    if action.shape[1] != expected_steps:
        raise RuntimeError(f"策略 chunk 长度应为 {expected_steps}，实际为 {action.shape[1]}")
    chunk = action[0].astype(np.float32)
    if not np.all(np.isfinite(chunk)):
        raise RuntimeError("策略 chunk 包含 NaN/Inf")
    return chunk


def training_stats(dataset) -> dict[str, np.ndarray]:
    replay = dataset.replay_buffer
    state = np.asarray(replay["state"][:], dtype=np.float32)
    action = np.asarray(replay["action"][:], dtype=np.float32)
    pcs = np.asarray(replay["point_cloud"][:: max(1, len(replay["point_cloud"]) // 2000)], dtype=np.float32)
    flat_pc = pcs.reshape(-1, 3)
    return {
        "state_min": state.min(axis=0),
        "state_max": state.max(axis=0),
        "action_min": action.min(axis=0),
        "action_max": action.max(axis=0),
        "action_delta_p99": np.percentile(np.abs(action - state), 99, axis=0),
        "pc_low": np.percentile(flat_pc, 0.1, axis=0),
        "pc_high": np.percentile(flat_pc, 99.9, axis=0),
    }


def point_cloud_outlier_fraction(
    point_cloud: np.ndarray,
    stats: dict[str, np.ndarray],
) -> float:
    """计算实时点云超出训练分位范围的比例，但不执行停止策略。"""

    outside = np.any(
        (point_cloud < stats["pc_low"][None])
        | (point_cloud > stats["pc_high"][None]),
        axis=1,
    )
    return float(outside.mean())


def safe_action(predicted: np.ndarray, current: np.ndarray, stats: dict[str, np.ndarray], args) -> tuple[np.ndarray, list[str]]:
    warnings: list[str] = []
    # 关节限位复用 contact 脚本的物理限位和训练分布 margin。
    lower = np.maximum(contact.PIPER_HARD_LOWER_RAD, stats["action_min"][:6] - args.dataset_margin_rad)
    upper = np.minimum(contact.PIPER_HARD_UPPER_RAD, stats["action_max"][:6] + args.dataset_margin_rad)
    if not args.skip_current_joint_range_check and (
        np.any(current[:6] < lower) or np.any(current[:6] > upper)
    ):
        raise RuntimeError("当前关节姿态超出训练分布安全范围，请先人工移动到示教初始姿态")
    if args.skip_current_joint_range_check:
        warnings.append("已跳过当前关节训练范围检查")
    target = predicted.copy()
    if np.any(target[:6] < lower) or np.any(target[:6] > upper):
        if not args.clip_actions:
            raise RuntimeError("策略关节目标超出训练范围，已停止；可显式传 --clip-actions")
        target[:6] = np.clip(target[:6], lower, upper)
        warnings.append("关节目标被裁剪到训练范围")
    max_joint_step = args.max_joint_speed_rad_s / args.rate
    delta_j = target[:6] - current[:6]
    if np.any(np.abs(delta_j) > max_joint_step):
        if not args.clip_actions:
            raise RuntimeError("策略关节单周期变化超过速度限制，已停止")
        target[:6] = current[:6] + np.clip(delta_j, -max_joint_step, max_joint_step)
        warnings.append("关节目标被速度限制器裁剪")

    if args.skip_gripper_safety_check:
        warnings.append("已跳过夹爪范围和速度检查")
    else:
        grip_lower = max(GRIPPER_LOWER_M, float(stats["action_min"][6]) - args.gripper_margin_m)
        grip_upper = min(GRIPPER_UPPER_M, float(stats["action_max"][6]) + args.gripper_margin_m)
        if not grip_lower <= current[6] <= grip_upper:
            raise RuntimeError(f"当前夹爪宽度 {current[6]:.4f}m 超出训练分布 [{grip_lower:.4f},{grip_upper:.4f}]m")
        if target[6] < grip_lower or target[6] > grip_upper:
            if not args.clip_actions:
                raise RuntimeError("策略夹爪目标超出训练范围，已停止；可显式传 --clip-actions")
            target[6] = np.clip(target[6], grip_lower, grip_upper)
            warnings.append("夹爪目标被裁剪到训练范围")
        max_grip_step = args.max_gripper_speed_m_s / args.rate
        if abs(float(target[6] - current[6])) > max_grip_step:
            if not args.clip_actions:
                raise RuntimeError("策略夹爪单周期变化超过速度限制，已停止")
            target[6] = current[6] + np.clip(target[6] - current[6], -max_grip_step, max_grip_step)
            warnings.append("夹爪目标被速度限制器裁剪")
    return target.astype(np.float32), warnings


def send_action(piper: Any, action: np.ndarray, speed_percent: int) -> None:
    joints_raw = np.rint(action[:6] * contact.RAD_TO_RAW).astype(np.int64)
    gripper_raw = int(round(float(action[6]) * GRIPPER_RAW_PER_M))
    piper.MotionCtrl_2(0x01, 0x01, int(speed_percent), 0x00)
    piper.JointCtrl(*[int(x) for x in joints_raw])
    piper.GripperCtrl(gripper_raw, GRIPPER_EFFORT, 0x01, 0)


def enable_gripper(
    piper: Any,
    reader: PiperPickPlaceStateReader,
    hold_width_m: float,
    require_homing: bool,
    timeout: float = 3.0,
) -> None:
    """Enable and hold the gripper, retrying SDK 0x01 until feedback confirms."""

    hold_raw = int(round(float(hold_width_m) * GRIPPER_RAW_PER_M))
    # 先读取真实反馈。已使能时绝不发 0x02，因为 Piper SDK 明确
    # 定义 0x02 为“失能清错”，会导致第二次推理启动时先把夹爪关掉。
    reader.read()
    status = reader.last_gripper_status
    if status.get("driver_enable_status") and (
        status.get("homing_status") or not require_homing
    ):
        piper.GripperCtrl(hold_raw, GRIPPER_EFFORT, 0x01, 0)
        print("[Piper] 夹爪已使能，保持当前宽度", flush=True)
        return

    # 仅在驱动未使能时清错一次，之后像 SDK piper_ctrl_gripper
    # demo 一样持续发送 0x01 使能/位置指令，而不是只发一帧。
    if not status.get("driver_enable_status"):
        piper.GripperCtrl(hold_raw, GRIPPER_EFFORT, 0x02, 0)
        time.sleep(0.1)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        piper.GripperCtrl(hold_raw, GRIPPER_EFFORT, 0x01, 0)
        time.sleep(0.02)
        reader.read()
        status = reader.last_gripper_status
        if status.get("driver_enable_status") and (
            status.get("homing_status") or not require_homing
        ):
            print(
                f"[Piper] 夹爪使能成功: enable=True, "
                f"homing={status.get('homing_status')}",
                flush=True,
            )
            return
    raise RuntimeError(
        "夹爪使能状态未就绪；请先按 Piper 流程检查驱动器，"
        f"当前状态={reader.last_gripper_status}"
    )


def offline_smoke(dataset, policy, device, use_cm: bool, expected_steps: int) -> None:
    sample = dataset[0]["obs"]
    obs = {k: v.unsqueeze(0).to(device) for k, v in sample.items()}
    with torch.no_grad():
        output = policy.predict_action(obs, deterministic=True, use_cm=use_cm)
    chunk = extract_action_chunk(output, expected_steps)
    print(f"[offline-smoke] 成功，反归一化 chunk shape={chunk.shape}", flush=True)
    print(f"[offline-smoke] 第一动作={np.round(chunk[0], 6)}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--policy-subdir", choices=["best", "bc", "best_cm"], default="best")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--offline-smoke", action="store_true")
    p.add_argument("--can", default="can0")
    p.add_argument("--piper-sdk-root", type=Path, default=None)
    p.add_argument("--rate", type=float, default=13.0)
    p.add_argument("--camera-fps", type=int, default=15)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument(
        "--chunk-exec-steps",
        type=int,
        default=1,
        help=(
            "每次策略推理后依次执行的 action chunk 步数，"
            "上限由权重配置中的 n_action_steps 决定。"
            "--rate 仍表示动作下发频率；默认 1 为逐步重新规划。"
        ),
    )
    p.add_argument("--execute", action="store_true")
    p.add_argument("--clip-actions", action="store_true")
    p.add_argument(
        "--skip-current-joint-range-check",
        action="store_true",
        help="允许当前关节姿态在训练范围外启动；仍保留策略目标和速度安全限制。",
    )
    p.add_argument("--dataset-margin-rad", type=float, default=0.03)
    p.add_argument("--gripper-margin-m", type=float, default=0.005)
    p.add_argument("--max-joint-speed-rad-s", type=float, default=0.15)
    p.add_argument("--max-gripper-speed-m-s", type=float, default=0.02)
    p.add_argument(
        "--skip-gripper-safety-check",
        action="store_true",
        help="跳过夹爪训练范围、当前宽度和速度检查；关节安全限制仍保留。",
    )
    p.add_argument("--speed-percent", type=int, default=10)
    p.add_argument("--reuse-diffusion-noise", action="store_true")
    p.add_argument("--diffusion-noise-seed", type=int, default=42)
    p.add_argument("--max-point-outlier-fraction", type=float, default=0.25)
    p.add_argument(
        "--skip-point-cloud-distribution-check",
        action="store_true",
        help="点云超出训练分布时仅记录 pc_out，不停止推理。",
    )
    p.add_argument("--state-stale-seconds", type=float, default=0.5)
    p.add_argument("--enable-timeout", type=float, default=5.0)
    p.add_argument(
        "--log",
        type=Path,
        default=TRAIN_ROOT / "data/piper_inference/pick_and_place_latest.json",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.rate <= 0 or args.rate > 20:
        raise ValueError("--rate 必须在 (0,20] Hz")
    if args.camera_fps < args.rate or args.max_steps <= 0:
        raise ValueError("camera-fps 必须不低于 rate，max-steps 必须为正")
    if args.chunk_exec_steps < 1:
        raise ValueError("--chunk-exec-steps 必须为正整数")
    if not 1 <= args.speed_percent <= 100:
        raise ValueError("--speed-percent 必须在 [1,100]")
    if args.rate > 15:
        print(
            "[时序警告] 当前 rate 高于采集相机的 15Hz；训练 transition 实际约 13Hz，"
            "20Hz 推理会产生观察历史和 action chunk 的时序分布偏移。",
            flush=True,
        )
    args.output_dir = args.output_dir.expanduser().resolve()
    cfg, dataset, policy, use_cm = contact.load_policy_and_dataset(args.output_dir, args.policy_subdir, args.device)
    if list(cfg.shape_meta.obs.agent_pos.shape) != [7] or list(cfg.shape_meta.action.shape) != [7]:
        raise RuntimeError(f"训练配置不是 7D：agent_pos={cfg.shape_meta.obs.agent_pos.shape}, action={cfg.shape_meta.action.shape}")
    n_obs_steps = int(cfg.n_obs_steps)
    n_action_steps = int(cfg.n_action_steps)
    horizon = int(cfg.horizon)
    if n_obs_steps < 1 or n_action_steps < 1:
        raise RuntimeError(
            f"训练配置的 n_obs_steps/n_action_steps 必须为正整数，"
            f"当前为 {n_obs_steps}/{n_action_steps}"
        )
    expected_horizon = n_obs_steps - 1 + n_action_steps
    if horizon != expected_horizon:
        raise RuntimeError(
            "当前推理脚本要求 horizon = n_obs_steps - 1 + n_action_steps，"
            f"当前为 {horizon} != {n_obs_steps}-1+{n_action_steps}="
            f"{expected_horizon}"
        )
    if args.chunk_exec_steps > n_action_steps:
        raise ValueError(
            f"--chunk-exec-steps={args.chunk_exec_steps} 超过模型 "
            f"n_action_steps={n_action_steps}"
        )
    device = torch.device(args.device)
    if args.offline_smoke:
        offline_smoke(
            dataset, policy, device, use_cm, expected_steps=n_action_steps
        )
        return

    from tools.teleop_off2off_data.realsense import RealSense

    # RealSense 必须在读取整套训练统计和启动 Piper CAN 后台线程之前
    # 打开。当前 ARM 主机上，training_stats() 之后再解析 UVC profile
    # 会稳定失败；这个顺序已用完整 BC 模型、D435i 和 can_right 验证。
    camera = RealSense(fps=args.camera_fps, color_width=640, color_height=480,
                       depth_width=640, depth_height=480, num_points=512,
                       point_cloud_frame="camera", align_depth_to_color=False)
    camera.start()
    try:
        stats = training_stats(dataset)
        print("[训练范围] state:", stats["state_min"], stats["state_max"], flush=True)
        print("[训练范围] action:", stats["action_min"], stats["action_max"], flush=True)
        C_PiperInterface_V2 = contact.import_piper_sdk(args.piper_sdk_root)
        piper = C_PiperInterface_V2(args.can)
        piper.ConnectPort()
        reader = PiperPickPlaceStateReader(piper)
    except Exception:
        camera.stop()
        raise
    print("[相机] RGB=RGB8预处理，Depth=未对齐原始深度，点云坐标=d435i_depth_optical_frame", flush=True)
    print("[模式]", "真机执行" if args.execute else "影子模式（不下发）", flush=True)
    print(
        f"[动作] 训练 chunk={n_action_steps}，每次推理依次执行前 "
        f"{args.chunk_exec_steps} 步；"
        f"动作下发 {args.rate:g}Hz，模型重规划约 "
        f"{args.rate / args.chunk_exec_steps:g}Hz",
        flush=True,
    )
    print("[人工] 输入 stop/Enter 停止并保持；输入 estop 发送 SDK 硬急停。", flush=True)
    if args.skip_current_joint_range_check:
        print("[安全警告] 已跳过当前关节训练范围检查，仅用于零点/人工初始姿态测试", flush=True)
    if args.skip_gripper_safety_check:
        print("[安全警告] 已跳过夹爪范围和速度检查，请确认夹爪机械限位与急停可用", flush=True)
    if args.skip_point_cloud_distribution_check:
        print("[安全警告] 已跳过点云分布停止检查，pc_out 仍会记录到终端和日志", flush=True)

    history = deque(maxlen=n_obs_steps)
    records = []
    operator = contact.OperatorConsole()
    robot_enabled = False
    hard_emergency = False
    stop_reason = "max_steps"
    last_state = None
    last_joint_ts = None
    last_grip_ts = None
    last_joint_fresh = time.monotonic()
    last_grip_fresh = time.monotonic()
    episode_noise = contact.make_episode_diffusion_noise(policy, device, args.diffusion_noise_seed) if args.reuse_diffusion_noise else None
    try:
        # 用实时数据填满训练配置中的观察窗口，不复制第一帧。
        for _ in range(n_obs_steps):
            state, _, _ = reader.read()
            frame = camera.get_frame(require_pc=False)
            image, pc = preprocess_camera_frame(frame)
            history.append({"agent_pos": state, "point_cloud": pc, "image": image})
            time.sleep(1.0 / args.rate)
        last_state = history[-1]["agent_pos"].copy()
        if args.execute:
            phrase = input("确认工作区安全后输入 EXECUTE PIPER 继续：").strip()
            if phrase != "EXECUTE PIPER":
                raise RuntimeError("确认短语不匹配，取消执行")
            contact.enable_robot_for_position_control(piper, args.speed_percent, args.enable_timeout)
            # 此时关节已使能；即使后续夹爪恢复失败，finally 也必须
            # 执行关节保持，不能因 robot_enabled 设置太晚而跳过。
            robot_enabled = True
            enable_gripper(
                piper,
                reader,
                float(last_state[6]),
                require_homing=not args.skip_gripper_safety_check,
            )
        operator.start()
        next_deadline = time.monotonic()
        active_chunk: np.ndarray | None = None
        inference_index = -1
        for step in range(args.max_steps):
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
            loop_start = time.monotonic()
            state, joint_ts, grip_ts = reader.read()
            if last_joint_ts != joint_ts:
                last_joint_fresh = loop_start
            elif loop_start - last_joint_fresh > args.state_stale_seconds:
                raise RuntimeError("Piper 关节反馈超过允许时间未更新")
            if last_grip_ts != grip_ts:
                last_grip_fresh = loop_start
            elif loop_start - last_grip_fresh > args.state_stale_seconds:
                raise RuntimeError("Piper 夹爪反馈超过允许时间未更新")
            last_joint_ts, last_grip_ts = joint_ts, grip_ts
            frame = camera.get_frame(require_pc=False)
            image, pc = preprocess_camera_frame(frame)
            if args.skip_point_cloud_distribution_check:
                fraction = point_cloud_outlier_fraction(pc, stats)
            else:
                fraction = contact.validate_point_cloud_distribution(
                    pc, contact.TrainingStats(stats["state_min"][:6], stats["state_max"][:6], stats["action_min"][:6], stats["action_max"][:6], stats["action_delta_p99"][:6], stats["pc_low"], stats["pc_high"]), args.max_point_outlier_fraction
                )
            history.append({"agent_pos": state, "point_cloud": pc, "image": image})
            chunk_step = step % args.chunk_exec_steps
            ran_inference = chunk_step == 0 or active_chunk is None
            if ran_inference:
                with torch.no_grad():
                    output = policy.predict_action(
                        build_obs(history, device),
                        deterministic=True,
                        use_cm=use_cm,
                        initial_noise=episode_noise,
                    )
                active_chunk = extract_action_chunk(
                    output, expected_steps=n_action_steps
                )
                inference_index += 1
                chunk_step = 0
            predicted_action = active_chunk[chunk_step]
            target, warnings = safe_action(predicted_action, state, stats, args)
            if robot_enabled:
                send_action(piper, target, args.speed_percent)
            record = {
                "step": step,
                "host_time": time.time(),
                "joint_gripper_state": state.tolist(),
                "predicted_chunk": active_chunk.tolist(),
                "chunk_step": chunk_step,
                "policy_inference": ran_inference,
                "inference_index": inference_index,
                "safe_action": target.tolist(),
                "point_outlier_fraction": fraction,
                "warnings": warnings,
                "executed": robot_enabled,
            }
            records.append(record)
            print(
                f"[step {step:04d}] state={np.round(state,4)} "
                f"chunk{chunk_step}={np.round(predicted_action,4)} "
                f"target={np.round(target,4)} pc_out={fraction:.1%} "
                f"{'INFER' if ran_inference else 'CACHED'} "
                f"{'EXEC' if robot_enabled else 'SHADOW'}",
                flush=True,
            )
            for warning in warnings:
                print(f"[安全警告] {warning}", flush=True)
            last_state = state.copy()
            next_deadline += 1.0 / args.rate
            remaining = next_deadline - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            elif time.monotonic() - next_deadline > 1.0 / args.rate:
                raise RuntimeError("推理循环连续落后超过一个控制周期")
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt_hold"
    except Exception:
        stop_reason = "exception_hold"
        raise
    finally:
        if robot_enabled and not hard_emergency and last_state is not None:
            try:
                # 停止时关节和夹爪都保持最后反馈姿态，不自动张开夹爪，避免方块掉落。
                for _ in range(10):
                    send_action(piper, last_state, args.speed_percent)
                    time.sleep(0.05)
            except Exception as exc:
                print(f"[保持] 发送保持动作失败: {exc}", file=sys.stderr, flush=True)
        try:
            camera.stop()
        except Exception as exc:
            print(f"[相机] 停止失败: {exc}", file=sys.stderr, flush=True)
        log_path = args.log.expanduser()
        if not log_path.is_absolute():
            log_path = REPO_ROOT / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps({"meta": {"output_dir": str(args.output_dir), "policy_subdir": args.policy_subdir, "use_cm": use_cm, "rate_hz": args.rate, "n_obs_steps": n_obs_steps, "horizon": horizon, "n_action_steps": n_action_steps, "chunk_exec_steps": args.chunk_exec_steps, "policy_replan_rate_hz": args.rate / args.chunk_exec_steps, "point_cloud_frame": "d435i_depth_optical_frame", "depth_aligned_to_color": False, "skip_current_joint_range_check": args.skip_current_joint_range_check, "skip_gripper_safety_check": args.skip_gripper_safety_check, "skip_point_cloud_distribution_check": args.skip_point_cloud_distribution_check, "stop_reason": stop_reason}, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[日志] 已保存: {log_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
