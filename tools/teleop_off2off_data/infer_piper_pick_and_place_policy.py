#!/usr/bin/env python3
"""Piper pick-and-place 7D policy 的安全推理脚本。

该脚本与 ``piper_rosbag_pick_place_to_rl100_zarr.py`` 使用同一套观测/动作约定：

* observation agent_pos: ``[joint_feedback_rad(6), gripper_width_m]``；
* policy action: ``[joint_target_rad(6), gripper_open_command(0/1)]``；
  推理安全层会把第7维解码为物理夹爪宽度后再限速下发；
* 从训练输出目录的 Hydra 配置自动读取 ``n_obs_steps``、``horizon``
  和 ``n_action_steps``，兼容不同长度的 action chunk。默认使用
  receding-horizon：每个控制周期重新推理，只执行当前 chunk 的第一步。
* 可选 temporal ensemble 会融合多个历史 chunk 对当前时刻的关节预测；
  夹爪命令仍使用最新预测，避免开合边沿被连续平均。

默认是 shadow 模式，只读机械臂/相机和打印动作；只有传入 ``--execute`` 并
输入确认短语才会使能 Piper。第一次上真机建议先运行 ``--offline-smoke``，
再运行 shadow，最后使用极低速、短时长的 execute。
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import multiprocessing as mp
import os
import sys
import threading
import time
import traceback
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


def _piper_process_worker(conn, can_name: str, sdk_root: str | None) -> None:
    """Own Piper SDK and its high-rate Python threads in a separate process."""

    try:
        # All Piper SDK threads inherit this affinity.  Reserve CPU 0 for CAN
        # receive/parse work so CUDA launch threads are not preempted by the
        # 200 Hz feedback stream on Thor's remaining cores.
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, {0})
        sdk_path = Path(sdk_root) if sdk_root is not None else None
        interface_cls = contact.import_piper_sdk(sdk_path)
        piper = interface_cls(can_name)
        conn.send(("ready", None))
        while True:
            request = conn.recv()
            if request is None:
                break
            method_name, args, kwargs = request
            try:
                result = getattr(piper, method_name)(*args, **kwargs)
                conn.send(("ok", result))
            except BaseException:
                conn.send(("error", traceback.format_exc()))
    except BaseException:
        try:
            conn.send(("fatal", traceback.format_exc()))
        except Exception:
            pass
    finally:
        conn.close()


class PiperProcessProxy:
    """Synchronous Piper SDK proxy that keeps CAN parser threads off the GIL."""

    def __init__(self, can_name: str, sdk_root: Path | None):
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        self._conn = parent_conn
        self._process = ctx.Process(
            target=_piper_process_worker,
            args=(child_conn, can_name, str(sdk_root) if sdk_root else None),
            daemon=True,
        )
        self._process.start()
        child_conn.close()
        status, payload = self._conn.recv()
        if status != "ready":
            self.close()
            raise RuntimeError(f"Piper 子进程启动失败:\n{payload}")

    def __getattr__(self, method_name: str):
        if method_name.startswith("_"):
            raise AttributeError(method_name)

        def call(*args, **kwargs):
            self._conn.send((method_name, args, kwargs))
            status, payload = self._conn.recv()
            if status != "ok":
                raise RuntimeError(
                    f"Piper 子进程调用 {method_name} 失败:\n{payload}"
                )
            return payload

        return call

    def close(self) -> None:
        conn = getattr(self, "_conn", None)
        process = getattr(self, "_process", None)
        if conn is not None:
            try:
                conn.send(None)
            except (BrokenPipeError, EOFError, OSError):
                pass
            try:
                conn.close()
            except OSError:
                pass
            self._conn = None
        if process is not None:
            process.join(timeout=1.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
            self._process = None


class LatestFrameCamera:
    """Continuously capture RealSense frames and expose the newest complete one."""

    def __init__(self, camera):
        self._camera = camera
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest: dict[str, Any] | None = None
        self._error: BaseException | None = None

    def start(self) -> None:
        self._camera.start()
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="realsense-latest-frame",
            daemon=True,
        )
        self._thread.start()
        self.get_frame(timeout=5.0)

    def _capture_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                frame = self._camera.get_frame(require_pc=False)
                with self._condition:
                    self._latest = frame
                    self._condition.notify_all()
        except BaseException as exc:
            if not self._stop_event.is_set():
                with self._condition:
                    self._error = exc
                    self._condition.notify_all()

    def get_frame(self, require_pc: bool = False, timeout: float = 1.0):
        if require_pc:
            raise ValueError("LatestFrameCamera 仅缓存原始 RGB-D 帧")
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._latest is None and self._error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("等待 RealSense 后台首帧超时")
                self._condition.wait(remaining)
            if self._error is not None:
                raise RuntimeError("RealSense 后台采集失败") from self._error
            return self._latest

    def stop(self) -> None:
        self._stop_event.set()
        try:
            self._camera.stop()
        finally:
            if self._thread is not None:
                self._thread.join(timeout=1.0)
                self._thread = None


class AsyncPolicyWorker:
    """Run FP32 policy inference concurrently and keep only the newest request."""

    def __init__(
        self,
        policy,
        device,
        use_cm: bool,
        initial_noise,
        expected_action_steps: int,
    ):
        self._policy = policy
        self._device = device
        self._use_cm = use_cm
        self._initial_noise = initial_noise
        self._expected_action_steps = int(expected_action_steps)
        if self._expected_action_steps < 1:
            raise ValueError("expected_action_steps 必须为正整数")
        self._condition = threading.Condition()
        self._pending = None
        self._latest = None
        self._next_request_id = 0
        self._stop = False
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run, name="cuda-policy-worker", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def submit(self, obs, control_step: int) -> int:
        with self._condition:
            request_id = self._next_request_id
            self._next_request_id += 1
            # Drop an observation that has not started yet; executing stale
            # observations is worse than using the most recent complete action.
            self._pending = (request_id, int(control_step), obs)
            self._condition.notify()
            return request_id

    def latest(self):
        with self._condition:
            if self._error is not None:
                raise RuntimeError("异步策略推理失败") from self._error
            return self._latest

    def _run(self) -> None:
        try:
            while True:
                with self._condition:
                    while self._pending is None and not self._stop:
                        self._condition.wait()
                    if self._stop:
                        return
                    request_id, control_step, obs = self._pending
                    self._pending = None
                start = time.monotonic()
                with torch.inference_mode():
                    output = predict_policy(
                        self._policy,
                        obs,
                        self._device,
                        self._use_cm,
                        self._initial_noise,
                    )
                if self._device.type == "cuda":
                    torch.cuda.synchronize(self._device)
                chunk = extract_action_chunk(
                    output, expected_steps=self._expected_action_steps
                )
                elapsed = time.monotonic() - start
                with self._condition:
                    self._latest = (
                        request_id,
                        control_step,
                        chunk,
                        elapsed,
                        time.monotonic(),
                    )
        except BaseException as exc:
            with self._condition:
                self._error = exc

    def stop(self) -> None:
        with self._condition:
            self._stop = True
            self._condition.notify_all()
        self._thread.join(timeout=2.0)


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


def predict_policy(policy, obs, device, use_cm, initial_noise=None):
    """Run policy inference without changing the configured DDIM step count."""
    return policy.predict_action(
        obs, deterministic=True, use_cm=use_cm, initial_noise=initial_noise
    )


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


def temporal_ensemble_action(
    chunk_history: deque[tuple[int, np.ndarray]],
    current_step: int,
    decay: float,
) -> tuple[np.ndarray, list[int], list[float]]:
    """融合历史 chunk 中对 ``current_step`` 的重叠预测。

    ``decay`` 越大越偏向最近一次推理。只融合六个关节维度；第七维
    夹爪采用最新 chunk 的时间对齐预测，防止二值开合命令被平均。
    """

    candidates: list[np.ndarray] = []
    ages: list[int] = []
    for origin_step, chunk in chunk_history:
        chunk_offset = current_step - origin_step
        if 0 <= chunk_offset < len(chunk):
            candidates.append(chunk[chunk_offset])
            ages.append(chunk_offset)
    if not candidates:
        raise RuntimeError("temporal ensemble 没有当前时刻的有效动作候选")

    weights = np.exp(-float(decay) * np.asarray(ages, dtype=np.float64))
    weights /= weights.sum()
    stacked = np.stack(candidates).astype(np.float32)
    result = stacked[-1].copy()
    result[:6] = np.sum(stacked[:, :6] * weights[:, None], axis=0)
    if not np.all(np.isfinite(result)):
        raise RuntimeError("temporal ensemble 结果包含 NaN/Inf")
    return result, ages, weights.astype(np.float32).tolist()


def training_stats(dataset) -> dict[str, np.ndarray]:
    replay = dataset.replay_buffer
    state = np.asarray(replay["state"][:], dtype=np.float32)
    # Piper数据配置使用policy_action；不要硬编码成旧数据集的action键。
    action_key = str(getattr(dataset, "action_key", "action"))
    if action_key not in replay:
        raise KeyError(
            f"数据集动作键{action_key!r}不存在，实际keys={list(replay.keys())}"
        )
    action = np.asarray(replay[action_key][:], dtype=np.float32)
    pcs = np.asarray(replay["point_cloud"][:: max(1, len(replay["point_cloud"]) // 2000)], dtype=np.float32)
    flat_pc = pcs.reshape(-1, 3)
    # 新数据的第7维是0/1开合命令；旧数据可能直接保存0~0.07m宽度。
    gripper_action_mode = (
        "command" if float(action[:, 6].max()) > GRIPPER_UPPER_M + 0.1 else "width"
    )
    return {
        "state_min": state.min(axis=0),
        "state_max": state.max(axis=0),
        "action_min": action.min(axis=0),
        "action_max": action.max(axis=0),
        "action_delta_p99": np.percentile(np.abs(action - state), 99, axis=0),
        "pc_low": np.percentile(flat_pc, 0.1, axis=0),
        "pc_high": np.percentile(flat_pc, 99.9, axis=0),
        "action_key": action_key,
        "gripper_action_mode": gripper_action_mode,
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

    # 原始7D diffusion直接回归0/1夹爪命令。predict_action()反归一化后，
    # 这里只做部署语义解码；前6维关节目标完全不受影响。
    command_coded = stats.get("gripper_action_mode") == "command"
    if command_coded:
        raw_gripper_command = float(target[6])
        gripper_open = raw_gripper_command >= args.gripper_command_threshold
        target[6] = (
            GRIPPER_UPPER_M
            if gripper_open
            else GRIPPER_LOWER_M
        )
        warnings.append(
            f"夹爪命令{raw_gripper_command:.3f}解码为"
            f"{'张开' if gripper_open else '闭合'}"
        )
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
        if command_coded:
            # 命令空间是0/1，物理安全范围必须来自反馈宽度state，而非命令值。
            grip_lower = max(
                GRIPPER_LOWER_M,
                float(stats["state_min"][6]) - args.gripper_margin_m,
            )
            grip_upper = min(
                GRIPPER_UPPER_M,
                float(stats["state_max"][6]) + args.gripper_margin_m,
            )
        else:
            grip_lower = max(GRIPPER_LOWER_M, float(stats["action_min"][6]) - args.gripper_margin_m)
            grip_upper = min(GRIPPER_UPPER_M, float(stats["action_max"][6]) + args.gripper_margin_m)
        # Zarr float32中的0.07可能表示为0.0700000003，给物理边界留数值容差。
        grip_eps = 1e-6
        if current[6] < grip_lower - grip_eps or current[6] > grip_upper + grip_eps:
            raise RuntimeError(f"当前夹爪宽度 {current[6]:.4f}m 超出训练分布 [{grip_lower:.4f},{grip_upper:.4f}]m")
        if target[6] < grip_lower - grip_eps or target[6] > grip_upper + grip_eps:
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
        output = predict_policy(policy, obs, device, use_cm)
    chunk = extract_action_chunk(output, expected_steps)
    print(f"[offline-smoke] 成功，反归一化 chunk shape={chunk.shape}", flush=True)
    print(f"[offline-smoke] 第一动作={np.round(chunk[0], 6)}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument(
        "--policy-subdir",
        default="best",
        help=(
            "output-dir下的推理权重子目录，例如bc、best_val、"
            "milestone_25/50/75。"
        ),
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--ddim-steps",
        type=int,
        default=None,
        help=(
            "覆盖 BC 策略的 DDIM 推理步数（1-10）。步数越少越快，"
            "但与训练时的 10-step 采样分布偏差越大；best_cm 不受影响。"
        ),
    )
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
    p.add_argument(
        "--gripper-command-threshold",
        type=float,
        default=0.5,
        help="第7维为0/1命令时的开合阈值；不改变训练模型。",
    )
    p.add_argument("--max-joint-speed-rad-s", type=float, default=0.15)
    p.add_argument("--max-gripper-speed-m-s", type=float, default=0.02)
    p.add_argument(
        "--skip-gripper-safety-check",
        action="store_true",
        help="跳过夹爪训练范围、当前宽度和速度检查；关节安全限制仍保留。",
    )
    p.add_argument("--speed-percent", type=int, default=10)
    p.add_argument(
        "--reuse-diffusion-noise",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "在一个真机 episode 内复用固定扩散初始噪声，避免相同观测因"
            "每次重新采样噪声而输出不同目标；默认关闭，以保持训练/旧推理"
            "的逐次随机采样语义。需要固定采样时显式传入此参数。"
        ),
    )
    p.add_argument("--diffusion-noise-seed", type=int, default=42)
    p.add_argument(
        "--temporal-ensemble",
        action="store_true",
        help="融合历史action chunk对当前时刻的重叠关节预测；夹爪仍采用最新预测。",
    )
    p.add_argument(
        "--temporal-ensemble-decay",
        type=float,
        default=0.5,
        help="temporal ensemble指数衰减系数；越大越偏向最新预测，0表示等权。",
    )
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
    policy_subdir_path = Path(args.policy_subdir)
    if policy_subdir_path.is_absolute() or ".." in policy_subdir_path.parts:
        raise ValueError("--policy-subdir 必须是output-dir下的相对子目录")
    if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
        available_cpus = set(os.sched_getaffinity(0))
        if 0 in available_cpus and len(available_cpus) > 1:
            os.sched_setaffinity(0, available_cpus - {0})
    if args.rate <= 0 or args.rate > 60:
        raise ValueError("--rate 必须在 (0,60] Hz")
    if args.camera_fps < args.rate or args.max_steps <= 0:
        raise ValueError("camera-fps 必须不低于 rate，max-steps 必须为正")
    if args.chunk_exec_steps < 1:
        raise ValueError("--chunk-exec-steps 必须为正整数")
    if not 1 <= args.speed_percent <= 100:
        raise ValueError("--speed-percent 必须在 [1,100]")
    if not 0.0 < args.gripper_command_threshold < 1.0:
        raise ValueError("--gripper-command-threshold 必须在 (0,1)")
    if args.temporal_ensemble_decay < 0:
        raise ValueError("--temporal-ensemble-decay 不能为负数")
    if args.rate > 15:
        print(
            "[时序警告] 当前 rate 高于采集相机的 15Hz；训练 transition 实际约 13Hz，"
            "高频推理会产生观察历史和 action chunk 的时序分布偏移。",
            flush=True,
        )
    args.output_dir = args.output_dir.expanduser().resolve()

    # On Jetson/Thor, resolving the D435i UVC profile can fail after CUDA has
    # initialized a large policy in unified memory. Open and start the camera
    # before loading the dataset/model; keeping the already-started pipeline
    # alive while CUDA initializes is reliable on this host.
    camera = None
    if not args.offline_smoke:
        from tools.teleop_off2off_data.realsense import RealSense

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
        camera = LatestFrameCamera(raw_camera)
        camera.start()
    try:
        cfg, dataset, policy, use_cm = contact.load_policy_and_dataset(
            args.output_dir, args.policy_subdir, args.device
        )
    except Exception:
        if camera is not None:
            camera.stop()
        raise
    if args.ddim_steps is not None:
        if not 1 <= args.ddim_steps <= 10:
            raise ValueError("--ddim-steps 必须在 [1,10] 范围内")
        if use_cm:
            print("[采样] best_cm 使用固定 CM 步数，忽略 --ddim-steps", flush=True)
        elif not hasattr(policy, "ddim_inference_steps"):
            raise RuntimeError("当前策略没有 ddim_inference_steps 属性，无法覆盖采样步数")
        else:
            original_steps = int(policy.ddim_inference_steps)
            policy.ddim_inference_steps = int(args.ddim_steps)
            print(
                f"[采样] BC DDIM steps: {original_steps} -> "
                f"{policy.ddim_inference_steps}",
                flush=True,
            )
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

    try:
        stats = training_stats(dataset)
        print("[训练范围] state:", stats["state_min"], stats["state_max"], flush=True)
        print("[训练范围] action:", stats["action_min"], stats["action_max"], flush=True)
        print(
            f"[夹爪动作] key={stats['action_key']}，"
            f"mode={stats['gripper_action_mode']}，"
            f"threshold={args.gripper_command_threshold:g}",
            flush=True,
        )
        # Piper SDK runs several high-rate Python CAN/parser threads.  Keeping
        # them in this process makes small CUDA kernel launches wait for the
        # GIL and increases 10-step DDIM latency from ~16 ms to ~52 ms on Thor.
        piper = PiperProcessProxy(args.can, args.piper_sdk_root)
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
    print("[策略时序] 异步策略推理 + 后台最新相机帧", flush=True)
    print(
        "[扩散噪声] "
        + (
            f"episode内固定复用，seed={args.diffusion_noise_seed}"
            if args.reuse_diffusion_noise
            else "每次推理重新随机采样（可能导致目标抖动）"
        ),
        flush=True,
    )
    if args.temporal_ensemble:
        print(
            f"[动作] temporal ensemble=ON，最多融合{n_action_steps}个重叠预测，"
            f"decay={args.temporal_ensemble_decay:g}；夹爪使用最新预测",
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
    policy_worker: AsyncPolicyWorker | None = None
    try:
        # 用实时数据填满训练配置中的观察窗口，不复制第一帧。
        for _ in range(n_obs_steps):
            state, _, _ = reader.read()
            frame = camera.get_frame(require_pc=False)
            image, pc = preprocess_camera_frame(frame)
            history.append({"agent_pos": state, "point_cloud": pc, "image": image})
            time.sleep(1.0 / args.rate)
        last_state = history[-1]["agent_pos"].copy()

        # CUDA 的第一次 forward 会初始化 context、cuDNN/cuBLAS kernels，
        # 延迟可能远超一个实时周期。必须在启动 deadline 计时和使能机械臂
        # 之前预热，否则 CPU 能通过而 CUDA 在第一步被误判为持续落后。
        if device.type == "cuda":
            print("[CUDA] 开始推理预热...", flush=True)
            warmup_obs = build_obs(history, device)
            with torch.no_grad():
                for _ in range(2):
                    predict_policy(
                        policy, warmup_obs, device, use_cm, episode_noise
                    )
            torch.cuda.synchronize(device)
            print("[CUDA] 推理预热完成", flush=True)
        else:
            with torch.inference_mode():
                predict_policy(
                    policy, build_obs(history, device), device, use_cm, episode_noise
                )
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

        # 人工确认可能持续数秒。确认后重新采集完整观察窗口，并同步计算首个
        # chunk，禁止把确认前的陈旧视觉和动作作为 step 0 下发。
        history.clear()
        for _ in range(n_obs_steps):
            state, _, _ = reader.read()
            frame = camera.get_frame(require_pc=False)
            image, pc = preprocess_camera_frame(frame)
            history.append({"agent_pos": state, "point_cloud": pc, "image": image})
            time.sleep(1.0 / args.rate)
        last_state = history[-1]["agent_pos"].copy()
        with torch.inference_mode():
            initial_output = predict_policy(
                policy, build_obs(history, device), device, use_cm, episode_noise
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        initial_chunk = extract_action_chunk(
            initial_output, expected_steps=n_action_steps
        )
        policy_worker = AsyncPolicyWorker(
            policy,
            device,
            use_cm,
            episode_noise,
            expected_action_steps=n_action_steps,
        )
        policy_worker.start()
        operator.start()
        next_deadline = time.monotonic()
        active_chunk: np.ndarray = initial_chunk
        active_chunk_origin_step = 0
        chunk_history: deque[tuple[int, np.ndarray]] = deque(maxlen=n_action_steps)
        chunk_history.append((active_chunk_origin_step, active_chunk.copy()))
        inference_index = 0
        last_policy_result_id = -1
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
            replan_step = step % args.chunk_exec_steps
            requested_inference = replan_step == 0
            ran_inference = False
            infer_elapsed = 0.0
            obs_build_elapsed = 0.0
            if requested_inference:
                obs_build_start = time.monotonic()
                policy_obs = build_obs(history, device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                obs_build_elapsed = time.monotonic() - obs_build_start
                policy_worker.submit(policy_obs, control_step=step)
            latest_policy = policy_worker.latest()
            if latest_policy is not None and latest_policy[0] > last_policy_result_id:
                (
                    last_policy_result_id,
                    active_chunk_origin_step,
                    active_chunk,
                    infer_elapsed,
                    _,
                ) = latest_policy
                chunk_history.append(
                    (active_chunk_origin_step, active_chunk.copy())
                )
                inference_index += 1
                ran_inference = True
            # 记录异步结果年龄用于诊断。模型 chunk 的离散时间来自约13Hz
            # 的训练数据，不能按30Hz控制周期直接跳到后续动作，否则会改变
            # 权重原本的动作语义。
            policy_age_steps = max(0, step - active_chunk_origin_step)
            chunk_step = replan_step
            latest_action = active_chunk[chunk_step]
            ensemble_ages: list[int] = [0]
            ensemble_weights: list[float] = [1.0]
            if args.temporal_ensemble:
                predicted_action, ensemble_ages, ensemble_weights = (
                    temporal_ensemble_action(
                        chunk_history,
                        current_step=step,
                        decay=args.temporal_ensemble_decay,
                    )
                )
            else:
                predicted_action = latest_action
            target, warnings = safe_action(predicted_action, state, stats, args)
            if robot_enabled:
                send_action(piper, target, args.speed_percent)
            record = {
                "step": step,
                "host_time": time.time(),
                "joint_gripper_state": state.tolist(),
                "predicted_chunk": active_chunk.tolist(),
                "latest_time_aligned_action": latest_action.tolist(),
                "temporal_ensemble_action": predicted_action.tolist(),
                "temporal_ensemble_candidate_ages": ensemble_ages,
                "temporal_ensemble_weights": ensemble_weights,
                "chunk_step": chunk_step,
                "policy_chunk_origin_step": active_chunk_origin_step,
                "policy_age_steps": policy_age_steps,
                "replan_step": replan_step,
                "policy_inference": ran_inference,
                "policy_inference_requested": requested_inference,
                "policy_result_id": last_policy_result_id,
                "inference_index": inference_index,
                "safe_action": target.tolist(),
                "point_outlier_fraction": fraction,
                "warnings": warnings,
                "executed": robot_enabled,
                "cycle_elapsed_s": time.monotonic() - loop_start,
                "inference_elapsed_s": infer_elapsed if ran_inference else None,
                "obs_build_elapsed_s": obs_build_elapsed if ran_inference else None,
            }
            records.append(record)
            print(
                f"[step {step:04d}] state={np.round(state,4)} "
                f"chunk{chunk_step}={np.round(predicted_action,4)} "
                f"te={len(ensemble_weights)} "
                f"target={np.round(target,4)} pc_out={fraction:.1%} "
                f"{'INFER' if ran_inference else 'CACHED'} "
                f"{'EXEC' if robot_enabled else 'SHADOW'} "
                f"dt={record['cycle_elapsed_s']*1000:.1f}ms"
                + (f" infer={infer_elapsed*1000:.1f}ms" if ran_inference else "")
                + (f" obs={obs_build_elapsed*1000:.1f}ms" if ran_inference else ""),
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
        if policy_worker is not None:
            policy_worker.stop()
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
        try:
            piper.close()
        except Exception as exc:
            print(f"[Piper] 子进程停止失败: {exc}", file=sys.stderr, flush=True)
        log_path = args.log.expanduser()
        if not log_path.is_absolute():
            log_path = REPO_ROOT / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps({"meta": {"output_dir": str(args.output_dir), "policy_subdir": args.policy_subdir, "use_cm": use_cm, "rate_hz": args.rate, "camera_fps": args.camera_fps, "speed_percent": args.speed_percent, "max_joint_speed_rad_s": args.max_joint_speed_rad_s, "max_gripper_speed_m_s": args.max_gripper_speed_m_s, "n_obs_steps": n_obs_steps, "horizon": horizon, "n_action_steps": n_action_steps, "chunk_exec_steps": args.chunk_exec_steps, "policy_replan_rate_hz": args.rate / args.chunk_exec_steps, "reuse_diffusion_noise": args.reuse_diffusion_noise, "diffusion_noise_seed": args.diffusion_noise_seed, "temporal_ensemble": args.temporal_ensemble, "temporal_ensemble_decay": args.temporal_ensemble_decay, "temporal_ensemble_gripper": "latest_prediction", "point_cloud_frame": "d435i_depth_optical_frame", "depth_aligned_to_color": False, "skip_current_joint_range_check": args.skip_current_joint_range_check, "skip_gripper_safety_check": args.skip_gripper_safety_check, "skip_point_cloud_distribution_check": args.skip_point_cloud_distribution_check, "stop_reason": stop_reason}, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[日志] 已保存: {log_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
