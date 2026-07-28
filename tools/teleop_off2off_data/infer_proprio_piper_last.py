#!/usr/bin/env python3
"""Run the chunk-4 Piper pick-and-place DP3 checkpoint on the real robot.

This file is deliberately self-contained at the deployment layer: model
construction, RealSense preprocessing, Piper communication, safety checks,
trajectory limiting, asynchronous inference and temporal ensembling all live
here.  It does not import another inference program.
"""
from __future__ import annotations

import argparse
import json
import math
import select
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
RL100_ROOT = REPO_ROOT / "RL-100"
for source_root in (RL100_ROOT, Path(__file__).resolve().parent):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from hydra.utils import instantiate
from realsense import RealSense


DEFAULT_RUN_DIR = (
    RL100_ROOT
    / "data/outputs/"
    "piper_pick_and_place_augmented_chunk4_control_clean_dp3_"
    "episode10_bs64_epoch4000_seed42"
)
DEFAULT_CHECKPOINT = DEFAULT_RUN_DIR / "checkpoints/latest.ckpt"
DEFAULT_OFFLINE_ZARR = (
    RL100_ROOT / "data/piper_pick_and_place_augmented_chunk4_control_clean.zarr"
)

OBS_STEPS = 3
ACTION_STEPS = 4
TRAINED_RATE = 15.0
ENSEMBLE_DECAY = 0.01
NUM_POINTS = 512
MIN_DEPTH_M = 0.1
MAX_DEPTH_M = 2.0

# Piper's documented joint limits, with no deployment-side widening.
PIPER_HARD_LOWER_RAD = np.deg2rad([-150.0, 0.0, -170.0, -100.0, -70.0, -120.0])
PIPER_HARD_UPPER_RAD = np.deg2rad([150.0, 180.0, 0.0, 100.0, 70.0, 120.0])


def load_checkpoint(path: Path, device: torch.device):
    """Instantiate the policy from the checkpoint's own Hydra config."""
    import dill

    payload = torch.load(path, map_location="cpu", pickle_module=dill)
    required = {"cfg", "state_dicts"}
    missing = required - payload.keys()
    if missing:
        raise KeyError(f"checkpoint缺少字段: {sorted(missing)}")
    cfg = payload["cfg"]
    expected = {"horizon": 6, "n_obs_steps": 3, "n_action_steps": 4}
    for key, value in expected.items():
        if int(cfg[key]) != value:
            raise RuntimeError(f"训练配置不匹配: {key}={cfg[key]!r}, 期望{value}")
    obs_meta = cfg.shape_meta.obs
    if tuple(obs_meta.agent_pos.shape) != (7,):
        raise RuntimeError(f"agent_pos维度错误: {obs_meta.agent_pos.shape}")
    if tuple(obs_meta.point_cloud.shape) != (NUM_POINTS, 3):
        raise RuntimeError(f"point_cloud维度错误: {obs_meta.point_cloud.shape}")
    if tuple(cfg.shape_meta.action.shape) != (7,):
        raise RuntimeError(f"action维度错误: {cfg.shape_meta.action.shape}")
    if cfg.policy.encoder_type != "dp3" or cfg.policy.model != "dp3":
        raise RuntimeError(
            f"仅支持本实验DP3 checkpoint，实际encoder/model="
            f"{cfg.policy.encoder_type}/{cfg.policy.model}"
        )
    if cfg.policy.backbone is not None:
        raise RuntimeError("该脚本按本实验backbone=null实现；checkpoint却需要RGB backbone")

    policy = instantiate(cfg.policy)
    state_dicts = payload["state_dicts"]
    state = state_dicts.get("ema_model")
    if state is None:
        raise KeyError("checkpoint没有state_dicts.ema_model")
    result = policy.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"EMA严格加载失败: {result}")
    policy.no_pre_action = bool(cfg.no_pre_action)
    policy.to(device).eval()
    print(
        f"[模型] EMA加载成功: {path} | h={cfg.horizon} "
        f"obs={cfg.n_obs_steps} action={cfg.n_action_steps}",
        flush=True,
    )
    return policy, payload


def policy_stats(policy):
    normalizer = policy.normalizer
    return {
        "state_min": normalizer["agent_pos"].get_input_stats()["min"].cpu().numpy().copy(),
        "state_max": normalizer["agent_pos"].get_input_stats()["max"].cpu().numpy().copy(),
        "action_min": normalizer["action"].get_input_stats()["min"].cpu().numpy().copy(),
        "action_max": normalizer["action"].get_input_stats()["max"].cpu().numpy().copy(),
    }


def depth_to_training_point_cloud(
    depth_raw: np.ndarray,
    depth_scale: float,
    intrinsics,
    num_points: int = NUM_POINTS,
) -> np.ndarray:
    """Exactly reproduce the rosbag-to-zarr depth point-cloud conversion."""
    depth_m = depth_raw.astype(np.float32) * np.float32(depth_scale)
    valid = (
        (depth_raw > 0)
        & (depth_raw < 65535)
        & (depth_m >= MIN_DEPTH_M)
        & (depth_m <= MAX_DEPTH_M)
    )
    v, u = np.nonzero(valid)
    if len(u) == 0:
        raise RuntimeError("RealSense深度帧在0.1--2.0m内没有有效像素")
    z = depth_m[v, u]
    fx, fy, cx, cy = map(float, intrinsics)
    points = np.stack(
        ((u.astype(np.float32) - cx) * z / fx,
         (v.astype(np.float32) - cy) * z / fy,
         z),
        axis=1,
    ).astype(np.float32)
    if len(points) >= num_points:
        indices = np.linspace(0, len(points) - 1, num_points, dtype=np.int64)
        return points[indices]
    repeats = math.ceil(num_points / len(points))
    return np.concatenate([points] * repeats, axis=0)[:num_points]


def make_observation(state: np.ndarray, frame: dict) -> dict:
    pc = depth_to_training_point_cloud(
        frame["depth"], frame["depth_scale"], frame["depth_intrinsics"]
    )
    return {"agent_pos": state.astype(np.float32).copy(), "point_cloud": pc}


def predict(policy, history, device):
    if len(history) != OBS_STEPS:
        raise RuntimeError(f"history必须有{OBS_STEPS}帧，实际{len(history)}")
    obs = {
        key: torch.from_numpy(np.stack([item[key] for item in history]))
        .unsqueeze(0).to(device)
        for key in ("point_cloud", "agent_pos")
    }
    start = time.monotonic()
    with torch.inference_mode():
        out = policy.predict_action(obs, deterministic=True, use_cm=False)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.monotonic() - start
    raw = out["action"].detach().cpu().numpy()
    if raw.shape != (1, ACTION_STEPS, 7):
        raise RuntimeError(f"action应为[1,{ACTION_STEPS},7]，实际为{raw.shape}")
    chunk = raw[0].astype(np.float32)
    if not np.all(np.isfinite(chunk)):
        raise RuntimeError("模型输出包含NaN/Inf")
    return chunk, elapsed


def ensemble(chunks, now, decay=ENSEMBLE_DECAY):
    vals, weights, ages = [], [], []
    for origin, chunk in chunks:
        age = max(0, int(math.floor((now - origin) * TRAINED_RATE)))
        if age < ACTION_STEPS:
            vals.append(chunk[age])
            weights.append(math.exp(-float(decay) * age))
            ages.append(age)
    if not vals:
        raise RuntimeError("没有可用动作chunk")
    fused = np.average(np.stack(vals), axis=0, weights=np.asarray(weights))
    return fused.astype(np.float32), len(vals), max(ages)


class AsyncPolicyWorker:
    """Run diffusion asynchronously, replacing queued work with fresh input."""

    def __init__(self, policy, device):
        self.policy, self.device = policy, device
        self.condition = threading.Condition()
        self.pending = self.latest_result = self.error = None
        self.next_id = 0
        self.stop_requested = False
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def submit(self, history, origin_time):
        snapshot = deque(
            ({k: v.copy() for k, v in item.items()} for item in history),
            maxlen=OBS_STEPS,
        )
        with self.condition:
            request_id = self.next_id
            self.next_id += 1
            self.pending = (request_id, float(origin_time), snapshot)
            self.condition.notify()
            return request_id

    def clear_pending(self):
        with self.condition:
            self.pending = None

    def latest(self):
        with self.condition:
            if self.error is not None:
                raise RuntimeError("异步DP3推理失败") from self.error
            return self.latest_result

    def _run(self):
        try:
            while True:
                with self.condition:
                    while self.pending is None and not self.stop_requested:
                        self.condition.wait()
                    if self.stop_requested:
                        return
                    request_id, origin, history = self.pending
                    self.pending = None
                chunk, elapsed = predict(self.policy, history, self.device)
                with self.condition:
                    self.latest_result = (
                        request_id, origin, chunk, elapsed, time.monotonic()
                    )
        except BaseException as exc:
            with self.condition:
                self.error = exc

    def stop(self):
        with self.condition:
            self.stop_requested = True
            self.condition.notify_all()
        self.thread.join(timeout=2.0)


class PiperRobot:
    """Minimal Piper SDK joint-position transport with SI units at its boundary."""

    def __init__(self, can_name: str, sdk_root: Path):
        sdk_root = sdk_root.expanduser().resolve()
        if not sdk_root.is_dir():
            raise FileNotFoundError(f"Piper SDK不存在: {sdk_root}")
        sys.path.insert(0, str(sdk_root))
        from piper_sdk import C_PiperInterface_V2

        self.arm = C_PiperInterface_V2(can_name)
        self.arm.ConnectPort()

    def read(self):
        joint_msg = self.arm.GetArmJointMsgs()
        grip_msg = self.arm.GetArmGripperMsgs()
        j = joint_msg.joint_state
        raw = np.asarray(
            [j.joint_1, j.joint_2, j.joint_3, j.joint_4, j.joint_5, j.joint_6],
            dtype=np.float64,
        )
        joints = raw * (0.001 * np.pi / 180.0)
        width = float(grip_msg.gripper_state.grippers_angle) * 1e-6
        state = np.r_[joints, width].astype(np.float32)
        return state, float(joint_msg.time_stamp), float(grip_msg.time_stamp)

    def enable(self, speed_percent: int, timeout: float):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.arm.EnablePiper():
                self.arm.MotionCtrl_2(0x01, 0x01, speed_percent, 0x00)
                return
            time.sleep(0.01)
        raise RuntimeError("Piper使能超时")

    def enable_gripper(self, current_width: float):
        # First command the measured width, preventing a jump on enable.
        self.arm.GripperCtrl(int(round(current_width * 1e6)), 1000, 0x01, 0)

    def send(self, action: np.ndarray, speed_percent: int):
        raw_joint = np.rint(np.rad2deg(action[:6]) * 1000.0).astype(np.int64)
        self.arm.MotionCtrl_2(0x01, 0x01, speed_percent, 0x00)
        self.arm.JointCtrl(*(int(x) for x in raw_joint))
        self.arm.GripperCtrl(int(round(float(action[6]) * 1e6)), 1000, 0x01, 0)

    def quick_stop(self):
        self.arm.EmergencyStop(0x01)

    def close(self):
        close = getattr(self.arm, "DisconnectPort", None)
        if close is not None:
            close()


class OperatorConsole:
    def poll(self):
        if not sys.stdin.isatty():
            return None
        readable, _, _ = select.select([sys.stdin], [], [], 0)
        return sys.stdin.readline().strip().lower() if readable else None


class JointTrajectoryPlanner:
    """Velocity/acceleration limited SI-unit command generator."""

    def __init__(self, state, max_speed, max_accel, max_gripper_speed, tracking_error):
        self.command = state.astype(np.float64).copy()
        self.velocity = np.zeros(6, dtype=np.float64)
        self.max_speed = float(max_speed)
        self.max_accel = float(max_accel)
        self.max_gripper_speed = float(max_gripper_speed)
        self.tracking_error = float(tracking_error)

    def step(self, target, feedback, dt):
        dt = float(np.clip(dt, 1e-4, 0.2))
        if np.max(np.abs(self.command[:6] - feedback[:6])) > self.tracking_error:
            self.command[:6] = feedback[:6]
            self.velocity[:] = 0.0
        desired_v = np.clip((target[:6] - self.command[:6]) / dt, -self.max_speed, self.max_speed)
        dv = np.clip(desired_v - self.velocity, -self.max_accel * dt, self.max_accel * dt)
        self.velocity = np.clip(self.velocity + dv, -self.max_speed, self.max_speed)
        self.command[:6] += self.velocity * dt
        dg = np.clip(
            target[6] - self.command[6],
            -self.max_gripper_speed * dt,
            self.max_gripper_speed * dt,
        )
        self.command[6] += dg
        return self.command.astype(np.float32).copy()

    def hold(self, feedback, dt):
        self.command = feedback.astype(np.float64).copy()
        self.velocity[:] = 0.0
        return feedback.astype(np.float32).copy()


def safe_action(predicted, current, stats, args):
    predicted = np.asarray(predicted, dtype=np.float64).copy()
    if not np.all(np.isfinite(predicted)):
        raise RuntimeError("动作包含NaN/Inf")
    lower = np.maximum(
        PIPER_HARD_LOWER_RAD,
        stats["action_min"][:6] - args.dataset_margin_rad,
    )
    upper = np.minimum(
        PIPER_HARD_UPPER_RAD,
        stats["action_max"][:6] + args.dataset_margin_rad,
    )
    current_lower = np.maximum(
        PIPER_HARD_LOWER_RAD, lower - args.startup_joint_margin_rad
    )
    current_upper = np.minimum(
        PIPER_HARD_UPPER_RAD, upper + args.startup_joint_margin_rad
    )
    if not args.skip_current_joint_range_check and (
        np.any(current[:6] < current_lower) or np.any(current[:6] > current_upper)
    ):
        raise RuntimeError("当前关节反馈超出训练范围/物理限位")
    warnings = []
    outside = np.any(predicted[:6] < lower) or np.any(predicted[:6] > upper)
    if outside and not args.clip_actions:
        raise RuntimeError("模型关节目标超出训练范围；只可人工确认后使用--clip-actions")
    if outside:
        predicted[:6] = np.clip(predicted[:6], lower, upper)
        warnings.append("关节目标已裁剪到训练范围")

    command = float(predicted[6])
    if command < -args.gripper_command_margin or command > 1 + args.gripper_command_margin:
        if not args.clip_actions:
            raise RuntimeError(f"夹爪命令超出[0,1]: {command:.4f}")
        warnings.append("夹爪命令已裁剪到[0,1]")
    command = float(np.clip(command, 0.0, 1.0))
    target = predicted.copy()
    target[6] = args.gripper_open_width_m if command >= 0.5 else args.gripper_close_width_m
    return target.astype(np.float32), warnings


def offline_smoke(args):
    import zarr

    device = torch.device(args.device)
    policy, _ = load_checkpoint(args.checkpoint, device)
    root = zarr.open(str(args.offline_zarr), mode="r")
    end = int(args.offline_index) + OBS_STEPS
    if args.offline_index < 0 or end > root["data/state"].shape[0]:
        raise IndexError("--offline-index超出数据集")
    history = deque(maxlen=OBS_STEPS)
    for i in range(args.offline_index, end):
        history.append({
            "agent_pos": root["data/state"][i].astype(np.float32),
            "point_cloud": root["data/point_cloud"][i].astype(np.float32),
        })
    chunk, elapsed = predict(policy, history, device)
    print("action_chunk", chunk)
    print(f"[offline-smoke] shape={chunk.shape} inference={elapsed*1000:.1f}ms")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--can", default="can_piper")
    p.add_argument(
        "--piper-sdk-root", type=Path,
        default=REPO_ROOT.parent / "piper_sdk",
    )
    p.add_argument("--rate", type=float, default=TRAINED_RATE)
    p.add_argument("--camera-fps", type=int, default=30)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--max-state-gap-ms", type=float, default=150.0)
    p.add_argument("--ensemble-decay", type=float, default=ENSEMBLE_DECAY)
    p.add_argument("--speed-percent", type=int, default=10)
    p.add_argument("--max-joint-speed-rad-s", type=float, default=0.15)
    p.add_argument("--max-joint-accel-rad-s2", type=float, default=0.20)
    p.add_argument("--max-gripper-speed-m-s", type=float, default=0.02)
    p.add_argument("--planner-tracking-error-rad", type=float, default=0.08)
    p.add_argument("--dataset-margin-rad", type=float, default=0.03)
    p.add_argument("--startup-joint-margin-rad", type=float, default=0.0)
    p.add_argument("--gripper-command-margin", type=float, default=0.05)
    p.add_argument("--gripper-open-threshold", type=float, default=0.55)
    p.add_argument("--gripper-close-threshold", type=float, default=0.45)
    p.add_argument("--gripper-open-width-m", type=float, default=0.07)
    p.add_argument("--gripper-close-width-m", type=float, default=0.0002)
    p.add_argument("--state-stale-seconds", type=float, default=0.5)
    p.add_argument("--enable-timeout", type=float, default=5.0)
    p.add_argument("--clip-actions", action="store_true")
    p.add_argument("--skip-current-joint-range-check", action="store_true")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--offline-smoke", action="store_true")
    p.add_argument("--offline-zarr", type=Path, default=DEFAULT_OFFLINE_ZARR)
    p.add_argument("--offline-index", type=int, default=0)
    p.add_argument(
        "--log", type=Path,
        default=REPO_ROOT / "data/piper_inference/dp3_chunk4_shadow.json",
    )
    return p.parse_args()


def run_robot(args):
    device = torch.device(args.device)
    policy, _ = load_checkpoint(args.checkpoint, device)
    stats = policy_stats(policy)
    robot = PiperRobot(args.can, args.piper_sdk_root)
    camera = RealSense(
        fps=args.camera_fps,
        depth_width=640, depth_height=480,
        color_width=640, color_height=480,
        num_points=NUM_POINTS,
        point_cloud_frame="camera",
        align_depth_to_color=False,
    )
    records, history = [], deque(maxlen=OBS_STEPS)
    chunks = deque(maxlen=ACTION_STEPS)
    worker = None
    robot_enabled = hard_emergency = False
    last_state = None
    stop_reason = "max_steps"
    camera_started = False
    try:
        camera.start()
        camera_started = True
        state, joint_ts, _ = robot.read()
        last_state = state.copy()
        for _ in range(OBS_STEPS):
            history.append(make_observation(state, camera.get_frame(require_pc=False)))
        _, warmup = predict(policy, history, device)
        print(f"[CUDA]预热完成: {warmup*1000:.1f}ms", flush=True)
        if args.execute:
            phrase = input("确认工作区安全后输入 EXECUTE DP3 PIPER 继续：").strip()
            if phrase != "EXECUTE DP3 PIPER":
                raise RuntimeError("确认短语不匹配")
            robot.enable(args.speed_percent, args.enable_timeout)
            robot.enable_gripper(float(state[6]))
            robot_enabled = True

        planner = JointTrajectoryPlanner(
            state, args.max_joint_speed_rad_s, args.max_joint_accel_rad_s2,
            args.max_gripper_speed_m_s, args.planner_tracking_error_rad,
        )
        worker = AsyncPolicyWorker(policy, device)
        worker.start()
        console = OperatorConsole()
        last_result_id = -1
        previous = next_tick = last_state_sample = time.monotonic()
        valid_result_after = previous
        gripper_open = bool(state[6] >= 0.035)

        for step in range(args.max_steps):
            now = time.monotonic()
            dt, previous = now - previous, now
            cmd = console.poll()
            if cmd in {"estop", "e", "emergency"}:
                if robot_enabled:
                    robot.quick_stop()
                    hard_emergency = True
                stop_reason = "hard_emergency"
                break
            if cmd in {"stop", "s", "q", "quit"}:
                stop_reason = "operator_hold"
                break

            state, new_joint_ts, _ = robot.read()
            last_state = state.copy()
            if new_joint_ts != joint_ts:
                gap_ms = (now - last_state_sample) * 1000.0
                joint_ts, last_state_sample = new_joint_ts, now
                if gap_ms > args.max_state_gap_ms:
                    history.clear(); chunks.clear(); worker.clear_pending()
                    valid_result_after = now
                    print(f"[状态门控] gap={gap_ms:.1f}ms，清空历史", flush=True)
            elif now - last_state_sample > args.state_stale_seconds:
                raise RuntimeError("Piper关节反馈超时")

            frame = camera.get_frame(require_pc=False)
            history.append(make_observation(state, frame))
            if len(history) == OBS_STEPS:
                worker.submit(history, now)

            result = worker.latest()
            new_prediction = False
            infer_elapsed = None
            if result is not None and result[0] != last_result_id:
                request_id, origin, raw, infer_elapsed, _ = result
                last_result_id = request_id
                if origin >= valid_result_after:
                    chunks.append((origin, raw))
                    new_prediction = True

            fused = safe = None
            warnings = []
            count, age = 0, None
            if chunks:
                try:
                    fused, count, age = ensemble(chunks, now, args.ensemble_decay)
                except RuntimeError:
                    chunks.clear()
            if fused is not None:
                if fused[6] >= args.gripper_open_threshold:
                    gripper_open = True
                elif fused[6] <= args.gripper_close_threshold:
                    gripper_open = False
                latched_target = fused.copy()
                latched_target[6] = 1.0 if gripper_open else 0.0
                safe, warnings = safe_action(latched_target, state, stats, args)
                planned = planner.step(safe, state, dt)
            else:
                planned = planner.hold(state, dt)
            if robot_enabled:
                robot.send(planned, args.speed_percent)

            records.append({
                "step": step, "host_time": time.time(),
                "joint_gripper_state": state.tolist(),
                "camera_timestamp": frame["timestamp"],
                "new_prediction": new_prediction,
                "fused_target": None if fused is None else fused.tolist(),
                "safe_target": None if safe is None else safe.tolist(),
                "planned_action": planned.tolist(),
                "ensemble_candidates": count,
                "max_chunk_age_steps": age,
                "inference_elapsed_s": infer_elapsed,
                "warnings": warnings, "executed": robot_enabled,
            })
            if step % max(1, int(round(args.rate / 5))) == 0:
                infer_text = "None" if infer_elapsed is None else f"{infer_elapsed*1000:.1f}"
                print(
                    f"[step {step:04d}] candidates={count} age={age} "
                    f"infer_ms={infer_text} "
                    f"{'EXEC' if robot_enabled else 'SHADOW'}",
                    flush=True,
                )
            next_tick += 1.0 / args.rate
            remain = next_tick - time.monotonic()
            if remain > 0:
                time.sleep(remain)
            elif remain < -1.0 / args.rate:
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt_hold"
    finally:
        if worker is not None:
            worker.stop()
        if robot_enabled and not hard_emergency and last_state is not None:
            for _ in range(10):
                robot.send(last_state, args.speed_percent)
                time.sleep(0.05)
        if camera_started:
            camera.stop()
        robot.close()
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(json.dumps({
            "meta": {
                "checkpoint": str(args.checkpoint),
                "obs_steps": OBS_STEPS, "action_steps": ACTION_STEPS,
                "point_cloud_frame": "d435i_depth_optical_frame",
                "depth_aligned_to_color": False,
                "point_cloud_sampling": "training_exact_uniform_512",
                "executed": robot_enabled, "stop_reason": stop_reason,
            },
            "records": records,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[日志] 已保存: {args.log}", flush=True)


def main():
    args = parse_args()
    for key in ("checkpoint", "offline_zarr", "piper_sdk_root", "log"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.startup_joint_margin_rad < 0:
        raise ValueError("--startup-joint-margin-rad不能为负")
    if not 0 < args.max_state_gap_ms <= 150:
        raise ValueError("--max-state-gap-ms必须在(0,150]内")
    if args.rate <= 0 or args.camera_fps <= 0:
        raise ValueError("rate/camera-fps必须为正")
    if not 0 <= args.gripper_close_threshold < args.gripper_open_threshold <= 1:
        raise ValueError("夹爪阈值必须满足0<=close<open<=1")
    if not 0 <= args.gripper_close_width_m < args.gripper_open_width_m <= 0.08:
        raise ValueError("夹爪宽度必须满足0<=close<open<=0.08m")
    if args.offline_smoke:
        if not args.offline_zarr.exists():
            raise FileNotFoundError(args.offline_zarr)
        offline_smoke(args)
    else:
        run_robot(args)


if __name__ == "__main__":
    main()
