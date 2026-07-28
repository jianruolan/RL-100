#!/usr/bin/env python3
"""Piper pick-and-place 2D RGB/RGB-D policy 的安全推理入口。

RGB实验支持84x84 DrQ和3x224x224 ImageNet/R3M ResNet；RGB-D实验使用
4x224x224 ImageNet ResNet18，第4通道是按固定深度范围量化的D435深度。
state/action 是 7 维，chunk 长度从权重配置读取（当前实验为 3/6/4）。

默认 shadow 模式不发送动作。执行模式必须显式传 ``--execute`` 并确认短语；
每个约 13 Hz 周期重新推理并执行 chunk[0]，以匹配这批 rosbag 的实际有效频率。
"""

from __future__ import annotations

import argparse
import json
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

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
TRAIN_ROOT = REPO_ROOT / "RL-100"
for import_path in (REPO_ROOT, TRAIN_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from tools.teleop_off2off_data import infer_piper_contact_policy as common
from tools.teleop_off2off_data.infer_piper_pick_and_place_policy import (
    GRIPPER_UPPER_M,
    LatestFrameCamera,
    PiperProcessProxy,
    PiperPickPlaceStateReader,
    enable_gripper,
    extract_action_chunk,
    resize_rgb_to_chw,
    safe_action,
    send_action,
    training_stats,
)

DEFAULT_OUTPUT_DIR = TRAIN_ROOT / "data/outputs/piper_pick_and_place_2d_chunk4_bc_cm_offline_seed42"
RGB224_IMAGE_SHAPE = [3, 224, 224]
RGB84_IMAGE_SHAPE = [3, 84, 84]
RGBD224_IMAGE_SHAPE = [4, 224, 224]
IMAGENET_MEAN_UINT8 = np.array([123, 116, 104], dtype=np.uint8)


def preprocess_rgb(frame: dict[str, Any]) -> np.ndarray:
    color_bgr = np.asarray(frame["color"])
    if color_bgr.shape != (480, 640, 3):
        raise RuntimeError(f"D435i RGB shape异常: {color_bgr.shape}，预期(480,640,3)")
    rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    # Zarr 转换脚本的输入已经是 RGB，resize 后转 CHW；DrQ直接接收84x84。
    return resize_rgb_to_chw(rgb, size=84)


def letterbox_rgb224_to_chw(rgb: np.ndarray) -> np.ndarray:
    """与RGB224 zarr转换一致：保持640x480比例，补ImageNet均值边框。"""

    if rgb.shape != (480, 640, 3):
        raise RuntimeError(f"D435i RGB shape异常: {rgb.shape}，预期(480,640,3)")
    height, width = rgb.shape[:2]
    scale = min(224 / width, 224 / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(
        rgb,
        (resized_width, resized_height),
        interpolation=interpolation,
    )
    output = np.empty((224, 224, 3), dtype=np.uint8)
    output[...] = IMAGENET_MEAN_UINT8
    left = (224 - resized_width) // 2
    top = (224 - resized_height) // 2
    output[top : top + resized_height, left : left + resized_width] = resized
    return np.transpose(output, (2, 0, 1)).astype(np.float32)


def preprocess_rgb224(frame: dict[str, Any]) -> np.ndarray:
    color_bgr = np.asarray(frame["color"])
    rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    return letterbox_rgb224_to_chw(rgb)


def preprocess_rgbd(frame: dict[str, Any]) -> np.ndarray:
    color_bgr = np.asarray(frame["color"])
    depth_mm = np.asarray(frame["depth"])
    if color_bgr.shape != (480, 640, 3) or depth_mm.shape != (480, 640):
        raise RuntimeError(f"D435i RGB-D shape异常: rgb={color_bgr.shape}, depth={depth_mm.shape}")
    rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

    # 与新RGB-D训练集完全一致：640x480按比例缩到224x168，再在上下各补28像素。
    # 不能直接拉伸到224x224，否则真机中的物体几何比例会偏离训练分布。
    def letterbox(image: np.ndarray, interpolation: int) -> np.ndarray:
        height, width = image.shape[:2]
        scale = min(224 / width, 224 / height)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized = cv2.resize(
            image, (resized_width, resized_height), interpolation=interpolation
        )
        output_shape = (224, 224) + (() if image.ndim == 2 else (image.shape[2],))
        output = np.zeros(output_shape, dtype=image.dtype)
        if image.ndim == 3:
            # 转换脚本用ImageNet均值填RGB边框，归一化后边框接近0。
            output[...] = np.array([123, 116, 104], dtype=image.dtype)
        x0 = (224 - resized_width) // 2
        y0 = (224 - resized_height) // 2
        output[y0:y0 + resized_height, x0:x0 + resized_width] = resized
        return output

    rgb_chw = letterbox(rgb, cv2.INTER_AREA)
    depth_m = depth_mm.astype(np.float32) * 0.001
    valid = np.isfinite(depth_m) & (depth_m >= 0.1) & (depth_m <= 2.0)
    depth = np.zeros_like(depth_m, dtype=np.float32)
    depth[valid] = (depth_m[valid] - 0.1) / 1.9
    depth = letterbox(
        np.rint(np.clip(depth, 0, 1) * 255).astype(np.uint8),
        cv2.INTER_NEAREST,
    )
    return np.concatenate([np.transpose(rgb_chw, (2, 0, 1)), depth[None]], axis=0)


def preprocess_frame(frame: dict[str, Any], image_shape: list[int]) -> np.ndarray:
    if image_shape == RGB84_IMAGE_SHAPE:
        return preprocess_rgb(frame)
    if image_shape == RGB224_IMAGE_SHAPE:
        return preprocess_rgb224(frame)
    if image_shape == RGBD224_IMAGE_SHAPE:
        return preprocess_rgbd(frame)
    raise RuntimeError(f"不支持的2D图像输入: {image_shape}")


def build_rgb_obs(history: deque[dict[str, np.ndarray]], device: torch.device) -> dict[str, torch.Tensor]:
    # 2D encoder 只读取 image 和 agent_pos；不读取 point_cloud。
    return {
        "agent_pos": torch.from_numpy(np.stack([x["agent_pos"] for x in history])).unsqueeze(0).to(device),
        "image": torch.from_numpy(np.stack([x["image"] for x in history])).unsqueeze(0).to(device),
    }


class AsyncRGBPolicyWorker:
    """后台运行2D/RGB policy推理，主控制循环只取最新完成结果。"""

    def __init__(self, policy, device: torch.device, use_cm: bool, expected_action_steps: int):
        self._policy = policy
        self._device = device
        self._use_cm = bool(use_cm)
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
            target=self._run,
            name="cuda-rgb-policy-worker",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def submit(self, obs: dict[str, torch.Tensor], control_step: int) -> int:
        with self._condition:
            request_id = self._next_request_id
            self._next_request_id += 1
            # 只保留尚未开始的最新观测；旧观测推理完成反而会制造控制滞后。
            self._pending = (request_id, int(control_step), obs)
            self._condition.notify()
            return request_id

    def latest(self):
        with self._condition:
            if self._error is not None:
                raise RuntimeError("异步2D策略推理失败") from self._error
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
                    output = self._policy.predict_action(
                        obs,
                        deterministic=True,
                        use_cm=self._use_cm,
                    )
                if self._device.type == "cuda":
                    torch.cuda.synchronize(self._device)
                chunk = extract_action_chunk(
                    output,
                    expected_steps=self._expected_action_steps,
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


def offline_smoke(dataset, policy, device, use_cm: bool, expected_steps: int) -> None:
    sample = dataset[0]["obs"]
    obs = {
        "agent_pos": sample["agent_pos"].unsqueeze(0).to(device),
        "image": sample["image"].unsqueeze(0).to(device),
    }
    with torch.no_grad():
        output = policy.predict_action(obs, deterministic=True, use_cm=use_cm)
    chunk = extract_action_chunk(output, expected_steps=expected_steps)
    print(f"[offline-smoke] 成功，2D输入 image={tuple(obs['image'].shape)}，chunk={chunk.shape}")
    print(f"[offline-smoke] 第一动作={np.round(chunk[0], 6)}")


def postprocess_action_chunk(
    chunk: np.ndarray,
    step: int,
    chunk_step: int,
    args: argparse.Namespace,
) -> np.ndarray:
    """扩展点：smooth入口可在这里做Temporal Ensemble等chunk级处理。"""

    return chunk


def extra_record_diagnostics() -> dict[str, Any]:
    """扩展点：smooth入口可给每条控制日志补充诊断字段。"""

    return {}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument(
        "--policy-subdir",
        choices=["best", "bc", "best_cm", "best_val", "milestone_25", "milestone_50", "milestone_75"],
        default="best",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--offline-smoke", action="store_true")
    p.add_argument("--can", default="can0")
    p.add_argument("--piper-sdk-root", type=Path, default=None)
    p.add_argument("--rate", type=float, default=13.0)
    p.add_argument("--camera-fps", type=int, default=15)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--chunk-exec-steps", type=int, default=1,
                   help="每次策略推理后依次执行的 action chunk 步数；必须不超过 n_action_steps。")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--clip-actions", action="store_true")
    p.add_argument("--skip-current-joint-range-check", action="store_true")
    p.add_argument("--dataset-margin-rad", type=float, default=0.03)
    p.add_argument("--gripper-margin-m", type=float, default=0.005)
    p.add_argument("--gripper-command-threshold", type=float, default=0.5,
                   help="第7维0/1夹爪命令的开合阈值。")
    p.add_argument("--max-joint-speed-rad-s", type=float, default=0.15)
    p.add_argument("--max-gripper-speed-m-s", type=float, default=0.02)
    p.add_argument("--skip-gripper-safety-check", action="store_true")
    p.add_argument("--skip-point-cloud-distribution-check", action="store_true",
                   help="2D策略不使用点云；保留该参数仅为兼容3D推理命令。")
    p.add_argument("--speed-percent", type=int, default=10)
    p.add_argument("--state-stale-seconds", type=float, default=0.5)
    p.add_argument("--enable-timeout", type=float, default=5.0)
    p.add_argument(
        "--max-consecutive-control-overruns",
        type=int,
        default=5,
        help="控制循环连续落后超过一个周期多少次后停止；同步2D推理默认允许短暂抖动。",
    )
    p.add_argument("--log", type=Path, default=TRAIN_ROOT / "data/piper_inference/pick_and_place_2d_latest.json")
    return p.parse_args()


def main():
    args = parse_args()
    if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
        available_cpus = set(os.sched_getaffinity(0))
        if 0 in available_cpus and len(available_cpus) > 1:
            os.sched_setaffinity(0, available_cpus - {0})
    if not 0 < args.rate <= 60 or args.camera_fps < args.rate:
        raise ValueError("需要 0<rate<=60 且 camera-fps 不低于 rate")
    if args.chunk_exec_steps < 1:
        raise ValueError("--chunk-exec-steps 必须为正整数")
    if not 0.0 < args.gripper_command_threshold < 1.0:
        raise ValueError("--gripper-command-threshold 必须在 (0,1)")
    if args.max_consecutive_control_overruns < 1:
        raise ValueError("--max-consecutive-control-overruns 必须为正整数")
    if args.rate > 15:
        print("[时序警告] 当前推理频率高于采集数据实际约13Hz", flush=True)
    args.output_dir = args.output_dir.expanduser().resolve()
    camera = None
    if not args.offline_smoke:
        # 和3D推理脚本一致：先启动RealSense，再初始化较大的CUDA模型。
        # Thor上模型加载/CUDA统一内存初始化后再resolve UVC profile，偶发失败率更高。
        from tools.teleop_off2off_data.realsense import RealSense

        raw_camera = RealSense(
            fps=args.camera_fps,
            color_width=640,
            color_height=480,
            depth_width=640,
            depth_height=480,
            num_points=512,
        )
        camera = LatestFrameCamera(raw_camera)
        camera.start()
    try:
        cfg, dataset, policy, use_cm = common.load_policy_and_dataset(
            args.output_dir,
            args.policy_subdir,
            args.device,
        )
    except Exception:
        if camera is not None:
            camera.stop()
        raise
    if list(cfg.shape_meta.obs.agent_pos.shape) != [7] or list(cfg.shape_meta.action.shape) != [7]:
        raise RuntimeError(f"配置不是7D：agent_pos={cfg.shape_meta.obs.agent_pos.shape}, action={cfg.shape_meta.action.shape}")
    image_shape = list(cfg.shape_meta.obs.image.shape)
    if image_shape not in (RGB84_IMAGE_SHAPE, RGB224_IMAGE_SHAPE, RGBD224_IMAGE_SHAPE):
        raise RuntimeError(
            f"Zarr/policy图像输入必须为[3,84,84]、[3,224,224]或[4,224,224]，实际为{image_shape}"
        )
    encoder_type = str(cfg.encoder_type).lower()
    if image_shape == RGB84_IMAGE_SHAPE:
        expected_encoders = {"drq"}
    elif image_shape == RGB224_IMAGE_SHAPE:
        expected_encoders = {"resnet", "resnet18", "r3m", "resnet18_r3m"}
    else:
        expected_encoders = {"resnet18_rgbd"}
    if encoder_type not in expected_encoders:
        raise RuntimeError(
            f"该推理脚本要求encoder_type属于{sorted(expected_encoders)}，当前为{cfg.encoder_type}"
        )
    n_obs_steps = int(cfg.n_obs_steps)
    n_action_steps = int(cfg.n_action_steps)
    horizon = int(cfg.horizon)
    expected_horizon = n_obs_steps - 1 + n_action_steps
    if n_obs_steps < 1 or n_action_steps < 1 or horizon != expected_horizon:
        raise RuntimeError(
            "训练配置必须满足 horizon=n_obs_steps-1+n_action_steps，"
            f"实际为{n_obs_steps}/{horizon}/{n_action_steps}"
        )
    if args.chunk_exec_steps > n_action_steps:
        raise ValueError(
            f"--chunk-exec-steps={args.chunk_exec_steps} 超过模型 n_action_steps={n_action_steps}"
        )
    device = torch.device(args.device)
    stats = training_stats(dataset)
    print(
        f"[夹爪动作] key={stats['action_key']}，mode={stats['gripper_action_mode']}，"
        f"threshold={args.gripper_command_threshold:g}",
        flush=True,
    )
    if args.offline_smoke:
        offline_smoke(dataset, policy, device, use_cm, expected_steps=n_action_steps)
        return

    assert camera is not None
    piper = None
    try:
        # Piper SDK高频CAN/parser线程放到子进程，避免和CUDA launch抢GIL。
        piper = PiperProcessProxy(args.can, args.piper_sdk_root)
        piper.ConnectPort()
        reader = PiperPickPlaceStateReader(piper)
    except Exception:
        camera.stop()
        if piper is not None:
            piper.close()
        raise
    history = deque(maxlen=n_obs_steps)
    records = []
    operator = common.OperatorConsole()
    robot_enabled = False
    hard_emergency = False
    stop_reason = "max_steps"
    last_state = None
    last_joint_ts, last_grip_ts = None, None
    last_joint_fresh, last_grip_fresh = time.monotonic(), time.monotonic()
    policy_worker: AsyncRGBPolicyWorker | None = None
    try:
        for _ in range(n_obs_steps):
            state, _, _ = reader.read()
            frame = camera.get_frame(require_pc=False)
            history.append({"agent_pos": state, "image": preprocess_frame(frame, image_shape)})
            time.sleep(1.0 / args.rate)
        last_state = history[-1]["agent_pos"].copy()

        # 与3D推理入口一致：CUDA首次forward会初始化context/kernels，
        # ResNet224 + diffusion第一次调用很容易超过实时周期。必须在
        # 启动控制deadline和使能机械臂前预热，否则会把一次性开销误判为
        # Thor推理速度不够。
        warmup_obs = build_rgb_obs(history, device)
        if device.type == "cuda":
            print("[CUDA] 开始2D推理预热...", flush=True)
            with torch.inference_mode():
                for _ in range(2):
                    policy.predict_action(warmup_obs, deterministic=True, use_cm=use_cm)
            torch.cuda.synchronize(device)
            print("[CUDA] 2D推理预热完成", flush=True)
        else:
            with torch.inference_mode():
                policy.predict_action(warmup_obs, deterministic=True, use_cm=use_cm)
        initial_start = time.monotonic()
        with torch.inference_mode():
            initial_output = policy.predict_action(
                build_rgb_obs(history, device),
                deterministic=True,
                use_cm=use_cm,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        initial_chunk = extract_action_chunk(
            initial_output,
            expected_steps=n_action_steps,
        )
        initial_infer_elapsed = time.monotonic() - initial_start
        initial_chunk = postprocess_action_chunk(initial_chunk, 0, 0, args)
        print(
            f"[异步推理] 初始2D action chunk完成: {initial_infer_elapsed*1000:.1f}ms；"
            "控制循环后续不会等待模型forward",
            flush=True,
        )
        print("[模式]", "真机执行" if args.execute else "影子模式（不下发）", flush=True)
        print(
            "[动作] 2D chunk=%d，每次推理依次执行前 %d 步；动作下发 %.2fHz，模型重规划约 %.2fHz"
            % (n_action_steps, args.chunk_exec_steps, args.rate, args.rate / args.chunk_exec_steps),
            flush=True,
        )
        if args.execute:
            phrase = input("确认工作区安全后输入 EXECUTE PIPER 继续：").strip()
            if phrase != "EXECUTE PIPER":
                raise RuntimeError("确认短语不匹配，取消执行")
            common.enable_robot_for_position_control(piper, args.speed_percent, args.enable_timeout)
            # 关节已使能；即使之后夹爪检查失败，finally 也必须保持关节。
            robot_enabled = True
            enable_gripper(
                piper,
                reader,
                float(last_state[6]),
                require_homing=not args.skip_gripper_safety_check,
                timeout=args.enable_timeout,
            )
        policy_worker = AsyncRGBPolicyWorker(
            policy,
            device,
            use_cm,
            expected_action_steps=n_action_steps,
        )
        policy_worker.start()
        operator.start()
        next_deadline = time.monotonic()
        consecutive_overruns = 0
        active_chunk: np.ndarray = initial_chunk
        active_chunk_origin_step = 0
        inference_index = 0
        last_policy_result_id = -1
        for step in range(args.max_steps):
            loop_start = time.monotonic()
            command = operator.poll()
            if command in {"estop", "e", "emergency"}:
                if robot_enabled:
                    common.quick_stop(piper)
                    hard_emergency = True
                stop_reason = "hard_emergency"
                break
            if command in {"stop", "s", "q", "quit"}:
                stop_reason = "operator_hold"
                break
            now = time.monotonic()
            state, joint_ts, grip_ts = reader.read()
            if joint_ts != last_joint_ts:
                last_joint_fresh = now
            elif now - last_joint_fresh > args.state_stale_seconds:
                raise RuntimeError("Piper关节反馈超过允许时间未更新")
            if grip_ts != last_grip_ts:
                last_grip_fresh = now
            elif now - last_grip_fresh > args.state_stale_seconds:
                raise RuntimeError("Piper夹爪反馈超过允许时间未更新")
            last_joint_ts, last_grip_ts = joint_ts, grip_ts
            camera_start = time.monotonic()
            frame = camera.get_frame(require_pc=False)
            camera_elapsed = time.monotonic() - camera_start
            preprocess_start = time.monotonic()
            history.append({"agent_pos": state, "image": preprocess_frame(frame, image_shape)})
            preprocess_elapsed = time.monotonic() - preprocess_start
            replan_step = step % args.chunk_exec_steps
            requested_inference = replan_step == 0
            ran_inference = False
            infer_elapsed = None
            obs_build_elapsed = None
            if requested_inference:
                obs_build_start = time.monotonic()
                obs = build_rgb_obs(history, device)
                obs_build_elapsed = time.monotonic() - obs_build_start
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                policy_worker.submit(obs, control_step=step)
            latest_policy = policy_worker.latest()
            if latest_policy is not None and latest_policy[0] > last_policy_result_id:
                (
                    last_policy_result_id,
                    active_chunk_origin_step,
                    active_chunk,
                    infer_elapsed,
                    _,
                ) = latest_policy
                inference_index += 1
                ran_inference = True
                active_chunk = postprocess_action_chunk(
                    active_chunk,
                    active_chunk_origin_step,
                    0,
                    args,
                )
            policy_age_steps = max(0, step - active_chunk_origin_step)
            chunk_step = replan_step
            predicted_action = active_chunk[chunk_step]
            target, warnings = safe_action(predicted_action, state, stats, args)
            if robot_enabled:
                send_action(piper, target, args.speed_percent)
            cycle_elapsed = time.monotonic() - loop_start
            record = {
                "step": step,
                "host_time": time.time(),
                "state": state.tolist(),
                "predicted_chunk": active_chunk.tolist(),
                "chunk_step": chunk_step,
                "policy_inference": ran_inference,
                "policy_inference_requested": requested_inference,
                "policy_age_steps": policy_age_steps,
                "policy_result_id": last_policy_result_id,
                "active_chunk_origin_step": active_chunk_origin_step,
                "inference_index": inference_index,
                "safe_action": target.tolist(),
                "warnings": warnings,
                "executed": robot_enabled,
                "cycle_elapsed_s": cycle_elapsed,
                "camera_elapsed_s": camera_elapsed,
                "preprocess_elapsed_s": preprocess_elapsed,
                "obs_build_elapsed_s": obs_build_elapsed if requested_inference else None,
                "inference_elapsed_s": infer_elapsed if ran_inference else None,
            }
            record.update(extra_record_diagnostics())
            records.append(record)
            print(
                f"[step {step:04d}] state={np.round(state,4)} "
                f"chunk{chunk_step}={np.round(predicted_action,4)} "
                f"target={np.round(target,4)} "
                f"{'INFER' if ran_inference else 'CACHED'} "
                f"{'EXEC' if robot_enabled else 'SHADOW'} "
                f"age={policy_age_steps} "
                f"dt={cycle_elapsed*1000:.1f}ms "
                f"cam={camera_elapsed*1000:.1f}ms "
                f"prep={preprocess_elapsed*1000:.1f}ms"
                + (f" obs={obs_build_elapsed*1000:.1f}ms" if requested_inference and obs_build_elapsed is not None else "")
                + (f" infer={infer_elapsed*1000:.1f}ms" if ran_inference and infer_elapsed is not None else ""),
                flush=True,
            )
            last_state = state.copy()
            next_deadline += 1.0 / args.rate
            remaining = next_deadline - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
                consecutive_overruns = 0
            else:
                lag = time.monotonic() - next_deadline
                if lag > 1.0 / args.rate:
                    consecutive_overruns += 1
                    print(
                        f"[时序警告] 2D控制循环落后{lag*1000:.1f}ms "
                        f"({consecutive_overruns}/{args.max_consecutive_control_overruns})，"
                        "已重同步deadline",
                        flush=True,
                    )
                    next_deadline = time.monotonic()
                    if consecutive_overruns >= args.max_consecutive_control_overruns:
                        raise RuntimeError(
                            "2D推理循环连续落后超过一个控制周期；"
                            "请查看日志中的cycle/camera/preprocess/inference耗时"
                        )
                else:
                    consecutive_overruns = 0
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt_hold"
    finally:
        if policy_worker is not None:
            policy_worker.stop()
        if robot_enabled and not hard_emergency and last_state is not None:
            for _ in range(10):
                send_action(piper, last_state, args.speed_percent)
                time.sleep(0.05)
        try:
            camera.stop()
        except Exception as exc:
            print(f"[相机] 停止失败: {exc}", file=sys.stderr)
        if piper is not None:
            piper.close()
        log_path = args.log.expanduser()
        if not log_path.is_absolute():
            log_path = TRAIN_ROOT / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps({"meta": {"output_dir": str(args.output_dir), "policy_subdir": args.policy_subdir, "rate_hz": args.rate, "n_obs_steps": n_obs_steps, "horizon": horizon, "n_action_steps": n_action_steps, "chunk_exec_steps": args.chunk_exec_steps, "policy_replan_rate_hz": args.rate / args.chunk_exec_steps, "image_shape": image_shape, "skip_point_cloud_distribution_check": args.skip_point_cloud_distribution_check, "stop_reason": stop_reason}, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[日志] 已保存: {log_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
