#!/usr/bin/env python3
"""Piper pick-and-place 2D RGB/RGB-D policy 的安全推理入口。

RGB实验使用84x84 DrQ；RGB-D实验使用4x224x224 ImageNet ResNet18，
第4通道是按固定深度范围量化的D435深度。
state/action 是 7 维，chunk 长度从权重配置读取（当前实验为 3/6/4）。

默认 shadow 模式不发送动作。执行模式必须显式传 ``--execute`` 并确认短语；
每个约 13 Hz 周期重新推理并执行 chunk[0]，以匹配这批 rosbag 的实际有效频率。
"""

from __future__ import annotations

import argparse
import json
import sys
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
    PiperPickPlaceStateReader,
    build_obs,
    enable_gripper,
    extract_action_chunk,
    resize_rgb_to_chw,
    safe_action,
    send_action,
    training_stats,
)

DEFAULT_OUTPUT_DIR = TRAIN_ROOT / "data/outputs/piper_pick_and_place_2d_chunk4_bc_cm_offline_seed42"


def preprocess_rgb(frame: dict[str, Any]) -> np.ndarray:
    color_bgr = np.asarray(frame["color"])
    if color_bgr.shape != (480, 640, 3):
        raise RuntimeError(f"D435i RGB shape异常: {color_bgr.shape}，预期(480,640,3)")
    rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    # Zarr 转换脚本的输入已经是 RGB，resize 后转 CHW；DrQ直接接收84x84。
    return resize_rgb_to_chw(rgb, size=84)


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
    if image_shape == [3, 84, 84]:
        return preprocess_rgb(frame)
    if image_shape == [4, 224, 224]:
        return preprocess_rgbd(frame)
    raise RuntimeError(f"不支持的2D图像输入: {image_shape}")


def build_rgb_obs(history: deque[dict[str, np.ndarray]], device: torch.device) -> dict[str, torch.Tensor]:
    # 2D encoder 只读取 image 和 agent_pos；不读取 point_cloud。
    return {
        "agent_pos": torch.from_numpy(np.stack([x["agent_pos"] for x in history])).unsqueeze(0).to(device),
        "image": torch.from_numpy(np.stack([x["image"] for x in history])).unsqueeze(0).to(device),
    }


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
    p.add_argument("--chunk-exec-steps", type=int, default=1,
                   help="每次策略推理后依次执行的 action chunk 步数；必须不超过 n_action_steps。")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--clip-actions", action="store_true")
    p.add_argument("--skip-current-joint-range-check", action="store_true")
    p.add_argument("--dataset-margin-rad", type=float, default=0.03)
    p.add_argument("--gripper-margin-m", type=float, default=0.005)
    p.add_argument("--max-joint-speed-rad-s", type=float, default=0.15)
    p.add_argument("--max-gripper-speed-m-s", type=float, default=0.02)
    p.add_argument("--skip-gripper-safety-check", action="store_true")
    p.add_argument("--skip-point-cloud-distribution-check", action="store_true",
                   help="2D策略不使用点云；保留该参数仅为兼容3D推理命令。")
    p.add_argument("--speed-percent", type=int, default=10)
    p.add_argument("--state-stale-seconds", type=float, default=0.5)
    p.add_argument("--enable-timeout", type=float, default=5.0)
    p.add_argument("--log", type=Path, default=TRAIN_ROOT / "data/piper_inference/pick_and_place_2d_latest.json")
    return p.parse_args()


def main():
    args = parse_args()
    if not 0 < args.rate <= 20 or args.camera_fps < args.rate:
        raise ValueError("需要 0<rate<=20 且 camera-fps 不低于 rate")
    if args.chunk_exec_steps < 1:
        raise ValueError("--chunk-exec-steps 必须为正整数")
    if args.rate > 15:
        print("[时序警告] 当前推理频率高于采集数据实际约13Hz", flush=True)
    args.output_dir = args.output_dir.expanduser().resolve()
    cfg, dataset, policy, use_cm = common.load_policy_and_dataset(args.output_dir, args.policy_subdir, args.device)
    if list(cfg.shape_meta.obs.agent_pos.shape) != [7] or list(cfg.shape_meta.action.shape) != [7]:
        raise RuntimeError(f"配置不是7D：agent_pos={cfg.shape_meta.obs.agent_pos.shape}, action={cfg.shape_meta.action.shape}")
    image_shape = list(cfg.shape_meta.obs.image.shape)
    if image_shape not in ([3, 84, 84], [4, 224, 224]):
        raise RuntimeError(f"Zarr/policy图像输入必须为[3,84,84]或[4,224,224]，实际为{image_shape}")
    expected_encoder = "drq" if image_shape == [3, 84, 84] else "resnet18_rgbd"
    if str(cfg.encoder_type).lower() != expected_encoder:
        raise RuntimeError(
            f"该推理脚本要求encoder_type={expected_encoder}，当前为{cfg.encoder_type}"
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
    if args.offline_smoke:
        offline_smoke(dataset, policy, device, use_cm, expected_steps=n_action_steps)
        return

    from tools.teleop_off2off_data.realsense import RealSense
    sdk_class = common.import_piper_sdk(args.piper_sdk_root)
    piper = sdk_class(args.can)
    piper.ConnectPort()
    reader = PiperPickPlaceStateReader(piper)
    camera = RealSense(fps=args.camera_fps, color_width=640, color_height=480,
                       depth_width=640, depth_height=480, num_points=512)
    camera.start()
    history = deque(maxlen=n_obs_steps)
    records = []
    operator = common.OperatorConsole()
    robot_enabled = False
    hard_emergency = False
    stop_reason = "max_steps"
    last_state = None
    last_joint_ts, last_grip_ts = None, None
    last_joint_fresh, last_grip_fresh = time.monotonic(), time.monotonic()
    try:
        for _ in range(n_obs_steps):
            state, _, _ = reader.read()
            frame = camera.get_frame(require_pc=False)
            history.append({"agent_pos": state, "image": preprocess_frame(frame, image_shape)})
            time.sleep(1.0 / args.rate)
        last_state = history[-1]["agent_pos"].copy()
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
        operator.start()
        next_deadline = time.monotonic()
        active_chunk: np.ndarray | None = None
        inference_index = -1
        for step in range(args.max_steps):
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
            frame = camera.get_frame(require_pc=False)
            history.append({"agent_pos": state, "image": preprocess_frame(frame, image_shape)})
            chunk_step = step % args.chunk_exec_steps
            ran_inference = chunk_step == 0 or active_chunk is None
            if ran_inference:
                with torch.no_grad():
                    output = policy.predict_action(build_rgb_obs(history, device), deterministic=True, use_cm=use_cm)
                active_chunk = extract_action_chunk(output, expected_steps=n_action_steps)
                inference_index += 1
                chunk_step = 0
            predicted_action = active_chunk[chunk_step]
            target, warnings = safe_action(predicted_action, state, stats, args)
            if robot_enabled:
                send_action(piper, target, args.speed_percent)
            records.append({"step": step, "host_time": time.time(), "state": state.tolist(), "predicted_chunk": active_chunk.tolist(), "chunk_step": chunk_step, "policy_inference": ran_inference, "inference_index": inference_index, "safe_action": target.tolist(), "warnings": warnings, "executed": robot_enabled})
            print(f"[step {step:04d}] state={np.round(state,4)} chunk{chunk_step}={np.round(predicted_action,4)} target={np.round(target,4)} {'INFER' if ran_inference else 'CACHED'} {'EXEC' if robot_enabled else 'SHADOW'}", flush=True)
            last_state = state.copy()
            next_deadline += 1.0 / args.rate
            remaining = next_deadline - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            elif time.monotonic() - next_deadline > 1.0 / args.rate:
                raise RuntimeError("2D推理循环连续落后超过一个控制周期")
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt_hold"
    finally:
        if robot_enabled and not hard_emergency and last_state is not None:
            for _ in range(10):
                send_action(piper, last_state, args.speed_percent)
                time.sleep(0.05)
        try:
            camera.stop()
        except Exception as exc:
            print(f"[相机] 停止失败: {exc}", file=sys.stderr)
        log_path = args.log.expanduser()
        if not log_path.is_absolute():
            log_path = TRAIN_ROOT / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps({"meta": {"output_dir": str(args.output_dir), "policy_subdir": args.policy_subdir, "rate_hz": args.rate, "n_obs_steps": n_obs_steps, "horizon": horizon, "n_action_steps": n_action_steps, "chunk_exec_steps": args.chunk_exec_steps, "policy_replan_rate_hz": args.rate / args.chunk_exec_steps, "image_shape": image_shape, "skip_point_cloud_distribution_check": args.skip_point_cloud_distribution_check, "stop_reason": stop_reason}, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[日志] 已保存: {log_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
