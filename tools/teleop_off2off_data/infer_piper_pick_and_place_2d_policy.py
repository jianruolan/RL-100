#!/usr/bin/env python3
"""Piper pick-and-place 2D RGB policy 的安全推理入口。

训练 Zarr 和模型都使用 84x84 RGB，由 ``DP3Encoder_with2D`` 内部的
``DrQEncoder([3,84,84], 32)`` 提取特征，不经过 R3M，也不上采样到 224。
state/action 是 7 维，chunk 配置固定为 3/6/4。

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


def build_rgb_obs(history: deque[dict[str, np.ndarray]], device: torch.device) -> dict[str, torch.Tensor]:
    # 2D encoder 只读取 image 和 agent_pos；不读取 point_cloud。
    return {
        "agent_pos": torch.from_numpy(np.stack([x["agent_pos"] for x in history])).unsqueeze(0).to(device),
        "image": torch.from_numpy(np.stack([x["image"] for x in history])).unsqueeze(0).to(device),
    }


def offline_smoke(dataset, policy, device, use_cm: bool) -> None:
    sample = dataset[0]["obs"]
    obs = {
        "agent_pos": sample["agent_pos"].unsqueeze(0).to(device),
        "image": sample["image"].unsqueeze(0).to(device),
    }
    with torch.no_grad():
        output = policy.predict_action(obs, deterministic=True, use_cm=use_cm)
    chunk = extract_action_chunk(output, expected_steps=4)
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
    p.add_argument("--execute", action="store_true")
    p.add_argument("--clip-actions", action="store_true")
    p.add_argument("--skip-current-joint-range-check", action="store_true")
    p.add_argument("--dataset-margin-rad", type=float, default=0.03)
    p.add_argument("--gripper-margin-m", type=float, default=0.005)
    p.add_argument("--max-joint-speed-rad-s", type=float, default=0.15)
    p.add_argument("--max-gripper-speed-m-s", type=float, default=0.02)
    p.add_argument("--skip-gripper-safety-check", action="store_true")
    p.add_argument("--speed-percent", type=int, default=10)
    p.add_argument("--state-stale-seconds", type=float, default=0.5)
    p.add_argument("--enable-timeout", type=float, default=5.0)
    p.add_argument("--log", type=Path, default=TRAIN_ROOT / "data/piper_inference/pick_and_place_2d_latest.json")
    return p.parse_args()


def main():
    args = parse_args()
    if not 0 < args.rate <= 20 or args.camera_fps < args.rate:
        raise ValueError("需要 0<rate<=20 且 camera-fps 不低于 rate")
    if args.rate > 15:
        print("[时序警告] 当前推理频率高于采集数据实际约13Hz", flush=True)
    args.output_dir = args.output_dir.expanduser().resolve()
    cfg, dataset, policy, use_cm = common.load_policy_and_dataset(args.output_dir, args.policy_subdir, args.device)
    if list(cfg.shape_meta.obs.agent_pos.shape) != [7] or list(cfg.shape_meta.action.shape) != [7]:
        raise RuntimeError(f"配置不是7D：agent_pos={cfg.shape_meta.obs.agent_pos.shape}, action={cfg.shape_meta.action.shape}")
    if list(cfg.shape_meta.obs.image.shape) != [3, 84, 84]:
        raise RuntimeError(f"Zarr/policy图像输入必须为[3,84,84]，实际为{cfg.shape_meta.obs.image.shape}")
    if str(cfg.encoder_type).lower() != "drq":
        raise RuntimeError(
            f"该推理脚本要求encoder_type=drq，当前为{cfg.encoder_type}；"
            "请勿把旧R3M实验误当作DrQEncoder(84)加载"
        )
    if int(cfg.n_obs_steps) != 3 or int(cfg.n_action_steps) != 4 or int(cfg.horizon) != 6:
        raise RuntimeError(f"chunk必须为3/6/4，实际为{cfg.n_obs_steps}/{cfg.horizon}/{cfg.n_action_steps}")
    device = torch.device(args.device)
    stats = training_stats(dataset)
    if args.offline_smoke:
        offline_smoke(dataset, policy, device, use_cm)
        return

    from tools.teleop_off2off_data.realsense import RealSense
    sdk_class = common.import_piper_sdk(args.piper_sdk_root)
    piper = sdk_class(args.can)
    piper.ConnectPort()
    reader = PiperPickPlaceStateReader(piper)
    camera = RealSense(fps=args.camera_fps, color_width=640, color_height=480,
                       depth_width=640, depth_height=480, num_points=512)
    camera.start()
    history = deque(maxlen=3)
    records = []
    operator = common.OperatorConsole()
    robot_enabled = False
    hard_emergency = False
    stop_reason = "max_steps"
    last_state = None
    last_joint_ts, last_grip_ts = None, None
    last_joint_fresh, last_grip_fresh = time.monotonic(), time.monotonic()
    try:
        for _ in range(3):
            state, _, _ = reader.read()
            frame = camera.get_frame(require_pc=False)
            history.append({"agent_pos": state, "image": preprocess_rgb(frame)})
            time.sleep(1.0 / args.rate)
        last_state = history[-1]["agent_pos"].copy()
        print("[模式]", "真机执行" if args.execute else "影子模式（不下发）", flush=True)
        print("[动作] 2D chunk=4，每 %.2fHz 重新推理并执行chunk[0]" % args.rate, flush=True)
        if args.execute:
            phrase = input("确认工作区安全后输入 EXECUTE PIPER 继续：").strip()
            if phrase != "EXECUTE PIPER":
                raise RuntimeError("确认短语不匹配，取消执行")
            common.enable_robot_for_position_control(piper, args.speed_percent, args.enable_timeout)
            enable_gripper(
                piper,
                reader,
                float(last_state[6]),
                require_homing=not args.skip_gripper_safety_check,
                timeout=args.enable_timeout,
            )
            robot_enabled = True
        operator.start()
        next_deadline = time.monotonic()
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
            history.append({"agent_pos": state, "image": preprocess_rgb(frame)})
            with torch.no_grad():
                output = policy.predict_action(build_rgb_obs(history, device), deterministic=True, use_cm=use_cm)
            chunk = extract_action_chunk(output, expected_steps=4)
            target, warnings = safe_action(chunk[0], state, stats, args)
            if robot_enabled:
                send_action(piper, target, args.speed_percent)
            records.append({"step": step, "host_time": time.time(), "state": state.tolist(), "predicted_chunk": chunk.tolist(), "safe_action": target.tolist(), "warnings": warnings, "executed": robot_enabled})
            print(f"[step {step:04d}] state={np.round(state,4)} chunk0={np.round(chunk[0],4)} target={np.round(target,4)} {'EXEC' if robot_enabled else 'SHADOW'}", flush=True)
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
        log_path.write_text(json.dumps({"meta": {"output_dir": str(args.output_dir), "policy_subdir": args.policy_subdir, "rate_hz": args.rate, "n_obs_steps": 3, "horizon": 6, "n_action_steps": 4, "image_shape": [3,84,84], "stop_reason": stop_reason}, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[日志] 已保存: {log_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
