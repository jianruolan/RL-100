#!/usr/bin/env python3
"""High-rate real-robot executor for the control-clean Piper DP3 BC policy.

The control, policy, and camera loops have independent clocks. Commands are
emitted at ``--rate``; genuine D435i frames alone update observation history;
``--policy-rate`` may independently resample the latest complete history.
An acceleration-limited planner holds a smooth command between policy results.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

SCRIPT_PATH = Path(__file__).resolve()
RL100_REPO = Path("/home/mtarch/Desktop/zyf/RL-100")
TRAIN_ROOT = RL100_REPO / "RL-100"
OUTPUT_DIR = TRAIN_ROOT / (
    "data/outputs/piper_pick_and_place_augmented_chunk4_control_clean_dp3_"
    "episode10_bs64_epoch4000_seed42"
)
for import_path in (RL100_REPO, TRAIN_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from tools.teleop_off2off_data import infer_piper_contact_policy as contact
from tools.teleop_off2off_data import infer_piper_pick_and_place_policy as base
from tools.teleop_off2off_data.realsense import RealSense
from infer_rl100_gap100_piper_last import JointTrajectoryPlanner


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help="DP3训练输出目录；默认是 control-clean 的70.79M模型。",
    )
    p.add_argument(
        "--policy-subdir", default="bc",
        help="权重子目录，例如 bc、best_val；必须位于 --output-dir 内。",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--can", default="can0")
    p.add_argument("--piper-sdk-root", type=Path)
    p.add_argument("--rate", "--control-rate", dest="rate", type=float, default=100.0)
    p.add_argument("--camera-fps", type=int, default=15)
    p.add_argument(
        "--policy-rate", type=float, default=0.0,
        help=(
            "独立策略请求频率（Hz）。0表示仅在新相机帧到达时请求；"
            "正数表示按该频率复用最新完整观测窗口，最大不超过--rate。"
        ),
    )
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--ddim-steps", type=int, default=None)
    p.add_argument("--speed-percent", type=int, default=10)
    p.add_argument("--max-joint-speed-rad-s", type=float, default=0.20)
    p.add_argument("--max-joint-accel-rad-s2", type=float, default=0.50)
    p.add_argument("--max-gripper-speed-m-s", type=float, default=0.02)
    p.add_argument("--planner-tracking-error-rad", type=float, default=0.10)
    p.add_argument("--max-camera-gap-ms", type=float, default=200.0)
    p.add_argument("--state-stale-seconds", type=float, default=0.5)
    p.add_argument("--dataset-margin-rad", type=float, default=0.03)
    p.add_argument("--gripper-margin-m", type=float, default=0.005)
    p.add_argument("--gripper-command-threshold", type=float, default=0.5)
    p.add_argument("--max-point-outlier-fraction", type=float, default=0.25)
    p.add_argument("--skip-current-joint-range-check", action="store_true")
    p.add_argument("--skip-gripper-safety-check", action="store_true")
    p.add_argument("--skip-point-cloud-distribution-check", action="store_true")
    p.add_argument("--clip-actions", action="store_true")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--offline-smoke", action="store_true")
    p.add_argument("--enable-timeout", type=float, default=5.0)
    p.add_argument("--log", type=Path, default=Path("piper_inference/control_clean_dp3.json"))
    return p.parse_args()


def hz_summary(times: list[float]) -> dict:
    intervals = np.diff(times)
    if len(intervals) == 0:
        return {"count": 0, "mean_hz": None, "median_hz": None}
    return {
        "count": int(len(intervals)),
        "mean_hz": float(1.0 / intervals.mean()),
        "median_hz": float(1.0 / np.median(intervals)),
        "p05_interval_s": float(np.percentile(intervals, 5)),
        "p95_interval_s": float(np.percentile(intervals, 95)),
    }


def frame_id(frame: dict) -> float:
    """D435 timestamps identify source frames; not a control-loop timestamp."""
    return float(frame["timestamp"])


def validate_args(args):
    if args.rate <= 0 or args.max_steps <= 0 or args.camera_fps <= 0:
        raise ValueError("--rate、--camera-fps 和 --max-steps 必须为正")
    if not 1 <= args.speed_percent <= 100:
        raise ValueError("--speed-percent 必须在 [1,100]")
    if args.max_joint_speed_rad_s <= 0 or args.max_joint_accel_rad_s2 <= 0:
        raise ValueError("规划器速度和加速度必须为正")
    if args.max_camera_gap_ms <= 0:
        raise ValueError("--max-camera-gap-ms 必须为正")
    if args.policy_rate < 0 or args.policy_rate > args.rate:
        raise ValueError("--policy-rate 必须为0或位于 (0,--rate] 范围内")


def main():
    args = parse_args()
    validate_args(args)
    args.output_dir = args.output_dir.expanduser().resolve()
    policy_subdir = Path(args.policy_subdir)
    if policy_subdir.is_absolute() or ".." in policy_subdir.parts or not policy_subdir.parts:
        raise ValueError("--policy-subdir 必须是 --output-dir 下的相对路径")
    args.policy_subdir = str(policy_subdir)
    for required in (
        ".hydra/config.yaml",
        policy_subdir / "model.pt",
        policy_subdir / "encoder.pt",
    ):
        path = args.output_dir / required
        if not path.is_file():
            raise FileNotFoundError(f"DP3训练输出不完整，缺少: {path}")
    if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
        cpus = set(os.sched_getaffinity(0))
        if 0 in cpus and len(cpus) > 1:
            os.sched_setaffinity(0, cpus - {0})

    camera = None
    piper = None
    worker = None
    records = []
    stop_reason = "max_steps"
    robot_enabled = False
    last_state = None
    control_times, send_times, policy_submit_times = [], [], []
    infer_done_times, infer_durations = [], []
    try:
        if not args.offline_smoke:
            raw_camera = RealSense(
                fps=args.camera_fps, color_width=640, color_height=480,
                depth_width=640, depth_height=480, num_points=512,
                point_cloud_frame="camera", align_depth_to_color=False,
            )
            camera = base.LatestFrameCamera(raw_camera)
            camera.start()

        cfg, dataset, policy, use_cm = contact.load_policy_and_dataset(
            args.output_dir, args.policy_subdir, args.device
        )
        n_obs = int(cfg.n_obs_steps)
        n_actions = int(cfg.n_action_steps)
        if int(cfg.horizon) != n_obs - 1 + n_actions:
            raise RuntimeError("DP3 horizon 与 n_obs_steps/n_action_steps 不一致")
        if args.ddim_steps is not None:
            if not 1 <= args.ddim_steps <= 10:
                raise ValueError("--ddim-steps 必须在 [1,10]")
            policy.ddim_inference_steps = args.ddim_steps
        if args.offline_smoke:
            base.offline_smoke(dataset, policy, torch.device(args.device), use_cm, n_actions)
            return

        stats = base.training_stats(dataset)
        device = torch.device(args.device)
        piper = base.PiperProcessProxy(args.can, args.piper_sdk_root)
        piper.ConnectPort()
        reader = base.PiperPickPlaceStateReader(piper)
        policy_mode = (
            "仅新相机帧触发"
            if args.policy_rate == 0
            else f"独立请求={args.policy_rate:g}Hz（复用最新真实观测窗口）"
        )
        print(f"[时序] 控制/下发={args.rate:g}Hz；相机={args.camera_fps}Hz；"
              f"策略{policy_mode}", flush=True)
        if args.policy_rate > args.camera_fps:
            print(
                "[时序警告] 策略请求快于相机；同一观测会被重复采样，"
                "适合吞吐测试，但随机扩散输出可能增加真机目标抖动。",
                flush=True,
            )
        print(f"[模型] n_obs_steps={n_obs}, n_action_steps={n_actions}, "
              f"DDIM={getattr(policy, 'ddim_inference_steps', 'default')}", flush=True)

        # Build an honest history: no repeated stale RGB-D frames.
        history = deque(maxlen=n_obs)
        last_cam_id = None
        while len(history) < n_obs:
            frame = camera.get_frame()
            fid = frame_id(frame)
            if fid == last_cam_id:
                time.sleep(min(0.005, 0.5 / args.camera_fps))
                continue
            state, _, _ = reader.read()
            image, pc = base.preprocess_camera_frame(frame)
            history.append({"agent_pos": state, "point_cloud": pc, "image": image,
                            "camera_id": fid, "captured_at": time.monotonic()})
            last_cam_id = fid
        warm_obs = base.build_obs(history, device)
        with torch.inference_mode():
            base.predict_policy(policy, warm_obs, device, use_cm)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        print("[CUDA] DP3预热完成", flush=True)

        state, _, _ = reader.read()
        last_state = state.copy()
        if args.execute:
            phrase = input("确认工作区安全后输入 EXECUTE DP3 PIPER 继续：").strip()
            if phrase != "EXECUTE DP3 PIPER":
                raise RuntimeError("确认短语不匹配，取消执行")
            contact.enable_robot_for_position_control(piper, args.speed_percent, args.enable_timeout)
            robot_enabled = True
            base.enable_gripper(piper, reader, float(state[6]),
                                require_homing=not args.skip_gripper_safety_check)

            # The operator confirmation can take seconds.  Never feed the
            # policy confirmation-era frames (or an inference queued from
            # them): collect a complete, temporally valid history again.
            history.clear()
            last_cam_id = None
            while len(history) < n_obs:
                frame = camera.get_frame()
                fid = frame_id(frame)
                if fid == last_cam_id:
                    time.sleep(min(0.005, 0.5 / args.camera_fps))
                    continue
                state, _, _ = reader.read()
                image, pc = base.preprocess_camera_frame(frame)
                history.append({"agent_pos": state, "point_cloud": pc, "image": image,
                                "camera_id": fid, "captured_at": time.monotonic()})
                last_cam_id = fid
            last_state = state.copy()
            print("[时序] 已在确认后重新收集完整的3帧观测历史", flush=True)

        # safe_action owns physical/training target bounds. The planner, rather
        # than this function's old one-control-period limiter, owns velocity.
        target_safety = copy.copy(args)
        target_safety.max_joint_speed_rad_s = 1e6
        target_safety.max_gripper_speed_m_s = 1e6
        target_safety.rate = 1.0
        planner = JointTrajectoryPlanner(
            state, args.max_joint_speed_rad_s, args.max_joint_accel_rad_s2,
            args.max_gripper_speed_m_s, args.planner_tracking_error_rad,
        )
        worker = base.AsyncPolicyWorker(policy, device, use_cm, None, n_actions)
        worker.start()
        latest_policy_obs = base.build_obs(history, device)
        worker.submit(latest_policy_obs, control_step=0)
        policy_submit_times.append(time.monotonic())
        active_target = None
        last_result_id = -1
        last_joint_ts = last_grip_ts = None
        joint_fresh = grip_fresh = time.monotonic()
        next_tick = time.monotonic()
        last_tick = next_tick
        next_policy_submit = next_tick + (
            1.0 / args.policy_rate if args.policy_rate > 0 else float("inf")
        )
        last_camera_mono = history[-1]["captured_at"]

        for step in range(args.max_steps):
            now = time.monotonic()
            if now < next_tick:
                time.sleep(next_tick - now)
            tick = time.monotonic()
            dt = tick - last_tick
            last_tick = tick
            next_tick += 1.0 / args.rate
            # Avoid accumulating a large catch-up burst after an OS stall.
            if next_tick < tick - 1.0 / args.rate:
                next_tick = tick + 1.0 / args.rate
            control_times.append(tick)

            state, joint_ts, grip_ts = reader.read()
            last_state = state.copy()
            if joint_ts != last_joint_ts: joint_fresh = tick
            if grip_ts != last_grip_ts: grip_fresh = tick
            last_joint_ts, last_grip_ts = joint_ts, grip_ts
            if tick - joint_fresh > args.state_stale_seconds or tick - grip_fresh > args.state_stale_seconds:
                raise RuntimeError("Piper反馈超过 --state-stale-seconds 未更新")

            frame = camera.get_frame()
            fid = frame_id(frame)
            new_frame = fid != last_cam_id
            pc_fraction = None
            if new_frame:
                capture_mono = tick
                camera_gap = capture_mono - last_camera_mono
                last_camera_mono = capture_mono
                last_cam_id = fid
                if camera_gap * 1000.0 > args.max_camera_gap_ms:
                    history.clear()
                    active_target = None
                    latest_policy_obs = None
                    print(f"[相机] 间隔 {camera_gap * 1000:.1f}ms 超限，清空观测历史", flush=True)
                image, pc = base.preprocess_camera_frame(frame)
                pc_fraction = base.point_cloud_outlier_fraction(pc, stats)
                if (not args.skip_point_cloud_distribution_check and
                        pc_fraction > args.max_point_outlier_fraction):
                    raise RuntimeError(f"点云训练分布外比例 {pc_fraction:.3f} 超限")
                history.append({"agent_pos": state, "point_cloud": pc, "image": image,
                                "camera_id": fid, "captured_at": capture_mono})
                if len(history) == n_obs:
                    latest_policy_obs = base.build_obs(history, device)
                    if args.policy_rate == 0:
                        worker.submit(latest_policy_obs, control_step=step)
                        policy_submit_times.append(time.monotonic())

            if (
                args.policy_rate > 0
                and latest_policy_obs is not None
                and tick >= next_policy_submit
            ):
                worker.submit(latest_policy_obs, control_step=step)
                policy_submit_times.append(time.monotonic())
                policy_period = 1.0 / args.policy_rate
                next_policy_submit += policy_period
                if next_policy_submit < tick:
                    next_policy_submit = tick + policy_period

            result = worker.latest()
            if result is not None and result[0] > last_result_id:
                result_id, origin_step, chunk, elapsed, completed_at = result
                last_result_id = result_id
                infer_done_times.append(completed_at)
                infer_durations.append(elapsed)
                active_target, warnings = base.safe_action(chunk[0], state, stats, target_safety)
                if warnings and step % max(1, int(args.rate)) == 0:
                    print("[安全] " + "；".join(warnings), flush=True)

            planned = planner.hold(state, dt) if active_target is None else planner.step(active_target, state, dt)
            if args.execute:
                send_start = time.monotonic()
                base.send_action(piper, planned, args.speed_percent)
                send_times.append(send_start)
            if step % max(1, int(args.rate)) == 0:
                age = None if active_target is None or not history else tick - history[-1]["captured_at"]
                print(f"[step {step:05d}] control={args.rate:g}Hz new_camera={new_frame} "
                      f"infer={hz_summary(infer_done_times)['mean_hz']}Hz policy_age={age}", flush=True)
            records.append({"step": step, "time_s": tick, "new_camera_frame": new_frame,
                            "camera_timestamp_s": fid, "pc_outlier_fraction": pc_fraction,
                            "policy_result_id": last_result_id,
                            "active_target": None if active_target is None else active_target.tolist(),
                            "planned_action": planned.tolist()})
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
    finally:
        if worker is not None: worker.stop()
        if piper is not None:
            try:
                if robot_enabled and last_state is not None:
                    # Keep the current pose rather than disabling motors or
                    # calling a hard stop when an episode ends normally.
                    for _ in range(10):
                        base.send_action(piper, last_state, args.speed_percent)
                        time.sleep(0.02)
            except Exception as exc:
                print(f"[Piper] 保持失败: {exc}", flush=True)
            piper.close()
        if camera is not None: camera.stop()
        payload = {"meta": {"output_dir": str(args.output_dir), "policy": args.policy_subdir,
                   "control_rate_hz": args.rate, "camera_fps_requested": args.camera_fps,
                   "policy_rate_requested_hz": args.policy_rate,
                   "n_obs_steps": locals().get("n_obs"), "n_action_steps": locals().get("n_actions"),
                   "stop_reason": stop_reason,
                   "cadence_monitor": {"control": hz_summary(control_times),
                                        "send_call": hz_summary(send_times),
                                        "policy_submit": hz_summary(policy_submit_times),
                                        "inference_completion": hz_summary(infer_done_times),
                                        "inference_duration_s": {"count": len(infer_durations),
                                                                 "mean_s": float(np.mean(infer_durations)) if infer_durations else None,
                                                                 "p95_s": float(np.percentile(infer_durations, 95)) if infer_durations else None}}},
                   "records": records}
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[日志] 已保存: {args.log.resolve()}", flush=True)


if __name__ == "__main__":
    main()
