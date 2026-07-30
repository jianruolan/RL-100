#!/usr/bin/env python3
"""Local RealSense/Piper client for a remote RL-100 DP3 inference server.

Shadow mode is the default. Real Piper commands require both ``--execute`` and
the exact interactive confirmation phrase. The server only proposes an action;
all freshness checks, training/physical bounds, trajectory limiting, watchdogs,
and final SDK calls remain local.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

import grpc
import numpy as np

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.local import piper_remote_runtime as runtime

EXPECTED_OUTPUT_NAME = (
    "piper_pick_and_place_augmented_chunk4_control_clean_dp3_medium_"
    "episode10_bs64_epoch4000_seed42"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="127.0.0.1:50051")
    parser.add_argument("--expected-output-name", default=EXPECTED_OUTPUT_NAME)
    parser.add_argument("--expected-policy-subdir", default="bc")
    parser.add_argument("--connect-timeout", type=float, default=5.0)
    parser.add_argument("--rpc-timeout", type=float, default=0.20)
    parser.add_argument("--max-action-age-ms", type=float, default=250.0)
    parser.add_argument("--offline-smoke", action="store_true")
    parser.add_argument("--check-server", action="store_true")
    parser.add_argument("--can", default="can0")
    parser.add_argument("--piper-sdk-root", type=Path, default=None)
    parser.add_argument("--rate", type=float, default=100.0)
    parser.add_argument("--camera-fps", type=int, default=15)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--ros-node-name", default="remote_dp3_piper_bridge")
    parser.add_argument("--ros-publish-rate", type=float, default=20.0)
    parser.add_argument("--speed-percent", type=int, default=10)
    parser.add_argument("--max-joint-speed-rad-s", type=float, default=0.20)
    parser.add_argument("--max-joint-accel-rad-s2", type=float, default=0.50)
    parser.add_argument("--max-gripper-speed-m-s", type=float, default=0.02)
    parser.add_argument("--planner-tracking-error-rad", type=float, default=0.10)
    parser.add_argument("--state-stale-seconds", type=float, default=0.5)
    parser.add_argument("--max-camera-gap-ms", type=float, default=200.0)
    parser.add_argument("--max-point-outlier-fraction", type=float, default=0.25)
    parser.add_argument("--dataset-margin-rad", type=float, default=0.03)
    parser.add_argument("--gripper-margin-m", type=float, default=0.005)
    parser.add_argument("--gripper-command-threshold", type=float, default=0.5)
    parser.add_argument("--clip-actions", action="store_true")
    parser.add_argument("--skip-current-joint-range-check", action="store_true")
    parser.add_argument("--skip-gripper-safety-check", action="store_true")
    parser.add_argument("--skip-point-cloud-distribution-check", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--enable-timeout", type=float, default=5.0)
    parser.add_argument(
        "--log",
        type=Path,
        default=REPO_ROOT / "data/piper_inference/remote_dp3_latest.json",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "rate": args.rate,
        "camera-fps": args.camera_fps,
        "max-steps": args.max_steps,
        "rpc-timeout": args.rpc_timeout,
        "max-action-age-ms": args.max_action_age_ms,
        "state-stale-seconds": args.state_stale_seconds,
        "ros-publish-rate": args.ros_publish_rate,
    }
    for name, value in positive.items():
        if float(value) <= 0:
            raise ValueError(f"--{name}必须为正数")
    if not 1 <= args.speed_percent <= 100:
        raise ValueError("--speed-percent必须在[1,100]")
    if args.max_joint_speed_rad_s <= 0 or args.max_joint_accel_rad_s2 <= 0:
        raise ValueError("轨迹速度和加速度必须为正数")
    if not 0 < args.gripper_command_threshold < 1:
        raise ValueError("--gripper-command-threshold必须在(0,1)")


def print_contract(contract: runtime.ServerContract) -> None:
    print(
        "[服务器] "
        f"model={contract.model_version}, "
        f"output={contract.output_dir}, policy={contract.policy_subdir}, "
        f"n_obs_steps={contract.n_obs_steps}, "
        f"n_action_steps={contract.n_action_steps}, "
        f"action_key={contract.stats.action_key}, "
        f"gripper={contract.stats.gripper_action_mode}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.offline_smoke:
        runtime.offline_smoke()
        return

    client = runtime.PolicyClient(args.server, args.connect_timeout)
    contract = client.get_contract(args.rpc_timeout)
    contract.validate_model_identity(
        args.expected_output_name, args.expected_policy_subdir
    )
    print_contract(contract)
    if args.check_server:
        print("[check-server] 协议、模型形状和安全元数据检查通过", flush=True)
        client.close()
        return

    episode_id = f"piper-{uuid.uuid4().hex}"
    client.reset_episode(episode_id, args.rpc_timeout)

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
    camera = runtime.LatestFrameCamera(raw_camera)
    ros_node = None
    piper = None
    reader = None
    worker = None
    robot_enabled = False
    last_state = None
    stop_reason = "max_steps"
    records: list[dict] = []
    request_count = response_count = send_count = 0
    try:
        from tools.local import ros_piper_bridge_node as ros_bridge

        ros_node = ros_bridge.init_ros_node(args.ros_node_name)
        print(
            "[ROS 2] node=/" + args.ros_node_name
            + ", feedback=/remote_dp3/joint_states_feedback, "
            + "target=/remote_dp3/joint_target",
            flush=True,
        )
        camera.start()
        interface_cls = runtime.import_piper_sdk(args.piper_sdk_root)
        piper = interface_cls(args.can)
        piper.ConnectPort()
        reader = runtime.PiperStateReader(piper)
        state, _, _ = reader.read()
        last_state = state.copy()

        print("[模式] " + ("真机执行" if args.execute else "SHADOW，不下发"), flush=True)
        print(
            f"[时序] control={args.rate:g}Hz, camera={args.camera_fps}Hz, "
            f"RPC timeout={args.rpc_timeout*1000:.0f}ms, "
            f"action age<={args.max_action_age_ms:.0f}ms",
            flush=True,
        )
        if args.execute:
            print("即将使能真实Piper；请确认工作区安全且急停可用。", flush=True)
            phrase = input("输入 EXECUTE REMOTE DP3 PIPER 继续：").strip()
            if phrase != "EXECUTE REMOTE DP3 PIPER":
                raise RuntimeError("确认短语不匹配，取消执行")
            runtime.enable_robot(piper, args.speed_percent, args.enable_timeout)
            robot_enabled = True
            runtime.enable_gripper(
                piper,
                reader,
                float(state[6]),
                require_homing=not args.skip_gripper_safety_check,
            )

        worker = runtime.AsyncPolicyClient(
            client, args.rpc_timeout, contract.n_action_steps
        )
        worker.start()
        console = runtime.OperatorConsole()
        console.start()
        print("[人工] Enter/stop停止并保持；estop发送硬急停。", flush=True)

        planner = runtime.JointTrajectoryPlanner(
            state,
            args.max_joint_speed_rad_s,
            args.max_joint_accel_rad_s2,
            args.max_gripper_speed_m_s,
            args.planner_tracking_error_rad,
        )
        sequence_id = 0
        last_frame_id = None
        last_camera_monotonic = time.monotonic()
        last_joint_timestamp = last_gripper_timestamp = None
        last_joint_fresh = last_gripper_fresh = time.monotonic()
        last_result_sequence = -1
        last_rpc_error_repr = None
        active_target = None
        active_result = None
        next_tick = time.monotonic()
        last_tick = next_tick
        ros_publish_period = 1.0 / args.ros_publish_rate
        next_ros_publish = next_tick

        for step in range(args.max_steps):
            now = time.monotonic()
            if now < next_tick:
                time.sleep(next_tick - now)
            tick = time.monotonic()
            dt = tick - last_tick
            last_tick = tick
            next_tick += 1.0 / args.rate
            if next_tick < tick - 1.0 / args.rate:
                next_tick = tick + 1.0 / args.rate

            ros_bridge.spin_ros_once(ros_node)
            if ros_node.estop_requested:
                stop_reason = "ros_estop"
                if robot_enabled:
                    piper.MotionCtrl_1(0x01, 0x00, 0x00)
                break
            if ros_node.stop_requested:
                stop_reason = "ros_operator_hold"
                break

            command = console.poll()
            if command in ("stop", "quit", "q"):
                stop_reason = "operator_hold"
                break
            if command == "estop":
                stop_reason = "operator_estop"
                if robot_enabled:
                    piper.MotionCtrl_1(0x01, 0x00, 0x00)
                break

            state, joint_timestamp, gripper_timestamp = reader.read()
            last_state = state.copy()
            if joint_timestamp != last_joint_timestamp:
                last_joint_fresh = tick
            if gripper_timestamp != last_gripper_timestamp:
                last_gripper_fresh = tick
            last_joint_timestamp = joint_timestamp
            last_gripper_timestamp = gripper_timestamp
            if (
                tick - last_joint_fresh > args.state_stale_seconds
                or tick - last_gripper_fresh > args.state_stale_seconds
            ):
                raise RuntimeError("Piper反馈超过允许时间未更新")

            frame = camera.get_frame()
            frame_id = float(frame["timestamp"])
            new_frame = frame_id != last_frame_id
            point_cloud_fraction = None
            if new_frame:
                camera_gap = tick - last_camera_monotonic
                last_camera_monotonic = tick
                last_frame_id = frame_id
                if request_count > 0 and camera_gap * 1000 > args.max_camera_gap_ms:
                    active_target = None
                    raise RuntimeError(
                        f"RealSense新帧间隔{camera_gap*1000:.1f}ms超限"
                    )
                image, point_cloud = runtime.preprocess_camera_frame(frame)
                point_cloud_fraction = runtime.point_cloud_outlier_fraction(
                    point_cloud, contract.stats
                )
                if (
                    not args.skip_point_cloud_distribution_check
                    and point_cloud_fraction > args.max_point_outlier_fraction
                ):
                    raise RuntimeError(
                        f"点云训练分布外比例{point_cloud_fraction:.3f}超限"
                    )
                sequence_id += 1
                request = runtime.make_request(
                    episode_id,
                    sequence_id,
                    time.time_ns(),
                    state,
                    image,
                    point_cloud,
                )
                worker.submit(request)
                request_count += 1

            result, rpc_error = worker.snapshot()
            if rpc_error is not None:
                error_repr = repr(rpc_error)
                if error_repr != last_rpc_error_repr:
                    print(f"[网络] RPC失败: {rpc_error}", flush=True)
                    last_rpc_error_repr = error_repr
                active_target = None
                if robot_enabled:
                    raise RuntimeError("远程推理RPC失败，停止真机策略") from rpc_error
            if result is not None and result.sequence_id > last_result_sequence:
                last_result_sequence = result.sequence_id
                response_count += 1
                last_rpc_error_repr = None
                if result.action_chunk is None:
                    active_target = None
                    active_result = None
                    print(
                        f"[服务器] seq={result.sequence_id}未就绪: "
                        f"{result.status_message}",
                        flush=True,
                    )
                else:
                    age_ms = (tick - result.submitted_monotonic) * 1000.0
                    if age_ms > args.max_action_age_ms:
                        active_target = None
                        if robot_enabled:
                            raise RuntimeError(
                                f"服务器动作到达时已过期: {age_ms:.1f}ms"
                            )
                    else:
                        active_target, warnings = runtime.safe_policy_target(
                            result.action_chunk[0], state, contract.stats, args
                        )
                        active_result = result
                        if warnings:
                            print("[安全] " + "；".join(warnings), flush=True)

            if active_result is not None:
                action_age_ms = (
                    tick - active_result.submitted_monotonic
                ) * 1000.0
                if action_age_ms > args.max_action_age_ms:
                    active_target = None
                    active_result = None
                    if robot_enabled:
                        raise RuntimeError("当前远程动作已过期，停止真机策略")

            planned = (
                planner.hold(state, dt)
                if active_target is None
                else planner.step(active_target, state, dt)
            )
            if robot_enabled:
                runtime.send_action(piper, planned, args.speed_percent)
                send_count += 1

            if tick >= next_ros_publish:
                ros_node.publish_cycle(
                    state,
                    planned,
                    executing=robot_enabled,
                    status={
                        "phase": "running",
                        "mode": "execute" if robot_enabled else "shadow",
                        "sequence_id": sequence_id,
                        "response_sequence_id": last_result_sequence,
                        "target": "ready" if active_target is not None else "hold",
                    },
                )
                next_ros_publish = tick + ros_publish_period

            if step % max(1, int(args.rate)) == 0:
                print(
                    f"[step {step:05d}] frame={new_frame} "
                    f"req={request_count} resp={response_count} "
                    f"target={'READY' if active_target is not None else 'HOLD'} "
                    f"{'EXEC' if robot_enabled else 'SHADOW'}",
                    flush=True,
                )
            records.append(
                {
                    "step": step,
                    "sequence_id": sequence_id,
                    "response_sequence_id": last_result_sequence,
                    "new_camera_frame": new_frame,
                    "point_cloud_outlier_fraction": point_cloud_fraction,
                    "state": state.tolist(),
                    "active_target": (
                        None if active_target is None else active_target.tolist()
                    ),
                    "planned": planned.tolist(),
                    "executed": robot_enabled,
                }
            )
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt_hold"
    except grpc.RpcError as exc:
        stop_reason = f"grpc_error:{exc.code().name}"
        raise
    except Exception:
        stop_reason = "exception_hold"
        raise
    finally:
        if worker is not None:
            worker.stop()
        if piper is not None:
            try:
                if (
                    robot_enabled
                    and last_state is not None
                    and "estop" not in stop_reason
                ):
                    for _ in range(10):
                        runtime.send_action(piper, last_state, args.speed_percent)
                        time.sleep(0.02)
                    print("[Piper] 已保持最后有效反馈姿态", flush=True)
            except Exception as exc:
                print(f"[Piper] 保持失败: {exc}", file=sys.stderr, flush=True)
            try:
                disconnect = getattr(piper, "DisconnectPort", None)
                if callable(disconnect):
                    disconnect()
            except Exception:
                pass
        if ros_node is not None:
            try:
                ros_node.publish_final(executing=False, stop_reason=stop_reason)
                ros_bridge.spin_ros_once(ros_node)
            except Exception as exc:
                print(f"[ROS 2] 最终状态发布失败: {exc}", file=sys.stderr)
            ros_bridge.shutdown_ros(ros_node)
        camera.stop()
        try:
            client.reset_episode(episode_id, args.rpc_timeout)
        except Exception:
            pass
        client.close()
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(
            json.dumps(
                {
                    "meta": {
                        "server": args.server,
                        "protocol_version": runtime.PROTOCOL_VERSION,
                        "model_version": contract.model_version,
                        "episode_id": episode_id,
                        "execute": args.execute,
                        "rate_hz": args.rate,
                        "camera_fps": args.camera_fps,
                        "rpc_timeout_s": args.rpc_timeout,
                        "max_action_age_ms": args.max_action_age_ms,
                        "request_count": request_count,
                        "response_count": response_count,
                        "send_count": send_count,
                        "stop_reason": stop_reason,
                    },
                    "records": records,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[日志] {args.log.resolve()}", flush=True)


if __name__ == "__main__":
    main()
