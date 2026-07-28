#!/usr/bin/env python3
"""带在线关节轨迹平滑的 Piper pick-and-place 推理入口。

该脚本复用 ``infer_piper_pick_and_place_policy.py`` 的模型、相机、异步推理、
安全检查和真机确认流程，仅在安全动作与 Piper 下发之间加入有状态的关节轨迹
规划器。原推理脚本的默认行为不会被修改。

默认启用轨迹平滑和 Temporal Ensemble；分别传入
``--no-trajectory-smoothing``、``--no-temporal-ensemble`` 可以独立关闭。
Temporal Ensemble 按绝对控制步对齐多次推理产生的重叠 action chunk，再由规划器
限制每个周期的关节速度、加速度和 jerk（加速度变化率）。夹爪默认使用最新
chunk 的预测，不做连续平均，避免开合命令被融合成不确定状态。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.teleop_off2off_data import infer_piper_pick_and_place_policy as base


class JerkLimitedJointPlanner:
    """离散时间、可在线更新目标的六关节轨迹平滑器。"""

    def __init__(
        self,
        rate_hz: float,
        max_velocity: float,
        max_acceleration: float,
        max_jerk: float,
        response_rad_s: float,
        feedback_resync_rad: float,
    ) -> None:
        self.dt = 1.0 / float(rate_hz)
        self.max_velocity = float(max_velocity)
        self.max_acceleration = float(max_acceleration)
        self.max_jerk = float(max_jerk)
        self.response_rad_s = float(response_rad_s)
        self.feedback_resync_rad = float(feedback_resync_rad)
        self.position: np.ndarray | None = None
        self.velocity = np.zeros(6, dtype=np.float64)
        self.acceleration = np.zeros(6, dtype=np.float64)
        self.resync_count = 0

    def reset(self, feedback_position: np.ndarray) -> None:
        self.position = np.asarray(feedback_position, dtype=np.float64).copy()
        self.velocity.fill(0.0)
        self.acceleration.fill(0.0)

    def step(
        self,
        target_position: np.ndarray,
        feedback_position: np.ndarray,
        lower_position: np.ndarray | None = None,
        upper_position: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        target = np.asarray(target_position, dtype=np.float64)
        feedback = np.asarray(feedback_position, dtype=np.float64)
        if self.position is None:
            self.reset(feedback)

        assert self.position is not None
        feedback_error = float(np.max(np.abs(feedback - self.position)))
        resynced = feedback_error > self.feedback_resync_rad
        if resynced:
            # 机械臂未跟上规划轨迹或被人工移动时，从真实反馈重新起步，避免
            # 继续追逐已经失效的内部轨迹。
            self.reset(feedback)
            self.resync_count += 1

        error = target - self.position

        # 临界阻尼二阶目标跟踪：a = w^2 * position_error - 2w * velocity。
        # 相比仅按制动距离反转速度，它在离散控制下不会围绕固定目标形成
        # 持续极限环。之后再依次施加 jerk、加速度和速度硬约束。
        omega = self.response_rad_s
        desired_acceleration = np.clip(
            omega * omega * error - 2.0 * omega * self.velocity,
            -self.max_acceleration,
            self.max_acceleration,
        )
        acceleration_delta = np.clip(
            desired_acceleration - self.acceleration,
            -self.max_jerk * self.dt,
            self.max_jerk * self.dt,
        )
        self.acceleration = np.clip(
            self.acceleration + acceleration_delta,
            -self.max_acceleration,
            self.max_acceleration,
        )
        self.velocity = np.clip(
            self.velocity + self.acceleration * self.dt,
            -self.max_velocity,
            self.max_velocity,
        )
        self.position = self.position + self.velocity * self.dt
        limit_clipped = np.zeros(6, dtype=bool)
        if lower_position is not None and upper_position is not None:
            lower = np.asarray(lower_position, dtype=np.float64)
            upper = np.asarray(upper_position, dtype=np.float64)
            clipped_position = np.clip(self.position, lower, upper)
            limit_clipped = clipped_position != self.position
            self.position = clipped_position
            # 触及硬边界时清除该轴的动态状态，防止积分状态持续把命令推向
            # 边界外。边界处的安全性优先于轨迹连续性。
            self.velocity[limit_clipped] = 0.0
            self.acceleration[limit_clipped] = 0.0

        diagnostics = {
            "trajectory_velocity_rad_s": self.velocity.astype(float).tolist(),
            "trajectory_acceleration_rad_s2": self.acceleration.astype(float).tolist(),
            "trajectory_feedback_error_rad": feedback_error,
            "trajectory_resynced": resynced,
            "trajectory_resync_count": self.resync_count,
            "trajectory_limit_clipped": limit_clipped.astype(bool).tolist(),
        }
        return self.position.astype(np.float32), diagnostics


class TemporalChunkEnsembler:
    """按绝对控制步融合不同推理产生的重叠 action chunk。"""

    def __init__(self, decay: float, max_predictions: int) -> None:
        self.decay = float(decay)
        self.max_predictions = int(max_predictions)
        self._chunks: list[tuple[int, np.ndarray]] = []

    def add(self, origin_step: int, chunk: np.ndarray) -> None:
        value = np.asarray(chunk, dtype=np.float32)
        if value.ndim != 2 or value.shape[1] != 7:
            raise RuntimeError(f"Temporal Ensemble要求action chunk形状为[T,7]，当前{value.shape}")
        self._chunks.append((int(origin_step), value.copy()))
        # 同一绝对动作时刻最多只可能被最近 T 个 chunk 覆盖，保留额外一份
        # 余量以兼容异步推理延迟。
        keep = max(self.max_predictions, int(value.shape[0])) + 1
        if len(self._chunks) > keep:
            self._chunks = self._chunks[-keep:]

    def build(
        self,
        block_start_step: int,
        output_steps: int,
    ) -> tuple[np.ndarray, list[int], list[list[int]], list[bool]]:
        if not self._chunks:
            raise RuntimeError("Temporal Ensemble尚未收到action chunk")

        newest_chunk = self._chunks[-1][1]
        result = np.empty((output_steps, 7), dtype=np.float32)
        prediction_counts: list[int] = []
        prediction_origins: list[list[int]] = []
        gripper_conflicts: list[bool] = []
        for output_index in range(output_steps):
            absolute_step = int(block_start_step + output_index)
            candidates: list[tuple[int, np.ndarray]] = []
            for origin, chunk in self._chunks:
                chunk_index = absolute_step - origin
                if 0 <= chunk_index < len(chunk):
                    candidates.append((origin, chunk[chunk_index]))
            if not candidates:
                # 异步延迟超过chunk长度时沿用最新chunk相同相对位置，避免
                # 因融合历史耗尽而中断真机控制。
                fallback_index = min(output_index, len(newest_chunk) - 1)
                candidates = [(absolute_step, newest_chunk[fallback_index])]

            # origin越大表示预测越新。仅保留最近若干预测；同一动作时刻下，
            # 新预测的 horizon offset 更小，因此指数权重更大。
            candidates.sort(key=lambda item: item[0], reverse=True)
            candidates = candidates[: self.max_predictions]
            origins = [item[0] for item in candidates]
            actions = np.stack([item[1] for item in candidates], axis=0)
            ages = absolute_step - np.asarray(origins, dtype=np.float64)
            weights = np.exp(-self.decay * ages)
            weights /= weights.sum()
            gripper_open = actions[:, 6] >= 0.5
            gripper_conflict = bool(np.any(gripper_open != gripper_open[0]))
            if gripper_conflict:
                # 不允许“最新夹爪阶段 + 旧阶段平均关节姿态”的组合。抓取或
                # 放置边界处整条7D动作采用最新预测，保持姿态与夹爪语义一致。
                result[output_index] = actions[0]
            else:
                result[output_index, :6] = np.sum(
                    actions[:, :6] * weights[:, None], axis=0
                )
                # 夹爪0/1命令不做平均，直接采用最新一次推理的判断。
                result[output_index, 6] = actions[0, 6]
            prediction_counts.append(len(candidates))
            prediction_origins.append(origins)
            gripper_conflicts.append(gripper_conflict)
        return result, prediction_counts, prediction_origins, gripper_conflicts


class GripperPoseCoordinator:
    """将离散夹爪切换与对应关节姿态对齐，并在切换后短暂保持。"""

    def __init__(
        self,
        pose_tolerance_rad: float,
        command_stable_steps: int,
        stable_steps: int,
        hold_steps: int,
        max_wait_steps: int,
    ) -> None:
        self.pose_tolerance_rad = float(pose_tolerance_rad)
        self.command_stable_steps = int(command_stable_steps)
        self.stable_steps = int(stable_steps)
        self.hold_steps = int(hold_steps)
        self.max_wait_steps = int(max_wait_steps)
        self.stable_width: float | None = None
        self.pending_width: float | None = None
        self.latched_joints: np.ndarray | None = None
        self.aligned_steps = 0
        self.wait_steps = 0
        self.hold_remaining = 0
        self.transition_count = 0
        self.candidate_width: float | None = None
        self.candidate_steps = 0

    @staticmethod
    def _binary_width(width: float) -> float:
        return (
            base.GRIPPER_UPPER_M
            if float(width) >= (base.GRIPPER_LOWER_M + base.GRIPPER_UPPER_M) / 2.0
            else base.GRIPPER_LOWER_M
        )

    def step(
        self,
        policy_target: np.ndarray,
        feedback: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        target = np.asarray(policy_target, dtype=np.float32).copy()
        if self.stable_width is None:
            self.stable_width = self._binary_width(float(feedback[6]))

        desired_width = self._binary_width(float(target[6]))
        phase = "tracking"
        pose_error = 0.0

        if self.hold_remaining > 0:
            assert self.latched_joints is not None
            target[:6] = self.latched_joints
            target[6] = self.stable_width
            self.hold_remaining -= 1
            phase = "post_transition_hold"
            if self.hold_remaining == 0:
                self.latched_joints = None
        else:
            if self.pending_width is None:
                if desired_width != self.stable_width:
                    target[6] = self.stable_width
                    if desired_width == self.candidate_width:
                        self.candidate_steps += 1
                    else:
                        self.candidate_width = desired_width
                        self.candidate_steps = 1
                    phase = "debouncing_gripper_command"
                    if self.candidate_steps >= self.command_stable_steps:
                        self.pending_width = desired_width
                        self.latched_joints = target[:6].copy()
                        self.aligned_steps = 0
                        self.wait_steps = 0
                        self.candidate_width = None
                        self.candidate_steps = 0
                else:
                    self.candidate_width = None
                    self.candidate_steps = 0

            if self.pending_width is not None:
                assert self.latched_joints is not None
                target[:6] = self.latched_joints
                target[6] = self.stable_width
                pose_error = float(
                    np.max(np.abs(feedback[:6] - self.latched_joints))
                )
                self.wait_steps += 1
                if pose_error <= self.pose_tolerance_rad:
                    self.aligned_steps += 1
                else:
                    self.aligned_steps = 0
                phase = "waiting_for_transition_pose"

                if self.aligned_steps >= self.stable_steps:
                    self.stable_width = self.pending_width
                    self.pending_width = None
                    target[6] = self.stable_width
                    self.hold_remaining = self.hold_steps
                    self.transition_count += 1
                    phase = "transition_committed"
                elif self.wait_steps >= self.max_wait_steps:
                    transition = "张开" if self.pending_width > 0 else "闭合"
                    raise RuntimeError(
                        f"夹爪{transition}等待姿态超时：最大关节误差"
                        f"{pose_error:.4f}rad，阈值{self.pose_tolerance_rad:.4f}rad"
                    )

        diagnostics = {
            "gripper_pose_sync_phase": phase,
            "gripper_pose_sync_pose_error_rad": pose_error,
            "gripper_pose_sync_wait_steps": self.wait_steps,
            "gripper_pose_sync_hold_remaining": self.hold_remaining,
            "gripper_pose_sync_transition_count": self.transition_count,
            "gripper_pose_sync_pending_width_m": self.pending_width,
            "gripper_pose_sync_candidate_steps": self.candidate_steps,
        }
        return target, diagnostics


_ORIGINAL_ARGUMENT_PARSER = argparse.ArgumentParser
_ORIGINAL_PARSE_ARGS = base.parse_args
_ORIGINAL_SAFE_ACTION = base.safe_action
_ORIGINAL_ASYNC_POLICY_WORKER = base.AsyncPolicyWorker
_planner: JerkLimitedJointPlanner | None = None
_gripper_coordinator: GripperPoseCoordinator | None = None
_last_args: Any = None
_last_diagnostics: dict[str, Any] = {}
_diagnostics_history: list[dict[str, Any]] = []
_initial_log_mtime_ns: int | None = None
_current_temporal_diagnostics: dict[str, Any] = {}


class _SmoothingArgumentParser(_ORIGINAL_ARGUMENT_PARSER):
    """给原脚本的 parser 注入平滑参数，同时保留完整原始 CLI。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.add_argument(
            "--trajectory-smoothing",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="启用速度/加速度/jerk在线轨迹规划；本脚本默认启用。",
        )
        self.add_argument(
            "--trajectory-max-joint-speed-rad-s",
            type=float,
            default=0.35,
            help="轨迹规划器的各关节最大速度，默认0.35 rad/s。",
        )
        self.add_argument(
            "--trajectory-max-joint-acceleration-rad-s2",
            type=float,
            default=0.8,
            help="轨迹规划器的各关节最大加速度，默认0.8 rad/s^2。",
        )
        self.add_argument(
            "--trajectory-max-joint-jerk-rad-s3",
            type=float,
            default=4.0,
            help="轨迹规划器的各关节最大jerk，默认4.0 rad/s^3。",
        )
        self.add_argument(
            "--trajectory-response-rad-s",
            type=float,
            default=2.0,
            help="临界阻尼目标跟踪响应频率；越小越平滑但滞后越大，默认2.0 rad/s。",
        )
        self.add_argument(
            "--trajectory-feedback-resync-rad",
            type=float,
            default=0.20,
            help="反馈与内部规划位置偏差超过该值时重置规划状态，默认0.20 rad。",
        )
        self.add_argument(
            "--temporal-ensemble",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="按绝对控制步融合重叠action chunk；本脚本默认启用。",
        )
        self.add_argument(
            "--temporal-ensemble-decay",
            type=float,
            default=0.5,
            help="Temporal Ensemble指数衰减系数；越小越接近均匀平均，默认0.5。",
        )
        self.add_argument(
            "--temporal-ensemble-max-predictions",
            type=int,
            default=4,
            help="同一动作时刻最多融合的预测数，默认4。",
        )
        self.add_argument(
            "--temporal-ensemble-alignment",
            choices=["execution", "observation"],
            default="execution",
            help=(
                "chunk时间对齐基准：execution按异步结果开始执行时刻对齐，"
                "与原推理语义一致；observation按观测提交时刻补偿延迟。"
                "默认execution。"
            ),
        )
        self.add_argument(
            "--gripper-pose-sync",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="夹爪开合前锁存并等待对应关节姿态；本脚本默认启用。",
        )
        self.add_argument(
            "--gripper-pose-tolerance-rad",
            type=float,
            default=0.025,
            help="允许执行夹爪切换的最大关节姿态误差，默认0.025 rad。",
        )
        self.add_argument(
            "--gripper-pose-stable-steps",
            type=int,
            default=2,
            help="姿态连续满足阈值的控制周期数，默认2。",
        )
        self.add_argument(
            "--gripper-command-stable-steps",
            type=int,
            default=2,
            help="夹爪新开合命令连续出现多少周期后才锁存姿态，默认2。",
        )
        self.add_argument(
            "--gripper-transition-hold-steps",
            type=int,
            default=4,
            help="夹爪切换后保持锁存姿态的控制周期数，默认4。",
        )
        self.add_argument(
            "--gripper-pose-max-wait-steps",
            type=int,
            default=60,
            help="等待抓取/放置姿态的最大周期数，超时停止，默认60。",
        )


class _ArgparseProxy:
    """仅替换 base 模块看到的 ArgumentParser，不修改标准库模块全局。"""

    ArgumentParser = _SmoothingArgumentParser

    def __getattr__(self, name: str):
        return getattr(argparse, name)


class TemporalEnsembleAsyncPolicyWorker(_ORIGINAL_ASYNC_POLICY_WORKER):
    """在主控制线程读取异步结果时，对齐并融合重叠chunk。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._control_step = -1
        self._last_ensemble_request_id = -1
        self._active_diagnostics: dict[str, Any] = {}
        self._ensembler = TemporalChunkEnsembler(
            decay=_last_args.temporal_ensemble_decay,
            max_predictions=_last_args.temporal_ensemble_max_predictions,
        )

    def latest(self):
        global _current_temporal_diagnostics
        self._control_step += 1
        raw = super().latest()
        if not _last_args.temporal_ensemble:
            _current_temporal_diagnostics = {
                "temporal_ensemble_enabled": False,
            }
            return raw
        if raw is None:
            _current_temporal_diagnostics = {
                "temporal_ensemble_enabled": True,
                "temporal_ensemble_updated": False,
                "temporal_ensemble_prediction_count": 0,
            }
            return None

        request_id, origin_step, chunk, elapsed, finished_time = raw
        if request_id <= self._last_ensemble_request_id:
            # base.main会继续执行之前保存的融合chunk；返回相同request id，
            # 它不会将底层worker的原始chunk误当成新结果。
            _current_temporal_diagnostics = self._active_diagnostics.copy()
            _current_temporal_diagnostics["temporal_ensemble_updated"] = False
            return raw

        self._last_ensemble_request_id = int(request_id)
        effective_origin_step = (
            self._control_step
            if _last_args.temporal_ensemble_alignment == "execution"
            else int(origin_step)
        )
        self._ensembler.add(effective_origin_step, chunk)
        exec_steps = int(_last_args.chunk_exec_steps)
        replan_step = self._control_step % exec_steps
        block_start = self._control_step - replan_step
        ensembled_chunk, counts, origins, gripper_conflicts = self._ensembler.build(
            block_start_step=block_start,
            output_steps=len(chunk),
        )
        self._active_diagnostics = {
            "temporal_ensemble_enabled": True,
            "temporal_ensemble_updated": True,
            "temporal_ensemble_prediction_count": counts[replan_step],
            "temporal_ensemble_prediction_origins": origins[replan_step],
            "temporal_ensemble_raw_origin_step": int(origin_step),
            "temporal_ensemble_effective_origin_step": int(effective_origin_step),
            "temporal_ensemble_action_step": int(block_start + replan_step),
            "temporal_ensemble_gripper_conflict": gripper_conflicts[replan_step],
        }
        _current_temporal_diagnostics = self._active_diagnostics.copy()
        return (
            request_id,
            block_start,
            ensembled_chunk,
            elapsed,
            finished_time,
        )


def _parse_args():
    global _last_args, _initial_log_mtime_ns
    args = _ORIGINAL_PARSE_ARGS()
    for name in (
        "trajectory_max_joint_speed_rad_s",
        "trajectory_max_joint_acceleration_rad_s2",
        "trajectory_max_joint_jerk_rad_s3",
        "trajectory_response_rad_s",
        "trajectory_feedback_resync_rad",
    ):
        if getattr(args, name) <= 0:
            option = "--" + name.replace("_", "-")
            raise ValueError(f"{option} 必须为正数")
    if args.temporal_ensemble_decay < 0:
        raise ValueError("--temporal-ensemble-decay 必须大于或等于0")
    if args.temporal_ensemble_max_predictions < 1:
        raise ValueError("--temporal-ensemble-max-predictions 必须为正整数")
    if args.gripper_pose_tolerance_rad <= 0:
        raise ValueError("--gripper-pose-tolerance-rad 必须为正数")
    for name in (
        "gripper_pose_stable_steps",
        "gripper_command_stable_steps",
        "gripper_transition_hold_steps",
        "gripper_pose_max_wait_steps",
    ):
        if getattr(args, name) < 1:
            option = "--" + name.replace("_", "-")
            raise ValueError(f"{option} 必须为正整数")
    _last_args = args
    log_path = Path(args.log).expanduser()
    if not log_path.is_absolute():
        log_path = base.REPO_ROOT / log_path
    _initial_log_mtime_ns = (
        log_path.stat().st_mtime_ns if log_path.is_file() else None
    )
    if args.trajectory_smoothing:
        print(
            "[轨迹平滑] 已启用 | "
            f"v_max={args.trajectory_max_joint_speed_rad_s:g}rad/s | "
            f"a_max={args.trajectory_max_joint_acceleration_rad_s2:g}rad/s² | "
            f"jerk_max={args.trajectory_max_joint_jerk_rad_s3:g}rad/s³ | "
            f"response={args.trajectory_response_rad_s:g}rad/s | "
            f"反馈重同步阈值={args.trajectory_feedback_resync_rad:g}rad",
            flush=True,
        )
    else:
        print("[轨迹平滑] 已关闭，使用原推理脚本动作下发行为", flush=True)
    if args.temporal_ensemble:
        print(
            "[Temporal Ensemble] 已启用 | "
            f"decay={args.temporal_ensemble_decay:g} | "
            f"最多融合={args.temporal_ensemble_max_predictions}个预测 | "
            f"对齐={args.temporal_ensemble_alignment} | "
            "夹爪采用最新预测",
            flush=True,
        )
    else:
        print("[Temporal Ensemble] 已关闭，直接使用最新action chunk", flush=True)
    if args.gripper_pose_sync:
        print(
            "[夹爪姿态同步] 已启用 | "
            f"姿态阈值={args.gripper_pose_tolerance_rad:g}rad | "
            f"命令去抖={args.gripper_command_stable_steps}步 | "
            f"连续稳定={args.gripper_pose_stable_steps}步 | "
            f"切换后保持={args.gripper_transition_hold_steps}步 | "
            f"最长等待={args.gripper_pose_max_wait_steps}步",
            flush=True,
        )
    else:
        print("[夹爪姿态同步] 已关闭，夹爪直接跟随策略命令", flush=True)
    return args


def _safe_action(predicted, current, stats, args):
    global _planner, _gripper_coordinator, _last_diagnostics
    safe_target, warnings = _ORIGINAL_SAFE_ACTION(predicted, current, stats, args)
    gripper_diagnostics: dict[str, Any] = {
        "gripper_pose_sync_enabled": bool(args.gripper_pose_sync)
    }
    if args.gripper_pose_sync:
        if _gripper_coordinator is None:
            _gripper_coordinator = GripperPoseCoordinator(
                pose_tolerance_rad=args.gripper_pose_tolerance_rad,
                command_stable_steps=args.gripper_command_stable_steps,
                stable_steps=args.gripper_pose_stable_steps,
                hold_steps=args.gripper_transition_hold_steps,
                max_wait_steps=args.gripper_pose_max_wait_steps,
            )
        safe_target, coordinator_state = _gripper_coordinator.step(
            safe_target, current
        )
        gripper_diagnostics.update(coordinator_state)
        if coordinator_state["gripper_pose_sync_phase"] == "transition_committed":
            warnings.append("夹爪目标姿态已稳定，执行开合并保持当前关节姿态")

    if not args.trajectory_smoothing:
        _last_diagnostics = _current_temporal_diagnostics.copy()
        _last_diagnostics.update(gripper_diagnostics)
        _diagnostics_history.append(_last_diagnostics.copy())
        return safe_target, warnings

    if _planner is None:
        _planner = JerkLimitedJointPlanner(
            rate_hz=args.rate,
            max_velocity=args.trajectory_max_joint_speed_rad_s,
            max_acceleration=args.trajectory_max_joint_acceleration_rad_s2,
            max_jerk=args.trajectory_max_joint_jerk_rad_s3,
            response_rad_s=args.trajectory_response_rad_s,
            feedback_resync_rad=args.trajectory_feedback_resync_rad,
        )
    lower = np.maximum(
        base.contact.PIPER_HARD_LOWER_RAD,
        stats["action_min"][:6] - args.dataset_margin_rad,
    )
    upper = np.minimum(
        base.contact.PIPER_HARD_UPPER_RAD,
        stats["action_max"][:6] + args.dataset_margin_rad,
    )
    planned_joints, _last_diagnostics = _planner.step(
        safe_target[:6], current[:6], lower, upper
    )
    _last_diagnostics.update(_current_temporal_diagnostics)
    _last_diagnostics.update(gripper_diagnostics)
    _diagnostics_history.append(_last_diagnostics.copy())
    safe_target = safe_target.copy()
    safe_target[:6] = planned_joints
    if _last_diagnostics["trajectory_resynced"]:
        warnings.append(
            "轨迹规划状态与关节反馈偏差过大，已从当前反馈重新规划"
        )
    if any(_last_diagnostics["trajectory_limit_clipped"]):
        warnings.append("轨迹规划输出触及关节安全边界，已裁剪并清除该轴动态状态")
    return safe_target, warnings


def _append_smoothing_metadata() -> None:
    """原脚本保存日志后补充本入口的规划参数。"""

    if _last_args is None:
        return
    if _last_args.offline_smoke:
        return
    log_path = Path(_last_args.log).expanduser()
    if not log_path.is_absolute():
        log_path = base.REPO_ROOT / log_path
    if not log_path.is_file():
        return
    if (
        _initial_log_mtime_ns is not None
        and log_path.stat().st_mtime_ns == _initial_log_mtime_ns
    ):
        # 启动阶段在原脚本写日志前失败时，不得把平滑元数据写进同路径旧日志。
        return
    try:
        payload = json.loads(log_path.read_text(encoding="utf-8"))
        payload.setdefault("meta", {}).update(
            {
                "trajectory_smoothing": _last_args.trajectory_smoothing,
                "trajectory_max_joint_speed_rad_s": _last_args.trajectory_max_joint_speed_rad_s,
                "trajectory_max_joint_acceleration_rad_s2": _last_args.trajectory_max_joint_acceleration_rad_s2,
                "trajectory_max_joint_jerk_rad_s3": _last_args.trajectory_max_joint_jerk_rad_s3,
                "trajectory_response_rad_s": _last_args.trajectory_response_rad_s,
                "trajectory_feedback_resync_rad": _last_args.trajectory_feedback_resync_rad,
                "trajectory_resync_count": _planner.resync_count if _planner else 0,
                "temporal_ensemble": _last_args.temporal_ensemble,
                "temporal_ensemble_decay": _last_args.temporal_ensemble_decay,
                "temporal_ensemble_max_predictions": _last_args.temporal_ensemble_max_predictions,
                "temporal_ensemble_alignment": _last_args.temporal_ensemble_alignment,
                "temporal_ensemble_gripper": "newest_prediction",
                "temporal_ensemble_phase_conflict_action": "newest_full_7d_action",
                "gripper_pose_sync": _last_args.gripper_pose_sync,
                "gripper_pose_tolerance_rad": _last_args.gripper_pose_tolerance_rad,
                "gripper_pose_stable_steps": _last_args.gripper_pose_stable_steps,
                "gripper_command_stable_steps": _last_args.gripper_command_stable_steps,
                "gripper_transition_hold_steps": _last_args.gripper_transition_hold_steps,
                "gripper_pose_max_wait_steps": _last_args.gripper_pose_max_wait_steps,
                "gripper_pose_sync_transition_count": (
                    _gripper_coordinator.transition_count
                    if _gripper_coordinator
                    else 0
                ),
            }
        )
        for record, diagnostics in zip(
            payload.get("records", []), _diagnostics_history
        ):
            record.update(diagnostics)
        log_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as exc:
        print(f"[轨迹平滑] 补充日志元数据失败: {exc}", flush=True)


def main() -> None:
    # 只替换 base 模块里的 argparse 引用；不能修改标准库模块自身的
    # ArgumentParser，否则 argparse 内部的 super() 查找会形成递归。
    base.argparse = _ArgparseProxy()
    base.parse_args = _parse_args
    base.safe_action = _safe_action
    base.AsyncPolicyWorker = TemporalEnsembleAsyncPolicyWorker
    try:
        base.main()
    finally:
        _append_smoothing_metadata()


if __name__ == "__main__":
    main()
