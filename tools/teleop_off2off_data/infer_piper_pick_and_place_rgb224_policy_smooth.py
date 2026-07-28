#!/usr/bin/env python3
"""带底层轨迹平滑的 RGB224 Piper pick-and-place policy 推理入口。

该脚本复用2D/RGB224推理流程，只在策略动作通过安全检查后、下发Piper前加入
在线关节轨迹规划和夹爪姿态同步。默认指向当前RGB224训练输出的 ``best_val``。
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
TRAIN_ROOT = REPO_ROOT / "RL-100"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.teleop_off2off_data import infer_piper_pick_and_place_2d_policy as base2d
from tools.teleop_off2off_data import infer_piper_contact_policy as contact
from tools.teleop_off2off_data.infer_piper_pick_and_place_policy_smooth import (
    GripperPoseCoordinator,
    JerkLimitedJointPlanner,
    TemporalChunkEnsembler,
)


DEFAULT_RGB224_OUTPUT_DIR = (
    TRAIN_ROOT
    / "data/outputs/piper_pick_and_place_augmented_rgb224_control_clean_resnet18r3m_dp3_episode10_bs64_epoch2000_seed42"
)

_ORIGINAL_ARGUMENT_PARSER = argparse.ArgumentParser
_ORIGINAL_PARSE_ARGS = base2d.parse_args
_ORIGINAL_SAFE_ACTION = base2d.safe_action
_ORIGINAL_POSTPROCESS_ACTION_CHUNK = base2d.postprocess_action_chunk
_ORIGINAL_EXTRA_RECORD_DIAGNOSTICS = base2d.extra_record_diagnostics
_planner: JerkLimitedJointPlanner | None = None
_gripper_coordinator: GripperPoseCoordinator | None = None
_temporal_ensembler: TemporalChunkEnsembler | None = None
_last_args: Any = None
_last_diagnostics: dict[str, Any] = {}
_current_temporal_diagnostics: dict[str, Any] = {}
_diagnostics_history: list[dict[str, Any]] = []
_initial_log_mtime_ns: int | None = None


class _SmoothingArgumentParser(_ORIGINAL_ARGUMENT_PARSER):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.add_argument(
            "--trajectory-smoothing",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="启用速度/加速度/jerk在线轨迹规划；本脚本默认启用。",
        )
        self.add_argument("--trajectory-max-joint-speed-rad-s", type=float, default=0.35)
        self.add_argument("--trajectory-max-joint-acceleration-rad-s2", type=float, default=0.8)
        self.add_argument("--trajectory-max-joint-jerk-rad-s3", type=float, default=4.0)
        self.add_argument("--trajectory-response-rad-s", type=float, default=2.0)
        self.add_argument("--trajectory-feedback-resync-rad", type=float, default=0.20)
        self.add_argument(
            "--temporal-ensemble",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="按绝对控制步融合重叠action chunk；本脚本默认启用。",
        )
        self.add_argument("--temporal-ensemble-decay", type=float, default=0.5)
        self.add_argument("--temporal-ensemble-max-predictions", type=int, default=4)
        self.add_argument(
            "--temporal-ensemble-alignment",
            choices=["execution", "observation"],
            default="execution",
            help="同步2D推理中两者等价；保留该参数以兼容3D smooth命令。",
        )
        self.add_argument(
            "--gripper-pose-sync",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="夹爪开合前锁存并等待对应关节姿态；本脚本默认启用。",
        )
        self.add_argument("--gripper-pose-tolerance-rad", type=float, default=0.025)
        self.add_argument("--gripper-pose-stable-steps", type=int, default=2)
        self.add_argument("--gripper-command-stable-steps", type=int, default=2)
        self.add_argument("--gripper-transition-hold-steps", type=int, default=4)
        self.add_argument("--gripper-pose-max-wait-steps", type=int, default=60)


class _ArgparseProxy:
    ArgumentParser = _SmoothingArgumentParser

    def __getattr__(self, name: str):
        return getattr(argparse, name)


def _has_option(names: set[str]) -> bool:
    return any(arg in names or any(arg.startswith(name + "=") for name in names) for arg in sys.argv[1:])


def _parse_args():
    global _last_args, _initial_log_mtime_ns
    if not _has_option({"--output-dir"}):
        sys.argv.extend(["--output-dir", str(DEFAULT_RGB224_OUTPUT_DIR)])
    if not _has_option({"--policy-subdir"}):
        sys.argv.extend(["--policy-subdir", "best_val"])
    args = _ORIGINAL_PARSE_ARGS()
    for name in (
        "trajectory_max_joint_speed_rad_s",
        "trajectory_max_joint_acceleration_rad_s2",
        "trajectory_max_joint_jerk_rad_s3",
        "trajectory_response_rad_s",
        "trajectory_feedback_resync_rad",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} 必须为正数")
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
            raise ValueError(f"--{name.replace('_', '-')} 必须为正整数")
    _last_args = args
    log_path = Path(args.log).expanduser()
    if not log_path.is_absolute():
        log_path = base2d.TRAIN_ROOT / log_path
    _initial_log_mtime_ns = log_path.stat().st_mtime_ns if log_path.is_file() else None
    if args.trajectory_smoothing:
        print(
            "[RGB224轨迹平滑] 已启用 | "
            f"v_max={args.trajectory_max_joint_speed_rad_s:g}rad/s | "
            f"a_max={args.trajectory_max_joint_acceleration_rad_s2:g}rad/s² | "
            f"jerk_max={args.trajectory_max_joint_jerk_rad_s3:g}rad/s³ | "
            f"response={args.trajectory_response_rad_s:g}rad/s | "
            f"反馈重同步阈值={args.trajectory_feedback_resync_rad:g}rad",
            flush=True,
        )
    else:
        print("[RGB224轨迹平滑] 已关闭", flush=True)
    if args.temporal_ensemble:
        print(
            "[RGB224 Temporal Ensemble] 已启用 | "
            f"decay={args.temporal_ensemble_decay:g} | "
            f"最多融合={args.temporal_ensemble_max_predictions}个预测 | "
            f"对齐={args.temporal_ensemble_alignment} | "
            "夹爪采用最新预测",
            flush=True,
        )
    else:
        print("[RGB224 Temporal Ensemble] 已关闭，直接使用最新action chunk", flush=True)
    if args.gripper_pose_sync:
        print(
            "[RGB224夹爪姿态同步] 已启用 | "
            f"姿态阈值={args.gripper_pose_tolerance_rad:g}rad | "
            f"命令去抖={args.gripper_command_stable_steps}步 | "
            f"连续稳定={args.gripper_pose_stable_steps}步 | "
            f"切换后保持={args.gripper_transition_hold_steps}步 | "
            f"最长等待={args.gripper_pose_max_wait_steps}步",
            flush=True,
        )
    else:
        print("[RGB224夹爪姿态同步] 已关闭", flush=True)
    return args


def _postprocess_action_chunk(chunk: np.ndarray, step: int, chunk_step: int, args):
    global _temporal_ensembler, _current_temporal_diagnostics
    chunk = _ORIGINAL_POSTPROCESS_ACTION_CHUNK(chunk, step, chunk_step, args)
    if not args.temporal_ensemble:
        _current_temporal_diagnostics = {
            "temporal_ensemble_enabled": False,
        }
        return chunk
    if _temporal_ensembler is None:
        _temporal_ensembler = TemporalChunkEnsembler(
            decay=args.temporal_ensemble_decay,
            max_predictions=args.temporal_ensemble_max_predictions,
        )
    # 2D脚本是同步推理，观测提交和结果执行没有异步跨步延迟；
    # 这里保留alignment字段用于日志/命令兼容，实际origin取当前控制步。
    effective_origin_step = int(step)
    _temporal_ensembler.add(effective_origin_step, chunk)
    block_start = int(step - chunk_step)
    ensembled_chunk, counts, origins, gripper_conflicts = _temporal_ensembler.build(
        block_start_step=block_start,
        output_steps=len(chunk),
    )
    _current_temporal_diagnostics = {
        "temporal_ensemble_enabled": True,
        "temporal_ensemble_updated": True,
        "temporal_ensemble_prediction_count": counts[chunk_step],
        "temporal_ensemble_prediction_origins": origins[chunk_step],
        "temporal_ensemble_raw_origin_step": int(step),
        "temporal_ensemble_effective_origin_step": int(effective_origin_step),
        "temporal_ensemble_action_step": int(block_start + chunk_step),
        "temporal_ensemble_gripper_conflict": gripper_conflicts[chunk_step],
    }
    return ensembled_chunk


def _extra_record_diagnostics() -> dict[str, Any]:
    diagnostics = _ORIGINAL_EXTRA_RECORD_DIAGNOSTICS()
    diagnostics.update(_current_temporal_diagnostics)
    diagnostics.update(_last_diagnostics)
    return diagnostics


def _safe_action(predicted, current, stats, args):
    global _planner, _gripper_coordinator, _last_diagnostics
    safe_target, warnings = _ORIGINAL_SAFE_ACTION(predicted, current, stats, args)
    diagnostics: dict[str, Any] = {
        "trajectory_smoothing_enabled": bool(args.trajectory_smoothing),
        "gripper_pose_sync_enabled": bool(args.gripper_pose_sync),
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
        safe_target, gripper_state = _gripper_coordinator.step(safe_target, current)
        diagnostics.update(gripper_state)
        if gripper_state["gripper_pose_sync_phase"] == "transition_committed":
            warnings.append("夹爪目标姿态已稳定，执行开合并保持当前关节姿态")
    if not args.trajectory_smoothing:
        _last_diagnostics = diagnostics
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
        contact.PIPER_HARD_LOWER_RAD,
        stats["action_min"][:6] - args.dataset_margin_rad,
    )
    upper = np.minimum(
        contact.PIPER_HARD_UPPER_RAD,
        stats["action_max"][:6] + args.dataset_margin_rad,
    )
    planned_joints, planner_state = _planner.step(safe_target[:6], current[:6], lower, upper)
    diagnostics.update(planner_state)
    _last_diagnostics = diagnostics
    _diagnostics_history.append(_last_diagnostics.copy())
    safe_target = safe_target.copy()
    safe_target[:6] = planned_joints
    if planner_state["trajectory_resynced"]:
        warnings.append("轨迹规划状态与关节反馈偏差过大，已从当前反馈重新规划")
    if any(planner_state["trajectory_limit_clipped"]):
        warnings.append("轨迹规划输出触及关节安全边界，已裁剪并清除该轴动态状态")
    return safe_target, warnings


def _append_smoothing_metadata() -> None:
    if _last_args is None or _last_args.offline_smoke:
        return
    log_path = Path(_last_args.log).expanduser()
    if not log_path.is_absolute():
        log_path = base2d.TRAIN_ROOT / log_path
    if not log_path.is_file():
        return
    if _initial_log_mtime_ns is not None and log_path.stat().st_mtime_ns == _initial_log_mtime_ns:
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
                    _gripper_coordinator.transition_count if _gripper_coordinator else 0
                ),
            }
        )
        for record, diagnostics in zip(payload.get("records", []), _diagnostics_history):
            record.update(diagnostics)
        log_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"[RGB224轨迹平滑] 补充日志元数据失败: {exc}", flush=True)


def main() -> None:
    base2d.argparse = _ArgparseProxy()
    base2d.parse_args = _parse_args
    base2d.safe_action = _safe_action
    base2d.postprocess_action_chunk = _postprocess_action_chunk
    base2d.extra_record_diagnostics = _extra_record_diagnostics
    try:
        base2d.main()
    finally:
        _append_smoothing_metadata()


if __name__ == "__main__":
    main()
