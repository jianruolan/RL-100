#!/usr/bin/env python3
"""安全运行带独立夹爪分类头的 Piper 3D pick-and-place 策略。

动作约定：

* diffusion ``action[..., :6]`` 是六关节绝对目标（rad）；
* diffusion ``action[..., 6]`` 是训练用的0/1夹爪命令状态，仅记录诊断；
* 真机夹爪由 ``gripper_prob`` 的未来12帧张开概率经过迟滞、连续确认、
  冷却和状态锁存后控制，不能把0/1直接当成米制宽度发送给 SDK。

默认仅运行 shadow mode。即使指定 ``--execute``，也只有再显式指定
``--allow-gripper-events``，分类头产生的开合事件才会发给真机。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
TRAIN_ROOT = REPO_ROOT / "RL-100"
for import_path in (REPO_ROOT, TRAIN_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from tools.teleop_off2off_data import infer_piper_contact_policy as contact
from tools.teleop_off2off_data import infer_piper_pick_and_place_policy as base

# contact 会把仓库内已编译的 aarch64 PyTorch3D fallback 加入 sys.path。
# 在加载 Hydra 策略前显式验证，避免缺失依赖被静默忽略后报出难定位的错误。
try:
    import pytorch3d.ops  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "无法导入 pytorch3d.ops；请检查 third_party/pytorch3d_simplified/"
        "pytorch3d/_C.cpython-310-aarch64-linux-gnu.so 是否与当前 Python/架构匹配"
    ) from exc


DEFAULT_OUTPUT_DIR = (
    TRAIN_ROOT / "data/outputs/piper_pick_and_place_gripper_chunk4_seed42"
)


@dataclass
class GripperDecision:
    """一次夹爪概率块的解码结果。"""

    score: float
    event: str | None
    latched_open: bool
    open_confirmations: int
    close_confirmations: int
    cooldown_remaining: int


class GripperEventDecoder:
    """把未来张开概率转换为带迟滞的一次性 open/close 事件。"""

    def __init__(
        self,
        initial_open: bool,
        close_threshold: float = 0.4,
        open_threshold: float = 0.6,
        consecutive: int = 2,
        cooldown_frames: int = 4,
        window_start: int = 0,
        window_end: int = 8,
    ):
        if not 0.0 <= close_threshold < open_threshold <= 1.0:
            raise ValueError("夹爪阈值必须满足 0<=close<open<=1")
        if consecutive < 1 or cooldown_frames < 0:
            raise ValueError("consecutive必须为正，cooldown_frames不能为负")
        if window_start < 0 or window_end <= window_start:
            raise ValueError("夹爪概率窗口必须满足 0<=start<end")
        self.latched_open = bool(initial_open)
        self.close_threshold = float(close_threshold)
        self.open_threshold = float(open_threshold)
        self.consecutive = int(consecutive)
        self.cooldown_frames = int(cooldown_frames)
        self.window_start = int(window_start)
        self.window_end = int(window_end)
        self.open_confirmations = 0
        self.close_confirmations = 0
        self.cooldown_remaining = 0

    def update(self, probabilities: np.ndarray, allow_transition: bool = True) -> GripperDecision:
        probabilities = np.asarray(probabilities, dtype=np.float32).reshape(-1)
        if self.window_end > len(probabilities):
            raise RuntimeError(
                f"夹爪概率只有{len(probabilities)}步，无法使用"
                f"[{self.window_start}:{self.window_end}]窗口"
            )
        if not np.all(np.isfinite(probabilities)) or np.any(
            (probabilities < 0.0) | (probabilities > 1.0)
        ):
            raise RuntimeError("夹爪概率包含NaN/Inf或超出[0,1]")
        score = float(probabilities[self.window_start:self.window_end].mean())
        event: str | None = None

        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            self.open_confirmations = 0
            self.close_confirmations = 0
        elif self.latched_open:
            self.open_confirmations = 0
            if score < self.close_threshold:
                self.close_confirmations = min(
                    self.consecutive, self.close_confirmations + 1
                )
                if self.close_confirmations >= self.consecutive:
                    if allow_transition:
                        self.latched_open = False
                        self.cooldown_remaining = self.cooldown_frames
                        self.close_confirmations = 0
                        event = "close"
                    else:
                        # 保持计数饱和，安全门恢复后下一次即可触发。
                        event = "close_blocked"
            else:
                self.close_confirmations = 0
        else:
            self.close_confirmations = 0
            if score > self.open_threshold:
                self.open_confirmations = min(
                    self.consecutive, self.open_confirmations + 1
                )
                if self.open_confirmations >= self.consecutive:
                    if allow_transition:
                        self.latched_open = True
                        self.cooldown_remaining = self.cooldown_frames
                        self.open_confirmations = 0
                        event = "open"
                    else:
                        event = "open_blocked"
            else:
                self.open_confirmations = 0

        return GripperDecision(
            score=score,
            event=event,
            latched_open=self.latched_open,
            open_confirmations=self.open_confirmations,
            close_confirmations=self.close_confirmations,
            cooldown_remaining=self.cooldown_remaining,
        )


def extract_policy_outputs(
    output: dict[str, torch.Tensor], expected_action_steps: int, expected_gripper_steps: int
) -> tuple[np.ndarray, np.ndarray]:
    """提取关节chunk和夹爪概率；明确丢弃diffusion的第7维控制值。"""

    chunk = base.extract_action_chunk(output, expected_steps=expected_action_steps)
    if chunk.shape[-1] not in (6, 7):
        raise RuntimeError(
            f"arm-only夹爪策略动作维度应为6或7，实际为{chunk.shape[-1]}"
        )
    if "gripper_prob" not in output:
        raise RuntimeError(
            "策略没有返回gripper_prob；请确认使用新任务配置和含gripper_head.pt的权重"
        )
    probabilities = output["gripper_prob"].detach().cpu().numpy()
    if probabilities.shape != (1, expected_gripper_steps):
        raise RuntimeError(
            f"gripper_prob应为[1,{expected_gripper_steps}]，实际为{probabilities.shape}"
        )
    probabilities = probabilities[0].astype(np.float32)
    if not np.all(np.isfinite(probabilities)):
        raise RuntimeError("gripper_prob包含NaN/Inf")
    return chunk, probabilities


def compute_training_stats(dataset) -> contact.TrainingStats:
    """只用前六维关节动作计算安全范围，第7维命令状态不按米解释。"""

    replay = dataset.replay_buffer
    action_key = getattr(dataset, "action_key", "action")
    state = np.asarray(replay["state"][:, :6], dtype=np.float32)
    action = np.asarray(replay[action_key][:, :6], dtype=np.float32)
    point_cloud = replay["point_cloud"]
    stride = max(1, len(point_cloud) // 2000)
    points = np.asarray(point_cloud[::stride], dtype=np.float32).reshape(-1, 3)
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) == 0:
        raise RuntimeError("训练点云没有有限点")
    return contact.TrainingStats(
        state_min=state.min(axis=0),
        state_max=state.max(axis=0),
        action_min=action.min(axis=0),
        action_max=action.max(axis=0),
        action_delta_p99=np.percentile(np.abs(action - state), 99, axis=0).astype(np.float32),
        point_cloud_low=np.percentile(points, 0.1, axis=0).astype(np.float32),
        point_cloud_high=np.percentile(points, 99.9, axis=0).astype(np.float32),
    )


def safe_joint_target(
    predicted: np.ndarray,
    current: np.ndarray,
    stats: contact.TrainingStats,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[str]]:
    """应用物理限位、训练分布限位和按实际控制周期计算的速度限制。"""

    predicted = np.asarray(predicted, dtype=np.float32)
    current = np.asarray(current, dtype=np.float32)
    if predicted.shape != (6,) or current.shape != (6,):
        raise ValueError(f"关节目标shape错误: {predicted.shape}/{current.shape}")
    warnings: list[str] = []
    lower = np.maximum(
        contact.PIPER_HARD_LOWER_RAD,
        stats.action_min - args.dataset_margin_rad,
    )
    upper = np.minimum(
        contact.PIPER_HARD_UPPER_RAD,
        stats.action_max + args.dataset_margin_rad,
    )
    if np.any(current < lower) or np.any(current > upper):
        if not args.skip_current_joint_range_check:
            raise RuntimeError(
                "当前关节姿态超出训练分布安全范围，请先人工移回示教初始区域"
            )
        warnings.append("已跳过当前关节训练范围检查")

    target = predicted.copy()
    if np.any(target < lower) or np.any(target > upper):
        if not args.clip_actions:
            raise RuntimeError("策略关节目标超出训练范围，默认停止")
        target = np.clip(target, lower, upper)
        warnings.append("关节目标被裁剪到训练范围")
    max_step = args.max_joint_speed_rad_s / args.rate
    delta = target - current
    if np.any(np.abs(delta) > max_step):
        if not args.clip_actions:
            raise RuntimeError("策略关节单周期变化超过速度限制，默认停止")
        target = current + np.clip(delta, -max_step, max_step)
        warnings.append("关节目标被速度限制器裁剪")
    if np.any(target < contact.PIPER_HARD_LOWER_RAD) or np.any(
        target > contact.PIPER_HARD_UPPER_RAD
    ):
        raise RuntimeError("安全过滤后的关节目标仍超出Piper物理限位")
    return target.astype(np.float32), warnings


def send_action(
    piper: Any,
    joint_target_rad: np.ndarray,
    gripper_width_m: float,
    speed_percent: int,
    gripper_effort: int,
) -> None:
    """发送六关节绝对目标和锁存后的米制夹爪宽度。"""

    joints_raw = np.rint(
        np.asarray(joint_target_rad, dtype=np.float64) * contact.RAD_TO_RAW
    ).astype(np.int64)
    gripper_raw = int(round(float(gripper_width_m) * base.GRIPPER_RAW_PER_M))
    piper.MotionCtrl_2(0x01, 0x01, int(speed_percent), 0x00)
    piper.JointCtrl(*[int(value) for value in joints_raw])
    piper.GripperCtrl(gripper_raw, int(gripper_effort), 0x01, 0)


def offline_smoke(dataset, policy, device, use_cm, n_action_steps, gripper_horizon, args):
    """不连接相机和机械臂，验证权重、normalizer和事件解码链路。"""

    sample = dataset[0]["obs"]
    obs = {key: value.unsqueeze(0).to(device) for key, value in sample.items()}
    with torch.no_grad():
        output = policy.predict_action(obs, deterministic=True, use_cm=use_cm)
    chunk, probabilities = extract_policy_outputs(
        output, n_action_steps, gripper_horizon
    )
    initial_open = bool(sample["agent_pos"][-1, 6].item() >= args.initial_open_threshold_m)
    decoder = GripperEventDecoder(
        initial_open=initial_open,
        close_threshold=args.close_threshold,
        open_threshold=args.open_threshold,
        consecutive=args.consecutive,
        cooldown_frames=args.cooldown_frames,
        window_start=args.window_start,
        window_end=args.window_end,
    )
    decision = decoder.update(probabilities)
    print(f"[offline-smoke] 关节chunk shape={chunk[:, :6].shape}", flush=True)
    if chunk.shape[-1] == 7:
        print(f"[offline-smoke] diffusion第7维={np.round(chunk[:, 6], 4)}（不下发）", flush=True)
    else:
        print("[offline-smoke] arm-only diffusion为6维，无第7维夹爪动作", flush=True)
    print(f"[offline-smoke] p_open={np.round(probabilities, 4)}", flush=True)
    print(f"[offline-smoke] score={decision.score:.4f}, event={decision.event}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--policy-subdir", choices=["best", "bc", "best_cm"], default="best")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--offline-smoke", action="store_true")
    parser.add_argument("--can", default="can0")
    parser.add_argument("--piper-sdk-root", type=Path, default=None)
    parser.add_argument("--rate", type=float, default=13.0)
    parser.add_argument("--camera-fps", type=int, default=15)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--chunk-exec-steps", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--allow-gripper-events",
        action="store_true",
        help="允许分类头事件实际控制夹爪；仅--execute不会启用策略夹爪事件。",
    )
    parser.add_argument("--clip-actions", action="store_true")
    parser.add_argument("--skip-current-joint-range-check", action="store_true")
    parser.add_argument("--dataset-margin-rad", type=float, default=0.03)
    parser.add_argument("--max-joint-speed-rad-s", type=float, default=0.15)
    parser.add_argument("--speed-percent", type=int, default=10)
    parser.add_argument(
        "--skip-gripper-safety-check",
        action="store_true",
        help=(
            "跳过夹爪训练宽度/回零要求；仍执行 SDK 夹爪使能并检查驱动器故障，"
            "用于已确认夹爪可用但 homing_status 未置位的测试。"
        ),
    )
    parser.add_argument("--open-width-m", type=float, default=0.07)
    parser.add_argument("--close-width-m", type=float, default=0.0)
    parser.add_argument("--gripper-effort", type=int, default=base.GRIPPER_EFFORT)
    parser.add_argument(
        "--initial-gripper-state",
        choices=["open", "feedback"],
        default="open",
        help="pick-and-place示教轨迹默认从open开始；feedback仅用于特殊任务。",
    )
    parser.add_argument("--initial-open-threshold-m", type=float, default=0.061)
    parser.add_argument("--close-threshold", type=float, default=0.4)
    parser.add_argument("--open-threshold", type=float, default=0.6)
    parser.add_argument("--consecutive", type=int, default=2)
    parser.add_argument("--cooldown-frames", type=int, default=4)
    parser.add_argument("--window-start", type=int, default=0)
    parser.add_argument(
        "--window-end",
        type=int,
        default=None,
        help="默认自动取min(8, gripper_horizon)；单帧分类头自动使用1。",
    )
    parser.add_argument(
        "--gripper-event-max-joint-speed-rad-s",
        type=float,
        default=0.10,
        help="任一关节反馈速度超过该值时暂时禁止真实开合事件。",
    )
    parser.add_argument("--reuse-diffusion-noise", action="store_true")
    parser.add_argument("--diffusion-noise-seed", type=int, default=42)
    parser.add_argument("--max-point-outlier-fraction", type=float, default=0.25)
    parser.add_argument("--skip-point-cloud-distribution-check", action="store_true")
    parser.add_argument("--state-stale-seconds", type=float, default=0.5)
    parser.add_argument("--enable-timeout", type=float, default=5.0)
    parser.add_argument(
        "--log",
        type=Path,
        default=TRAIN_ROOT / "data/piper_inference/pick_and_place_gripper_latest.json",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0 < args.rate <= 20:
        raise ValueError("--rate必须在(0,20]Hz")
    if args.camera_fps < args.rate or args.max_steps < 1:
        raise ValueError("camera-fps不能低于rate，max-steps必须为正")
    if args.chunk_exec_steps < 1 or not 1 <= args.speed_percent <= 100:
        raise ValueError("chunk-exec-steps必须为正，speed-percent必须在[1,100]")
    if not 0 <= args.close_width_m < args.open_width_m <= base.GRIPPER_UPPER_M:
        raise ValueError("夹爪宽度必须满足0<=close<open<=0.07m")
    if args.gripper_effort < 0:
        raise ValueError("gripper-effort不能为负")
    if args.gripper_event_max_joint_speed_rad_s <= 0:
        raise ValueError("gripper-event-max-joint-speed-rad-s必须为正")
    if not 0.0 <= args.close_threshold < args.open_threshold <= 1.0:
        raise ValueError("夹爪阈值必须满足0<=close<open<=1")
    if args.consecutive < 1 or args.cooldown_frames < 0:
        raise ValueError("consecutive必须为正，cooldown-frames不能为负")
    if args.window_start < 0 or (
        args.window_end is not None and args.window_end <= args.window_start
    ):
        raise ValueError("夹爪概率窗口必须满足0<=start<end")
    if not args.close_width_m < args.initial_open_threshold_m < args.open_width_m:
        raise ValueError("initial-open-threshold-m必须位于close/open宽度之间")


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir = args.output_dir.expanduser().resolve()
    cfg, dataset, policy, use_cm = contact.load_policy_and_dataset(
        args.output_dir, args.policy_subdir, args.device
    )
    if not bool(getattr(policy, "use_gripper_head", False)) or policy.gripper_head is None:
        raise RuntimeError("所选权重没有启用独立夹爪分类头")
    if list(cfg.shape_meta.obs.agent_pos.shape) != [7] or list(cfg.shape_meta.action.shape) not in ([6], [7]):
        raise RuntimeError("当前脚本只支持agent_pos=7维、action=6或7维Piper配置")
    if getattr(dataset, "action_key", None) != "policy_action":
        raise RuntimeError("数据集没有使用policy_action，拒绝把旧夹爪宽度权重用于本脚本")

    n_obs_steps = int(cfg.n_obs_steps)
    n_action_steps = int(cfg.n_action_steps)
    horizon = int(cfg.horizon)
    gripper_horizon = int(policy.gripper_horizon)
    if args.window_end is None:
        args.window_end = min(8, gripper_horizon)
    if horizon != n_obs_steps - 1 + n_action_steps:
        raise RuntimeError("horizon与n_obs_steps/n_action_steps不匹配")
    if args.chunk_exec_steps > n_action_steps:
        raise ValueError("chunk-exec-steps超过模型n_action_steps")
    if args.window_end > gripper_horizon:
        raise ValueError("夹爪概率窗口超过模型gripper_horizon")
    device = torch.device(args.device)
    if args.offline_smoke:
        offline_smoke(
            dataset, policy, device, use_cm, n_action_steps, gripper_horizon, args
        )
        return

    from tools.teleop_off2off_data.realsense import RealSense

    # ARM主机上先启动UVC，再扫描完整训练统计和启动CAN线程更稳定。
    camera = RealSense(
        fps=args.camera_fps,
        color_width=640,
        color_height=480,
        depth_width=640,
        depth_height=480,
        num_points=512,
        point_cloud_frame="camera",
        align_depth_to_color=False,
    )
    camera.start()
    try:
        stats = compute_training_stats(dataset)
        C_PiperInterface_V2 = contact.import_piper_sdk(args.piper_sdk_root)
        piper = C_PiperInterface_V2(args.can)
        piper.ConnectPort()
        reader = base.PiperPickPlaceStateReader(piper)
    except Exception:
        camera.stop()
        raise

    print("[模式]", "真机执行" if args.execute else "shadow（不下发）", flush=True)
    print(
        f"[模型] obs={n_obs_steps}, joint_chunk={n_action_steps}, "
        f"gripper_horizon={gripper_horizon}, rate={args.rate:g}Hz",
        flush=True,
    )
    print(
        f"[夹爪解码] mean(p_open[{args.window_start}:{args.window_end}]), "
        f"close<{args.close_threshold}, open>{args.open_threshold}, "
        f"consecutive={args.consecutive}, cooldown={args.cooldown_frames}",
        flush=True,
    )
    print(
        "[夹爪警告] 0.4/0.6阈值来自参考BC实验；新RL-100权重必须先用shadow日志校准，"
        "不能仅凭默认值直接判断真机安全。",
        flush=True,
    )
    if args.execute and not args.allow_gripper_events:
        print("[安全] 机械臂可执行，但策略夹爪事件被禁止，夹爪保持初始宽度", flush=True)
    if args.skip_gripper_safety_check:
        print(
            "[安全警告] 已跳过夹爪回零/训练宽度检查；仍要求驱动器使能且无故障，"
            "请确认机械限位和急停可用",
            flush=True,
        )
    print("[人工] Enter/stop保持停止；estop发送SDK快速停止。", flush=True)

    history: deque[dict[str, np.ndarray]] = deque(maxlen=n_obs_steps)
    records: list[dict[str, Any]] = []
    operator = contact.OperatorConsole()
    robot_enabled = False
    hard_emergency = False
    stop_reason = "max_steps"
    last_state: np.ndarray | None = None
    last_joint_ts = last_grip_ts = None
    last_joint_fresh = last_grip_fresh = time.monotonic()
    previous_state: np.ndarray | None = None
    previous_state_time: float | None = None
    desired_gripper_width: float | None = None
    episode_noise = (
        contact.make_episode_diffusion_noise(policy, device, args.diffusion_noise_seed)
        if args.reuse_diffusion_noise else None
    )

    try:
        for _ in range(n_obs_steps):
            state, _, _ = reader.read()
            frame = camera.get_frame(require_pc=False)
            image, point_cloud = base.preprocess_camera_frame(frame)
            history.append(
                {"agent_pos": state, "point_cloud": point_cloud, "image": image}
            )
            time.sleep(1.0 / args.rate)
        last_state = history[-1]["agent_pos"].copy()
        initial_open = (
            args.initial_gripper_state == "open"
            or float(last_state[6]) >= args.initial_open_threshold_m
        )
        desired_gripper_width = (
            args.open_width_m if initial_open else float(last_state[6])
        )
        decoder = GripperEventDecoder(
            initial_open=initial_open,
            close_threshold=args.close_threshold,
            open_threshold=args.open_threshold,
            consecutive=args.consecutive,
            cooldown_frames=args.cooldown_frames,
            window_start=args.window_start,
            window_end=args.window_end,
        )

        if args.execute:
            expected_phrase = (
                "EXECUTE"
                if args.allow_gripper_events else "EXECUTE PIPER"
            )
            phrase = input(f"确认工作区安全后输入 {expected_phrase}：").strip()
            if phrase != expected_phrase:
                raise RuntimeError("确认短语不匹配，取消执行")
            contact.enable_robot_for_position_control(
                piper, args.speed_percent, args.enable_timeout
            )
            robot_enabled = True
            base.enable_gripper(
                piper,
                reader,
                desired_gripper_width,
                require_homing=not args.skip_gripper_safety_check,
                timeout=args.enable_timeout,
            )
            if initial_open:
                # 训练轨迹全部从张开状态开始；先等待夹爪反馈稳定，再建立
                # n_obs_steps历史，避免用“启动时闭合”的帧污染策略输入。
                time.sleep(max(0.3, 2.0 / args.rate))
                history.clear()
                for _ in range(n_obs_steps):
                    refreshed_state, _, _ = reader.read()
                    refreshed_frame = camera.get_frame(require_pc=False)
                    refreshed_image, refreshed_pc = base.preprocess_camera_frame(
                        refreshed_frame
                    )
                    history.append(
                        {
                            "agent_pos": refreshed_state,
                            "point_cloud": refreshed_pc,
                            "image": refreshed_image,
                        }
                    )
                    time.sleep(1.0 / args.rate)
                last_state = history[-1]["agent_pos"].copy()
        operator.start()

        next_deadline = time.monotonic()
        active_chunk: np.ndarray | None = None
        active_probabilities: np.ndarray | None = None
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
                raise RuntimeError("Piper关节反馈超时")
            if last_grip_ts != grip_ts:
                last_grip_fresh = loop_start
            elif loop_start - last_grip_fresh > args.state_stale_seconds:
                raise RuntimeError("Piper夹爪反馈超时")
            last_joint_ts, last_grip_ts = joint_ts, grip_ts

            if previous_state is None or previous_state_time is None:
                measured_max_joint_speed = 0.0
            else:
                dt = max(loop_start - previous_state_time, 1e-6)
                measured_max_joint_speed = float(
                    np.max(np.abs(state[:6] - previous_state[:6])) / dt
                )
            previous_state = state.copy()
            previous_state_time = loop_start

            frame = camera.get_frame(require_pc=False)
            image, point_cloud = base.preprocess_camera_frame(frame)
            if args.skip_point_cloud_distribution_check:
                outside = np.any(
                    (point_cloud < stats.point_cloud_low[None])
                    | (point_cloud > stats.point_cloud_high[None]),
                    axis=1,
                )
                point_outlier_fraction = float(outside.mean())
            else:
                point_outlier_fraction = contact.validate_point_cloud_distribution(
                    point_cloud, stats, args.max_point_outlier_fraction
                )
            history.append(
                {"agent_pos": state, "point_cloud": point_cloud, "image": image}
            )

            chunk_step = step % args.chunk_exec_steps
            ran_inference = chunk_step == 0 or active_chunk is None
            decision: GripperDecision | None = None
            if ran_inference:
                with torch.no_grad():
                    output = policy.predict_action(
                        base.build_obs(history, device),
                        deterministic=True,
                        use_cm=use_cm,
                        initial_noise=episode_noise,
                    )
                active_chunk, active_probabilities = extract_policy_outputs(
                    output, n_action_steps, gripper_horizon
                )
                inference_index += 1
                chunk_step = 0
                speed_gate = (
                    measured_max_joint_speed
                    <= args.gripper_event_max_joint_speed_rad_s
                )
                # shadow模式始终解码事件；真机模式只有显式授权且速度门通过才提交锁存状态。
                allow_transition = (
                    not robot_enabled
                    or (args.allow_gripper_events and speed_gate)
                )
                decision = decoder.update(
                    active_probabilities, allow_transition=allow_transition
                )
                if decision.event == "close":
                    desired_gripper_width = args.close_width_m
                elif decision.event == "open":
                    desired_gripper_width = args.open_width_m

            predicted_arm = active_chunk[chunk_step]
            joint_target, warnings = safe_joint_target(
                predicted_arm[:6], state[:6], stats, args
            )
            if decision is not None and decision.event in {
                "close_blocked", "open_blocked"
            }:
                warnings.append(
                    f"夹爪事件{decision.event}：未授权或关节速度"
                    f"{measured_max_joint_speed:.3f}rad/s超过门限"
                )

            if robot_enabled:
                # 重复发送同一个锁存宽度是位置控制保活，不等于重复产生开合事件。
                send_action(
                    piper,
                    joint_target,
                    desired_gripper_width,
                    args.speed_percent,
                    args.gripper_effort,
                )

            record = {
                "step": step,
                "host_time": time.time(),
                "state": state.tolist(),
                "predicted_joint_chunk": active_chunk[:, :6].tolist(),
                "diffusion_gripper_state_chunk": (
                    active_chunk[:, 6].tolist() if active_chunk.shape[-1] == 7 else None
                ),
                "gripper_prob": active_probabilities.tolist(),
                "gripper_score": None if decision is None else decision.score,
                "gripper_event": None if decision is None else decision.event,
                "latched_open": decoder.latched_open,
                "desired_gripper_width_m": desired_gripper_width,
                "measured_max_joint_speed_rad_s": measured_max_joint_speed,
                "chunk_step": chunk_step,
                "policy_inference": ran_inference,
                "inference_index": inference_index,
                "safe_joint_target": joint_target.tolist(),
                "point_outlier_fraction": point_outlier_fraction,
                "warnings": warnings,
                "executed": robot_enabled,
                "gripper_events_authorized": bool(args.allow_gripper_events),
            }
            records.append(record)
            event_text = "-" if decision is None or decision.event is None else decision.event
            score_text = "-" if decision is None else f"{decision.score:.3f}"
            print(
                f"[step {step:04d}] joint={np.round(joint_target,4)} "
                f"p_open={score_text} event={event_text} "
                f"grip={desired_gripper_width:.3f}m "
                f"speed={measured_max_joint_speed:.3f} pc_out={point_outlier_fraction:.1%} "
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
                # 停止时保持反馈关节和当前反馈夹爪宽度，禁止自动张开导致物体掉落。
                for _ in range(10):
                    send_action(
                        piper,
                        last_state[:6],
                        float(last_state[6]),
                        args.speed_percent,
                        args.gripper_effort,
                    )
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
        payload = {
            "meta": {
                "output_dir": str(args.output_dir),
                "policy_subdir": args.policy_subdir,
                "use_cm": use_cm,
                "rate_hz": args.rate,
                "n_obs_steps": n_obs_steps,
                "n_action_steps": n_action_steps,
                "gripper_horizon": gripper_horizon,
                "chunk_exec_steps": args.chunk_exec_steps,
                "close_threshold": args.close_threshold,
                "open_threshold": args.open_threshold,
                "consecutive": args.consecutive,
                "cooldown_frames": args.cooldown_frames,
                "probability_window": [args.window_start, args.window_end],
                "allow_gripper_events": args.allow_gripper_events,
                "initial_gripper_state": args.initial_gripper_state,
                "open_width_m": args.open_width_m,
                "close_width_m": args.close_width_m,
                "stop_reason": stop_reason,
            },
            "records": records,
        }
        log_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[日志] 已保存: {log_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
