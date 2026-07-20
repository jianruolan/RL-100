#!/usr/bin/env python3
"""在 Piper 单臂上安全地运行 RL-100 接触策略。

默认是影子模式：连接机械臂和 D435i、执行策略推理、打印预测动作，
但绝不使能机械臂，也不下发任何运动命令。只有显式传入 ``--execute``
并在终端输入确认短语后，才会进入 20 Hz 的关节位置控制。

第一轮真机测试建议按以下顺序进行：

1. ``--offline-smoke``：只用 zarr 中的一条样本验证模型能加载；
2. 默认影子模式：读取实时相机/关节，但不下发动作；
3. ``--execute``：低速、短时、人工随时按 Enter 停止。

注意：本脚本假设训练数据中的 action 是下一时刻六关节绝对位置（弧度）。
采集脚本通过 ``action[t] = state[t+1]`` 构造 action，因此不能把模型输出
当作关节增量再次叠加到当前关节角上。
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
TRAIN_ROOT = REPO_ROOT / "RL-100"
DEFAULT_OUTPUT_DIR = TRAIN_ROOT / "data/outputs/piper_soft_block_contact_50_seed42"
DEFAULT_PIPER_SDK_ROOT = REPO_ROOT.parent / "piper_sdk"

# 保证从任意工作目录启动时都能导入 RL-100 和 tools。
for path in (str(REPO_ROOT), str(TRAIN_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

RAD_TO_RAW = 180.0 / math.pi * 1000.0
RAW_TO_RAD = math.pi / 180.0 / 1000.0

# Piper SDK 文档给出的宽松物理关节范围。真正执行时还会使用更严格的
# “训练数据范围 + margin”限制，避免策略离开示教分布。
PIPER_HARD_LOWER_RAD = np.deg2rad(
    np.array([-150.0, -1.0, -170.0, -100.0, -70.0, -120.0], dtype=np.float32)
)
PIPER_HARD_UPPER_RAD = np.deg2rad(
    np.array([150.0, 180.0, 1.0, 100.0, 70.0, 120.0], dtype=np.float32)
)


@dataclass
class TrainingStats:
    """部署安全检查使用的训练数据统计。"""

    state_min: np.ndarray
    state_max: np.ndarray
    action_min: np.ndarray
    action_max: np.ndarray
    action_delta_p99: np.ndarray
    point_cloud_low: np.ndarray
    point_cloud_high: np.ndarray


class PiperStateReader:
    """只读取 Piper 当前六关节位置，不对机器人状态做任何修改。"""

    def __init__(self, piper: Any):
        self.piper = piper

    def read_joint_rad(self) -> tuple[np.ndarray, float]:
        msg = self.piper.GetArmJointMsgs()
        joint_state = msg.joint_state
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
        joint_rad = (raw * RAW_TO_RAD).astype(np.float32)
        if not np.all(np.isfinite(joint_rad)):
            raise RuntimeError(f"机械臂关节状态包含 NaN/Inf: {joint_rad}")
        sdk_timestamp = float(getattr(msg, "time_stamp", 0.0))
        return joint_rad, sdk_timestamp


class OperatorConsole:
    """后台读取人工停止命令，避免控制循环阻塞在 input()。"""

    def __init__(self):
        self.commands: queue.Queue[str] = queue.Queue()
        self.thread = threading.Thread(target=self._read_loop, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _read_loop(self) -> None:
        while True:
            try:
                line = sys.stdin.readline()
            except Exception:
                return
            if line == "":
                return
            command = line.strip().lower()
            # 空行即直接按 Enter，视为普通停止。
            self.commands.put(command or "stop")

    def poll(self) -> str | None:
        try:
            return self.commands.get_nowait()
        except queue.Empty:
            return None


def import_piper_sdk(sdk_root: Path):
    """从用户指定目录导入 Piper SDK，避免依赖全局 Python 环境。"""

    sdk_root = sdk_root.expanduser().resolve()
    if not sdk_root.exists():
        raise FileNotFoundError(f"Piper SDK 目录不存在: {sdk_root}")
    if str(sdk_root) not in sys.path:
        sys.path.insert(0, str(sdk_root))
    try:
        from piper_sdk import C_PiperInterface_V2
    except ImportError as exc:
        raise RuntimeError(f"无法从 {sdk_root} 导入 piper_sdk") from exc
    return C_PiperInterface_V2


def resolve_training_path(path_value: str | Path) -> Path:
    """训练配置里的相对路径以 RL-100 内层训练目录为基准。"""

    path = Path(path_value).expanduser()
    return path.resolve() if path.is_absolute() else (TRAIN_ROOT / path).resolve()


def load_policy_and_dataset(
    output_dir: Path,
    policy_subdir: str,
    device: str,
):
    """加载保存的 Hydra 配置、训练集 normalizer 和策略权重。"""

    config_path = output_dir / ".hydra/config.yaml"
    policy_dir = output_dir / policy_subdir
    if not config_path.exists():
        raise FileNotFoundError(f"找不到训练配置: {config_path}")
    for filename in ("model.pt", "encoder.pt"):
        if not (policy_dir / filename).exists():
            raise FileNotFoundError(f"找不到策略文件: {policy_dir / filename}")

    # 保存的 Hydra 配置仍包含 ${eval:...}，需要注册和训练入口相同的 resolver。
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver(
        "now", lambda fmt: datetime.datetime.now().strftime(fmt), replace=True
    )
    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)

    # AdroitDataset 只是本项目通用的 zarr reader；Piper 配置通过
    # controlled_dims=6 丢弃恒定的第七维夹爪通道。
    old_cwd = Path.cwd()
    os.chdir(TRAIN_ROOT)
    try:
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        normalizer = dataset.get_normalizer()
        policy = hydra.utils.instantiate(cfg.policy)
    finally:
        os.chdir(old_cwd)

    policy.set_normalizer(normalizer)
    policy.load(str(policy_dir))
    policy.to(torch.device(device))
    policy.eval()

    dataset_path = resolve_training_path(cfg.task.dataset.zarr_path)
    print(f"[模型] 配置: {config_path}", flush=True)
    print(f"[模型] 权重: {policy_dir}", flush=True)
    print(f"[模型] normalizer 数据集: {dataset_path}", flush=True)
    print(
        f"[模型] n_obs_steps={cfg.n_obs_steps}, "
        f"n_action_steps={cfg.n_action_steps}, action_dim={cfg.shape_meta.action.shape[0]}",
        flush=True,
    )
    return cfg, dataset, policy


def compute_training_stats(dataset) -> TrainingStats:
    """从本次训练 zarr 计算部署范围，禁止跨数据版本复用统计。"""

    replay = dataset.replay_buffer
    state = np.asarray(replay["state"][:, :6], dtype=np.float32)
    action = np.asarray(replay["action"][:, :6], dtype=np.float32)
    delta = np.abs(action - state)

    # 点云较大，等间隔抽取至多 2000 帧计算分位数即可用于分布告警。
    point_cloud = replay["point_cloud"]
    stride = max(1, len(point_cloud) // 2000)
    point_sample = np.asarray(point_cloud[::stride], dtype=np.float32).reshape(-1, 3)
    point_sample = point_sample[np.all(np.isfinite(point_sample), axis=1)]
    if len(point_sample) == 0:
        raise RuntimeError("训练点云没有任何有限点")

    return TrainingStats(
        state_min=state.min(axis=0),
        state_max=state.max(axis=0),
        action_min=action.min(axis=0),
        action_max=action.max(axis=0),
        action_delta_p99=np.percentile(delta, 99, axis=0).astype(np.float32),
        point_cloud_low=np.percentile(point_sample, 0.1, axis=0).astype(np.float32),
        point_cloud_high=np.percentile(point_sample, 99.9, axis=0).astype(np.float32),
    )


def print_training_stats(stats: TrainingStats) -> None:
    np.set_printoptions(precision=5, suppress=True)
    print("[数据] state min:", stats.state_min, flush=True)
    print("[数据] state max:", stats.state_max, flush=True)
    print("[数据] action min:", stats.action_min, flush=True)
    print("[数据] action max:", stats.action_max, flush=True)
    print("[数据] |action-state| p99:", stats.action_delta_p99, flush=True)
    print("[数据] point cloud 0.1%:", stats.point_cloud_low, flush=True)
    print("[数据] point cloud 99.9%:", stats.point_cloud_high, flush=True)


def preprocess_camera_frame(frame: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """完全复用后处理脚本的 RGB 格式：BGR→RGB、84×84、CHW。"""

    color_bgr = np.asarray(frame["color"])
    point_cloud = np.asarray(frame["point_cloud"], dtype=np.float32)
    if color_bgr.ndim != 3 or color_bgr.shape[2] < 3:
        raise RuntimeError(f"D435i RGB shape 异常: {color_bgr.shape}")
    if point_cloud.shape != (512, 3):
        raise RuntimeError(f"D435i point cloud shape 异常: {point_cloud.shape}")
    if not np.all(np.isfinite(point_cloud)):
        raise RuntimeError("实时点云包含 NaN/Inf")

    rgb = cv2.cvtColor(color_bgr[..., :3], cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (84, 84), interpolation=cv2.INTER_AREA)
    rgb_chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32)
    return rgb_chw, point_cloud


def build_policy_obs(history: deque[dict[str, np.ndarray]], device: torch.device):
    """把三帧历史组装为 [B,T,...]，与训练 dataset 输出保持一致。"""

    agent_pos = np.stack([item["agent_pos"] for item in history], axis=0)
    point_cloud = np.stack([item["point_cloud"] for item in history], axis=0)
    image = np.stack([item["image"] for item in history], axis=0)
    return {
        "agent_pos": torch.from_numpy(agent_pos).unsqueeze(0).to(device),
        "point_cloud": torch.from_numpy(point_cloud).unsqueeze(0).to(device),
        "image": torch.from_numpy(image).unsqueeze(0).to(device),
    }


def extract_first_action(action_dict: dict[str, torch.Tensor]) -> np.ndarray:
    """当前 n_action_steps=1，只执行策略返回的第一步六关节目标。"""

    if "action" not in action_dict:
        raise RuntimeError(f"predict_action() 没有返回 action，keys={list(action_dict)}")
    action = action_dict["action"].detach().to("cpu").numpy()
    if action.shape[-1] != 6:
        raise RuntimeError(f"策略 action 维度不是 6: {action.shape}")
    target = action.reshape(-1, 6)[0].astype(np.float32)
    if not np.all(np.isfinite(target)):
        raise RuntimeError(f"策略输出包含 NaN/Inf: {target}")
    return target


def validate_point_cloud_distribution(
    point_cloud: np.ndarray,
    stats: TrainingStats,
    max_outlier_fraction: float,
) -> float:
    """检查实时点云是否明显偏离训练分布，异常时停止而不是继续执行。"""

    outside = np.any(
        (point_cloud < stats.point_cloud_low[None])
        | (point_cloud > stats.point_cloud_high[None]),
        axis=1,
    )
    fraction = float(outside.mean())
    if fraction > max_outlier_fraction:
        raise RuntimeError(
            f"实时点云有 {fraction:.1%} 的点超出训练分布，"
            f"阈值为 {max_outlier_fraction:.1%}；检查相机位置、深度和外参"
        )
    return fraction


def safe_joint_target(
    predicted: np.ndarray,
    current: np.ndarray,
    stats: TrainingStats,
    rate_hz: float,
    dataset_margin_rad: float,
    max_joint_speed_rad_s: float,
    clip_actions: bool,
) -> tuple[np.ndarray, list[str]]:
    """执行硬限位、示教分布限位和单周期速度限位。"""

    warnings: list[str] = []
    lower = np.maximum(PIPER_HARD_LOWER_RAD, stats.action_min - dataset_margin_rad)
    upper = np.minimum(PIPER_HARD_UPPER_RAD, stats.action_max + dataset_margin_rad)

    if np.any(current < lower) or np.any(current > upper):
        raise RuntimeError(
            "当前关节姿态已经超出训练数据安全范围；请人工移动到示教初始分布后重试。"
            f"\ncurrent={current}\nlower={lower}\nupper={upper}"
        )

    target = predicted.copy()
    if np.any(target < lower) or np.any(target > upper):
        if not clip_actions:
            raise RuntimeError(
                "策略输出超出训练数据安全范围，默认停止。"
                f"\npredicted={predicted}\nlower={lower}\nupper={upper}"
            )
        target = np.clip(target, lower, upper)
        warnings.append("策略输出被裁剪到训练数据范围")

    max_step = max_joint_speed_rad_s / rate_hz
    delta = target - current
    if np.any(np.abs(delta) > max_step):
        if not clip_actions:
            raise RuntimeError(
                "策略单步关节变化超过速度限制，默认停止。"
                f"\ndelta={delta}\nmax_step={max_step:.6f} rad"
            )
        target = current + np.clip(delta, -max_step, max_step)
        warnings.append("策略输出被速度限制器裁剪")

    if np.any(target < PIPER_HARD_LOWER_RAD) or np.any(target > PIPER_HARD_UPPER_RAD):
        raise RuntimeError("安全过滤后的动作仍超出 Piper 物理关节限位")
    return target.astype(np.float32), warnings


def send_joint_target(piper: Any, target_rad: np.ndarray) -> None:
    """Piper JointCtrl 使用 0.001 degree 整数单位。"""

    raw = np.rint(np.asarray(target_rad, dtype=np.float64) * RAD_TO_RAW).astype(np.int64)
    piper.JointCtrl(*[int(value) for value in raw])


def enable_robot_for_position_control(piper: Any, speed_percent: int, timeout: float) -> None:
    """显式使能并进入关节位置控制；调用前必须通过人工确认。"""

    piper.EnablePiper()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if bool(piper.EnablePiper()):
            break
        time.sleep(0.1)
    else:
        raise RuntimeError("Piper 在超时时间内未成功使能")

    # 解除可能残留的急停状态，然后选择 CAN 控制 + 关节位置模式。
    piper.MotionCtrl_1(0x02, 0x00, 0x00)
    piper.MotionCtrl_2(0x01, 0x01, int(speed_percent), 0x00)


def quick_stop(piper: Any) -> None:
    """请求 Piper 快速停止；故意不 Disable，避免机械臂失去保持力。"""

    try:
        piper.MotionCtrl_1(0x01, 0x00, 0x00)
    except Exception as exc:
        print(f"[安全] quick stop 发送失败: {exc}", file=sys.stderr, flush=True)


def save_run_log(log_path: Path, records: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "records": records}
    with log_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    print(f"[日志] 已保存: {log_path}", flush=True)


def offline_smoke_test(dataset, policy, device: torch.device) -> None:
    """不连接任何硬件，用训练数据验证模型、normalizer 和权重。"""

    sample = dataset[0]["obs"]
    obs = {key: value.unsqueeze(0).to(device) for key, value in sample.items()}
    with torch.no_grad():
        action_dict = policy.predict_action(obs, deterministic=True, use_cm=False)
    action = extract_first_action(action_dict)
    print("[offline-smoke] 推理成功，第一步反归一化 action(rad):", action, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--policy-subdir",
        choices=["best", "bc"],
        default="best",
        help="默认部署 offline RL best；首次真机测试建议也用 bc 做对照。",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--offline-smoke", action="store_true")
    parser.add_argument("--can", default="can0")
    parser.add_argument("--piper-sdk-root", type=Path, default=DEFAULT_PIPER_SDK_ROOT)
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--clip-actions",
        action="store_true",
        help="允许裁剪越界动作；默认更安全的行为是立即停止。",
    )
    parser.add_argument("--dataset-margin-rad", type=float, default=0.03)
    parser.add_argument(
        "--max-joint-speed-rad-s",
        type=float,
        default=0.15,
        help="第一轮测试的保守关节速度上限；确认安全后再逐步提高。",
    )
    parser.add_argument("--max-point-outlier-fraction", type=float, default=0.25)
    parser.add_argument("--state-stale-seconds", type=float, default=0.5)
    parser.add_argument("--enable-timeout", type=float, default=5.0)
    parser.add_argument("--speed-percent", type=int, default=10)
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("data/piper_inference/latest_run.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.rate <= 0 or args.rate > 20:
        raise ValueError("--rate 必须在 (0,20] Hz；本数据按 20 Hz 采集")
    if args.camera_fps < args.rate:
        raise ValueError("--camera-fps 不能低于控制频率")
    if args.max_steps <= 0:
        raise ValueError("--max-steps 必须为正整数")
    if not 1 <= args.speed_percent <= 100:
        raise ValueError("--speed-percent 必须在 [1,100]")

    cfg, dataset, policy = load_policy_and_dataset(
        args.output_dir,
        args.policy_subdir,
        args.device,
    )
    device = torch.device(args.device)
    stats = compute_training_stats(dataset)
    print_training_stats(stats)

    if int(cfg.n_obs_steps) != 3 or int(cfg.n_action_steps) != 1:
        raise RuntimeError(
            "该脚本按 n_obs_steps=3、n_action_steps=1 审核；"
            f"当前配置为 {cfg.n_obs_steps}/{cfg.n_action_steps}"
        )

    if args.offline_smoke:
        offline_smoke_test(dataset, policy, device)
        return

    # 离线 smoke test 不应依赖 RealSense/pyrealsense2；只有连接真机时再导入。
    from tools.teleop_off2off_data.realsense import RealSense

    C_PiperInterface_V2 = import_piper_sdk(args.piper_sdk_root)
    piper = C_PiperInterface_V2(args.can)
    piper.ConnectPort()
    state_reader = PiperStateReader(piper)

    camera = RealSense(
        fps=args.camera_fps,
        color_width=640,
        color_height=480,
        depth_width=640,
        depth_height=480,
        num_points=512,
    )
    camera.start()

    print("\n[模式]", "真机执行" if args.execute else "影子模式（绝不下发动作）", flush=True)
    print("[人工] 运行中直接按 Enter：普通停止；输入 q/estop：快速急停。", flush=True)
    print("[安全] 请确保急停可触达、机械臂周围无人、工作空间无障碍物。", flush=True)

    robot_enabled_by_script = False
    history: deque[dict[str, np.ndarray]] = deque(maxlen=3)
    records: list[dict[str, Any]] = []
    operator = OperatorConsole()
    last_sdk_timestamp: float | None = None
    last_fresh_state_host_time = time.monotonic()
    next_deadline = time.monotonic()

    try:
        # 连续采集三帧，而不是把第一帧简单复制三次，尽量匹配训练时间窗。
        for _ in range(3):
            joint_rad, sdk_timestamp = state_reader.read_joint_rad()
            frame = camera.get_frame(require_pc=True)
            image, point_cloud = preprocess_camera_frame(frame)
            validate_point_cloud_distribution(
                point_cloud, stats, args.max_point_outlier_fraction
            )
            history.append(
                {
                    "agent_pos": joint_rad,
                    "point_cloud": point_cloud,
                    "image": image,
                }
            )
            time.sleep(1.0 / args.rate)

        initial_joint = history[-1]["agent_pos"]
        # 即使在影子模式也检查初始姿态，提前暴露训练/部署范围不一致。
        safe_joint_target(
            initial_joint,
            initial_joint,
            stats,
            args.rate,
            args.dataset_margin_rad,
            args.max_joint_speed_rad_s,
            False,
        )

        if args.execute:
            print("\n即将使能并控制真实 Piper。", flush=True)
            print("确认以下条件：低速、急停可用、方块和机械臂均已正确复位。", flush=True)
            phrase = input("请输入 EXECUTE PIPER 继续：").strip()
            if phrase != "EXECUTE PIPER":
                raise RuntimeError("确认短语不匹配，取消真机执行")
            enable_robot_for_position_control(
                piper, args.speed_percent, args.enable_timeout
            )
            robot_enabled_by_script = True

        # 必须放在 execute 的确认 input() 之后启动，否则后台 stdin 线程可能
        # 抢先读走确认短语，导致主线程一直等待。
        operator.start()

        next_deadline = time.monotonic()
        for step in range(args.max_steps):
            command = operator.poll()
            if command in {"q", "quit", "e", "estop", "emergency"}:
                if robot_enabled_by_script:
                    quick_stop(piper)
                print("[人工] 收到急停命令", flush=True)
                break
            if command in {"stop", "s"}:
                print("[人工] 收到普通停止命令", flush=True)
                break

            loop_start = time.monotonic()
            current_joint, sdk_timestamp = state_reader.read_joint_rad()
            if last_sdk_timestamp is None or sdk_timestamp != last_sdk_timestamp:
                last_sdk_timestamp = sdk_timestamp
                last_fresh_state_host_time = loop_start
            elif loop_start - last_fresh_state_host_time > args.state_stale_seconds:
                raise RuntimeError("Piper 关节状态超过允许时间未更新")

            frame = camera.get_frame(require_pc=True)
            image, point_cloud = preprocess_camera_frame(frame)
            outlier_fraction = validate_point_cloud_distribution(
                point_cloud, stats, args.max_point_outlier_fraction
            )
            history.append(
                {
                    "agent_pos": current_joint,
                    "point_cloud": point_cloud,
                    "image": image,
                }
            )

            obs = build_policy_obs(history, device)
            with torch.no_grad():
                action_dict = policy.predict_action(
                    obs,
                    deterministic=True,
                    use_cm=False,
                )
            predicted = extract_first_action(action_dict)
            target, safety_warnings = safe_joint_target(
                predicted,
                current_joint,
                stats,
                args.rate,
                args.dataset_margin_rad,
                args.max_joint_speed_rad_s,
                args.clip_actions,
            )

            if robot_enabled_by_script:
                send_joint_target(piper, target)

            record = {
                "step": step,
                "host_time": time.time(),
                "camera_timestamp": float(frame["timestamp"]),
                "joint_rad": current_joint.tolist(),
                "predicted_action_rad": predicted.tolist(),
                "safe_target_rad": target.tolist(),
                "point_outlier_fraction": outlier_fraction,
                "safety_warnings": safety_warnings,
                "executed": robot_enabled_by_script,
            }
            records.append(record)
            print(
                f"[step {step:04d}] q={np.round(current_joint, 4)} "
                f"pred={np.round(predicted, 4)} "
                f"target={np.round(target, 4)} "
                f"pc_out={outlier_fraction:.1%} "
                f"{'EXEC' if robot_enabled_by_script else 'SHADOW'}",
                flush=True,
            )
            for warning in safety_warnings:
                print(f"[安全警告] {warning}", flush=True)

            # 以绝对 deadline 控制 20 Hz，避免推理耗时累计漂移。
            next_deadline += 1.0 / args.rate
            remaining = next_deadline - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            elif time.monotonic() - next_deadline > 1.0 / args.rate:
                raise RuntimeError(
                    "控制循环连续落后超过一个周期；降低推理频率或优化相机/模型延迟"
                )

    except KeyboardInterrupt:
        print("\n[人工] Ctrl-C，停止推理", flush=True)
        if robot_enabled_by_script:
            quick_stop(piper)
    except Exception:
        if robot_enabled_by_script:
            quick_stop(piper)
        raise
    finally:
        if robot_enabled_by_script:
            quick_stop(piper)
        try:
            camera.stop()
        except Exception as exc:
            print(f"[相机] 停止失败: {exc}", file=sys.stderr, flush=True)

        log_path = args.log.expanduser()
        if not log_path.is_absolute():
            log_path = REPO_ROOT / log_path
        save_run_log(
            log_path.resolve(),
            records,
            {
                "output_dir": str(args.output_dir),
                "policy_subdir": args.policy_subdir,
                "rate_hz": args.rate,
                "execute": args.execute,
                "max_joint_speed_rad_s": args.max_joint_speed_rad_s,
                "dataset_margin_rad": args.dataset_margin_rad,
            },
        )


if __name__ == "__main__":
    main()
