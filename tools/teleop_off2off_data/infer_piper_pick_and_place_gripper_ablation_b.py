#!/usr/bin/env python3
"""推理消融B：6维机械臂diffusion + detach独立单帧夹爪头。

默认加载实验B的BC结果目录。共享推理实现会把六维diffusion输出仅用于
机械臂关节目标，夹爪只接受分类头产生的迟滞事件，不会访问不存在的
第七维动作。
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "RL-100/data/outputs/piper_pick_and_place_gripper_ablation_arm6_h1_bc_seed42"
)


def _has_option(name: str) -> bool:
    return name in sys.argv or any(arg.startswith(f"{name}=") for arg in sys.argv[1:])


if __name__ == "__main__":
    if not _has_option("--output-dir"):
        sys.argv.extend(["--output-dir", str(DEFAULT_OUTPUT_DIR)])
    if not _has_option("--policy-subdir"):
        sys.argv.extend(["--policy-subdir", "bc"])
    sys.argv.extend(
        [
            "--expected-action-dims", "6",
            "--expected-gripper-horizon", "1",
            "--require-gripper-detach",
        ]
    )
    from tools.teleop_off2off_data.infer_piper_pick_and_place_gripper_policy import main

    main()
