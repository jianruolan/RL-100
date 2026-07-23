#!/usr/bin/env python3
"""推理消融A：7维diffusion + detach单帧夹爪分类头。

默认加载完整BC结果目录。所有shadow/真机安全参数均由共享的
``infer_piper_pick_and_place_gripper_policy.py``实现；本入口额外锁定实验A
的动作维度、夹爪头长度和detach设置，防止误加载实验B权重。
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "RL-100/data/outputs/piper_pick_and_place_gripper_ablation_detach_h1_bc_seed42"
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
            "--expected-action-dims", "7",
            "--expected-gripper-horizon", "1",
            "--require-gripper-detach",
        ]
    )
    from tools.teleop_off2off_data.infer_piper_pick_and_place_gripper_policy import main

    main()
