#!/usr/bin/env python3
"""RGB224 Piper pick-and-place policy 推理入口。

这是 ``infer_piper_pick_and_place_2d_policy.py`` 的轻量包装，默认指向
当前RGB224训练输出和 ``best_val`` 权重；其它真机参数仍可通过命令行覆盖。
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
TRAIN_ROOT = REPO_ROOT / "RL-100"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.teleop_off2off_data import infer_piper_pick_and_place_2d_policy as base


DEFAULT_RGB224_OUTPUT_DIR = (
    TRAIN_ROOT
    / "data/outputs/piper_pick_and_place_augmented_rgb224_control_clean_resnet18r3m_dp3_episode10_bs64_epoch2000_seed42"
)


def _has_option(names: set[str]) -> bool:
    return any(arg in names or any(arg.startswith(name + "=") for name in names) for arg in sys.argv[1:])


def main() -> None:
    if not _has_option({"--output-dir"}):
        sys.argv.extend(["--output-dir", str(DEFAULT_RGB224_OUTPUT_DIR)])
    if not _has_option({"--policy-subdir"}):
        sys.argv.extend(["--policy-subdir", "best_val"])
    base.main()


if __name__ == "__main__":
    main()
