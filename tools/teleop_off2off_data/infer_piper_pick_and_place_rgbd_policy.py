#!/usr/bin/env python3
"""RGB-D ResNet18策略推理入口，具体安全控制逻辑复用2D RGB/RGB-D脚本。"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.teleop_off2off_data.infer_piper_pick_and_place_2d_policy import main

if __name__ == "__main__":
    if "--output-dir" not in sys.argv:
        root = REPO_ROOT / "RL-100"
        sys.argv.extend([
            "--output-dir",
            str(root / "data/outputs/piper_pick_and_place_augmented_rgbd_resnet18_chunk4_seed42"),
        ])
    # 本实验只训练BC，最终可部署权重写入bc/，不会生成offline RL的best/。
    if "--policy-subdir" not in sys.argv:
        sys.argv.extend(["--policy-subdir", "bc"])
    main()
