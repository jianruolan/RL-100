#!/usr/bin/env bash
# 等待一个已有训练进程结束，然后依次训练 Piper pick-and-place 3D chunk 8 和 chunk 12。
#
# 用法：
#   bash queue_piper_pick_and_place_chunk8_chunk12.sh <当前训练主进程PID>
#
# 队列采用 set -e：chunk 8 失败时不会继续启动 chunk 12，避免重复消耗算力。
set -euo pipefail

wait_pid="${1:?请传入当前训练主进程PID}"
repo_root="$(cd "$(dirname "$0")/../../../.." && pwd)"
launcher="${repo_root}/scripts/Diffusion/Offline/3D/train_policy_piper_pick_and_place.sh"
dataset_path="data/piper_pick_and_place.zarr"
seed=42

# 后台启动时显式使用与当前实验相同的 Conda 环境，不依赖交互式 conda activate。
export PATH="/home/mtarch/miniforge3/envs/rl100_test/bin:${PATH}"
export CONDA_PREFIX="/home/mtarch/miniforge3/envs/rl100_test"
export CONDA_DEFAULT_ENV="rl100_test"

timestamp() {
  date '+%F %T'
}

echo "[$(timestamp)] 等待当前训练进程 PID=${wait_pid} 完成。"
while kill -0 "${wait_pid}" 2>/dev/null; do
  sleep 30
done
echo "[$(timestamp)] 当前训练已结束，开始 3D chunk=8 实验。"

cd "${repo_root}"

GPU_ID=0 \
OFFLINE=True \
RESUME=False \
DISTILL_PHASE=after_dp \
N_OBS_STEPS=3 \
ACTION_STEPS=8 \
HORIZON=10 \
DATASET_PATH="${dataset_path}" \
RUN_DIR=data/outputs/piper_pick_and_place_3d_chunk8_bc_cm_offline_seed42 \
bash "${launcher}" \
  rl100 \
  piper_pick_and_place \
  chunk8-bc-cm-offline \
  "${seed}"

echo "[$(timestamp)] 3D chunk=8 已完成，开始 3D chunk=12 实验。"

GPU_ID=0 \
OFFLINE=True \
RESUME=False \
DISTILL_PHASE=after_dp \
N_OBS_STEPS=3 \
ACTION_STEPS=12 \
HORIZON=14 \
DATASET_PATH="${dataset_path}" \
RUN_DIR=data/outputs/piper_pick_and_place_3d_chunk12_bc_cm_offline_seed42 \
bash "${launcher}" \
  rl100 \
  piper_pick_and_place \
  chunk12-bc-cm-offline \
  "${seed}"

echo "[$(timestamp)] 3D chunk=8 和 chunk=12 实验均已完成。"
