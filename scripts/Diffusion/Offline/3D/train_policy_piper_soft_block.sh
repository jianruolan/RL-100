#!/usr/bin/env bash
# Fixed single-run offline training for the Piper soft-block contact dataset.
# Usage: bash scripts/Diffusion/Offline/3D/train_policy_piper_soft_block.sh \
#   [alg_name] [task_name] [addition_info] [seed]
set -euo pipefail

alg_name="${1:-rl100}"
task_name="${2:-piper_soft_block_contact}"
addition_info="${3:-fixed}"
seed="${4:-42}"
exp_name="${task_name}-${alg_name}-${addition_info}"
gpu_id="${GPU_ID:-0}"

cd "$(dirname "$0")/../../../.."
cd RL-100

export MUJOCO_GL=egl
export MUJOCO_PY_MUJOCO_PATH="${MUJOCO_PY_MUJOCO_PATH:-$HOME/.mujoco/mujoco210}"
export LD_LIBRARY_PATH="${MUJOCO_PY_MUJOCO_PATH}/bin:${MUJOCO_PY_MUJOCO_PATH}/lib:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"
export HYDRA_FULL_ERROR=1
export MUJOCO_EGL_DEVICE_ID="${gpu_id}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"

# Deliberately no hyperparameter loops: one reproducible configuration.
python train.py --config-name=rl100_3d_epsilon.yaml \
  task="${task_name}" \
  hydra.run.dir="data/outputs/${exp_name}_seed${seed}" \
  training.debug=False \
  training.seed="${seed}" \
  training.device=cuda:0 \
  exp_name="${exp_name}" \
  logging.mode=online \
  use_wandb=True \
  checkpoint.save_ckpt=True \
  training.resume=False \
  horizon=3 n_obs_steps=3 n_action_steps=1 \
  chunk_as_single_action=False \
  only_bc=True \
  offline=False online=False \
  policy._target_=rl_100.policy.rl100_3d.RL1003D \
  policy.encoder_type=dp3vib \
  policy.model=skipnet \
  policy.act=relu \
  policy.use_agent_pos=True \
  policy.use_vib=True \
  policy.use_recon=True \
  training.num_epochs=600 \
  task.env_runner=null
