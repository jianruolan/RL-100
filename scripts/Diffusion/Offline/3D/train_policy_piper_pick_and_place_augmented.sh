#!/usr/bin/env bash
# 扩增数据上的原始7D diffusion BC，chunk=4，不启用夹爪专用head。
set -euo pipefail
alg_name="${1:-rl100}"
task_name="${2:-piper_pick_and_place_augmented}"
addition_info="${3:-chunk4-bc}"
seed="${4:-42}"
gpu_id="${GPU_ID:-0}"
run_dir="${RUN_DIR:-data/outputs/piper_pick_and_place_augmented_chunk4_seed${seed}}"
resume="${RESUME:-False}"
dataset_path="${DATASET_PATH:-data/piper_pick_and_place_augmented_chunk4.zarr}"

cd "$(dirname "$0")/../../../.."
cd RL-100
export HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES="${gpu_id}"
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID="${gpu_id}"
export MUJOCO_PY_MUJOCO_PATH="${MUJOCO_PY_MUJOCO_PATH:-$HOME/.mujoco/mujoco210}"
export LD_LIBRARY_PATH="${MUJOCO_PY_MUJOCO_PATH}/bin:${MUJOCO_PY_MUJOCO_PATH}/lib:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"

python train.py --config-name=rl100_3d_epsilon.yaml \
  task="${task_name}" hydra.run.dir="${run_dir}" \
  training.debug=False training.seed="${seed}" training.device=cuda:0 \
  exp_name="${task_name}-${alg_name}-${addition_info}" \
  logging.mode=online use_wandb=True checkpoint.save_ckpt=True \
  training.resume="${resume}" +stop_after_bc=True \
  horizon=6 n_obs_steps=3 n_action_steps=4 chunk_as_single_action=True \
  dynamics.prediction_mode=full only_bc=True offline=False online=False \
  policy._target_=rl_100.policy.rl100_3d.RL1003D \
  policy.encoder_type=dp3vib policy.model=skipnet policy.act=relu \
  policy.use_agent_pos=True policy.use_vib=True policy.use_recon=True \
  +policy.use_gripper_head=False \
  training.num_epochs=600 task.env_runner=null \
  task.dataset.zarr_path="${dataset_path}" task.norm_dataset.zarr_path="${dataset_path}" \
  task.critic_dataset.zarr_path="${dataset_path}" task.scale_dataset.zarr_path="${dataset_path}"
