#!/usr/bin/env bash
# 扩增数据上的RGB-D 2D BC，使用ImageNet预训练ResNet18，图像为4x224x224。
set -euo pipefail
alg_name="${1:-rl100}"
task_name="${2:-piper_pick_and_place_augmented_rgbd}"
addition_info="${3:-rgbd-resnet18-chunk4-bc}"
seed="${4:-42}"
gpu_id="${GPU_ID:-0}"
run_dir="${RUN_DIR:-data/outputs/piper_pick_and_place_augmented_rgbd_resnet18_chunk4_seed${seed}}"
resume="${RESUME:-False}"
dataset_path="${DATASET_PATH:-data/piper_pick_and_place_augmented_rgbd224.zarr}"
python_bin="${PYTHON_BIN:-python}"
checkpoint_every="${CHECKPOINT_EVERY:-10}"

cd "$(dirname "$0")/../../../.."
cd RL-100
export HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES="${gpu_id}"
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID="${gpu_id}"
export MUJOCO_PY_MUJOCO_PATH="${MUJOCO_PY_MUJOCO_PATH:-$HOME/.mujoco/mujoco210}"
export LD_LIBRARY_PATH="${MUJOCO_PY_MUJOCO_PATH}/bin:${MUJOCO_PY_MUJOCO_PATH}/lib:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"

"${python_bin}" train.py --config-name=rl100_2d_epsilon.yaml \
  task="${task_name}" hydra.run.dir="${run_dir}" \
  training.debug=False training.seed="${seed}" training.device=cuda:0 \
  exp_name="${task_name}-${alg_name}-${addition_info}" \
  logging.mode=online use_wandb=True checkpoint.save_ckpt=True \
  training.checkpoint_every="${checkpoint_every}" \
  training.resume="${resume}" +stop_after_bc=True \
  horizon=6 n_obs_steps=3 n_action_steps=4 chunk_as_single_action=True \
  dynamics.prediction_mode=full only_bc=True offline=False online=False eval=False \
  policy._target_=rl_100.policy.rl100_2d.RL1002D feature_type=2D use_agent_pos=True \
  policy.use_visual=True policy.w_pc=False policy.model=skipnet policy.act=relu \
  policy.scheduler_type=ddim policy.use_aug=False \
  encoder_type=resnet18_rgbd use_pretrained_2DEncoder=True use_vib=False use_recon=False \
  policy.img_shape="[4,224,224]" task.shape_meta.obs.image.shape="[4,224,224]" \
  dataloader.batch_size="${BATCH_SIZE:-32}" val_dataloader.batch_size="${BATCH_SIZE:-32}" \
  training.num_epochs=600 training.num_critic_epochs=600 \
  dynamics_type=diffusion dynamics.dynamics_max_epochs=350 \
  unio4.bppo_steps=6000 unio4.idql_eval=False unio4.use_ema_eval=False \
  task.env_runner=null \
  task.dataset.zarr_path="${dataset_path}" task.norm_dataset.zarr_path="${dataset_path}" \
  task.critic_dataset.zarr_path="${dataset_path}" task.scale_dataset.zarr_path="${dataset_path}"
