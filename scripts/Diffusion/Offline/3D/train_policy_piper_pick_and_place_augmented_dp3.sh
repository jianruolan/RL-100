#!/usr/bin/env bash
# Standard-DP3 architecture ablation on the augmented Piper chunk-4 dataset.
set -euo pipefail

alg_name="${1:-rl100}"
task_name="${2:-piper_pick_and_place_augmented}"
addition_info="${3:-chunk4-standard-dp3-epoch800}"
seed="${4:-42}"
gpu_id="${GPU_ID:-0}"
run_dir="${RUN_DIR:-data/outputs/piper_pick_and_place_augmented_chunk4_dp3_episode4_epoch800_seed${seed}}"
dataset_path="${DATASET_PATH:-data/piper_pick_and_place_augmented_chunk4.zarr}"
python_bin="${PYTHON_BIN:-python}"
num_epochs="${NUM_EPOCHS:-800}"
windows_per_episode="${WINDOWS_PER_EPISODE:-4}"
batch_size="${BATCH_SIZE:-256}"
use_wandb="${USE_WANDB:-True}"
wandb_mode="${WANDB_MODE:-online}"

if ! [[ "${num_epochs}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_EPOCHS必须是正数，当前为: ${num_epochs}" >&2
  exit 2
fi
if ! [[ "${windows_per_episode}" =~ ^[1-9][0-9]*$ ]]; then
  echo "WINDOWS_PER_EPISODE必须是正数，当前为: ${windows_per_episode}" >&2
  exit 2
fi
if ! [[ "${batch_size}" =~ ^[1-9][0-9]*$ ]]; then
  echo "BATCH_SIZE必须是正数，当前为: ${batch_size}" >&2
  exit 2
fi

cd "$(dirname "$0")/../../../.."
cd RL-100
export HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES="${gpu_id}"
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID="${gpu_id}"
export MUJOCO_PY_MUJOCO_PATH="${MUJOCO_PY_MUJOCO_PATH:-$HOME/.mujoco/mujoco210}"
export LD_LIBRARY_PATH="${MUJOCO_PY_MUJOCO_PATH}/bin:${MUJOCO_PY_MUJOCO_PATH}/lib:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"

echo "[训练配置] standard DP3 | DDIM | no VIB/reconstruction/CM"
echo "[训练配置] dataset=${dataset_path} epochs=${num_epochs} windows/episode=${windows_per_episode} batch_size=${batch_size}"
echo "[训练预计] 启动后根据前20个epoch的实测速度输出ETA"

"${python_bin}" train.py --config-name=rl100_3d_epsilon.yaml \
  task="${task_name}" hydra.run.dir="${run_dir}" \
  training.debug=False training.seed="${seed}" training.device=cuda:0 \
  exp_name="${task_name}-${alg_name}-${addition_info}" \
  logging.mode="${wandb_mode}" use_wandb="${use_wandb}" \
  checkpoint.save_ckpt=True training.resume=False +stop_after_bc=True \
  horizon=6 n_obs_steps=3 n_action_steps=4 chunk_as_single_action=True \
  dynamics.prediction_mode=full only_bc=True offline=False online=False \
  distill_phase=null kl_annealing=False \
  policy._target_=rl_100.policy.rl100_3d.RL1003D \
  policy.encoder_type=dp3 policy.model=dp3 policy.scheduler_type=ddim \
  policy.use_agent_pos=True policy.use_vib=False policy.use_recon=False \
  policy.use_cm=False +policy.use_gripper_head=False \
  policy.encoder_output_dim=64 policy.diffusion_step_embed_dim=256 \
  policy.down_dims="[256,512,1024]" \
  training.num_epochs="${num_epochs}" \
  dataloader.batch_size="${batch_size}" \
  training.val_every=20 training.sample_every=20 training.checkpoint_every=100 \
  task.dataset.windows_per_episode_per_epoch="${windows_per_episode}" \
  task.env_runner=null \
  task.dataset.zarr_path="${dataset_path}" task.norm_dataset.zarr_path="${dataset_path}" \
  task.critic_dataset.zarr_path="${dataset_path}" task.scale_dataset.zarr_path="${dataset_path}"
