#!/usr/bin/env bash
# Control-clean Piper pick-and-place RGB224 BC with RL100 R3M ResNet18 encoder.
set -euo pipefail

alg_name="${1:-rl100}"
task_name="${2:-piper_pick_and_place_augmented_rgb224_control_clean}"
addition_info="${3:-rgb224-r3m-resnet18-dp3-episode10-bs64-epoch2000}"
seed="${4:-42}"
gpu_id="${GPU_ID:-0}"
run_dir="${RUN_DIR:-data/outputs/piper_pick_and_place_augmented_rgb224_control_clean_resnet18r3m_dp3_episode10_bs64_epoch2000_seed${seed}}"
dataset_path="${DATASET_PATH:-data/piper_pick_and_place_augmented_chunk4_control_clean_rgb224.zarr}"
python_bin="${PYTHON_BIN:-python}"
num_epochs="${NUM_EPOCHS:-2000}"
resume="${RESUME:-False}"
windows_per_episode="${WINDOWS_PER_EPISODE:-10}"
batch_size="${BATCH_SIZE:-64}"
use_wandb="${USE_WANDB:-True}"
wandb_mode="${WANDB_MODE:-online}"
save_fraction_milestones="${SAVE_FRACTION_MILESTONES:-True}"
save_best_val="${SAVE_BEST_VAL:-True}"
n_obs_steps="${N_OBS_STEPS:-3}"
action_steps="${ACTION_STEPS:-4}"
horizon="${HORIZON:-$((n_obs_steps - 1 + action_steps))}"

if ! [[ "${num_epochs}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_EPOCHS must be positive, got: ${num_epochs}" >&2
  exit 2
fi
case "${resume}" in
  True|False) ;;
  *) echo "RESUME must be True or False, got: ${resume}" >&2; exit 2 ;;
esac
if ! [[ "${windows_per_episode}" =~ ^[1-9][0-9]*$ ]]; then
  echo "WINDOWS_PER_EPISODE must be positive, got: ${windows_per_episode}" >&2
  exit 2
fi
if ! [[ "${batch_size}" =~ ^[1-9][0-9]*$ ]]; then
  echo "BATCH_SIZE must be positive, got: ${batch_size}" >&2
  exit 2
fi
expected_horizon=$((n_obs_steps - 1 + action_steps))
if ! [[ "${horizon}" =~ ^[1-9][0-9]*$ ]] || [ "${horizon}" -ne "${expected_horizon}" ]; then
  echo "HORIZON must equal N_OBS_STEPS+ACTION_STEPS-1=${expected_horizon}, got: ${horizon}" >&2
  exit 2
fi

cd "$(dirname "$0")/../../../.."
cd RL-100
export HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES="${gpu_id}"
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID="${gpu_id}"
export MUJOCO_PY_MUJOCO_PATH="${MUJOCO_PY_MUJOCO_PATH:-$HOME/.mujoco/mujoco210}"
export LD_LIBRARY_PATH="${MUJOCO_PY_MUJOCO_PATH}/bin:${MUJOCO_PY_MUJOCO_PATH}/lib:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"

if [[ "${dataset_path}" = /* ]]; then
  dataset_abs="${dataset_path}"
else
  dataset_abs="${PWD}/${dataset_path}"
fi
if [ ! -e "${dataset_abs}" ]; then
  echo "Dataset not found: ${dataset_abs}" >&2
  exit 2
fi

"${python_bin}" - "${dataset_abs}" <<'PY'
import sys
import zarr

root = zarr.open(sys.argv[1], mode="r")
expected = {
    "data/img": (3, 224, 224),
    "data/state": (7,),
    "data/policy_action": (7,),
}
for key, tail in expected.items():
    shape = tuple(root[key].shape)
    if shape[1:] != tail:
        raise SystemExit(f"{key} shape must be (N,{tail}), got {shape}")
print(
    f"[Piper RGB224] img={root['data/img'].shape} "
    f"state={root['data/state'].shape} policy_action={root['data/policy_action'].shape}",
    flush=True,
)
PY

echo "[训练配置] RGB224 R3M ResNet18 encoder | standard DP3 denoiser | DDIM | no VIB/reconstruction/CM"
echo "[训练配置] dataset=${dataset_path} epochs=${num_epochs} resume=${resume} windows/episode=${windows_per_episode} batch_size=${batch_size} obs_steps=${n_obs_steps} action_steps=${action_steps} horizon=${horizon}"
echo "[训练预计] 启动后根据前20个epoch的实测速度输出ETA"

"${python_bin}" train.py --config-name=rl100_2d_epsilon.yaml \
  task="${task_name}" hydra.run.dir="${run_dir}" \
  training.debug=False training.seed="${seed}" training.device=cuda:0 \
  exp_name="${task_name}-${alg_name}-${addition_info}" \
  logging.mode="${wandb_mode}" use_wandb="${use_wandb}" \
  checkpoint.save_ckpt=True \
  +checkpoint.save_fraction_milestones="${save_fraction_milestones}" \
  +checkpoint.save_best_val="${save_best_val}" \
  training.resume="${resume}" +stop_after_bc=True \
  horizon="${horizon}" n_obs_steps="${n_obs_steps}" n_action_steps="${action_steps}" \
  chunk_as_single_action=True dynamics.prediction_mode=full \
  only_bc=True offline=False online=False eval=False distill_phase=null kl_annealing=False \
  policy._target_=rl_100.policy.rl100_2d.RL1002D \
  feature_type=2D use_agent_pos=True encoder_type=resnet \
  policy.use_visual=True policy.w_pc=False policy.model=dp3 policy.scheduler_type=ddim \
  policy.use_cm=False policy.use_aug=False \
  use_vib=False use_recon=False use_pretrained_2DEncoder=False \
  policy.img_shape="[3,224,224]" task.shape_meta.obs.image.shape="[3,224,224]" \
  policy.diffusion_step_embed_dim=256 policy.down_dims="[256,512,1024]" \
  training.num_epochs="${num_epochs}" training.num_critic_epochs="${num_epochs}" \
  dataloader.batch_size="${batch_size}" val_dataloader.batch_size="${batch_size}" \
  training.val_every=20 training.sample_every=20 training.checkpoint_every=100 \
  task.dataset.windows_per_episode_per_epoch="${windows_per_episode}" \
  task.env_runner=null \
  task.dataset.zarr_path="${dataset_path}" task.norm_dataset.zarr_path="${dataset_path}" \
  task.critic_dataset.zarr_path="${dataset_path}" task.scale_dataset.zarr_path="${dataset_path}"
