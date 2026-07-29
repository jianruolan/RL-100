#!/usr/bin/env bash
# Train the 19.14M medium DP3 model on trajectories truncated three frames after release.
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../../../.." && pwd)"
project_root="${repo_root}/RL-100"
dataset_path="data/piper_pick_and_place_augmented_chunk4_control_clean_release3.zarr"
run_dir="data/outputs/piper_pick_and_place_augmented_chunk4_control_clean_release3_dp3_medium_episode10_bs64_epoch4000_seed42"
status_dir="${project_root}/data/outputs/piper_pick_and_place_augmented_chunk4_control_clean_release3_dp3_medium_status"

mkdir -p "${status_dir}"
exec > >(tee -a "${status_dir}/train.log") 2>&1

if [[ -e "${project_root}/${run_dir}" ]]; then
    echo "Refusing to overwrite existing run directory: ${project_root}/${run_dir}" >&2
    exit 2
fi

rm -f "${status_dir}/COMPLETE" "${status_dir}/FAILED"
trap 'status=$?; printf "%s exit=%s\n" "$(date -Is)" "$status" >> "${status_dir}/status.log"; touch "${status_dir}/FAILED"; exit "$status"' ERR

cd "${project_root}"
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export MUJOCO_PY_MUJOCO_PATH=/home/mtarch/.mujoco/mujoco210
export LD_LIBRARY_PATH="${MUJOCO_PY_MUJOCO_PATH}/bin:${MUJOCO_PY_MUJOCO_PATH}/lib:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"

printf "%s start medium release3 dataset=%s run_dir=%s\n" \
    "$(date -Is)" "${dataset_path}" "${run_dir}" | tee -a "${status_dir}/status.log"

python train.py --config-name=rl100_3d_epsilon.yaml \
    task=piper_pick_and_place_augmented hydra.run.dir="${run_dir}" \
    training.debug=False training.seed=42 training.device=cuda:0 \
    exp_name=piper-pick-place-release3-medium-dp3-bs64 \
    logging.mode=online use_wandb=True checkpoint.save_ckpt=True \
    +checkpoint.save_fraction_milestones=True +checkpoint.save_best_val=True \
    training.resume=False +stop_after_bc=True \
    horizon=6 n_obs_steps=3 n_action_steps=4 chunk_as_single_action=True \
    dynamics.prediction_mode=full only_bc=True offline=False online=False \
    distill_phase=null kl_annealing=False \
    policy._target_=rl_100.policy.rl100_3d.RL1003D \
    policy.encoder_type=dp3 policy.model=dp3 policy.scheduler_type=ddim \
    policy.use_agent_pos=True policy.use_vib=False policy.use_recon=False \
    policy.use_cm=False +policy.use_gripper_head=False \
    policy.encoder_output_dim=64 policy.diffusion_step_embed_dim=128 \
    policy.down_dims='[128,256,512]' \
    training.num_epochs=4000 dataloader.batch_size=64 \
    training.val_every=20 training.sample_every=20 training.checkpoint_every=100 \
    task.dataset.windows_per_episode_per_epoch=10 task.env_runner=null \
    task.dataset.zarr_path="${dataset_path}" \
    task.norm_dataset.zarr_path="${dataset_path}" \
    task.critic_dataset.zarr_path="${dataset_path}" \
    task.scale_dataset.zarr_path="${dataset_path}"

printf "%s complete medium release3\n" "$(date -Is)" | tee -a "${status_dir}/status.log"
touch "${status_dir}/COMPLETE"
