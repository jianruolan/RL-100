#!/usr/bin/env bash
# Train three compact DP3 variants sequentially on one GPU.
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../../../.." && pwd)"
project_root="${repo_root}/RL-100"
python_bin="${PYTHON_BIN:-python}"
gpu_id="${GPU_ID:-0}"
queue_dir="${project_root}/data/outputs/piper_pick_and_place_augmented_chunk4_control_clean_dp3_compact_queue"
dataset_path="data/piper_pick_and_place_augmented_chunk4_control_clean.zarr"

mkdir -p "${queue_dir}"
rm -f "${queue_dir}/FAILED" "${queue_dir}/COMPLETE" "${queue_dir}/ALL_COMPLETE" "${queue_dir}/MEDIUM_FAILED"
stage="initializing"
trap 'status=$?; printf "%s stage=%s exit=%s\n" "$(date -Is)" "$stage" "$status" >> "${queue_dir}/queue.log"; touch "${queue_dir}/FAILED"; exit "$status"' ERR

cd "${project_root}"
export HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES="${gpu_id}"
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID="${gpu_id}"
export MUJOCO_PY_MUJOCO_PATH="${MUJOCO_PY_MUJOCO_PATH:-$HOME/.mujoco/mujoco210}"
export LD_LIBRARY_PATH="${MUJOCO_PY_MUJOCO_PATH}/bin:${MUJOCO_PY_MUJOCO_PATH}/lib:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"

run_model() {
    local label="$1"
    local dims="$2"
    local embed_dim="$3"
    local run_dir="data/outputs/piper_pick_and_place_augmented_chunk4_control_clean_dp3_${label}_episode10_bs64_epoch4000_seed42"

    if [[ -e "${run_dir}" ]]; then
        echo "Refusing to overwrite existing run directory: ${run_dir}" >&2
        return 2
    fi
    stage="${label}"
    printf "%s start=%s dims=%s embed=%s run_dir=%s\n" "$(date -Is)" "$label" "$dims" "$embed_dim" "$run_dir" | tee -a "${queue_dir}/queue.log"

    "${python_bin}" train.py --config-name=rl100_3d_epsilon.yaml \
        task=piper_pick_and_place_augmented hydra.run.dir="${run_dir}" \
        training.debug=False training.seed=42 training.device=cuda:0 \
        exp_name="piper_pick_and_place_augmented-compact-${label}-dp3-bs64" \
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
        policy.encoder_output_dim=64 policy.diffusion_step_embed_dim="${embed_dim}" \
        policy.down_dims="${dims}" \
        training.num_epochs=4000 dataloader.batch_size=64 \
        training.val_every=20 training.sample_every=20 training.checkpoint_every=100 \
        task.dataset.windows_per_episode_per_epoch=10 task.env_runner=null \
        task.dataset.zarr_path="${dataset_path}" task.norm_dataset.zarr_path="${dataset_path}" \
        task.critic_dataset.zarr_path="${dataset_path}" task.scale_dataset.zarr_path="${dataset_path}"

    printf "%s complete=%s\n" "$(date -Is)" "$label" | tee -a "${queue_dir}/queue.log"
}

# Train the two smaller variants first, then the 19.14M medium variant.
run_model "small96" "[96,192,384]" 128
run_model "tiny64" "[64,128,256]" 128
run_model "medium" "[128,256,512]" 128
stage="complete"
printf "%s queue_complete\n" "$(date -Is)" | tee -a "${queue_dir}/queue.log"
touch "${queue_dir}/COMPLETE"
touch "${queue_dir}/ALL_COMPLETE"
