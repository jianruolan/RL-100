#!/usr/bin/env bash
# Piper pick-and-place 84x84 RGB + 7D 状态/动作的固定单次 2D 训练入口。
# 默认：obs=3、action chunk=4、horizon=6，BC后可继续CM和offline RL。
set -euo pipefail

alg_name="${1:-rl100}"
task_name="${2:-piper_pick_and_place_2d}"
addition_info="${3:-chunk4}"
seed="${4:-42}"

exp_name="${task_name}-${alg_name}-${addition_info}"
gpu_id="${GPU_ID:-0}"
run_dir="${RUN_DIR:-data/outputs/${exp_name}_seed${seed}}"
resume="${RESUME:-False}"
offline="${OFFLINE:-False}"
dataset_path="${DATASET_PATH:-data/piper_pick_and_place.zarr}"
image_size="${IMAGE_SIZE:-84}"
batch_size="${BATCH_SIZE:-128}"
action_steps="${ACTION_STEPS:-4}"
n_obs_steps="${N_OBS_STEPS:-3}"
horizon="${HORIZON:-6}"
distill_phase="${DISTILL_PHASE:-null}"

case "${offline}" in True|False) ;; *) echo "OFFLINE必须为True或False: ${offline}" >&2; exit 2;; esac
case "${resume}" in True|False) ;; *) echo "RESUME必须为True或False: ${resume}" >&2; exit 2;; esac
case "${distill_phase}" in null|after_dp|after_offline) ;; *) echo "DISTILL_PHASE必须为null、after_dp或after_offline: ${distill_phase}" >&2; exit 2;; esac
if [ "${image_size}" != 84 ]; then
  echo "本任务固定使用84x84 Zarr，IMAGE_SIZE必须为84，当前为: ${image_size}" >&2
  exit 2
fi
if ! [[ "${batch_size}" =~ ^[1-9][0-9]*$ ]]; then
  echo "BATCH_SIZE必须是正整数: ${batch_size}" >&2; exit 2
fi
if ! [[ "${action_steps}" =~ ^[1-9][0-9]*$ ]] || ! [[ "${n_obs_steps}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ACTION_STEPS和N_OBS_STEPS必须是正整数" >&2; exit 2
fi
expected_horizon=$((action_steps + n_obs_steps - 1))
if ! [[ "${horizon}" =~ ^[1-9][0-9]*$ ]] || [ "${horizon}" -ne "${expected_horizon}" ]; then
  echo "HORIZON必须等于N_OBS_STEPS+ACTION_STEPS-1=${expected_horizon}，当前为${horizon}" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../../.." && pwd)"
cd "${repo_root}/RL-100"

export MUJOCO_GL=egl
export MUJOCO_PY_MUJOCO_PATH="${MUJOCO_PY_MUJOCO_PATH:-$HOME/.mujoco/mujoco210}"
export LD_LIBRARY_PATH="${MUJOCO_PY_MUJOCO_PATH}/bin:${MUJOCO_PY_MUJOCO_PATH}/lib:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"
export HYDRA_FULL_ERROR=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/rl100-matplotlib}"
export MUJOCO_EGL_DEVICE_ID="${gpu_id}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"

if [[ "${dataset_path}" = /* ]]; then dataset_abs="${dataset_path}"; else dataset_abs="${PWD}/${dataset_path}"; fi
if [ ! -e "${dataset_abs}" ]; then echo "找不到数据集: ${dataset_abs}" >&2; exit 2; fi

# 启动训练前严格检查图像、state、action shape。
python - "${dataset_abs}" <<'PY'
import sys, zarr
r = zarr.open(sys.argv[1], mode="r")
expected = {"data/img": (3, 84, 84), "data/state": (7,), "data/action": (7,)}
for key, tail in expected.items():
    shape = tuple(r[key].shape)
    if shape[1:] != tail:
        raise SystemExit(f"{key} shape应为(N,{','.join(map(str, tail))})，实际为{shape}")
print(f"[Piper pick-place 2D] img={r['data/img'].shape}, state={r['data/state'].shape}, action={r['data/action'].shape}")
PY

echo "[Piper pick-place 2D] task=${task_name} output=${run_dir} seed=${seed}"
echo "[Piper pick-place 2D] dataset=${dataset_path} encoder=DrQEncoder(84) batch=${batch_size}"
echo "[Piper pick-place 2D] obs=${n_obs_steps} action_chunk=${action_steps} horizon=${horizon} distill=${distill_phase} offline=${offline} resume=${resume}"

python train.py --config-name=rl100_2d_epsilon.yaml \
  task="${task_name}" \
  hydra.run.dir="${run_dir}" \
  training.debug=False training.seed="${seed}" training.device=cuda:0 \
  exp_name="${exp_name}" logging.mode=online use_wandb=True \
  checkpoint.save_ckpt=True training.resume="${resume}" \
  distill_phase="${distill_phase}" +run_validation=True env_num=1 \
  horizon="${horizon}" n_obs_steps="${n_obs_steps}" n_action_steps="${action_steps}" \
  chunk_as_single_action=True dynamics.prediction_mode=full \
  only_bc=True offline="${offline}" online=False eval=False \
  policy._target_=rl_100.policy.rl100_2d.RL1002D feature_type=2D use_agent_pos=True \
  policy.use_visual=True policy.w_pc=False policy.model=skipnet policy.act=relu \
  policy.scheduler_type=ddim policy.use_aug=False \
  encoder_type=drq use_pretrained_2DEncoder=False \
  use_vib=False use_recon=False policy.img_shape="[3,84,84]" \
  task.shape_meta.obs.image.shape="[3,84,84]" \
  dataloader.batch_size="${batch_size}" val_dataloader.batch_size="${batch_size}" \
  training.num_epochs=600 training.num_critic_epochs=600 \
  dynamics_type=diffusion dynamics.dynamics_max_epochs=350 \
  unio4.bppo_steps=6000 unio4.idql_eval=False unio4.use_ema_eval=False \
  task.env_runner=null \
  task.dataset.zarr_path="${dataset_path}" task.norm_dataset.zarr_path="${dataset_path}" \
  task.critic_dataset.zarr_path="${dataset_path}" task.scale_dataset.zarr_path="${dataset_path}"
