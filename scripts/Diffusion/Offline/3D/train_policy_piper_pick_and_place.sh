#!/usr/bin/env bash
# Piper pick-and-place 的单次、固定配置 3D 离线训练入口。
#
# 用法：
#   bash scripts/Diffusion/Offline/3D/train_policy_piper_pick_and_place.sh \
#     [alg_name] [task_name] [addition_info] [seed]
#
# 可选环境变量：
#   GPU_ID=0
#   OFFLINE=True       BC 后继续 Q/Value、dynamics 和 offline RL
#   RESUME=False       从 RUN_DIR/checkpoints/latest.ckpt 恢复
#   RUN_DIR=...        本次实验输出目录（相对于内层 RL-100 项目）
#   DATASET_PATH=...   临时覆盖 YAML 中的 Zarr 路径
#   ACTION_STEPS=1     action chunk 中参与训练的连续绝对动作数
#   N_OBS_STEPS=3      历史观察帧数
#   HORIZON=3          必须等于 N_OBS_STEPS+ACTION_STEPS-1
#   DISTILL_PHASE=null 一致性蒸馏阶段；默认关闭
set -euo pipefail

alg_name="${1:-rl100}"
task_name="${2:-piper_pick_and_place}"
addition_info="${3:-fixed}"
seed="${4:-42}"
exp_name="${task_name}-${alg_name}-${addition_info}"
gpu_id="${GPU_ID:-0}"
run_dir="${RUN_DIR:-data/outputs/${exp_name}_seed${seed}}"
resume="${RESUME:-False}"
offline="${OFFLINE:-False}"
distill_phase="${DISTILL_PHASE:-null}"
dataset_path="${DATASET_PATH:-}"
action_steps="${ACTION_STEPS:-1}"
horizon="${HORIZON:-3}"
n_obs_steps="${N_OBS_STEPS:-3}"

# 无论从仓库外层还是其他当前目录启动，都进入实际 Python 项目根目录。
cd "$(dirname "$0")/../../../.."
cd RL-100

export MUJOCO_GL=egl
export MUJOCO_PY_MUJOCO_PATH="${MUJOCO_PY_MUJOCO_PATH:-$HOME/.mujoco/mujoco210}"
export LD_LIBRARY_PATH="${MUJOCO_PY_MUJOCO_PATH}/bin:${MUJOCO_PY_MUJOCO_PATH}/lib:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"
export HYDRA_FULL_ERROR=1
export MUJOCO_EGL_DEVICE_ID="${gpu_id}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"

if ! [[ "${action_steps}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ACTION_STEPS必须是正整数，当前为: ${action_steps}" >&2
  exit 2
fi
if ! [[ "${n_obs_steps}" =~ ^[1-9][0-9]*$ ]]; then
  echo "N_OBS_STEPS必须是正整数，当前为: ${n_obs_steps}" >&2
  exit 2
fi
expected_horizon=$((action_steps + n_obs_steps - 1))
if ! [[ "${horizon}" =~ ^[1-9][0-9]*$ ]] || [ "${horizon}" -ne "${expected_horizon}" ]; then
  echo "当前N_OBS_STEPS=${n_obs_steps}时，HORIZON必须等于N_OBS_STEPS+ACTION_STEPS-1=${expected_horizon}，当前为: ${horizon}/${action_steps}" >&2
  exit 2
fi

dataset_overrides=()
if [ -n "${dataset_path}" ]; then
  dataset_overrides+=(
    "task.dataset.zarr_path=${dataset_path}"
    "task.norm_dataset.zarr_path=${dataset_path}"
    "task.critic_dataset.zarr_path=${dataset_path}"
    "task.scale_dataset.zarr_path=${dataset_path}"
  )
fi

# 不执行超参数组合：每次调用只启动一个可复现实验。
python train.py --config-name=rl100_3d_epsilon.yaml \
  task="${task_name}" \
  hydra.run.dir="${run_dir}" \
  training.debug=False \
  training.seed="${seed}" \
  training.device=cuda:0 \
  exp_name="${exp_name}" \
  logging.mode=online \
  use_wandb=True \
  checkpoint.save_ckpt=True \
  training.resume="${resume}" \
  distill_phase="${distill_phase}" \
  horizon="${horizon}" n_obs_steps="${n_obs_steps}" n_action_steps="${action_steps}" \
  chunk_as_single_action=True \
  dynamics.prediction_mode=full \
  only_bc=True \
  offline="${offline}" online=False \
  policy._target_=rl_100.policy.rl100_3d.RL1003D \
  policy.encoder_type=dp3vib \
  policy.model=skipnet \
  policy.act=relu \
  policy.use_agent_pos=True \
  policy.use_vib=True \
  policy.use_recon=True \
  training.num_epochs=600 \
  task.env_runner=null \
  "${dataset_overrides[@]}"
