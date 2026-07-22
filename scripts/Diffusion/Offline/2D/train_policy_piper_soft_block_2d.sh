#!/usr/bin/env bash
# Piper 腕部 RGB 图像 + 六维关节状态的固定单次 2D Diffusion 训练。
#
# 用法：
#   bash scripts/Diffusion/Offline/2D/train_policy_piper_soft_block_2d.sh \
#     [alg_name] [task_name] [addition_info] [seed]
#
# 可选环境变量：
#   GPU_ID=0                 使用的物理 GPU
#   OFFLINE=True             True: BC 后继续 Q/Value、dynamics、offline RL
#   RESUME=False             是否从 RUN_DIR 恢复
#   RUN_DIR=...              Hydra 输出目录（相对 RL-100/ 内层训练目录）
#   DATASET_PATH=...         覆盖所有训练阶段使用的 Zarr 路径
#   IMAGE_SIZE=84            Zarr 中图像的方形边长
#   BATCH_SIZE=128           2D ResNet 训练 batch size
#   RGB_WEIGHTS=r3m          r3m / IMAGENET1K_V1 / null
#   ACTION_STEPS=1           每次策略预测的连续绝对动作数
#   N_OBS_STEPS=3           历史观测帧数
#   HORIZON=3                轨迹窗口；须等于N_OBS_STEPS+ACTION_STEPS-1
#   DISTILL_PHASE=null       null / after_dp / after_offline
set -euo pipefail

alg_name="${1:-rl100}"
task_name="${2:-piper_soft_block_contact_2d}"
addition_info="${3:-fixed}"
seed="${4:-42}"

exp_name="${task_name}-${alg_name}-${addition_info}"
gpu_id="${GPU_ID:-0}"
run_dir="${RUN_DIR:-data/outputs/${exp_name}_seed${seed}}"
resume="${RESUME:-False}"
offline="${OFFLINE:-False}"
dataset_path="${DATASET_PATH:-data/piper_soft_block_contact_50_trimmed.zarr}"
image_size="${IMAGE_SIZE:-84}"
batch_size="${BATCH_SIZE:-128}"
# 与仓库现有 2D 训练脚本保持一致，默认使用机器人操作视频预训练的 R3M。
rgb_weights="${RGB_WEIGHTS:-r3m}"
action_steps="${ACTION_STEPS:-1}"
horizon="${HORIZON:-3}"
n_obs_steps="${N_OBS_STEPS:-3}"
distill_phase="${DISTILL_PHASE:-null}"

case "${offline}" in
  True|False) ;;
  *) echo "OFFLINE 必须为 True 或 False，当前为: ${offline}" >&2; exit 2 ;;
esac
case "${resume}" in
  True|False) ;;
  *) echo "RESUME 必须为 True 或 False，当前为: ${resume}" >&2; exit 2 ;;
esac
if ! [[ "${image_size}" =~ ^[0-9]+$ ]] || [ "${image_size}" -le 0 ]; then
  echo "IMAGE_SIZE 必须为正整数，当前为: ${image_size}" >&2
  exit 2
fi
if ! [[ "${batch_size}" =~ ^[0-9]+$ ]] || [ "${batch_size}" -le 0 ]; then
  echo "BATCH_SIZE 必须为正整数，当前为: ${batch_size}" >&2
  exit 2
fi
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
case "${distill_phase}" in
  null|after_dp|after_offline) ;;
  *) echo "DISTILL_PHASE必须为null、after_dp或after_offline，当前为: ${distill_phase}" >&2; exit 2 ;;
esac

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

if [[ "${rgb_weights,,}" == "r3m" ]]; then
  if ! python -c "import r3m" >/dev/null 2>&1; then
    echo "RGB_WEIGHTS=r3m，但当前 Python 环境没有安装 r3m。" >&2
    echo "请先安装 r3m，或使用 RGB_WEIGHTS=null / IMAGENET1K_V1。" >&2
    exit 2
  fi
fi

if [[ "${dataset_path}" = /* ]]; then
  dataset_abs="${dataset_path}"
else
  dataset_abs="${PWD}/${dataset_path}"
fi
if [ ! -e "${dataset_abs}" ]; then
  echo "找不到2D训练数据集: ${dataset_abs}" >&2
  echo "当前仓库的84x84数据是: ${PWD}/data/piper_soft_block_contact_50_trimmed.zarr" >&2
  exit 2
fi

# 在启动耗时的模型和WandB初始化前，验证Zarr图像尺寸与IMAGE_SIZE一致。
python - "${dataset_abs}" "${image_size}" <<'PY'
import sys
import zarr

path = sys.argv[1]
expected = int(sys.argv[2])
root = zarr.open(path, mode="r")
if "data/img" not in root:
    raise SystemExit(f"Zarr中找不到data/img: {path}")
shape = tuple(root["data/img"].shape)
if len(shape) != 4:
    raise SystemExit(f"data/img应为4维，实际为{shape}: {path}")
if shape[1] == 3:       # NCHW
    height, width = shape[2], shape[3]
elif shape[-1] == 3:    # NHWC
    height, width = shape[1], shape[2]
else:
    raise SystemExit(f"无法识别data/img通道维，实际为{shape}: {path}")
if (height, width) != (expected, expected):
    raise SystemExit(
        f"IMAGE_SIZE={expected}与Zarr图像{height}x{width}不一致: {path}"
    )
print(f"[Piper 2D] dataset image shape={shape}")
PY

# Hydra 的 YAML anchor 在组合后是独立节点。覆盖数据集路径时必须同步修改
# dataset / norm_dataset / critic_dataset / scale_dataset，避免不同阶段误读旧数据。
dataset_overrides=(
  "task.dataset.zarr_path=${dataset_path}"
  "task.norm_dataset.zarr_path=${dataset_path}"
  "task.critic_dataset.zarr_path=${dataset_path}"
  "task.scale_dataset.zarr_path=${dataset_path}"
)

echo "[Piper 2D] task=${task_name} seed=${seed} gpu=${gpu_id}"
echo "[Piper 2D] output=${run_dir} offline=${offline} resume=${resume}"
echo "[Piper 2D] image=${image_size}x${image_size} batch_size=${batch_size}"
echo "[Piper 2D] RGB weights=${rgb_weights}"
echo "[Piper 2D] dataset=${dataset_path}"
echo "[Piper 2D] horizon=${horizon} action_steps=${action_steps} distill_phase=${distill_phase}"

# 固定单次配置，不包含学习率、rollout length 或其他超参数循环。
python train.py --config-name=rl100_2d_epsilon.yaml \
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
  +run_validation=True \
  env_num=1 \
  horizon="${horizon}" n_obs_steps="${n_obs_steps}" n_action_steps="${action_steps}" \
  chunk_as_single_action=True \
  dynamics.prediction_mode=full \
  only_bc=True \
  offline="${offline}" online=False eval=False \
  policy._target_=rl_100.policy.rl100_2d.RL1002D \
  feature_type=2D \
  use_agent_pos=True \
  policy.use_visual=True \
  policy.w_pc=False \
  policy.model=skipnet \
  policy.act=relu \
  policy.scheduler_type=ddim \
  policy.use_aug=False \
  encoder_type=resnet \
  encoders.resnet.rgb_model.weights="${rgb_weights}" \
  encoders.resnet.resize_shape="[224,224]" \
  use_vib=False \
  use_recon=False \
  policy.img_shape="[3,${image_size},${image_size}]" \
  task.shape_meta.obs.image.shape="[3,${image_size},${image_size}]" \
  dataloader.batch_size="${batch_size}" \
  val_dataloader.batch_size="${batch_size}" \
  training.num_epochs=600 \
  training.num_critic_epochs=600 \
  dynamics_type=diffusion \
  dynamics.dynamics_max_epochs=350 \
  unio4.bppo_steps=6000 \
  unio4.idql_eval=False \
  unio4.use_ema_eval=False \
  task.env_runner=null \
  "${dataset_overrides[@]}"
