#!/usr/bin/env bash
# 23:30之后等待GPU空闲，依次运行扩增数据的3D BC和RGB-D 2D BC。
set -uo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
python_bin="${PYTHON_BIN:-/home/mtarch/miniforge3/envs/rl100_test/bin/python}"
log_dir="${repo_root}/RL-100/data/outputs/piper_augmented_bc_queue"
mkdir -p "${log_dir}"
queue_log="${log_dir}/queue.log"
pid_file="${log_dir}/queue.pid"
target_epoch="$(date -d "$(date +%F) 23:30:00" +%s)"
gpu_id="${GPU_ID:-0}"
gpu_poll_seconds="${GPU_POLL_SECONDS:-60}"

# 防止监控器重启或人工误操作后同时存在两份队列、重复占用GPU。
exec 9>"${log_dir}/queue.lock"
if ! flock -n 9; then
  echo "[$(date '+%F %T')] 已有训练队列持有锁，当前实例退出" | tee -a "${queue_log}"
  exit 1
fi
echo "$$" > "${pid_file}"
trap 'rm -f "${pid_file}"' EXIT

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${queue_log}"
}

wait_until_allowed() {
  while [ "$(date +%s)" -lt "${target_epoch}" ]; do
    remaining=$((target_epoch - $(date +%s)))
    interval=1800
    if [ "${remaining}" -lt "${interval}" ]; then
      interval="${remaining}"
    fi
    log "23:30前不启动训练，${interval}秒后再检查"
    sleep "${interval}"
  done
}

wait_for_gpu() {
  while true; do
    if ! nvidia-smi -i "${gpu_id}" >/dev/null 2>&1; then
      log "nvidia-smi暂不可用，${gpu_poll_seconds}秒后重试"
      sleep "${gpu_poll_seconds}"
      continue
    fi
    # Thor使用统一内存，活跃CUDA进程也可能被nvidia-smi报告为0 MiB；因此
    # 不能按显存数值过滤。只排除已知的桌面服务，任何其他compute app都
    # 一律视为GPU占用，避免与别人的Python训练并发。
    active="$(
      nvidia-smi -i "${gpu_id}" \
        --query-compute-apps=pid,process_name,used_gpu_memory \
        --format=csv,noheader 2>/dev/null |
      awk -F',' '
        {
          process=$2
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", process)
          if (process !~ /(gnome-remote-desktop-daemon|rustdesk)$/) print $0
        }
      '
    )"
    if [ -z "${active}" ]; then
      log "GPU ${gpu_id}空闲"
      return 0
    fi
    log "GPU ${gpu_id}仍有计算进程=${active//$'\n'/;}，${gpu_poll_seconds}秒后重试"
    sleep "${gpu_poll_seconds}"
  done
}

run_with_resume() {
  local name="$1"
  local command="$2"
  local checkpoint_path="${3:-}"
  local run_log="${log_dir}/${name}.log"
  local resume=False
  if [ -n "${checkpoint_path}" ] && [ -f "${checkpoint_path}" ]; then
    resume=True
    log "检测到${name} checkpoint=${checkpoint_path}，将从断点恢复"
  fi
  while true; do
    wait_for_gpu
    log "启动${name}，RESUME=${resume}"
    set +e
    RESUME="${resume}" bash -c "${command}" 2>&1 | tee -a "${run_log}"
    status=${PIPESTATUS[0]}
    set -e
    if [ "${status}" -eq 0 ]; then
      log "${name}正常完成"
      return 0
    fi
    log "${name}异常退出(status=${status})，保留日志并在60秒后从checkpoint恢复"
    resume=True
    sleep 60
  done
}

wait_until_allowed
cd "${repo_root}"

run_with_resume "3d_chunk4_bc" \
  "PYTHON_BIN=${python_bin} GPU_ID=${gpu_id} RUN_DIR=data/outputs/piper_pick_and_place_augmented_chunk4_seed42 bash scripts/Diffusion/Offline/3D/train_policy_piper_pick_and_place_augmented.sh rl100 piper_pick_and_place_augmented chunk4-bc 42" \
  "${repo_root}/RL-100/data/outputs/piper_pick_and_place_augmented_chunk4_seed42/checkpoints/latest.ckpt"

run_with_resume "2d_rgbd_resnet18_chunk4_bc" \
  "PYTHON_BIN=${python_bin} GPU_ID=${gpu_id} BATCH_SIZE=32 RUN_DIR=data/outputs/piper_pick_and_place_augmented_rgbd_resnet18_chunk4_seed42 bash scripts/Diffusion/Offline/2D/train_policy_piper_pick_and_place_rgbd.sh rl100 piper_pick_and_place_augmented_rgbd rgbd-resnet18-chunk4-bc 42" \
  "${repo_root}/RL-100/data/outputs/piper_pick_and_place_augmented_rgbd_resnet18_chunk4_seed42/checkpoints/latest.ckpt"

log "两个BC实验均已完成"
