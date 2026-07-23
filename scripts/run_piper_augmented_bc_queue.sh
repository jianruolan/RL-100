#!/usr/bin/env bash
# 23:30之后等待GPU空闲，依次运行扩增数据的3D BC和RGB-D 2D BC。
set -uo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
python_bin="${PYTHON_BIN:-/home/mtarch/miniforge3/envs/rl100_test/bin/python}"
log_dir="${repo_root}/RL-100/data/outputs/piper_augmented_bc_queue"
mkdir -p "${log_dir}"
queue_log="${log_dir}/queue.log"
target_epoch="$(date -d "$(date +%F) 23:30:00" +%s)"
gpu_id="${GPU_ID:-0}"
gpu_poll_seconds="${GPU_POLL_SECONDS:-60}"

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
    # GNOME等显示服务偶尔会被列为compute app，但实际显存为0 MiB。
    # 只将显存大于0 MiB（或无法解析显存）的进程视作真正GPU占用。
    active="$(
      nvidia-smi -i "${gpu_id}" \
        --query-compute-apps=pid,process_name,used_gpu_memory \
        --format=csv,noheader 2>/dev/null |
      awk -F',' '
        {
          memory=$3
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", memory)
          gsub(/[[:space:]]*MiB$/, "", memory)
          if (memory ~ /^[0-9]+$/) {
            if (memory + 0 > 0) print $0
          } else {
            print $0
          }
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
  local run_log="${log_dir}/${name}.log"
  local resume=False
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
  "PYTHON_BIN=${python_bin} GPU_ID=${gpu_id} RUN_DIR=data/outputs/piper_pick_and_place_augmented_chunk4_seed42 bash scripts/Diffusion/Offline/3D/train_policy_piper_pick_and_place_augmented.sh rl100 piper_pick_and_place_augmented chunk4-bc 42"

run_with_resume "2d_rgbd_resnet18_chunk4_bc" \
  "PYTHON_BIN=${python_bin} GPU_ID=${gpu_id} BATCH_SIZE=32 RUN_DIR=data/outputs/piper_pick_and_place_augmented_rgbd_resnet18_chunk4_seed42 bash scripts/Diffusion/Offline/2D/train_policy_piper_pick_and_place_rgbd.sh rl100 piper_pick_and_place_augmented_rgbd rgbd-resnet18-chunk4-bc 42"

log "两个BC实验均已完成"
