#!/usr/bin/env bash
# Append one health snapshot every 30 minutes until all three DP3 variants finish.
set -u

repo_root="$(cd "$(dirname "$0")/../../../.." && pwd)"
project_root="${repo_root}/RL-100"
queue_dir="${project_root}/data/outputs/piper_pick_and_place_augmented_chunk4_control_clean_dp3_compact_queue"
log_file="${queue_dir}/monitor_30min.log"

mkdir -p "${queue_dir}"
while [[ ! -e "${queue_dir}/ALL_COMPLETE" && ! -e "${queue_dir}/FAILED" && ! -e "${queue_dir}/MEDIUM_FAILED" ]]; do
    {
        date -Is
        printf 'main_train_process='
        pgrep -aof '^/.*python train.py .*piper_pick_and_place_augmented' || echo 'none'
        df -h "${project_root}" | tail -1
        for marker in FAILED COMPLETE MEDIUM_FAILED ALL_COMPLETE; do
            [[ -e "${queue_dir}/${marker}" ]] && printf 'marker=%s\n' "${marker}"
        done
        for run in "${project_root}"/data/outputs/piper_pick_and_place_augmented_chunk4_control_clean_dp3_{small96,tiny64,medium}_episode10_bs64_epoch4000_seed42; do
            [[ -d "${run}" ]] || continue
            printf 'run=%s\n' "${run##*/}"
            find "${run}/checkpoints" -maxdepth 1 -type f -printf '%TY-%Tm-%TdT%TH:%TM:%TS %f\n' 2>/dev/null | sort | tail -3
        done
        printf '%s\n' 'launcher_tail:'
        tr '\r' '\n' < "${queue_dir}/launcher.log" 2>/dev/null | tail -20 || true
        if [[ -s "${queue_dir}/medium_supervisor.log" ]]; then
            printf '%s\n' 'medium_supervisor_tail:'
            tr '\r' '\n' < "${queue_dir}/medium_supervisor.log" 2>/dev/null | tail -20 || true
        fi
    } >> "${log_file}" 2>&1
    sleep 1800
done
{
    date -Is
    [[ -e "${queue_dir}/ALL_COMPLETE" ]] && echo 'queue=all_complete'
    [[ -e "${queue_dir}/FAILED" ]] && echo 'queue=failed'
    [[ -e "${queue_dir}/MEDIUM_FAILED" ]] && echo 'queue=medium_failed'
} >> "${log_file}"
