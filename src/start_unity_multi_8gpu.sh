#!/usr/bin/env bash
set -euo pipefail

export DISPLAY="${DISPLAY:-:1}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"

SIMULATOR_BIN="${MECOBENCH_SIMULATOR_BIN:-${UNITY_BIN:-}}"
BASE_PORT="${UNITY_BASE_PORT:-8001}"
NUM_PER_GPU="${UNITY_NUM_PER_GPU:-10}"
GPU_LIST="${UNITY_GPUS:-0,1}"

if [[ -z "${SIMULATOR_BIN}" ]]; then
  echo "error: MECOBENCH_SIMULATOR_BIN is not set. Example: export MECOBENCH_SIMULATOR_BIN=/path/to/linux.x86_64" >&2
  exit 1
fi

if [[ ! -x "${SIMULATOR_BIN}" ]]; then
  echo "error: MECoBench Simulator binary not found or not executable: ${SIMULATOR_BIN}" >&2
  exit 1
fi

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"

i=0
for gpu in "${GPUS[@]}"; do
  for slot in $(seq 0 $((NUM_PER_GPU - 1))); do
    port=$((BASE_PORT + i * NUM_PER_GPU + slot))
    echo "starting force-device-index=${gpu} port=${port}"
    "${SIMULATOR_BIN}" -batchmode -http-port="${port}" -force-device-index "${gpu}" &
  done
  i=$((i + 1))
done

echo "launched $((${#GPUS[@]} * NUM_PER_GPU)) simulators starting at port ${BASE_PORT}"
wait
