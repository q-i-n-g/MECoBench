#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
START_SCREEN_SCRIPT="${REPO_ROOT}/src/start_screen.sh"
START_UNITY_SCRIPT="${REPO_ROOT}/src/start_unity_multi_8gpu.sh"

SCREEN_XVFB="${SCREEN_XVFB:-mecobench-xvfb}"
SCREEN_UNITY="${SCREEN_UNITY:-mecobench-unity}"
SLEEP_BETWEEN_SEC="${SLEEP_BETWEEN_SEC:-300}"
TASKS="${VWAH_TASKS:-parallel,sequential}"
USE_FULL_TASKS="${VWAH_USE_FULL_TASKS:-0}"

stop_infra_screens() {
  screen -S "${SCREEN_XVFB}" -X quit 2>/dev/null || true
  screen -S "${SCREEN_UNITY}" -X quit 2>/dev/null || true
}

start_infra_screens() {
  stop_infra_screens
  chmod +x "${START_SCREEN_SCRIPT}" "${START_UNITY_SCRIPT}" 2>/dev/null || true

  screen -dmS "${SCREEN_XVFB}" bash -lc "exec bash '${START_SCREEN_SCRIPT}'"
  screen -dmS "${SCREEN_UNITY}" bash -lc "cd '${REPO_ROOT}' && exec bash '${START_UNITY_SCRIPT}'"

  echo "[infra] started screen sessions: ${SCREEN_XVFB} (Xvfb), ${SCREEN_UNITY} (Unity)"
  echo "[infra] waiting ${SLEEP_BETWEEN_SEC}s before running evaluation"
  sleep "${SLEEP_BETWEEN_SEC}"
}

after_experiment_cooldown() {
  stop_infra_screens
  echo "[infra] stopped screen sessions; waiting ${SLEEP_BETWEEN_SEC}s"
  sleep "${SLEEP_BETWEEN_SEC}"
}

run_task() {
  local task_name="$1"
  local task_file="data/examples/${task_name}_5.json"
  if [[ "${USE_FULL_TASKS}" == "1" || "${USE_FULL_TASKS}" == "true" ]]; then
    task_file="data/task/${task_name}.json"
  fi

  if [[ ! -f "${REPO_ROOT}/${task_file}" ]]; then
    echo "error: task file not found: ${task_file}" >&2
    return 1
  fi

  echo ""
  echo "========================================"
  echo "  MECoBench ${task_name}"
  echo "========================================"

  local task_id_list="${VWAH_TASK_ID_LIST:-}"
  if [[ -z "${task_id_list}" ]]; then
    task_id_list="$(python3 - "${REPO_ROOT}/${task_file}" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    tasks = json.load(f)
print(",".join(str(i) for i in range(len(tasks))))
PY
)"
  fi

  (
    cd "${REPO_ROOT}"
    env \
      VWAH_TASK_FILE="${task_file}" \
      VWAH_NO_COMMUNICATION="${VWAH_NO_COMMUNICATION:-false}" \
      VWAH_COMMUNICATION_MODE="${VWAH_COMMUNICATION_MODE:-leader_worker}" \
      VWAH_NUM_AGENTS="${VWAH_NUM_AGENTS:-2}" \
      VWAH_STEPS_THRESHOLD="${VWAH_STEPS_THRESHOLD:-40}" \
      UNITY_NUM_SIMULATORS="${UNITY_NUM_SIMULATORS:-20}" \
      VWAH_GOAL_MODE="${VWAH_GOAL_MODE:-goal_with_location_and_desc}" \
      VWAH_LEADER_COMM_MODE="${VWAH_LEADER_COMM_MODE:-text_only}" \
      VLM_MAIN_MODEL="${VLM_MAIN_MODEL:-qwen-8b-vl}" \
      VLM_LEADER_MODEL="${VLM_LEADER_MODEL:-qwen-32b-vl}" \
      VLM_WORKER_MODEL="${VLM_WORKER_MODEL:-}" \
      VLM_EMBEDDING_MODEL="${VLM_EMBEDDING_MODEL:-text-embedding-3-large}" \
      VLM_RESOLVE_MODEL="${VLM_RESOLVE_MODEL:-gpt-5-mini-2}" \
      VLM_HISTORICAL_DIALOGUE_ROUNDS="${VLM_HISTORICAL_DIALOGUE_ROUNDS:-10}" \
      UNITY_BASE_PORT="${UNITY_BASE_PORT:-8001}" \
      VWAH_TASK_ID_LIST="${task_id_list}" \
      VWAH_OUTPUT_ROOT="${VWAH_OUTPUT_ROOT:-outputs/${task_name}}" \
      python eval/main.py
  )
}

main() {
  start_infra_screens
  local rc=0
  IFS=',' read -r -a task_names <<< "${TASKS}"
  for task_name in "${task_names[@]}"; do
    run_task "${task_name}" || rc=$?
    if [[ "${rc}" -ne 0 ]]; then
      break
    fi
  done
  after_experiment_cooldown
  return "${rc}"
}

main "$@"
