set -euo pipefail

export HF_HOME=/workspace/cache

RUN_CMD="python run.py \
  --model_name Qwen/Qwen3-4B \
  --method latent_mas \
  --task gsm8k \
  --max_new_tokens 512 \
  --prompt sequential \
  --max_samples 200 \
  --generate_bs 1 \
  --latent_steps 64 \
  --latent_space_realign \
  --custom_prompt_file config/test.json \
  --resume"

LOG_FILE="output.log"
DONE_PATTERN="[Done]"

is_done() {
  grep -qF "${DONE_PATTERN}" "${LOG_FILE}" 2>/dev/null
}

attempt=0

while true; do
  if is_done; then
    echo "[watch_run] Task already completed, exiting watch_run."
    break
  fi

  attempt=$((attempt + 1))
  
  echo "[watch_run] The ${attempt} attempt."
  CMD="${RUN_CMD}"

  echo "" >> "${LOG_FILE}"
  echo "========== [watch_run] attempt=${attempt} $(date '+%Y-%m-%d %H:%M:%S') ==========" >> "${LOG_FILE}"

  set +e
  ${CMD} >> "${LOG_FILE}" 2>&1
  EXIT_CODE=$?
  set -e

  if is_done; then
    echo "[watch_run] Task completed (process exited normally), exiting watch_run."
    break
  fi

  if [[ ${EXIT_CODE} -eq 0 ]]; then
    echo "[watch_run] Process exited normally (exit code 0), but no completion flag detected, restarting with --resume..."
  else
    echo "[watch_run] Process exited abnormally (exit ${EXIT_CODE}), restarting with --resume..."
  fi

  sleep 5
done