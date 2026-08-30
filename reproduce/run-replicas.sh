#!/bin/sh
# Launch the validated DP8 + EP8 DeepSeek-V4-Flash-0731 serving profile: one
# API server per DP rank (each on its own loopback port), behind a session-
# affinity reverse proxy (see nginx.conf).
#
# This covers the two engine configurations in results/serving-envelope.json:
# SPECULATION=off and SPECULATION=k5. It is intentionally minimal and
# secret-free. Replace the placeholders below with your own bindings and
# credentials. The measurement assumed a dedicated 8x B200 node.

set -eu

# ---- placeholders: set in your environment, or edit inline ----
CONTAINER_NAME="${CONTAINER_NAME:-vllm-dsv4flash}"
IMAGE="${IMAGE:-vllm/vllm-openai@sha256:e1c1ff1af9a15921bfa11d1d95047258c1797392cdbfa296e7639da446b23f97}"
MODEL_REV="${MODEL_REV:-7872f01b1d1fe23eabc4c98b48bffcef5a386062}"
MODEL_PATH="${MODEL_PATH:-/models/deepseek-ai/DeepSeek-V4-Flash-0731/$MODEL_REV}"
# Read the model weights for this revision into $MODEL_PATH before launch.
# Example: huggingface-cli download deepseek-ai/DeepSeek-V4-Flash-0731 \
#   --revision "$MODEL_REV" --local-dir "$MODEL_PATH"
ENV_FILE="${ENV_FILE:-/path/to/env.list}"   # must set VLLM_API_KEY=<your value>
SHM_SIZE="${SHM_SIZE:-34359738368}"          # 32 GiB host shared memory
READY_TIMEOUT="${READY_TIMEOUT:-900}"
REPLACE_EXISTING="${REPLACE_EXISTING:-0}"
SPECULATION="${SPECULATION:-k5}"

for command_name in sudo docker curl grep tr date; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Required command is unavailable: $command_name" >&2
    exit 1
  fi
done
case "$READY_TIMEOUT" in
  ""|*[!0-9]*)
    echo "READY_TIMEOUT must be a positive integer: $READY_TIMEOUT" >&2
    exit 1
    ;;
esac
if [ "$READY_TIMEOUT" -le 0 ]; then
  echo "READY_TIMEOUT must be positive: $READY_TIMEOUT" >&2
  exit 1
fi
if [ ! -r "$ENV_FILE" ]; then
  echo "ENV_FILE is not readable: $ENV_FILE" >&2
  exit 1
fi
if ! grep -q '^VLLM_API_KEY=.' "$ENV_FILE"; then
  echo "ENV_FILE must contain a nonempty VLLM_API_KEY entry." >&2
  exit 1
fi
if [ ! -d "$MODEL_PATH" ]; then
  echo "MODEL_PATH is not a directory: $MODEL_PATH" >&2
  exit 1
fi
if [ ! -r "$MODEL_PATH/.model-revision" ] || [ "$(tr -d '\r\n' < "$MODEL_PATH/.model-revision")" != "$MODEL_REV" ]; then
  echo "MODEL_PATH does not carry the required .model-revision marker: $MODEL_REV" >&2
  exit 1
fi
case "$IMAGE" in
  *@sha256:*) ;;
  *)
    echo "IMAGE must be digest-pinned: $IMAGE" >&2
    exit 1
    ;;
esac

set -- \
  "$IMAGE" \
  "$MODEL_PATH" \
  --served-model-name deepseek-v4-flash-0731 \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --enable-expert-parallel \
  --data-parallel-size 8 \
  --data-parallel-size-local 8 \
  --data-parallel-multi-port-external-lb \
  --port 8100 \
  --attention_config.use_fp4_indexer_cache=True \
  --moe-backend deep_gemm_mega_moe \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 256 \
  --gpu-memory-utilization 0.85

case "$SPECULATION" in
  off) ;;
  k5)
    set -- "$@" --speculative-config \
      '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"greedy"}'
    ;;
  *)
    echo "SPECULATION must be off or k5: $SPECULATION" >&2
    exit 1
    ;;
esac

if sudo docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  if [ "$REPLACE_EXISTING" != "1" ]; then
    echo "Container already exists: $CONTAINER_NAME" >&2
    echo "Set REPLACE_EXISTING=1 to remove that exact container before launch." >&2
    exit 1
  fi
  echo "Removing exact existing container: $CONTAINER_NAME"
  sudo docker rm -f "$CONTAINER_NAME"
fi

sudo docker run -d --name "$CONTAINER_NAME" \
  --restart unless-stopped \
  --gpus all --ipc host --shm-size "$SHM_SIZE" \
  --ulimit nofile=1048576:1048576 \
  --env-file "$ENV_FILE" \
  -v "$MODEL_PATH":"$MODEL_PATH":ro \
  -p 127.0.0.1:8100-8107:8100-8107 -p 127.0.0.1:9256:9256 \
  "$@"

# Readiness: gate on HTTP 200 across every rank port (8100-8107), not on the
# "Application startup complete" line, which fires well before the engines are
# ready to serve. The controllers are served on 9256.
echo "Launched $CONTAINER_NAME; waiting for rank health checks."
started_at="$(date +%s)"
while :; do
  all_ready=1
  port=8100
  while [ "$port" -le 8107 ]; do
    if ! curl --fail --silent --show-error --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      all_ready=0
      break
    fi
    port=$((port + 1))
  done
  if [ "$all_ready" -eq 1 ]; then
    echo "Ready: all rank health checks returned HTTP 200."
    exit 0
  fi
  now="$(date +%s)"
  if [ $((now - started_at)) -ge "$READY_TIMEOUT" ]; then
    echo "Readiness timed out after ${READY_TIMEOUT}s; container remains available for diagnosis." >&2
    exit 1
  fi
  sleep 5
done
