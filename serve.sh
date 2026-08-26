#!/usr/bin/env bash
# Launch DeepSeek-V4-Flash-0731 on 8x B200: DP8 + EP8 + DSpark k=5.
# See README.md for why each flag is set the way it is.
set -euo pipefail

MODEL_CACHE="${MODEL_CACHE:-/data/hf-cache}"
MODEL_REV="7872f01b1d1fe23eabc4c98b48bffcef5a386062"
MODEL_PATH="/root/.cache/huggingface/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/${MODEL_REV}"
IMAGE="${IMAGE:-vllm/vllm-openai:v0.25.0}"
IMAGE_DIGEST="sha256:e1c1ff1af9a15921bfa11d1d95047258c1797392cdbfa296e7639da446b23f97"
CONTAINER="${CONTAINER:-v4flash-serve-dpm}"

# VLLM_API_KEY is read from the environment and never written to disk by this script.
# Supply it from your own secret store, e.g. export VLLM_API_KEY="$(your-secret-tool get ...)"
: "${VLLM_API_KEY:?set VLLM_API_KEY from your secret store}"

test -d "${MODEL_CACHE}" || { echo "model cache ${MODEL_CACHE} not found" >&2; exit 1; }

# The v0.25.0 tag has been observed to drift. Warn if the local tag no longer
# matches the digest this recipe was validated against.
if ! docker images --digests vllm/vllm-openai | grep -q "${IMAGE_DIGEST%%:*}\|${IMAGE_DIGEST#sha256:}"; then
  echo "WARNING: validated digest ${IMAGE_DIGEST} not found locally." >&2
  echo "         Pull explicitly: docker pull vllm/vllm-openai@${IMAGE_DIGEST}" >&2
fi

docker rm -f "${CONTAINER}" 2>/dev/null || true

docker run -d --name "${CONTAINER}" --restart unless-stopped \
  --gpus all --ipc host --shm-size 34359738368 \
  --ulimit nofile=1048576:1048576 \
  -e VLLM_API_KEY="${VLLM_API_KEY}" \
  -v "${MODEL_CACHE}:/root/.cache/huggingface" \
  -p 127.0.0.1:8100-8107:8100-8107 -p 127.0.0.1:9256:9256 \
  "${IMAGE}" \
  "${MODEL_PATH}" \
  --served-model-name deepseek-v4-flash-0731 --trust-remote-code \
  --kv-cache-dtype fp8 --block-size 256 \
  --enable-expert-parallel --data-parallel-size 8 --data-parallel-size-local 8 \
  --data-parallel-multi-port-external-lb --port 8100 \
  --attention_config.use_fp4_indexer_cache=True \
  --moe-backend deep_gemm_mega_moe \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --max-num-batched-tokens 8192 --max-num-seqs 256 \
  --gpu-memory-utilization 0.85 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"greedy"}'

echo "Started ${CONTAINER}."
echo "Engine init takes several minutes. Run ./verify.sh to gate on readiness."
