#!/usr/bin/env bash
# Post-launch gates for the DP8 + DSpark k=5 serve.
# Exits non-zero on any failed gate; never reports success it did not observe.
set -uo pipefail

CONTAINER="${CONTAINER:-v4flash-serve-dpm}"
EXPECTED_KV="17076237"
KV_TOLERANCE_PCT="0.1"
RANKS=(8100 8101 8102 8103 8104 8105 8106 8107)
: "${VLLM_API_KEY:?set VLLM_API_KEY}"

fail=0

# Gate 1 - per-rank health.
# An accepted TCP connection is NOT readiness: this image accepts then resets
# for ~4 minutes of pre-startup. Gate on HTTP 200, not on an open socket.
echo "== Gate 1: per-rank health =="
for p in "${RANKS[@]}"; do
  code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://127.0.0.1:${p}/health" || echo 000)"
  if [ "$code" = "200" ]; then
    echo "  rank ${p}: 200"
  else
    echo "  rank ${p}: ${code} NOT READY"; fail=1
  fi
done

# Gate 2 - KV invariant.
# Guards against wrong image, wrong weights, or wrong flags, which move KV by
# orders of magnitude. Sub-0.1% deltas are CUDA-graph profiling variance.
echo "== Gate 2: KV invariant (expect 8x ${EXPECTED_KV}) =="
# Portable array fill (mapfile is bash 4+; macOS ships bash 3.2).
kv=()
while IFS= read -r _line; do kv+=("$_line"); done < <(docker logs "${CONTAINER}" 2>&1 \
  | grep -oE 'GPU KV cache size: [0-9,]+ tokens' \
  | grep -oE '[0-9,]+' | tr -d ',')
if [ "${#kv[@]}" -ne 8 ]; then
  echo "  found ${#kv[@]} KV lines, expected 8 (engines may still be starting)"; fail=1
else
  for v in "${kv[@]}"; do
    delta=$(awk -v a="$v" -v b="$EXPECTED_KV" 'BEGIN{printf "%.4f", (a>b?a-b:b-a)/b*100}')
    within=$(awk -v d="$delta" -v t="$KV_TOLERANCE_PCT" 'BEGIN{print (d<=t)?1:0}')
    if [ "$within" = "1" ]; then
      echo "  ${v} ok (${delta}% delta)"
    else
      echo "  ${v} OUT OF TOLERANCE (${delta}% delta) - stop and diagnose"; fail=1
    fi
  done
fi

# Gate 3 - a real completion through rank 0.
echo "== Gate 3: live completion =="
resp="$(curl -s --max-time 120 http://127.0.0.1:8100/v1/chat/completions \
  -H "Authorization: Bearer ${VLLM_API_KEY}" -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash-0731","messages":[{"role":"user","content":"Reply with the single word: ready"}],"max_tokens":16,"temperature":0}')"
if echo "$resp" | grep -qi 'ready'; then
  echo "  completion ok"
else
  echo "  completion FAILED"; echo "$resp" | head -c 400; fail=1
fi

echo
if [ "$fail" -eq 0 ]; then echo "ALL GATES PASSED"; else echo "GATES FAILED"; fi
exit "$fail"
