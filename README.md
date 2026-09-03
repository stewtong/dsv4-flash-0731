<!-- register: public-github-repo | reader: an ml infra engineer standing this up on one 8x b200 node | consumed: reproduce first, then inspect the evidence -->

# DeepSeek-V4-Flash-0731 on B200

This repository publishes a text-only serving recipe for DeepSeek-V4-Flash-0731 on one dedicated 8x B200 node. The validated topology is 1-way tensor parallelism, 8-way data-parallel attention, and 8-way expert parallelism (TP1, DP8, EP8). It uses vLLM's multi-port external load-balancing mode, FP8 KV cache, an FP4 indexer cache, and DSpark speculative decoding with five draft tokens and greedy draft sampling.

The matched benchmark found that DSpark k=5 reduced pooled warm median (p50) normalized 800-token wall time by **55.1% at 128K** and **52.2% at 256K** versus speculative decoding disabled. All six session-level mean comparisons favored k=5. The result covers serial requests with session affinity. Aggregate throughput and model quality remain unmeasured.

This is an individual engineering contribution, not an official statement from any vendor.

## Reproduce the endpoint

Install Docker with NVIDIA GPU support, curl, Python 3, and nginx on a dedicated 8x B200 node. Download the exact model revision and write the revision marker required by the launcher:

```bash
export MODEL_REV=7872f01b1d1fe23eabc4c98b48bffcef5a386062
export MODEL_PATH="/models/deepseek-ai/DeepSeek-V4-Flash-0731/$MODEL_REV"
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash-0731 \
  --revision "$MODEL_REV" --local-dir "$MODEL_PATH"
printf '%s\n' "$MODEL_REV" | sudo tee "$MODEL_PATH/.model-revision" >/dev/null
```

Create an env file containing `VLLM_API_KEY`, restrict it to its owner with `chmod 600`, then launch the validated k=5 profile:

```bash
export ENV_FILE=/path/to/env.list
export MODEL_PATH MODEL_REV
SPECULATION=k5 ./reproduce/run-replicas.sh
```

The launcher uses a digest-pinned vLLM image, validates the model revision, and waits for all eight DP rank health checks. It refuses to replace a container with the same name unless `REPLACE_EXISTING=1` is set. Review the exact container before using that flag.

Install [the affinity balancer](reproduce/nginx.conf) and [authenticated ingress](reproduce/nginx-ingress.conf). Supply secrets through the root-only include described in the ingress file, then run `nginx -t` in the target environment. Both references bind to loopback by default. External access requires an operator-owned network edge or a deliberate listener change with TLS, firewalling, and access controls.

## Verify the service and evidence

Check the data-parallel supervisor's aggregated health endpoint and one DP rank before putting traffic through the gateway:

```bash
curl --fail http://127.0.0.1:9256/health
curl --fail http://127.0.0.1:8100/health
```

Verify every published result bundle and checksum:

```bash
python3 reproduce/derive-results.py --check-all
shasum -a 256 -c results/SHA256SUMS
```

Linux systems can use `sha256sum -c results/SHA256SUMS`.

The verifier rejects malformed records, failed requests, missing or duplicate turns, mismatched paired inputs, undeclared configuration differences, incomplete prefix-affinity sessions, and summary drift.

Run a new serial measurement into a new path after the endpoint is ready:

```bash
export DSV4FLASH_API_KEY="<your-key>"
python3 reproduce/benchmark.py \
  --host http://127.0.0.1:8080 \
  --model deepseek-v4-flash-0731 \
  --leg k5-local --shapes 131072,262144 \
  --turns 6 --sessions 3 --max-tokens 1024 --prewarm \
  --out /tmp/dsv4-k5-local.jsonl
```

The harness calibrates prefixes through `/v1/messages/count_tokens`, starts timing before request dispatch, requires exact usage from the Server-Sent Events (SSE) stream, and returns nonzero on any calibration, prewarm, HTTP, parsing, or usage failure. Compare runs only when their manifests and request order match.

## Why this profile was selected

| Decision | Evidence | Selected setting |
| --- | --- | --- |
| Attention topology | No topology won every cold cell. DP8 led most short and medium cells and retained the strongest overall operating profile. | TP1, DP8 attention, EP8 experts |
| Prefix placement | Later turns stayed fast when a session returned to the DP rank holding its cached prefix. | Consistent session hashing with no inference retry |
| Draft length | k=5 preserved most of k=7's decode gain and behaved better under concurrent long-context load. | DSpark with five draft tokens and greedy draft sampling |
| Maximum batched tokens | Raising `--max-num-batched-tokens` from 8,192 to 16,384 cut KV capacity by 32.6% and produced mixed latency changes. | 8,192 |
| Memory headroom | The final k=5 setting recorded 17,076,237 KV tokens per rank. | GPU memory utilization 0.85 |

The full decision record is in [BENCHMARKS.md](BENCHMARKS.md).

## Matched DSpark result

The matched run held image, model revision, TP1 with DP8 and EP8 topology, request fixture, session order, `--max-num-batched-tokens`, `--max-num-seqs`, and GPU memory utilization constant. The control disabled speculative decoding; the candidate used DSpark k=5. The k=5 draft head reduced KV capacity from 17,837,368 to 17,076,237 tokens per rank.

Each configuration has three six-turn sessions at each context. Turn 1 is cold. The warm aggregate pools turns 2 through 6, producing 15 warm observations per configuration and context. Turns within a session reuse the same cache and are correlated, so the pooled values are descriptive rather than independent session-level estimates.

| Context | Speculative decoding disabled, normalized 800-token wall p50 | k=5 normalized 800-token wall p50 | Delta | Speculative decoding disabled, decode rate p50 | k=5 decode rate p50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128K | 8.71 s | 3.91 s | **-55.1%** | 97.16 tok/s | 227.93 tok/s |
| 256K | 8.53 s | 4.07 s | **-52.2%** | 100.87 tok/s | 236.43 tok/s |

Normalized 800-token wall time is calculated per request, then aggregated:

```text
ttfvt_seconds + 800 / decode_tokens_per_second
```

The configurations produced different output lengths at temperature 0. Normalized wall time provides the equal-output-length comparison.

Median (p50) time to first visible token (TTFVT) changed from 0.4160 to 0.4100 seconds at 128K and from 0.6485 to 0.6827 seconds at 256K. This measurement stops at the first visible text delta, which can differ from standard time to first token (TTFT) for reasoning models. Two warm control turns in the first two 128K sessions took 12.6058 and 10.1781 seconds to produce visible text. No k=5 warm turn exceeded three seconds. All 72 requests returned HTTP 200 with zero recorded errors.

The earlier 52.4% and 54.8% figures came from an August 21 run that combined different GPU memory utilization settings and aggregated percentile components before normalization. They remain historical context. The matched run provides the current comparison.

## Prefix locality

Each vLLM DP rank has an independent KV cache. A later turn routed to another rank must prefill the shared prefix again. The consistent-hash gateway kept every turn of all six measured sessions on one DP rank.

| Path | 128K-class late-turn median TTFVT | 256K-class late-turn median TTFVT |
| --- | ---: | ---: |
| Direct rank pin, serial | 0.467 to 0.519 s | 0.972 to 1.037 s |
| Direct rank pin, four concurrent sessions | 0.934 to 1.189 s | 2.162 to 2.793 s |
| Consistent-hash gateway, serial | 0.493 to 0.570 s | 1.005 to 1.085 s |

This prefix-affinity campaign used speculative decoding disabled and GPU memory utilization 0.95, so its absolute timings should not be combined with the matched DSpark result. It verifies the routing behavior required by the validated profile.

The prefix fixture's observed prompt counts were 222,099 for the 128K-class label and 446,049 for the 256K-class label. The context labels name campaign fixtures rather than exact token counts.

## Gateway contract

The ingress authenticates bearer keys, clears any external rank override, enforces eight in-flight connections per authenticated user, and allows:

- `/v1/chat/completions`
- `/v1/completions`
- `/v1/models`
- `/v1/messages`
- `/v1/messages/count_tokens`

`/health` is an unauthenticated liveness route. Other routes return 404. The OpenAI Responses API is outside the reference.

The external load balancer chooses `X-Session-Id`, then Claude Code session or agent identity, then the authenticated user label. Traffic without an identity uses round robin. Rank and session headers are stripped before the request reaches vLLM. Both proxies disable buffering and upstream retries, so a dispatched inference POST is never replayed on another DP rank.

The model's maximum sequence length is 1,048,576 tokens, counting prompt and output tokens together. The public gateway is text-only and omits the private deployment's media normalization component.

## Client examples

OpenAI-compatible chat:

```bash
export DSV4FLASH_API_KEY="<your-key>"
export DSV4FLASH_OPENAI_BASE_URL="https://<your-endpoint-host>/v1"

curl "$DSV4FLASH_OPENAI_BASE_URL/chat/completions" \
  -H "Authorization: Bearer $DSV4FLASH_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash-0731","messages":[{"role":"user","content":"hello"}],"max_tokens":64}'
```

Claude Code 2.1.229 was the recorded client build. Keep `ANTHROPIC_BASE_URL` at the bare host because the client appends `/v1/messages`:

```bash
export DSV4FLASH_API_KEY="<your-key>"
export DSV4FLASH_HOST="https://<your-endpoint-host>"

env \
  ANTHROPIC_BASE_URL="$DSV4FLASH_HOST" \
  ANTHROPIC_AUTH_TOKEN="$DSV4FLASH_API_KEY" \
  ANTHROPIC_DEFAULT_OPUS_MODEL=deepseek-v4-flash-0731 \
  ANTHROPIC_DEFAULT_SONNET_MODEL=deepseek-v4-flash-0731 \
  ANTHROPIC_DEFAULT_HAIKU_MODEL=deepseek-v4-flash-0731 \
  CLAUDE_CODE_MAX_CONTEXT_TOKENS=1000000 \
  CLAUDE_CODE_MAX_OUTPUT_TOKENS=32768 \
  API_TIMEOUT_MS=3000000 \
  claude
```

## Failure behavior and data handling

- Missing or incorrect bearer credentials return 401 at the ingress.
- Routes outside the allowlist return 404.
- HTTP 413 means a deployed hop still has a body-size cap. The reference nginx files remove their caps.
- HTTP 429 is the ingress policy of eight in-flight connections per authenticated user. It does not report vLLM capacity.
- Live nginx errors used HTML. Anthropic-compatible JSON error envelopes were unverified.
- The ingress can read request and response content. Its access log excludes bodies and credentials, but remote addresses and user labels still require an appropriate retention policy.

The [August 30 deployment verification](results/verification-2026-08-30.md) records runtime versions, the Anthropic Messages matrix, and limited failure-path tests. It adds no performance measurement. Do not send protected material to a deployment whose data-handling policy you have not verified.

## Measured limits

- The matched result covers serial, warm requests with session affinity at two long-context shapes.
- Concurrency above 16, open-loop arrivals, decode-heavy workloads, accuracy, and model quality were not measured.
- Prefix affinity can skew load when a few sessions dominate.
- Clean recreation from the pinned image remains unverified.
- Model weights and upstream software retain their own licenses. This repository's scripts and prose are MIT licensed.

The [method](reproduce/METHOD.md), [environment record](results/environment.md), and [claim ledger](CLAIM-LEDGER.md) define the calculation, campaign boundaries, and source for every material claim.
