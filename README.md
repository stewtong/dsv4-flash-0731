<!-- register: public-github-repo | reader: an ml infra engineer standing this up on their own 8x b200 | consumed: read once top-to-bottom, then reproduce -->

# DeepSeek-V4-Flash-0731 on B200

This repository publishes a text-only serving recipe for
DeepSeek-V4-Flash-0731 on one dedicated 8x B200 node. It includes the engine
launch profile, an authenticated session-affine nginx reference, a benchmark
harness, 72 per-request records, and a deterministic verifier.

This is an individual engineering contribution, not an official statement from
any vendor.

## Measured warm-turn operating point

The validated profile used DP8 attention, EP8 experts, DSpark speculative
decoding at k=5 greedy, `--max-num-batched-tokens 8192`,
`--max-num-seqs 256`, and `--gpu-memory-utilization 0.85`. Each DP rank served
one loopback port from 8100 through 8107. Requests were serial, temperature 0,
and pinned by session to the rank holding the prefix cache.

| Context label | First visible token, p50 | Decode, p50 |
| ---: | ---: | ---: |
| 128K | 0.390 s | 221.1 tok/s |
| 256K | 0.730 s | 240.8 tok/s |

The public records contain three sessions of six turns for each of four cells:
speculation off and k=5 at 128K and 256K. Turn 1 is the cold reference. Each
cell's warm aggregate pools turns 2 through 6 across the three sessions, for 15
warm observations per cell and 60 warm observations overall. A session is a
cache-placement cluster, not an independent scalar replicate.

Two k=5 warm turns per context paid a long first-visible-token stall. Their
timings are disclosed as `warm_stall_turns_s` in
`results/serving-envelope.json`. The headline reports p50 only and does not
describe tail latency.

First visible token is measured from request dispatch to the first visible text
delta. Decode rate is `output_tokens / (wall - ttfvt)` and uses the output-token
count reported by the server. It is a per-request decode-window rate, not node
throughput.

The validated profile records 17,076,237 KV tokens per rank. The exact value is
a retained startup invariant, not a fresh boot-log read in this repository.
Reconcile a materially different value before comparing results.

### Boundaries

- These are warm serial-request latency results, not aggregate throughput,
  saturation, accuracy, or model-quality results.
- Concurrency above 16, open-loop arrivals, and decode-heavy workloads were not
  measured.
- Affinity improves multi-turn latency by preserving prefix locality. It can
  skew load when a few sessions dominate.
- The reference gateway accepts text-only payloads. It does not include the
  private deployment's media-normalization component.

The [August 30 deployment verification](results/verification-2026-08-30.md)
records a later fixed-k5 restart, current runtime versions, the live Anthropic
Messages matrix, and bounded failure-path characterization. It does not add or
change any performance measurement.

## DSpark result

Against the same DP8 profile with speculation off, k=5 reduced the normalized
wall for an equal 800-token answer by **52.4% at 128K and 54.8% at 256K**.

Normalized wall is expressed in milliseconds:

```
1000 * ttfvt_seconds + 800 * (1000 / decode_tokens_per_second)
```

The two arms produced different output lengths at temperature 0, so their raw
wall times are not directly comparable. The normalized calculation uses each
arm's p50 first-visible-token time and p50 decode rate. Run the verifier below
to reproduce the result from `results/raw/`.

The retained source run also recorded a seven-prompt functional suite at 7 of 7
and six verbatim identity-echo checks at 6 of 6. Those artifacts are not
published here, so these are labeled recorded gates in `CLAIM-LEDGER.md`, not
publicly re-derivable results. They are smoke tests and make no accuracy claim.

## Why session affinity is required

This model uses multi-head latent attention with a single KV head. TP8 would
replicate that KV pool, while DP8 gives each rank an independent prefix cache.
The design rationale is discussed in
[vllm-project/vllm#51454](https://github.com/vllm-project/vllm/issues/51454).
That issue is design evidence, not performance evidence for this profile.

DP8 external-LB mode exposes one API server per rank. A later turn routed to a
different rank loses prefix locality and pays another prefill. The reference
balancer hashes the strongest available session identity, then uses the
authenticated user label as a fallback. Traffic without an identity uses round
robin.

## Reproduce the engine and benchmark

Install Docker with GPU support, curl, Python 3, and nginx on a dedicated 8x
B200 node. Download the exact model revision and record the marker required by
the launcher:

```bash
export MODEL_REV=7872f01b1d1fe23eabc4c98b48bffcef5a386062
export MODEL_PATH="/models/deepseek-ai/DeepSeek-V4-Flash-0731/$MODEL_REV"
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash-0731 \
  --revision "$MODEL_REV" --local-dir "$MODEL_PATH"
printf '%s\n' "$MODEL_REV" | sudo tee "$MODEL_PATH/.model-revision" >/dev/null
```

Create a restrictive env file, readable by the invoking operator, containing
`VLLM_API_KEY`, then launch:

```bash
export ENV_FILE=/path/to/env.list
export MODEL_PATH MODEL_REV
SPECULATION=off ./reproduce/run-replicas.sh
```

The script requires a digest-pinned image, refuses to replace an existing
same-named container unless `REPLACE_EXISTING=1`, and exits only after all eight
rank health checks return HTTP 200. Review the exact target before setting the
replacement flag.

Install and validate `reproduce/nginx.conf` and
`reproduce/nginx-ingress.conf` with secrets supplied through the root-only
include described in the ingress file. Run `nginx -t` in the target
environment. The public staging environment did not have nginx, so native
configuration validation is not claimed here.

Both public nginx files bind loopback by default. The benchmark below runs on
the serving node through the cleartext loopback ingress. External service
requires an operator-owned network edge or a deliberate listener change with
firewalling, TLS, and access controls.

Measure each configuration into a new file. Do not write into `results/raw/`:

```bash
export DSV4FLASH_API_KEY="<your-key>"

python3 reproduce/benchmark.py \
  --host http://127.0.0.1:8080 \
  --model deepseek-v4-flash-0731 \
  --leg specoff --shapes 131072,262144 \
  --turns 6 --sessions 3 --max-tokens 1024 --prewarm \
  --out /tmp/dsv4-specoff.jsonl

# Review the exact existing container, then relaunch the k=5 arm with
# REPLACE_EXISTING=1 SPECULATION=k5 ./reproduce/run-replicas.sh
python3 reproduce/benchmark.py \
  --host http://127.0.0.1:8080 \
  --model deepseek-v4-flash-0731 \
  --leg cellK5-mnbt8192 --shapes 131072,262144 \
  --turns 6 --sessions 3 --max-tokens 1024 --prewarm \
  --out /tmp/dsv4-k5.jsonl

python3 reproduce/derive-results.py \
  --input /tmp/dsv4-specoff.jsonl \
  --input /tmp/dsv4-k5.jsonl --json
```

The harness calibrates every synthetic prefix through
`/v1/messages/count_tokens`, starts timing before the HTTP request is sent,
requires exact input and output usage from the SSE stream, and returns nonzero
if calibration, prewarm, or any measured request fails. The verifier rejects
errors, missing sessions, missing turns, and duplicate turns.

Verify the bundled evidence without changing it:

```bash
python3 reproduce/derive-results.py \
  --check-envelope results/serving-envelope.json
shasum -a 256 -c results/SHA256SUMS
```

On Linux, use `sha256sum -c results/SHA256SUMS` for the manifest check.

## Gateway contract

The two nginx files are replaceable references with separate responsibilities:

- `nginx-ingress.conf` terminates TLS, authenticates bearer keys, enforces the
  exact route allowlist and per-key connection limit, supplies the authenticated
  user label as the affinity fallback, and clears any external
  `X-data-parallel-rank` header.
- `nginx.conf` selects one of the eight rank servers by trusted explicit rank or
  consistent session hash. It strips the rank and all session-identity headers
  before proxying to the engine.
- vLLM serves the OpenAI-compatible and Anthropic Messages routes on the eight
  loopback rank ports.

Both proxies disable buffering and upstream retries for inference requests. A
dispatched POST must not be replayed on another rank. Both set
`client_max_body_size 0` because nginx's default 1 MB cap can reject a large
long-context request with HTTP 413.

The authenticated API allowlist is `/v1/chat/completions`, `/v1/completions`,
`/v1/models`, `/v1/messages`, and `/v1/messages/count_tokens`. `/health` is a
separate unauthenticated liveness route. Other routes return 404. The OpenAI
Responses API is not included.

The ingress access log includes remote address, method, route, status, request
size, response size, and user label. It does not log bodies or credentials.
Treat remote addresses and user labels as operational metadata that still
requires an appropriate retention policy.

### Session and rank headers

The balancer chooses the strongest available identity in this order:

1. `X-Session-Id`
2. Claude Code session or agent identity
3. the authenticated user label supplied as `X-Session-Key`

The public ingress clears an external `X-data-parallel-rank`. A trusted
loopback caller may select ranks 0 through 7 explicitly. The balancer consumes
and strips that header because a per-rank API process sees local
`data-parallel-size 1` and can reject a forwarded rank 1 through 7 with HTTP
400.

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

Claude Code 2.1.229 was the recorded client build. Keep
`ANTHROPIC_BASE_URL` at the bare host because the client appends
`/v1/messages`:

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

The public reference supports text-only streaming Messages and token counting.
The [August 30 deployment verification](results/verification-2026-08-30.md)
records non-streaming Messages, version and beta headers, and exact live error
classes through the private deployment. Those checks do not certify the
sanitized nginx references or live timeout and mid-stream behavior. A
`thinking` request was accepted in the retained source run but did not emit a
thinking block; output was text-only. Later client or engine versions may
differ.

## Failure behavior and data handling

- Missing or incorrect bearer credentials return 401 at the ingress.
- Routes outside the allowlist return 404.
- HTTP 413 indicates that a body-size cap still exists on one of the deployed
  hops. The reference nginx files remove their caps.
- HTTP 429 is the reference ingress policy of eight in-flight connections per
  authenticated user, not an engine or GPU limit.
- Live 401, 404, 413, and 429 responses were nginx-generated HTML, not
  Anthropic-compatible JSON envelopes. The August 30 record separates these
  live results from isolated sidecar failure tests.
- TLS terminates at the ingress, which can read request and response content.
  The reference logging configuration is metadata-only, but operators remain
  responsible for access controls and retention.
- Do not send protected material to a deployment whose data-handling policy you
  have not verified.

## Repository layout

```text
README.md
LICENSE
CLAIM-LEDGER.md
reproduce/
  run-replicas.sh
  nginx.conf
  nginx-ingress.conf
  benchmark.py
  derive-results.py
  METHOD.md
results/
  raw/
  serving-envelope.json
  environment.md
  verification-2026-08-30.md
  SHA256SUMS
```

Performance figures re-derive from the published raw records. Configuration and
retained-run claims are classified separately in `CLAIM-LEDGER.md`.
