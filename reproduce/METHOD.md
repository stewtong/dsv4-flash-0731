<!-- register: public-github-repo | reader: an ml infra engineer reproducing the result | consumed: read as reference when reproducing, then rare -->

# Methodology and provenance

This file defines the published performance metrics, aggregation rules,
request shape, environment evidence, and known gaps.

## Metrics

All published headline statistics are p50 values. Percentiles use linear
interpolation. Time is measured at the client running on the serving node.

| Metric | Definition |
| --- | --- |
| First visible token (`ttfvt`) | Seconds from request dispatch, before opening the HTTP response, to the first nonempty visible text delta. |
| Turn wall (`wall`) | Seconds from dispatch to the end of the SSE response. |
| Decode rate | `output_tokens / (wall - ttfvt)`, using server-reported output tokens. |
| End-to-end output rate | `output_tokens / wall`. |
| Normalized wall | `1000 * ttfvt + 800 * (1000 / decode_rate)`, in milliseconds. |

The speculation-on and speculation-off arms produced different output lengths
at temperature 0. Normalized wall compares an equal 800-token answer using each
arm's p50 first-visible-token time and p50 decode rate. It does not estimate
aggregate node throughput.

## Cell structure and warm-turn aggregation

Each cell contains three sessions of six sequential turns. Turn 1 is the cold
reference. The warm aggregate pools turns 2 through 6 across all three sessions,
giving 15 warm observations per cell. Session identity controls cache placement,
so observations within one session are clustered and are not treated as
independent session-level estimates.

The public bundle contains four cells and 72 requests total:

- speculation off at 128K and 256K
- DSpark k=5 with `--max-num-batched-tokens 8192` at 128K and 256K

All 72 published records report HTTP 200. The warm aggregates contain 60
requests total. `derive-results.py` validates the exact three-by-six structure,
rejects any failed request, and fails on missing or duplicate turns.

A discarded prewarm request per context shape runs before a measurement series.
It prevents first-use kernel initialization from being attributed to one arm.
The public harness provides this behavior through `--prewarm`.

## Request shape

Traffic is a deterministic synthetic ASCII code-context fixture sent to
`/v1/messages` with streaming SSE and a stable session header. The nominal
labels are 131,072 and 262,144 input tokens. The bundled source records report
132,598 and 264,801 observed input tokens for their respective fixtures.

The current public harness calibrates each generated prefix against
`/v1/messages/count_tokens` until it is within the configured tolerance. It
does not estimate input tokens from characters or truncate the count request.
Each stream must report exact input and output usage. Any calibration, prewarm,
HTTP, parsing, or usage failure causes a nonzero exit.

## Derivation

Verify the bundled result without writing a file:

```bash
python3 reproduce/derive-results.py \
  --check-envelope results/serving-envelope.json
```

Derive separate new runs by passing each JSONL file explicitly:

```bash
python3 reproduce/derive-results.py \
  --input /tmp/dsv4-specoff.jsonl \
  --input /tmp/dsv4-k5.jsonl --json
```

Writing an envelope requires the explicit `--json-out PATH` option. The default
command reads the 12 bundled raw files and prints a human-readable summary.

The k=5 cells each include two warm turns with first-visible-token time above
two seconds. `serving-envelope.json` discloses their exact timings under
`warm_stall_turns_s`. No p95 is published because the retained source
summarizer's p95 rule does not reconcile with a linear-interpolation p95 of the
raw warm aggregate.

## Hardware and software record

Measurement took place on one dedicated 8x B200 node on 2026-08-21. The final
profile was DP8 attention with EP8 experts, FP8 KV cache, FP4 routed experts,
the FP4 indexer cache, `deep_gemm_mega_moe`, DSpark k=5 greedy,
`--max-num-batched-tokens 8192`, `--max-num-seqs 256`, and
`--gpu-memory-utilization 0.85`.

The retained deployment used the `v0.25.0` image tag. The reproducibility plan
records these coordinates:

- image digest `sha256:e1c1ff1af9a15921bfa11d1d95047258c1797392cdbfa296e7639da446b23f97`
- vLLM build `dd10e03f95f94edbea1975c67ace3a35ec9a8a40`

The digest is a recorded reproducibility anchor because the deployed launcher
used the mutable tag; it was not independently recovered from the published
request records.

Model revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062` is recorded in the
launcher and model configuration, and `run-replicas.sh` requires a matching
local revision marker before launch. `results/environment.md` lists the full
record and unavailable patch-level versions.

## Gateway provenance and scope

The public nginx files are sanitized references derived from the retained
gateway configuration. They contain no real keys. The ingress supplies an
authenticated user label for fallback affinity and clears an external rank
override. The affinity layer selects a rank, then strips rank and session
headers before the engine.

The private deployment included a separate request transformer for unsupported
media. Its code is not published here and it is not required by the public
reference, which accepts text-only payloads. The public repository therefore
does not claim complete gateway parity with that deployment.

Claude Code 2.1.229 is the recorded client version for the retained Anthropic
Messages observations. The public reference does not certify later client
versions, non-streaming Messages, beta-query behavior, or native nginx error
envelopes as Anthropic-compatible.

## Evidence classes

- **Published**: recomputable or directly inspectable in this repository.
- **Configuration**: implemented by a published reference file, but not proven
  by the performance records alone.
- **Recorded**: retained in the source run material but not included as public
  evidence. Recorded claims are not publicly re-derivable.

`CLAIM-LEDGER.md` assigns one of these classes to every nontrivial README claim.

## Unverified items

The repository does not claim verification of:

- live `nginx -t` against the target installation;
- exact 413, 429, connection-refusal, timeout, or mid-stream error bodies;
- non-streaming and beta-query Anthropic compatibility through the complete
  gateway;
- a fresh boot-log read of 17,076,237 KV tokens per rank;
- patch-level NVIDIA driver, CUDA runtime, PyTorch, or NCCL versions;
- clean recreation of the measured result from the recorded image digest.

Model weights and upstream software retain their own licenses. This
repository's scripts and prose are MIT licensed. It republishes no model
weights.
