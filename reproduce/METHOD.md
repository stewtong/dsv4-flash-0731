<!-- register: public-github-repo | reader: an ml infra engineer reproducing or auditing the result | consumed: reference during setup and result comparison -->

# Method and provenance

This repository contains two recomputable evidence bundles, one matched DSpark comparison and one prefix-affinity validation. It also includes a machine-readable summary of earlier topology, token-budget, speculative-width, and TP4 campaigns. Campaigns with different controls are reported separately.

## Metric definitions

Time is measured by a client running on the serving node. Percentiles use linear interpolation.

| Metric | Definition |
| --- | --- |
| First visible token (`ttfvt`) | Seconds from request dispatch, before opening the HTTP response, to the first nonempty visible text delta. |
| Turn wall (`wall`) | Seconds from dispatch to the end of the SSE response. |
| Decode rate | `output_tokens / (wall - ttfvt)`, using server-reported output tokens. |
| End-to-end output rate | `output_tokens / wall`. |
| Normalized 800-token wall | `ttfvt + 800 / decode_rate`, in seconds. |

Normalized wall time is calculated for every request before aggregation because the two arms can produce different answer lengths at temperature 0. Raw wall time remains in the evidence; normalized wall time provides the equal-answer-length comparison.

Decode rate is a per-request decode-window rate. Aggregate node throughput requires a concurrent node-level measurement.

## Matched DSpark campaign

The September 3 campaign compared speculation off with DSpark k=5 greedy. Both arms used:

- one dedicated 8x B200 node;
- vLLM 0.25.0 from the same image ID;
- model revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062`;
- DP8 attention with EP8 experts and external load balancing;
- FP8 KV cache with 256-token blocks and the FP4 indexer cache;
- `deep_gemm_mega_moe`;
- maximum batched tokens 8,192, maximum sequences 256, and GPU memory utilization 0.85;
- serial Messages API requests, temperature 0, maximum output 1,024 tokens, and stable session affinity.

The intended arm difference was speculative configuration. The resulting KV capacity was 17,837,368 tokens per rank with speculation off and 17,076,237 with k=5. The speculation-off value is physically consistent with the shared 0.85 memory setting after removing the draft head. It is not the 20,571,235 value associated with the earlier 0.95 memory setting.

Each arm and context contains three sessions of six sequential turns. Turn 1 is the cold reference. The warm aggregate pools turns 2 through 6 across all three sessions, giving 15 observations per arm and context. The full bundle contains 72 requests. Session identity controls cache placement, so observations within a session are clustered. Pooled medians and means are descriptive rather than six independent estimates.

Input-token counts are paired by context, session index, and turn. The verifier requires exact equality across arms. It also requires 36 successful records per arm, the complete session and turn grid, finite positive timings, matching manifest controls, no speculative counter movement for the off arm, and five-position acceptance evidence for k=5.

Two speculation-off warm turns at 128K exceeded three seconds to first visible text. Their timings and positions are retained in the derived summary. No k=5 warm turn exceeded that threshold.

Evidence lives under `results/dspark-matched/`. The selected public records exclude prompts, generated text, session identifiers, private host details, and restart timestamps.

## Prefix-affinity campaign

The prefix-affinity campaign used DP8 with EP8, speculation off, GPU memory utilization 0.95, maximum batched tokens 8,192, and 20,571,235 KV tokens per rank. Its separate controls make it routing evidence for the matched DSpark result.

Every session contains six turns. Turn 1 is cold. Each reported late-turn value is the median of turns 2 through 6 for that session. The direct-rank bundle contains serial and four-session concurrent measurements on ranks 0 and 1. The gateway bundle contains three sessions at each of two context classes and records whether every turn remained on the same rank.

The verifier requires complete six-turn sessions, zero recorded errors, and one serving rank for all six turns of every gateway session. Evidence lives under `results/prefix-affinity/`.

## Earlier campaigns

`results/benchmark-summary.json` preserves selected values and boundaries from four retained campaigns:

- the August 19 cold topology search across 11 context and concurrency cells;
- the 8,192 versus 16,384 batched-token comparison;
- the August 21 DSpark width search;
- the August 29 TP4 and EP4 comparison.

These are summary-only records. Their raw artifacts are not included, so the individual request-level values are not publicly re-derivable.

The earlier 52.4% and 54.8% normalized-wall reductions came from August 21 records that used different GPU memory utilization settings across the compared profiles. They were also calculated by inserting separately aggregated p50 TTFVT and p50 decode rate into the normalization formula. They remain historical results and are superseded as the reference DSpark comparison by the matched per-request calculation.

## Request fixture

Traffic uses a deterministic synthetic ASCII code-context fixture sent to `/v1/messages` with streaming SSE and a stable session header. Nominal context labels are campaign shorthand rather than exact tokenizer counts.

The public benchmark harness calibrates every generated prefix through `/v1/messages/count_tokens` until it falls within the configured tolerance. It starts timing before sending the HTTP request and requires exact input and output usage from the SSE stream. Calibration, prewarm, HTTP, parsing, or usage failure returns a nonzero exit.

A discarded prewarm request per context runs before a measured series. It keeps first-use kernel initialization out of one measured arm.

## Re-derive and verify

Verify the complete published bundle without modifying it:

```bash
python3 reproduce/derive-results.py --check-all
shasum -a 256 -c results/SHA256SUMS
```

Individual checks are also available:

```bash
python3 reproduce/derive-results.py --check-envelope results/serving-envelope.json
python3 reproduce/derive-results.py --check-matched
python3 reproduce/derive-results.py --check-prefix-affinity
python3 reproduce/derive-results.py --check-benchmark-summary
```

Derive a new legacy-format run by passing its JSONL files explicitly:

```bash
python3 reproduce/derive-results.py \
  --input /tmp/dsv4-specoff.jsonl \
  --input /tmp/dsv4-k5.jsonl --json
```

Writing output requires an explicit output option. The verifier returns nonzero for malformed, incomplete, failed, duplicated, or control-drifted evidence.

## Environment and gateway provenance

The digest-pinned launcher encodes the reference deployment and validates a local model-revision marker. The matched manifest records the image ID used in both measured arms. [`results/environment.md`](../results/environment.md) separates the campaign environments and later runtime verification.

The public nginx files are sanitized references. The ingress authenticates requests and provides a trusted fallback affinity key. The balancer selects a rank, then strips rank and session headers before proxying. The private deployment included a separate media-normalization component that is not published or required for the text-only reference.

## Evidence classes

- **Published raw:** recomputable from selected request or session records in this repository.
- **Published configuration:** directly inspectable in a launcher, nginx reference, or manifest.
- **Published summary:** machine-readable retained values whose source request records are not included.
- **Recorded:** described by a dated verification record while the underlying operational artifact remains private.
- **First-party design:** an upstream issue or source that supports architecture rationale rather than this repository's measured performance.

[`CLAIM-LEDGER.md`](../CLAIM-LEDGER.md) assigns a class and boundary to every material claim.

## Unverified scope

This repository does not claim:

- live `nginx -t` validation of the sanitized references on another installation;
- patch-level runtime versions for the August 19 and August 21 campaign containers beyond the retained record;
- Anthropic-compatible JSON parity for nginx-generated errors;
- complete gateway behavior under timeout or mid-stream fault injection;
- clean recreation of any campaign from the pinned image;
- accuracy, model quality, saturated throughput, or arbitrary production-load performance.

Model weights and upstream software retain their own licenses. This repository republishes no model weights.
