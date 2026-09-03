<!-- register: public-github-repo | reader: an ml infra engineer comparing serving profiles | consumed: scan tables, then inspect relevant campaign boundaries -->

# Benchmark guide

The reference deployment uses data-parallel attention across eight ranks, expert parallelism across eight ranks, DSpark speculative decoding with five draft tokens and greedy draft sampling, a maximum batched-token budget of 8,192, and GPU memory utilization 0.85. Topology, prefix-cache placement, token-budget, speculative-width, and TP4 comparison tests informed the configuration.

## Matched k=5 comparison

The September 3 matched run changed only speculative decoding and the KV capacity that follows from it. Both arms used the same image, model revision, DP8 and EP8 topology, request fixtures, session order, batch-token budget, maximum sequences, and GPU memory utilization 0.85.

Each cell contains three six-turn sessions. Turn 1 is cold. The warm aggregate contains turns 2 through 6, giving 15 observations per arm and context. Time to first visible token (TTFVT) stops at the first nonempty visible text delta. Normalized 800-token wall time is calculated for every request as `ttfvt_s + 800 / tps_decode`, then aggregated.

| Context | Arm | Normalized 800 wall p50 | Normalized 800 wall mean | Decode p50 | TTFVT p50 | Warm stalls over 3 s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 128K | Speculation off | 8.7055 s | 10.0807 s | 97.16 tok/s | 0.4160 s | 2 |
| 128K | k=5 | 3.9072 s | 3.7788 s | 227.93 tok/s | 0.4100 s | 0 |
| 256K | Speculation off | 8.5306 s | 8.5433 s | 100.87 tok/s | 0.6485 s | 0 |
| 256K | k=5 | 4.0742 s | 4.0045 s | 236.43 tok/s | 0.6827 s | 0 |

| Context | Normalized p50 delta | Decode p50 delta | TTFVT p50 delta |
| --- | ---: | ---: | ---: |
| 128K | **-55.12%** | +134.59% | -1.45% |
| 256K | **-52.24%** | +134.40% | +5.28% |

All six per-session mean comparisons favored k=5. The deltas ranged from -57.10% to -64.95% at 128K and from -49.20% to -56.72% at 256K. The two 128K speculation-off stalls occurred on turn 2 of sessions 0 and 1 at 12.6058 and 10.1781 seconds. All 72 requests returned HTTP 200, and no k=5 warm turn exceeded three seconds.

The output lengths differed between arms. Normalized wall time provides the equal-answer-length comparison. The pooled values are descriptive because turns within a session share cache placement. Saturation, aggregate throughput, accuracy, and model quality remain unmeasured.

Evidence: [`results/dspark-matched/summary.json`](results/dspark-matched/summary.json), [`results/dspark-matched/manifest.json`](results/dspark-matched/manifest.json), and the selected records under `results/dspark-matched/raw/`.

## Cold topology search

The August 19 cold campaign compared TP8, TP4 with DP2, TP2 with DP4, and DP8 across 11 context and concurrency cells. Every request generated up to 64 tokens. Each topology processed 276 requests with zero failures.

| Context | Concurrency | TP8 p95 TTFT | TP4 with DP2 | TP2 with DP4 | DP8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32K | 4 | 3.30 s | 3.97 s | 3.16 s | **1.90 s** |
| 32K | 8 | 6.50 s | 3.82 s | 3.18 s | **2.40 s** |
| 32K | 16 | 12.92 s | 18.16 s | 5.64 s | **3.73 s** |
| 128K | 4 | 14.71 s | 12.77 s | 9.02 s | **8.38 s** |
| 128K | 8 | 29.15 s | 15.36 s | 22.44 s | **9.98 s** |
| 128K | 16 | 58.04 s | 30.22 s | **17.94 s** | 28.50 s |
| 256K | 4 | 33.56 s | 17.87 s | **10.77 s** | 20.98 s |
| 256K | 8 | 66.54 s | 34.82 s | 30.31 s | **22.08 s** |
| 256K | 16 | 132.39 s | 68.89 s | 39.42 s | **33.51 s** |
| 512K | 4 | 86.56 s | **45.40 s** | 51.31 s | 48.23 s |
| 1M | 4 | 250.96 s | **130.08 s** | 143.78 s | 134.87 s |

No topology won every cell. DP8 won most short and medium cells, TP2 with DP4 had the best p95 at two stressed cells, and TP4 with DP2 held the p95 edge at 512K and 1M. At 1M, TP4 with DP2 had the best p95 but a 129-second median, versus 75.9 seconds for TP2 with DP4 and 72.0 seconds for DP8. The apparent TP4 edge was therefore a variance result rather than a broadly faster distribution.

The 512K one-token outputs were retained as an anomaly and excluded from quality interpretation. This summary-only campaign is recorded in [`results/benchmark-summary.json`](results/benchmark-summary.json). Its broader design context is in [vllm-project/vllm#51454](https://github.com/vllm-project/vllm/issues/51454).

Issue #51454 is the earlier broad campaign. Its matched speculative-decoding cell used TP8 at 1M. Its early DSpark with DP8 launch failure predates the working DP8 with EP8 measurements published here.

## Prefix-affinity validation

The prefix campaign used DP8 with EP8, speculation off, GPU memory utilization 0.95, and a batched-token budget of 8,192. Turn 1 populated the cache. The reported range covers each session's median over turns 2 through 6.

| Path | Sessions per context | 128K-class late median TTFVT range | 256K-class late median TTFVT range |
| --- | ---: | ---: | ---: |
| Direct rank pin, serial | 6 | 0.467 to 0.519 s | 0.972 to 1.037 s |
| Direct rank pin, four concurrent sessions | 8 | 0.934 to 1.189 s | 2.162 to 2.793 s |
| Consistent-hash gateway, serial | 3 | 0.493 to 0.570 s | 1.005 to 1.085 s |

The gateway kept all six turns of every measured session on one rank. The direct-rank and gateway results agree closely under serial load. Four concurrent sessions widened the later-turn range, but it remained well below the corresponding cold first turn for every recorded session.

The observed gateway prompt counts were 222,099 for the 128K-class label and 446,049 for the 256K-class label. The labels identify campaign fixtures rather than exact token counts.

These timings come from a different memory setting and speculation state than the matched k=5 campaign. They support the routing choice but should not be combined with the matched performance figures.

Evidence: [`results/prefix-affinity/summary.json`](results/prefix-affinity/summary.json), [`results/prefix-affinity/manifest.json`](results/prefix-affinity/manifest.json), and selected records under `results/prefix-affinity/raw/`.

## Batched-token budget

The budget campaign compared 8,192 with 16,384 maximum batched tokens across the same 11 cold cells. Raising the budget changed p95 TTFT in both directions and reduced KV capacity from 20,971,965 to 14,138,549 tokens per rank, a 32.6% loss.

| Budget | KV tokens per rank | Result |
| ---: | ---: | --- |
| 8,192 | 20,971,965 | Preserved long-context headroom and became the reference setting |
| 16,384 | 14,138,549 | Mixed latency changes with materially lower KV capacity |

The decision favors the smaller budget because the endpoint serves long-context sessions and the larger budget did not provide a consistent latency gain. Full cell values are in [`results/benchmark-summary.json`](results/benchmark-summary.json).

## DSpark width search

The width campaign compared k=3, k=4, k=5, and k=7 before the final memory-headroom adjustment.

| Width | Max batched tokens | Effective target budget | KV tokens per rank | Decode at 128K | Decode at 256K | Mean accepted tokens |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| k=3 | 8,192 | 7,680 | 17,076,237 | 179.1 tok/s | 193.7 tok/s | 2.316 |
| k=4 | 8,960 | 8,192 | 16,353,196 | 202.5 tok/s | 212.8 tok/s | 2.797 |
| k=5 | 9,216 | 8,192 | 16,138,640 | 223.1 tok/s | 237.5 tok/s | 3.065 |
| k=7 | 9,728 | 8,192 | 15,677,138 | 260.5 tok/s | 259.0 tok/s | Not reported |

k=7 improved normalized serial wall time versus k=5 by 13.0% at 128K and 6.4% at 256K, but performed worse under concurrent long-context load and consumed more KV headroom. k=5 retained most of the decode gain with better concurrent behavior and more KV headroom.

The final k=5 profile lowered maximum batched tokens to 8,192, giving an effective target budget of 7,168 and 17,076,237 KV tokens per rank. That change increased KV capacity by 5.8% versus the k=5 width-search cell and had less than a 5% serial effect in the retained comparison.

An MTP attempt failed during model load because the checkpoint lacked `mtp_block.main_norm.weight`. No MTP performance claim is made.

## TP4 comparison

A final fixed-output test compared DP8 with two TP4 and EP4 replicas. The TP4 profile improved decode time per output token by 15.3% at 1M, but full wall time was 69.58 seconds versus 67.05 seconds for DP8, a 3.8% regression.

The TP4 profile also showed a persistent slow warm range of 16.74 to 24.49 seconds. Under four forced concurrent 1.03M sessions on one replica, only one session completed all turns before the ten-minute stop. The planned mixed 12-session arm was skipped after that safety stop. The reference deployment retained DP8.

This campaign rejects the tested TP4 profile. Other TP4 configurations remain unmeasured. Summary values and boundaries are in [`results/benchmark-summary.json`](results/benchmark-summary.json).

## Evidence boundaries

- Matched DSpark and prefix-affinity claims are recomputable from selected public records.
- Cold topology, token-budget, DSpark-width, and TP4 comparison values are summary-only records from retained campaigns.
- Cold TTFT, warm TTFVT, normalized equal-length wall time, and decode-window rate measure different parts of request execution and are not interchangeable.
- None of these campaigns measures model quality, accuracy, aggregate throughput at saturation, or production behavior under arbitrary traffic.
- The [method](reproduce/METHOD.md), [environment record](results/environment.md), and [claim ledger](CLAIM-LEDGER.md) define the calculation and provenance for each claim.
