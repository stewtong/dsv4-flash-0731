<!-- register: public-github-repo | reader: a reviewer checking that every number is sourced | consumed: spot-check against the re-derivation and citations -->

# Claim to source ledger

Evidence classes:

- **Published** claims re-derive from or are directly inspectable in this tree.
- **Configuration** claims describe behavior implemented by the public reference
  files; they are not performance measurements.
- **Recorded** claims come from retained source-run material that is not included
  in this repository and therefore cannot be independently re-derived here.

| Claim | Class | Source and boundary |
| --- | --- | --- |
| 128K first visible token 0.390 s and decode 221.1 tok/s | Published | `results/raw/cellK5-mnbt8192-serial-128k_b1_s*.jsonl`, aggregated by `reproduce/derive-results.py` |
| 256K first visible token 0.730 s and decode 240.8 tok/s | Published | `results/raw/cellK5-mnbt8192-serial-256k_b1_s*.jsonl`, aggregated by `reproduce/derive-results.py` |
| k=5 normalized-wall reduction is 52.4% at 128K and 54.8% at 256K | Published | `results/serving-envelope.json`; formula is `1000 * ttfvt_seconds + 800 * (1000 / decode_tokens_per_second)` |
| Four cells contain 72 requests, including 60 warm turns, with zero recorded errors | Published | 12 JSONL files with six rows each; `derive-results.py` validates three sessions by six turns per cell before aggregation |
| Two k=5 warm stalls per context | Published | Raw `ttfvt` values and `warm_stall_turns_s` in `results/serving-envelope.json` |
| Seven-prompt functional suite passed 7 of 7 | Recorded | Retained source-run suite summary; underlying suite artifact is not published and the result is not accuracy evidence |
| Verbatim identity echo passed 6 of 6 | Recorded | Retained source-run identity-echo record; underlying artifact is not published |
| KV capacity is 17,076,237 tokens per rank | Recorded | Retained serving run record; no final boot-log line is published |
| Model revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062` | Configuration and recorded | `reproduce/run-replicas.sh` enforces a revision marker; the retained model configuration records the same revision |
| Image digest `sha256:e1c1ff1af9a15921bfa11d1d95047258c1797392cdbfa296e7639da446b23f97` | Configuration and recorded | Digest-pinned public launcher and retained reproducibility plan; the historical deployed launcher used tag `v0.25.0` |
| vLLM build `dd10e03f95f94edbea1975c67ace3a35ec9a8a40` | Recorded | Retained reproducibility plan; not derivable from the request records |
| DP8 and EP8 external-LB profile on ports 8100 through 8107 | Configuration | `reproduce/run-replicas.sh` |
| DSpark k=5 greedy, max batched tokens 8192, max sequences 256, GPU memory utilization 0.85 | Configuration | `reproduce/run-replicas.sh` |
| FP8 KV, FP4 indexer cache, and `deep_gemm_mega_moe` | Configuration | `reproduce/run-replicas.sh`; broader weight-precision description is recorded in `results/environment.md` |
| Five authenticated API routes, separate unauthenticated `/health`, 401 auth rejection, 404 fallback, and per-user connection limit 8 | Configuration | `reproduce/nginx-ingress.conf` |
| External rank headers are cleared; rank and session headers are stripped before the engine | Configuration | `reproduce/nginx-ingress.conf` and `reproduce/nginx.conf` |
| No inference retry after dispatch | Configuration | Both nginx files set `proxy_next_upstream off` and one try |
| No nginx body-size cap in the reference | Configuration | Both nginx files set `client_max_body_size 0`; a default cap can cause 413 on large requests |
| Single-KV-head design favors independent DP caches over TP KV replication | First-party design evidence | [vllm-project/vllm#51454](https://github.com/vllm-project/vllm/issues/51454); the issue is not performance evidence for this profile |
| Claude Code 2.1.229 was the observed client build | Recorded | Retained prefix-affinity plan; later client versions are outside the claim |
| A `thinking` request was accepted but no thinking block was emitted | Recorded | Retained `/v1/messages` observation; not re-tested through the public reference |
| August 30 fixed-k5 deployment passed all rank, aggregate, balancer, proxy, and public health boundaries | Recorded | `results/verification-2026-08-30.md`; raw operational artifacts are not published |
| Fresh August 30 startup recorded 17,076,237 KV tokens on each of eight ranks | Recorded | `results/verification-2026-08-30.md`; boot logs are retained but not published |
| August 30 public-gateway suite passed 7 of 7 and the Anthropic success/error matrix returned the documented statuses | Recorded | `results/verification-2026-08-30.md`; synthetic inputs, with no accuracy claim |
| Current deployment used NVIDIA 580.173.02, CUDA 13.0, PyTorch 2.11.0+cu130, NCCL 2.28.9, and vLLM 0.25.0 | Recorded | `results/verification-2026-08-30.md`; these versions postdate the performance run |
| Live nginx errors were HTML; isolated sidecar tests returned the documented 502 and truncated-stream behavior | Recorded | `results/verification-2026-08-30.md`; live gateway and isolated-component boundaries are separate |

## Deliberately excluded claims

- No p95 is published. The retained summarizer's p95 rule does not reconcile
  with a linear-interpolation p95 of the raw warm aggregate.
- The private media-normalization component is not published. The public
  reference is text-only and makes no media-transformation claim.
- No Anthropic-compatible JSON parity is claimed for nginx-generated errors.
- Timeout and mid-stream failure were not injected through the standing
  endpoint. Their documented behavior comes from isolated tests of source that
  was byte-identical to the live sidecar.
- No claim is made that the full measured environment has been recreated from
  the digest-pinned public launcher.
