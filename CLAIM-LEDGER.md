<!-- register: public-github-repo | reader: a technical reviewer checking every material claim | consumed: spot-check against evidence and derivation code -->

# Claim to source ledger

Evidence classes are defined in [`reproduce/METHOD.md`](reproduce/METHOD.md). Published raw claims re-derive from selected records. Published configuration claims are inspectable in this tree. Published summary claims retain campaign-level values without public request-level records. Recorded claims describe private operational evidence. First-party design claims support architecture rationale rather than measured performance.

## Standing profile and matched DSpark result

| Claim | Class | Source and boundary |
| --- | --- | --- |
| Standing profile is DP8 attention, EP8 experts, DSpark k=5 greedy, maximum batched tokens 8,192, maximum sequences 256, and GPU memory utilization 0.85 | Published configuration | `reproduce/run-replicas.sh` |
| k=5 cut pooled warm normalized 800-token wall p50 by 55.12% at 128K and 52.24% at 256K | Published raw | `results/dspark-matched/raw/specoff.json`, `k5.json`, and `reproduce/derive-results.py`; 15 warm observations per arm and context |
| Normalized p50 changed from 8.7055 to 3.9072 seconds at 128K and from 8.5306 to 4.0742 seconds at 256K | Published raw | `results/dspark-matched/summary.json`; calculated per request as `ttfvt + 800 / decode_rate` before aggregation |
| Decode p50 increased 134.59% at 128K and 134.40% at 256K | Published raw | Matched records and derived summary; per-request decode-window rate, not node throughput |
| TTFT p50 changed by -1.45% at 128K and +5.28% at 256K | Published raw | Matched records and derived summary |
| All six per-session normalized-wall mean comparisons favored k=5 | Published raw | `session_normalized_800_wall_mean_deltas_pct` in the matched summary |
| All 72 requests returned HTTP 200 with zero errors | Published raw | 36 records per arm; verifier rejects any failed or incomplete record |
| Two speculation-off 128K warm turns stalled at 12.6058 and 10.1781 seconds; no k=5 warm turn exceeded three seconds | Published raw | Matched raw records and `stalls` in the derived summary |
| Arms shared image, model revision, topology, request fixture, session order, batch-token budget, maximum sequences, and GPU memory utilization | Published raw and configuration | `results/dspark-matched/manifest.json`; verifier rejects drift outside the declared differences |
| Speculation-off recorded 17,837,368 KV tokens per rank and k=5 recorded 17,076,237 | Published configuration | Matched manifest; the draft head explains the capacity difference at shared GPU memory utilization 0.85 |
| Off-arm speculative counters did not move and k=5 counters did | Published raw | Sanitized engine summaries under `results/dspark-matched/raw/`; checked by verifier |
| The old 52.4% and 54.8% figures are historical and confounded by different GPU memory utilization settings | Published raw and recorded | Legacy raw bundle and serving envelope preserve the calculation; `reproduce/METHOD.md` records the control boundary |

## Prefix locality and gateway behavior

| Claim | Class | Source and boundary |
| --- | --- | --- |
| Direct serial late-turn median TTFT ranged from 0.467 to 0.519 seconds at 128K-class and 0.972 to 1.037 seconds at 256K-class | Published raw | Selected direct-rank session summaries under `results/prefix-affinity/raw/` |
| Four-session concurrent late-turn ranges were 0.934 to 1.189 seconds and 2.162 to 2.793 seconds | Published raw | Selected direct-rank concurrent summaries and derived prefix summary |
| Gateway serial late-turn ranges were 0.493 to 0.570 seconds and 1.005 to 1.085 seconds | Published raw | `gateway-sessions.json` and derived prefix summary |
| Every turn of all six gateway sessions stayed on one rank | Published raw | Gateway session records; verifier requires six turns on one serving rank |
| Prefix campaign used speculation off, GPU memory utilization 0.95, and 20,571,235 KV tokens per rank | Published configuration | `results/prefix-affinity/manifest.json`; not a control arm for the matched k=5 campaign |
| Ingress allowlist, unauthenticated health, bearer authentication, per-user connection limit 8, and external rank-header clearing | Published configuration | `reproduce/nginx-ingress.conf` |
| Session hashing, trusted explicit-rank handling, and rank and session header stripping | Published configuration | `reproduce/nginx.conf` |
| Inference requests are not retried after dispatch | Published configuration | Both nginx files set one try and disable upstream retry |
| The reference removes nginx body-size caps | Published configuration | Both nginx files set `client_max_body_size 0` |
| Model context limit is 1,048,576 combined input and output tokens | Recorded | Retained model configuration; the launcher relies on the model default and does not override `max-model-len` |

## Earlier campaign decisions

| Claim | Class | Source and boundary |
| --- | --- | --- |
| Cold topology campaign covered 11 cells and 276 requests per topology with zero failures | Published summary | `results/benchmark-summary.json`; request-level artifacts are not included |
| No topology won every cold cell; DP8 led most short and medium cells, while narrower TP variants led selected stressed or boundary cells | Published summary | Cold-topology p50 and p95 table in the benchmark summary |
| Raising maximum batched tokens from 8,192 to 16,384 reduced KV capacity from 20,971,965 to 14,138,549, a 32.6% loss, with mixed latency changes | Published summary | Batched-token campaign in the benchmark summary |
| k=5 was selected over k=7 because k=7's incremental serial gain narrowed at 256K and its concurrent long-context posture regressed | Published summary | DSpark-width campaign and selection reason in the benchmark summary |
| Final k=5 at maximum batched tokens 8,192 recovered 5.8% KV capacity versus the k=5 width-search cell with less than 5% serial effect | Published summary | DSpark-width selected profile and selection reason |
| MTP failed to reach readiness because the checkpoint lacked `mtp_block.main_norm.weight` | Published summary | DSpark-width campaign; no MTP performance claim |
| TP4 with EP4 improved 1M decode time per token by 15.3% but regressed wall time by 3.8%, showed sticky slow warm turns, and failed the forced concurrent boundary | Published summary | TP4 challenger in the benchmark summary; mixed 12-session arm was skipped |

## Runtime and compatibility record

| Claim | Class | Source and boundary |
| --- | --- | --- |
| Model revision is `7872f01b1d1fe23eabc4c98b48bffcef5a386062` | Published configuration | Launcher requires the revision marker; manifests record the same revision |
| Reproduction image is digest-pinned | Published configuration | `reproduce/run-replicas.sh`; measured campaign manifests separately record image identity |
| Single-KV-head design motivates independent DP caches over TP KV replication | First-party design | [vllm-project/vllm#51454](https://github.com/vllm-project/vllm/issues/51454); architecture rationale, not performance evidence |
| Issue #51454's matched DSpark cell used TP8 at 1M, and its early DSpark with DP8 launch failure predates the working DP8 with EP8 campaigns here | First-party design and published raw | Issue #51454 for the earlier cells; this repository's matched and prefix bundles for the later working deployment |
| August 30 restart exposed all service boundaries and eight 17,076,237-token KV records | Recorded | `results/verification-2026-08-30.md`; operational logs are not published |
| August 30 synthetic functional suite passed 7 of 7 | Recorded | Verification record; not an accuracy result |
| August 30 runtime used NVIDIA 580.173.02, CUDA 13.0, PyTorch 2.11.0+cu130, NCCL 2.28.9, and vLLM 0.25.0 | Recorded | Verification record; later than the performance campaigns |
| Anthropic Messages success cases and live 401, 404, 413, and 429 status classes matched the verification table | Recorded | Verification record; raw gateway artifacts are private |
| Live nginx errors were HTML; isolated sidecar checks produced documented 502 and truncated-stream behavior | Recorded | Verification record; complete live fault injection was not performed |
| Claude Code 2.1.229 was the observed client build | Recorded | `results/environment.md`; later clients are outside the claim |

## Excluded claims

- No p95 is reported for the matched warm DSpark result.
- No model-quality, accuracy, saturated-throughput, or open-loop claim is made.
- No complete public-gateway parity is claimed because the private media-normalization component is omitted.
- No Anthropic-compatible JSON parity is claimed for nginx-generated errors.
- No clean recreation from the pinned image is claimed.
- No broad claim is made that every TP4 configuration loses to DP8.
