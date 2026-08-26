<!-- register: public GitHub repo | reader: an ML infra engineer standing this up on their own 8xB200 | consumed: read once top-to-bottom, then copy-paste -->
# DeepSeek-V4-Flash-0731 on 8x B200 — DP8 + EP8 + DSpark

A reproducible vLLM serving recipe for `deepseek-ai/DeepSeek-V4-Flash-0731` on a single 8x NVIDIA B200 node, tuned for multi-user interactive coding traffic.

The configuration here runs **data-parallel attention (DP8) with expert parallelism (EP8) and DSpark speculative decoding at k=5**, in vLLM's per-rank external-load-balancer mode. Public report [vllm-project/vllm#51454](https://github.com/vllm-project/vllm/issues/51454) recorded that DSpark and DP8 together failed to launch in a single attempt, with no compatibility claim made either way. They do run together, and the argv below is a working combination — it has served as a standing endpoint since 2026-08-21.

What is not established is *why* the earlier attempt failed. That was a single try with no isolated cause, and this configuration differs from it in more than one way, notably `--data-parallel-multi-port-external-lb` with `--data-parallel-size-local 8`. Treat the argv as a known-good recipe, not as a diagnosis.

**Not an official Nebius statement.** Independent benchmarking on Nebius AI Cloud hardware, published as an individual contributor.

| Stage | Status |
| --- | --- |
| Serving config | **Live** — standing endpoint since 2026-08-21 |
| Correctness on this config | **PASS** — 7-prompt suite 7/7, identity-echo 6/6 |
| Performance on this config | **Partial** — coding-turn wall measured; no concurrency ladder yet |
| Long-context envelope on this config | **Not measured** — see [What is not measured](#what-is-not-measured) |

## Metadata

| Field | Value |
| --- | --- |
| Model | [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) |
| Model revision | `7872f01b1d1fe23eabc4c98b48bffcef5a386062` |
| Model on disk | 155.4 GiB, 48 safetensors shards |
| Draft model | none — DSpark ships inside the base checkpoint |
| Serving image | `vllm/vllm-openai:v0.25.0`, pinned by digest (the tag has been observed to drift) |
| Image digest (amd64) | `sha256:e1c1ff1af9a15921bfa11d1d95047258c1797392cdbfa296e7639da446b23f97` |
| vLLM version | 0.25.0, V1 engine, build `dd10e03f95f94edbea1975c67ace3a35ec9a8a40` |
| GPUs | 8x NVIDIA B200 SXM6, SM100, 183,359 MiB each |
| Driver | 580.126.09 / 580.173.02 (both tested) |
| CUDA / PyTorch | 13.0 / 2.11.0+cu130 |
| NCCL / Triton / FlashInfer | 2.28.9 / 3.6.0 / 0.6.13 |
| Parallelism | DP8 + EP8, per-rank external LB |
| Precision | FP4 experts, FP8 KV cache (`fp8_ds_mla`), MXFP4 indexer cache |
| Speculative decoding | DSpark, `num_speculative_tokens=5`, `draft_sample_method=greedy` |
| Context length | 1,048,576 |
| KV cache | **17,076,237 tokens per rank** (16.29x at 1M per request) |
| Ports | 8100–8107 per-rank API servers, 9256 aggregated health |

## Quickstart

### 1. Download the model

```bash
export MODEL_DIR=/data/hf-cache

pip install -U "huggingface_hub[cli]"
hf download deepseek-ai/DeepSeek-V4-Flash-0731 \
  --revision 7872f01b1d1fe23eabc4c98b48bffcef5a386062 \
  --cache-dir "$MODEL_DIR"
```

`--cache-dir` matters. With it, the snapshot lands at `$MODEL_DIR/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/<rev>`, which is the path the launch command expects. Downloading with a default `HF_HOME` instead puts it one level deeper, under `hub/`, and the container path will not resolve. `serve.sh` checks for this and tells you which case you are in.

### 2. Pull the image by digest

The `v0.25.0` tag has been observed to drift. Pin by digest and verify before serving.

```bash
IMAGE=vllm/vllm-openai@sha256:e1c1ff1af9a15921bfa11d1d95047258c1797392cdbfa296e7639da446b23f97
docker pull "$IMAGE"
docker image inspect "$IMAGE" --format '{{join .RepoDigests "\n"}}'
```

### 3. Launch

`VLLM_API_KEY` is read from the environment and is never written to disk by these scripts. Supply it from your own secret store.

```bash
export VLLM_API_KEY="$(your-secret-tool get vllm-api-key)"
./serve.sh
```

Or inline — this is the complete argv, nothing elided:

```bash
docker run -d --name v4flash-serve-dpm --restart unless-stopped \
  --gpus all --ipc host --shm-size 34359738368 \
  --ulimit nofile=1048576:1048576 \
  -e VLLM_API_KEY="$VLLM_API_KEY" \
  -v /data/hf-cache:/root/.cache/huggingface \
  -p 127.0.0.1:8100-8107:8100-8107 -p 127.0.0.1:9256:9256 \
  vllm/vllm-openai@sha256:e1c1ff1af9a15921bfa11d1d95047258c1797392cdbfa296e7639da446b23f97 \
  /root/.cache/huggingface/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062 \
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
```

### 4. Verify before sending traffic

Run this on the serve host, with `VLLM_API_KEY` exported, once the container has been up for a few minutes.

```bash
./verify.sh
```

Three checks, and all three matter:

1. **Do not trust an accepted TCP connection on port 8100.** This image accepts then resets connections for roughly four minutes of pre-startup, so a bare port check false-positives.
2. **Do not gate on the `Application startup complete` log line.** It fires about ten minutes before the engines are ready. Gate on HTTP 200 from all of `8100–8107`.
3. **Check the KV invariant.** Startup logs must show eight `GPU KV cache size: 17,076,237 tokens` lines, one per `EngineCore_DP0..7`. A materially different number means wrong image, wrong weights, or wrong flags. Deltas around 0.001% are CUDA-graph memory-profiling variance and are noise; anything beyond ~0.1% is a stop-and-diagnose.

## Why this configuration

### DP8 instead of TP8

DeepSeek-V4-Flash uses Multi-head Latent Attention, which has a single KV head. Under tensor parallelism that KV cache cannot be sharded, so TP8 replicates it eight times. Under `--data-parallel-size 8` with `--enable-expert-parallel`, each GPU holds an independent KV pool and runs the full attention stack, while MoE expert weights stay sharded EP8 across GPUs and expert traffic crosses via all-to-all.

The startup log makes the difference visible: TP8 reports 20.43x maximum concurrency for 1M-token requests, DP8 reports 156.9x across eight replicas. Measured TTFT at 1M with 8 concurrent requests improves 3.4x on a same-node A/B (3.2–3.7x depending on metric and run). Full ladder in [#51454](https://github.com/vllm-project/vllm/issues/51454).

The trade-off is at the low end: a single 1M request is comparable or slightly slower under DP8, because one GPU does the prefill instead of eight. Break-even is around 3 concurrent requests.

### DSpark at k=5, not k=7

k=5 and k=7 were measured against a 10% turn-wall bar on this endpoint. k=7 cleared the bar in only one of two serial cells and made a 256K-concurrent regression worse, so k=5 is the standing value.

Note that `verifier_accept_len` and `dspark_block_size` are nightly-only speculative-config fields; vLLM v0.25.0 rejects them. There is no built-in way to read acceptance length from the serving API on this tag, which is why the k comparison was run against end-to-end turn wall rather than acceptance rate.

### gpu-memory-utilization 0.85

The DSpark draft head needs roughly 6 GiB per GPU. At 0.85 this config serves 17,076,237 KV tokens per rank against 20,571,235 for the same DP8 shape without speculation — a deliberate trade of KV headroom for decode latency.

### External-LB mode

`--data-parallel-multi-port-external-lb` starts one API server per DP rank on ports 8100–8107, with aggregated health on 9256. It requires `--data-parallel-size-local >= 2`, which its own help text does not state.

This shape is what makes prefix-locality routing possible. Each rank keeps an independent prefix cache, so a client that lands on a different rank each turn re-prefills from cold. Routing a conversation to a fixed rank — hash a session ID to a rank and put it in front of these eight ports — turns that into a cache hit. Without it, a byte-identical request cold-prefills once per rank before going reliably warm.

### max-num-batched-tokens stays at 8192

Raising it to 16384 produced no TTFT improvement and cost 32% of KV capacity.

## Measured on this exact configuration

`dp8-spec-k5` — DP8+EP8, DSpark k=5 greedy, util 0.85.

| Result | Value |
| --- | --- |
| KV cache | 17,076,237 tokens/rank, 16.29x at 1M |
| Coding-turn wall vs speculation off | **−51% to −55%** normalized |
| 7-prompt correctness suite | 7/7 PASS |
| Identity echo | 6/6 PASS |
| k=7 comparison | rejected on measurement |

## Adjacent measurements from #51454

Every number below was measured on a **different configuration** than the one this repo deploys. They are the evidence behind the design choices, not a performance claim for `dp8-spec-k5`. Configuration labels:

- `tp8-plain` — TP8+EP8, no speculation, util 0.95
- `dp8-plain` — DP8+EP8, no speculation, util 0.95 (KV 20,571,235/rank)
- `tp8-spec` — TP8+EP8, DSpark k=5 **probabilistic**, util 0.85

### Long-context TTFT, single request (`tp8-plain`)

Cache-busted real-prose prompts, steady-state medians.

| Context | TTFT p50 |
| --- | ---: |
| 128K | 3.9s |
| 256K | 8.8s |
| 512K | 22.7s |
| 768K | 40.3s |
| 1M | 65.2s |

### Concurrency, DP8 vs TP8 (`dp8-plain` vs `tp8-plain`)

3 reps per cell, cache-busted prose, `max_tokens=64`, streaming.

| Context | c | `dp8-plain` TTFT p50 | `tp8-plain` TTFT p50 | Speedup |
| --- | ---: | ---: | ---: | ---: |
| 128K | 8 | 6.2s | 18.5s | 3.0x |
| 128K | 32 | 16.6s | 61.8s | 3.7x |
| 128K | 64 | 22.0s | 119.8s | 5.4x |
| 256K | 32 | 30.2s | 141.3s | 4.7x |
| 512K | 16 | 32.6s | 172.5s | 5.3x |
| 1M | 8 | 84.7s | 314.6s | 3.7x |
| 1M | 16 | 152.1s | 314.5s (27/48 ok) | 2.1x + 100% ok |

`tp8-plain` at 1M c=16 admits only 56% of requests within a 600s client timeout; its surviving-request statistics carry survivorship bias. Read that row as an admission boundary, not a latency measurement.

On sample sizes: cells are 3 batch reps, and requests inside a rep start synchronized and share the batch. The rep is the unit of replication, so "24/24 ok" is a success count and not 24 independent samples.

### DSpark decode window (`tp8-spec`)

Matched 0.85/0.85 utilization A/B, `max_tokens=512`, decode-window TPOT = `(wall − TTFT) / (completion_tokens − 1)`.

| Cell | Vanilla | DSpark | Multiplier |
| --- | ---: | ---: | ---: |
| 1M c=1 | 8.44ms | 2.08ms | **4.1x** |
| 8K c=1 | 7.82ms | 4.97ms | **1.6x** |

**Streaming TPOT is not a valid per-token metric under speculative decoding.** At 1M c=1, DSpark's streaming inter-arrival TPOT reads 11.18ms against a true decode-window TPOT of 2.08ms, because 512 tokens arrive in roughly 96 chunks and the inter-arrival mean divides by chunks rather than tokens. Vanilla decode does not have this problem — tokens arrive individually, and its streaming figure (8.97ms) tracks its decode-window figure (8.44ms).

At 8K under `tp8-spec`, mean TPOT jumps from 7.8ms at c=3 to 64.2ms at c=4 while p50 stays single-digit; the inflation comes from stragglers rather than uniform degradation. That measurement is TP8 at 8K and has not been repeated on `dp8-spec-k5`.

### Prefix cache (`tp8-plain`)

One 1M-token document, then five sequential follow-up questions: 66.3s cold, ~3.0s warm, a 22x collapse. A second probe loaded two distinct 1M documents and re-sent the first at 3.2s, confirming both coexist in a 21.4M-token budget without eviction. The eviction boundary was not found.

### Levers that did nothing (`tp8-plain`, v0.25.0)

| Lever | Effect |
| --- | --- |
| `--max-num-batched-tokens 16384` | no TTFT gain, −32% KV |
| `--long-prefill-token-threshold 65536` | accepted, no effect on short-request TTFT |
| `--scheduling-policy priority` + per-request priority | no effect; running prefills are not preempted |
| `--max-num-partial-prefills > 1` | `NotImplementedError` |
| `VLLM_USE_BREAKABLE_CUDAGRAPH=0` | torch.compile unsupported for this model; vLLM warns |

Short requests queued behind a 1M prefill wait the full prefill duration under `tp8-plain`. DP8 sidesteps this for the single-long-prefill case through replica isolation, taking short-request TTFT p50 from 63.9s to 0.57s on a same-node A/B.

## What is not measured

Stated plainly so nobody reads the tables above as describing this deployment:

- **No concurrency ladder on `dp8-spec-k5`.** The DP8 ladder is `dp8-plain` at util 0.95 without speculation. Adding a draft head and dropping to 0.85 changes KV budget by 17%, and its effect on the ladder is unquantified.
- **No long-context TTFT on `dp8-spec-k5`.** The 128K–1M ladder is `tp8-plain`.
- **No high-concurrency speculation data.** The c>=4 tail-latency collapse was measured at 8K under `tp8-spec`. Whether `dp8-spec-k5` shows it is unknown, and replica isolation is a reason it might not.
- **`greedy` vs `probabilistic` draft sampling** was A/B tested under `tp8-spec` across 7 cells with no significant difference (greedy 0–16% slower at c=1, identical at c=2+). This repo runs `greedy`, matching the HF model card.
- **No accuracy benchmark suite.** Correctness here is a 7-prompt functional suite, not AIME, NIAH, or an OpenAI-compatibility matrix.

## Files

| File | Purpose |
| --- | --- |
| `serve.sh` | The launch command, parameterized by environment. Pins the image by digest, and fails before launch if the model snapshot path or image digest is wrong. |
| `verify.sh` | Post-launch gates: per-rank health, KV invariant, a live completion |

## References

- [vllm-project/vllm#51454](https://github.com/vllm-project/vllm/issues/51454) — DP8 vs TP8 for single-KV-head MLA, full benchmark report
- [vllm-project/vllm#43753](https://github.com/vllm-project/vllm/issues/43753) — DeepSeek-V4-Pro 1M on the same host class
- [vllm-project/recipes#762](https://github.com/vllm-project/recipes/pull/762) — recipes-page proposal from this work
