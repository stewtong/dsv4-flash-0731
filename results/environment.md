<!-- register: public-github-repo | reader: an ml infra engineer comparing or reproducing a campaign | consumed: check controls before comparing results -->

# Environment record

All campaigns used one dedicated 8x B200 SXM node with approximately 160 vCPUs and 1,792 GB of host memory. The model was `deepseek-ai/DeepSeek-V4-Flash-0731` at revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062`.

## Reference deployment and matched comparison

The September 3 matched comparison and reference deployment share these settings:

| Field | Value |
| --- | --- |
| Server engine | vLLM 0.25.0 |
| Container tag | `vllm/vllm-openai:v0.25.0` |
| Measured image ID | `sha256:d5219758abb32a8fa60cf18fc6f8a5b2984aea0ad4004f71ae9ca835bc15c8fd` |
| Reproduction image digest | `sha256:e1c1ff1af9a15921bfa11d1d95047258c1797392cdbfa296e7639da446b23f97` |
| vLLM build record | `dd10e03f95f94edbea1975c67ace3a35ec9a8a40` |
| Attention and expert topology | DP8 attention, EP8 experts, external load balancing |
| KV cache | FP8, 256-token blocks |
| Model precision | FP4 routed experts; other weights FP8 e4m3 with 128x128 block scaling |
| Indexer cache | FP4 enabled |
| MoE backend | `deep_gemm_mega_moe` |
| Batched-token budget | 8,192 |
| Maximum sequences | 256 |
| GPU memory utilization | 0.85 |
| Speculative decoding | DSpark k=5 greedy |
| Rank endpoints | Eight loopback API servers on ports 8100 through 8107 |
| Aggregate controller | Loopback port 9256 |

The measured image ID identifies the local Docker image used by both matched arms. The launcher pins the registry manifest digest used for reproduction. Docker reports these as different identifiers.

The matched speculation-off arm recorded 17,837,368 KV tokens per rank. The k=5 arm recorded 17,076,237. The difference follows from the k=5 draft head. Both arms used GPU memory utilization 0.85.

## August 19 topology and prefix campaigns

The cold topology, batched-token, and prefix-affinity campaigns used vLLM 0.25.0, FP8 KV cache, and GPU memory utilization 0.95. The prefix-affinity DP8 profile recorded 20,571,235 KV tokens per rank with speculation off and maximum batched tokens 8,192.

The topology campaign varied attention parallelism among TP8, TP4 with DP2, TP2 with DP4, and DP8. Expert parallelism followed the retained campaign configurations. [`benchmark-summary.json`](benchmark-summary.json) gives the result boundaries.

## August 21 historical DSpark campaign

The original DSpark width work used the same model revision and vLLM 0.25.0. Its off and k=5 comparison did not hold GPU memory utilization constant, so the older 52.4% and 54.8% normalized-wall figures are historical rather than the reference comparison.

The width search varied speculative width and batched-token budget. The selected final k=5 profile used maximum batched tokens 8,192, maximum sequences 256, GPU memory utilization 0.85, and 17,076,237 KV tokens per rank.

## August 30 runtime verification

A later fixed-k5 restart recorded:

| Runtime field | Value |
| --- | --- |
| NVIDIA driver | 580.173.02 |
| CUDA runtime | 13.0 |
| PyTorch | 2.11.0+cu130 |
| NCCL | 2.28.9 |
| vLLM | 0.25.0 |

These versions describe the August 30 deployment, not a patch-level reconstruction of earlier campaign environments. The [August 30 verification](verification-2026-08-30.md) records the service and protocol checks without adding performance measurements.

## Version boundaries

The node disk-image record was Ubuntu 24.04 with CUDA 13.0.0.2.1216. Patch-level NVIDIA driver, CUDA runtime, PyTorch, and NCCL versions were not retained for every earlier campaign and are not reconstructed from the later deployment.

Claude Code 2.1.229 was the observed client build for the retained Anthropic Messages checks. Later client behavior is outside the claim.
