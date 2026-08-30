<!-- register: public-github-repo | reader: an ml infra engineer reproducing the result | consumed: check before a run and when comparing results -->

# Environment record

Measured on one dedicated 8x B200 node, 2026-08-21. Values are the exact record
from the serving run. Where a patch-level version is not retained in a
first-party artifact on this project, it is marked unavailable and is not
reconstructed.

| Field | Value |
| --- | --- |
| GPU | 8x B200 SXM |
| Node shape | 8 GPU, 160 vCPU, ~1792 GB host memory |
| Node type | Dedicated, non-pre-emptible |
| Server engine | vLLM 0.25.0 |
| Container image | Retained deployment used tag `v0.25.0`; reproducibility plan pins `vllm/vllm-openai@sha256:e1c1ff1af9a15921bfa11d1d95047258c1797392cdbfa296e7639da446b23f97` (built 2026-07-11) |
| vLLM build | `dd10e03f95f94edbea1975c67ace3a35ec9a8a40` |
| Model | `deepseek-ai/DeepSeek-V4-Flash-0731`, revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062` |
| Host/user-space base image | Ubuntu 24.04 with CUDA 13.0.0.2.1216 (from the node disk image) |
| NVIDIA driver | Unavailable (not retained in a first-party artifact) |
| CUDA runtime / PyTorch / NCCL patch versions | Unavailable (not retained in a first-party artifact) |
| KV cache dtype | FP8, 256-token blocks |
| Model precision | FP4 routed experts (top-level `expert_dtype`); other weights FP8 e4m3 with 128x128 block scaling; FP4 indexer cache on |
| MoE backend | `deep_gemm_mega_moe` |
| Serving profile | `--data-parallel-size 8 --data-parallel-size-local 8 --data-parallel-multi-port-external-lb --port 8100`, `--enable-expert-parallel`, DSpark k=5 greedy, `--max-num-batched-tokens 8192`, `--max-num-seqs 256`, `--gpu-memory-utilization 0.85` |
| KV capacity | 17,076,237 tokens per rank (recorded startup invariant; provenance in METHOD.md) |
| Ports | Eight loopback rank API servers on 8100-8107, controllers on 9256; affinity balancer on loopback 8000 |
| Recorded Anthropic client | Claude Code 2.1.229 |

The Claude Code version is an observed point. Environment-variable names and
gateway behavior are reported for that build and are not a promise for later
versions.
