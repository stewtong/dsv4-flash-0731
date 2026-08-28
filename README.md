<!-- register: public github repo | reader: an ml infra engineer standing this up on their own 8x b200 | consumed: read once top-to-bottom, then copy-paste -->
# DeepSeek-V4-Flash-0731 on B200

Serving recipe and performance notes for running DeepSeek-V4-Flash-0731 on a dedicated 8x B200 node behind an OpenAI-compatible gateway. The model is text-only. The reference endpoint accepts OpenAI-compatible Chat Completions and the Anthropic Messages API.

| Field | Value |
| --- | --- |
| Model | `deepseek-v4-flash-0731`, text-only |
| Deployment | Eight DP attention workers with EP8 experts and session-affine routing |
| Context | 1,048,576 tokens configured |
| Validated client profile | 1,000,000 input plus 32,768 output tokens |
| Concurrency | Reference-gateway policy: 8 in-flight requests per key |
| OpenAI base URL | `https://<your-endpoint-host>/v1` |
| Anthropic base URL | `https://<your-endpoint-host>` |
| Authentication | `Authorization: Bearer <your key>` |

Validated on August 21, 2026 on one 8x B200 node. The reference deployment terminates TLS with a publicly trusted certificate; an independent deployment must provide its own trusted TLS termination. This is an individual contribution and not an official statement from Nebius.

Concurrency of eight in-flight requests per key is a policy enforced by the reference gateway, not a limit of vLLM or the GPU.

## Performance

Measured on the validated serving profile described below: data-parallel attention over eight ranks (DP8) with EP8 experts, DSpark speculative decoding at k=5 greedy, `max-num-batched-tokens 8192`, and `gpu-memory-utilization 0.85`. Requests were serial, at temperature 0, with identical prompts, on a warm prefix cache. The warm aggregate is turns 2 through 6 across 3 sessions, so 15 warm turns per cell; the session is the unit of replication. Cells are medians (p50). Decode is the per-session decode-window output rate in tokens per second.

| Context | First visible token | Decode |
| ---: | ---: | ---: |
| 128K | 0.390 s | 221.1 tok/s |
| 256K | 0.730 s | 240.8 tok/s |

Against the same DP8 shape with speculation off, DSpark at k=5 cut the normalized wall for an equal 800-token answer by 52.4% at 128K and 54.8% at 256K. Normalized turn wall is time to first visible token plus 800 multiplied by milliseconds per token. At temperature 0 the two arms generate different-length answers for the same prompt, so raw wall time is not comparable between them.

The exact k=5 configuration passed a seven-prompt functional suite 7/7 and six verbatim identity-echo checks 6/6. The suite covers exact arithmetic, long-context needle recovery, and code generation whose output must compile; it is a functional smoke test, not an accuracy benchmark. The identity echo guards against the identifier-corruption class reported in [vllm-project/vllm#52404](https://github.com/vllm-project/vllm/issues/52404) and [sgl-project/sglang#34959](https://github.com/sgl-project/sglang/issues/34959), which did not reproduce on this k=5 configuration.

The validated cell holds 17,076,237 KV tokens per rank. That number is a startup invariant for this serving profile; reconcile a materially different value before comparing results.

## Further evidence

[vllm-project/vllm#51454](https://github.com/vllm-project/vllm/issues/51454) records the full DP8-versus-TP8 analysis and the benchmark ladder behind the design choices above. Its tables use adjacent configurations, not the validated serving profile here; treat them as design evidence, not as a performance claim for this endpoint.

## OpenAI-compatible use

```bash
export DSV4FLASH_API_KEY="<your-key>"
export DSV4FLASH_OPENAI_BASE_URL="https://<your-endpoint-host>/v1"

curl "$DSV4FLASH_OPENAI_BASE_URL/chat/completions" \
  -H "Authorization: Bearer $DSV4FLASH_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash-0731","messages":[{"role":"user","content":"hello"}],"max_tokens":64}'
```

Python with the OpenAI SDK:

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url=os.environ["DSV4FLASH_OPENAI_BASE_URL"],
    api_key=os.environ["DSV4FLASH_API_KEY"],
    timeout=180,
)
response = client.chat.completions.create(
    model="deepseek-v4-flash-0731",
    messages=[{"role": "user", "content": "hello"}],
    max_tokens=64,
)
print(response.choices[0].message.content)
```

Give the response budget enough room to return visible content reliably. Supported routes are `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/messages`, and `/v1/messages/count_tokens`. The OpenAI SDK `responses.create()` route returns 404; use `chat.completions.create()`.

## Claude Code

Run Claude Code with variables scoped to that process:

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

Keep `ANTHROPIC_BASE_URL` at the bare host. Claude Code appends `/v1/messages`; adding `/v1` to the base URL produces `/v1/v1/messages` and a 404. These variables apply only to the launched process. The recipe was verified against Claude Code as of 2026-08-21; environment variable names can change between releases.

The model is text-only. A bare model endpoint does not support image or document blocks. The validated reference gateway replaces unsupported media blocks with text markers so a Claude Code session remains usable. An independent operator must implement equivalent filtering or keep clients text-only. Extract image or PDF text locally before sending it.

## Under the hood

- The serving image is `vllm/vllm-openai:v0.25.0`, pinned to `sha256:e1c1ff1af9a15921bfa11d1d95047258c1797392cdbfa296e7639da446b23f97` (vLLM build `dd10e03f95f94edbea1975c67ace3a35ec9a8a40`). Model weights are pinned to revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062`.
- vLLM serves eight data-parallel API workers with expert parallelism across all eight B200s. Each DP rank runs its own attention stack and a shard of the MoE experts, with expert traffic crossing ranks via all-to-all. One API server runs per rank behind the gateway.
- Routed experts use FP4, other weights use FP8 e4m3, and the KV cache uses FP8 with 256-token blocks. The FP4 indexer cache and `deep_gemm_mega_moe` backend are enabled.
- DSpark speculative decoding runs at k=5 with greedy draft sampling, 8,192 batched tokens, 256 maximum sequences, and `gpu-memory-utilization 0.85`.
- The reference gateway keeps a conversation on one DP rank so its prefix cache stays warm across turns. An independent operator reproduces this with session-affine routing (below).

## Reverse-proxy requirements

The eight model workers are private and expect an authenticated ingress in front of them. These are operator requirements on that gateway, not generic vLLM behavior:

- Bind the model workers to private endpoints and put authenticated ingress in front of them.
- Keep TLS verification enabled.
- Disable response buffering for streamed token delivery, or the first byte arrives late and a coding terminal feels dead.
- Preserve or generate a stable session key so repeated turns reach the same DP rank.
- Set a request-body cap large enough for the chosen context profile.
- Apply concurrency and rate policy at the gateway, and label those values as policy.
- Log metadata only if that is the operator's declared data-handling posture.

## Limits and data handling

- Stream long responses and use a client timeout of at least 180 seconds for large documents.
- On the reference deployment, a `429` normally means the per-key limit of 8 in-flight requests was reached. A boundary `429` can occur at exactly 8 because of connection accounting. Back off and retry. Independent gateways may enforce different policies.
- The model is text-only. The reference gateway strips unsupported blocks; an independent operator must implement that behavior or require text-only requests.
- TLS terminates at the gateway edge, which can read request and response content.
- In the reference deployment the gateway logs metadata only and does not store prompt or response content. Those are properties of that deployment, not of the model flags; an independent operator must set its own guards.
- Do not send customer data, credentials, or material covered by an NDA to a deployment whose data-handling policy you have not verified.

| Symptom | Likely cause |
| --- | --- |
| `401` | Missing or incorrect bearer key |
| `404` | Unsupported route, or `/v1` added to the Anthropic base URL |
| `429` | The reference gateway's per-key concurrency limit was reached; back off and retry |
| Timeout on a large document | Client timeout below 180 seconds |
| Media omission marker | The gateway removed an image or document block; extract the text locally |
| Model error after media | The client bypassed the text-only guard |
| `certificate verify failed` | A local CA override or trust setting is active |
