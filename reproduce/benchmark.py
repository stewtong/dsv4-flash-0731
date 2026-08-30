#!/usr/bin/env python3
"""Measure text-only Anthropic Messages requests through a gateway.

The harness calibrates each synthetic prefix with ``count_tokens``, starts the
first-visible-token clock at dispatch, uses response-reported token counts, and
returns nonzero if any measured request fails. Credentials are read only from
``DSV4FLASH_API_KEY``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ANTHROPIC_VERSION = "2023-06-01"
CALIBRATION_BLOCK = (
    "def helper(value):\n"
    "    return value * 2\n\n"
    "# Deterministic ASCII repository context.\n"
)


def request_headers(session: str | None = None) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + os.environ["DSV4FLASH_API_KEY"],
        "anthropic-version": ANTHROPIC_VERSION,
    }
    if session:
        headers["x-claude-code-session-id"] = session
    return headers


def build_messages(prefix: str, turn: int) -> list[dict]:
    suffix = f"\nTurn {turn:02d}: return a concise description of helper()."
    return [{"role": "user", "content": prefix + suffix}]


def count_tokens(host: str, model: str, messages: list[dict], timeout: float) -> int:
    url = host.rstrip("/") + "/v1/messages/count_tokens"
    body = json.dumps({"model": model, "messages": messages}).encode()
    request = urllib.request.Request(url, data=body, method="POST", headers=request_headers())
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if int(response.status) != 200:
            raise RuntimeError(f"count_tokens returned HTTP {response.status}")
        payload = json.load(response)
    value = payload.get("input_tokens")
    if not isinstance(value, int) or value <= 0:
        raise RuntimeError("count_tokens response did not contain a positive input_tokens value")
    return value


def calibrate_prefix(host: str, model: str, target_tokens: int, session: str,
                     timeout: float, tolerance: int) -> tuple[str, int]:
    repeats = max(1, target_tokens // 20)
    observed = 0
    for _ in range(10):
        prefix = f"# session {session}\n" + CALIBRATION_BLOCK * repeats
        observed = count_tokens(host, model, build_messages(prefix, 1), timeout)
        if abs(observed - target_tokens) <= tolerance:
            return prefix, observed
        repeats = max(1, round(repeats * target_tokens / observed))
    raise RuntimeError(
        f"could not calibrate {target_tokens} tokens within +-{tolerance}; last count was {observed}"
    )


def stream_read(response, dispatch_start: float) -> tuple[float, str, int, int]:
    first_visible = None
    visible: list[str] = []
    input_tokens = None
    output_tokens = None
    for raw in response:
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        event = json.loads(data)
        if event.get("type") == "error":
            error_type = event.get("error", {}).get("type", "unknown")
            raise RuntimeError(f"stream returned an error event: {error_type}")
        if event.get("type") == "message_start":
            usage = event.get("message", {}).get("usage", {})
            if isinstance(usage.get("input_tokens"), int):
                input_tokens = usage["input_tokens"]
            if isinstance(usage.get("output_tokens"), int):
                output_tokens = usage["output_tokens"]
        elif event.get("type") == "message_delta":
            usage = event.get("usage", {})
            if isinstance(usage.get("output_tokens"), int):
                output_tokens = usage["output_tokens"]
        elif event.get("type") == "content_block_delta":
            delta = event.get("delta", {})
            if delta.get("type") == "text_delta" and delta.get("text"):
                if first_visible is None:
                    first_visible = time.monotonic() - dispatch_start
                visible.append(delta["text"])
    if first_visible is None:
        raise RuntimeError("stream ended without a visible text delta")
    if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
        raise RuntimeError("stream did not report exact input_tokens and output_tokens usage")
    return first_visible, "".join(visible), input_tokens, output_tokens


def run_turn(host: str, model: str, session: str, prefix: str, turn: int,
             max_tokens: int, timeout: float) -> dict:
    url = host.rstrip("/") + "/v1/messages"
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "messages": build_messages(prefix, turn),
    }
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), method="POST", headers=request_headers(session)
    )
    dispatch_start = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status_code = int(response.status)
            if status_code != 200:
                raise RuntimeError(f"messages returned HTTP {status_code}")
            ttfvt, visible, input_tokens, output_tokens = stream_read(response, dispatch_start)
        wall = time.monotonic() - dispatch_start
        decode_window = wall - ttfvt
        if decode_window <= 0:
            raise RuntimeError("decode window is not positive")
        return {
            "status": "ok", "status_code": status_code, "error": None,
            "ttfvt": ttfvt, "wall": wall, "decode_window_s": decode_window,
            "tps_turn": output_tokens / wall,
            "tps_decode": output_tokens / decode_window,
            "visible_chars": len(visible), "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
    except urllib.error.HTTPError as exc:
        return {
            "status": "error", "status_code": int(exc.code),
            "error": f"HTTPError: {exc.reason}",
            "ttfvt": None,
            "wall": time.monotonic() - dispatch_start,
            "decode_window_s": None, "tps_turn": None, "tps_decode": None,
            "visible_chars": 0, "input_tokens": 0, "output_tokens": 0,
        }
    except Exception as exc:
        return {
            "status": "error", "status_code": 0,
            "error": f"{type(exc).__name__}: {exc}",
            "ttfvt": None,
            "wall": time.monotonic() - dispatch_start,
            "decode_window_s": None, "tps_turn": None, "tps_decode": None,
            "visible_chars": 0, "input_tokens": 0, "output_tokens": 0,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--model", default="deepseek-v4-flash-0731")
    parser.add_argument("--leg", required=True, help="configuration label stored in every record")
    parser.add_argument("--shapes", default="131072", help="comma-separated target token counts")
    parser.add_argument("--turns", type=int, default=6)
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--tolerance", type=int, default=128)
    parser.add_argument("--prewarm", action="store_true", help="run one discarded kernel warmup per shape")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if not os.environ.get("DSV4FLASH_API_KEY"):
        print("error: set DSV4FLASH_API_KEY", file=sys.stderr)
        return 2
    if args.turns < 2 or args.sessions < 1:
        print("error: turns must be at least 2 and sessions at least 1", file=sys.stderr)
        return 2

    try:
        shapes = [int(value) for value in args.shapes.split(",")]
    except ValueError:
        print("error: shapes must be comma-separated integers", file=sys.stderr)
        return 2
    if (not shapes or any(value <= 0 for value in shapes) or args.max_tokens <= 0
            or args.timeout <= 0 or args.tolerance < 0):
        print("error: shapes, max-tokens, and timeout must be positive; tolerance cannot be negative",
              file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    try:
        output_file = args.out.open("x")
    except FileExistsError:
        print(f"error: output already exists: {args.out}", file=sys.stderr)
        return 2
    with output_file as output:
        for shape in shapes:
            if args.prewarm:
                warm_session = f"prewarm_{args.leg}_{shape}"
                prefix, _ = calibrate_prefix(
                    args.host, args.model, shape, warm_session, args.timeout, args.tolerance
                )
                warm = run_turn(
                    args.host, args.model, warm_session, prefix, 0, args.max_tokens, args.timeout
                )
                if warm["status"] != "ok":
                    print(f"error: prewarm failed for shape {shape}: {warm['error']}", file=sys.stderr)
                    return 1
            for index in range(args.sessions):
                session = f"bench_{args.leg}_{shape}_{index}"
                try:
                    prefix, calibrated_tokens = calibrate_prefix(
                        args.host, args.model, shape, session, args.timeout, args.tolerance
                    )
                except Exception as exc:
                    print(f"error: calibration failed for {session}: {exc}", file=sys.stderr)
                    return 1
                for turn in range(1, args.turns + 1):
                    record = run_turn(
                        args.host, args.model, session, prefix, turn,
                        args.max_tokens, args.timeout,
                    )
                    record.update({
                        "leg": args.leg, "session": session, "turn": turn,
                        "shape_tok": shape, "target_shape_tok": shape,
                        "calibrated_input_tokens": calibrated_tokens,
                    })
                    output.write(json.dumps(record) + "\n")
                    output.flush()
                    if record["status"] != "ok":
                        failures += 1

    if failures:
        print(f"error: wrote {args.out} with {failures} failed request(s)", file=sys.stderr)
        return 1
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
