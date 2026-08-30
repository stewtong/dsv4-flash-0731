#!/usr/bin/env python3
"""Derive the published serving envelope from per-request JSONL records.

The default input is ``results/raw``. Use repeated ``--input`` arguments to
derive a new run. Validation fails on request errors, missing sessions, missing
turns, duplicate turns, or a schema mismatch. No file is written unless
``--json-out`` is supplied.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW = ROOT / "results" / "raw"
DEFAULT_LEGS = ("specoff", "cellK5-mnbt8192")
LABELS = {"specoff": "specoff", "cellK5-mnbt8192": "k5-mnbt8192"}
REQUIRED_FIELDS = {
    "leg", "session", "turn", "shape_tok", "status", "status_code",
    "ttfvt", "wall", "tps_decode", "tps_turn", "output_tokens",
}


def percentile(values: list[float], p: float) -> float:
    """Return a linear-interpolation percentile."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    if len(ordered) == 1:
        return ordered[0]
    index = p / 100.0 * (len(ordered) - 1)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def default_paths(raw_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for leg in DEFAULT_LEGS:
        paths.extend(Path(p) for p in sorted(glob.glob(str(raw_dir / f"{leg}-serial-*.jsonl"))))
    return paths


def load_records(paths: list[Path]) -> list[dict]:
    if not paths:
        raise ValueError("no JSONL inputs found")
    records: list[dict] = []
    for path in paths:
        if not path.is_file():
            raise ValueError(f"input does not exist: {path}")
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            missing = sorted(REQUIRED_FIELDS - set(record))
            if missing:
                raise ValueError(f"{path}:{lineno}: missing fields: {', '.join(missing)}")
            record["_source"] = f"{path}:{lineno}"
            records.append(record)
    if not records:
        raise ValueError("JSONL inputs contain no records")
    return records


def validate_cell(records: list[dict], expected_sessions: int, expected_turns: int) -> None:
    errors = [r for r in records if r["status"] != "ok" or r["status_code"] != 200]
    if errors:
        samples = ", ".join(r["_source"] for r in errors[:3])
        raise ValueError(f"cell contains {len(errors)} failed request(s): {samples}")

    for record in records:
        for field in ("ttfvt", "wall", "tps_decode", "tps_turn", "output_tokens"):
            try:
                value = float(record[field])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{record['_source']}: {field} is not numeric") from exc
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{record['_source']}: {field} must be finite and positive")
        if float(record["wall"]) <= float(record["ttfvt"]):
            raise ValueError(f"{record['_source']}: wall must exceed ttfvt")

    by_session: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_session[str(record["session"])].append(record)
    if len(by_session) != expected_sessions:
        raise ValueError(
            f"expected {expected_sessions} sessions, found {len(by_session)}: "
            + ", ".join(sorted(by_session))
        )

    expected = set(range(1, expected_turns + 1))
    for session, rows in sorted(by_session.items()):
        turns = [int(r["turn"]) for r in rows]
        if len(turns) != len(set(turns)):
            raise ValueError(f"session {session} has duplicate turns: {turns}")
        if set(turns) != expected:
            raise ValueError(f"session {session} turns {sorted(turns)} do not match {sorted(expected)}")


def summarize(records: list[dict], expected_sessions: int, expected_turns: int) -> dict:
    validate_cell(records, expected_sessions, expected_turns)
    warm = [r for r in records if int(r["turn"]) > 1]
    cold = [r for r in records if int(r["turn"]) == 1]
    stalls = sorted(float(r["ttfvt"]) for r in warm if float(r["ttfvt"]) > 2.0)
    return {
        "leg": str(records[0]["leg"]),
        "shape_tok": int(records[0]["shape_tok"]),
        "n_sessions": expected_sessions,
        "n_turns_total": len(records),
        "n_warm_turns": len(warm),
        "pct_ok": 100.0,
        "ttfvt_s_p50": round(percentile([float(r["ttfvt"]) for r in warm], 50), 4),
        "warm_stall_turns_s": [round(value, 3) for value in stalls],
        "tps_decode_p50": round(percentile([float(r["tps_decode"]) for r in warm], 50), 2),
        "tps_turn_p50": round(percentile([float(r["tps_turn"]) for r in warm], 50), 2),
        "wall_s_p50": round(percentile([float(r["wall"]) for r in warm], 50), 3),
        "cold_turn1_ttfvt_s_p50": round(percentile([float(r["ttfvt"]) for r in cold], 50), 4),
        "output_tokens_turn_median": round(
            percentile([float(r["output_tokens"]) for r in warm], 50), 1
        ),
        "n_errors": 0,
    }


def normalized_wall_ms(summary: dict, output_tokens: int = 800) -> float:
    """Return first-visible-token plus decode time, in milliseconds."""
    return 1000.0 * float(summary["ttfvt_s_p50"]) + output_tokens * (
        1000.0 / float(summary["tps_decode_p50"])
    )


def derive(records: list[dict], expected_sessions: int, expected_turns: int,
           baseline_leg: str, candidate_leg: str) -> dict:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for record in records:
        grouped[(str(record["leg"]), int(record["shape_tok"]))].append(record)

    results: dict = {"cells": {}}
    summaries: dict[tuple[str, int], dict] = {}
    for (leg, shape_tok), rows in sorted(grouped.items()):
        summary = summarize(rows, expected_sessions, expected_turns)
        summaries[(leg, shape_tok)] = summary
        label = LABELS.get(leg, leg)
        results["cells"][f"{label}:{shape_tok // 1024}k"] = summary

    baseline_shapes = {shape for leg, shape in summaries if leg == baseline_leg}
    candidate_shapes = {shape for leg, shape in summaries if leg == candidate_leg}
    if not baseline_shapes:
        raise ValueError(f"no records found for baseline leg: {baseline_leg}")
    if not candidate_shapes:
        raise ValueError(f"no records found for candidate leg: {candidate_leg}")
    if baseline_shapes != candidate_shapes:
        raise ValueError(
            f"baseline shapes {sorted(baseline_shapes)} do not match "
            f"candidate shapes {sorted(candidate_shapes)}"
        )

    results["normalized_wall"] = {}
    for shape_tok in sorted(baseline_shapes):
        baseline_ms = normalized_wall_ms(summaries[(baseline_leg, shape_tok)])
        candidate_ms = normalized_wall_ms(summaries[(candidate_leg, shape_tok)])
        reduction = 100.0 * (1.0 - candidate_ms / baseline_ms)
        results["normalized_wall"][f"{shape_tok // 1024}k"] = {
            "specoff_ms": round(baseline_ms, 1),
            "k5_ms": round(candidate_ms, 1),
            "reduction_pct": round(reduction, 1),
        }
    return results


def print_summary(results: dict) -> None:
    for key, cell in results["cells"].items():
        print(
            f"{key:22s} ttfvt_p50={cell['ttfvt_s_p50']:.4f}s "
            f"tps_decode_p50={cell['tps_decode_p50']:.2f} "
            f"wall_p50={cell['wall_s_p50']:.3f}s "
            f"warm_turns={cell['n_warm_turns']} errors={cell['n_errors']}"
        )
    for shape, row in results.get("normalized_wall", {}).items():
        print(
            f"{shape} normalized wall: off={row['specoff_ms']:.1f}ms "
            f"candidate={row['k5_ms']:.1f}ms reduction={row['reduction_pct']:.1f}%"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, help="JSONL input; repeat as needed")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--expected-sessions", type=int, default=3)
    parser.add_argument("--expected-turns", type=int, default=6)
    parser.add_argument("--baseline-leg", default="specoff")
    parser.add_argument("--candidate-leg", default="cellK5-mnbt8192")
    parser.add_argument("--json", action="store_true", help="print JSON to stdout")
    parser.add_argument("--json-out", type=Path, help="write JSON to this explicit path")
    parser.add_argument("--check-envelope", type=Path, help="compare derived JSON with this file")
    args = parser.parse_args()

    paths = args.input if args.input else default_paths(args.raw_dir)
    try:
        records = load_records(paths)
        results = derive(records, args.expected_sessions, args.expected_turns,
                         args.baseline_leg, args.candidate_leg)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    rendered = json.dumps(results, indent=2) + "\n"
    if args.check_envelope:
        try:
            expected = json.loads(args.check_envelope.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"error: cannot read envelope {args.check_envelope}: {exc}", file=sys.stderr)
            return 1
        if expected != results:
            print(f"error: derived result differs from {args.check_envelope}", file=sys.stderr)
            return 1
        print(f"verified {args.check_envelope}")
    elif args.json:
        sys.stdout.write(rendered)
    else:
        print_summary(results)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered)
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
