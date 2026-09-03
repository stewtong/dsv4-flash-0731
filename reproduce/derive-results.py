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
MATCHED_DIR = ROOT / "results" / "dspark-matched"
PREFIX_DIR = ROOT / "results" / "prefix-affinity"
BENCHMARK_SUMMARY = ROOT / "results" / "benchmark-summary.json"
REQUIRED_FIELDS = {
    "leg", "session", "turn", "shape_tok", "status", "status_code",
    "ttfvt", "wall", "tps_decode", "tps_turn", "output_tokens",
}
MATCHED_REQUIRED_FIELDS = {
    "arm", "context", "shape_tok", "session_index", "turn", "status",
    "status_code", "input_tokens", "output_tokens", "ttfvt_s", "wall_s",
    "decode_window_s", "tps_turn", "tps_decode", "stop_reason",
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


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc


def finite_positive(value, source: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} must be numeric") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{source} must be finite and positive")
    return number


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("mean requires at least one value")
    return sum(values) / len(values)


def rounded_range(values: list[float], digits: int = 4) -> list[float]:
    return [round(min(values), digits), round(max(values), digits)]


def validate_matched_records(records: list[dict], arm: str, path: Path) -> None:
    if not isinstance(records, list) or len(records) != 36:
        raise ValueError(f"{path}: expected 36 records, found {len(records) if isinstance(records, list) else 'non-list'}")
    expected = {
        (context, session, turn)
        for context in ("128K", "256K")
        for session in range(3)
        for turn in range(1, 7)
    }
    observed = set()
    for index, record in enumerate(records):
        source = f"{path}[{index}]"
        if not isinstance(record, dict):
            raise ValueError(f"{source}: record must be an object")
        missing = sorted(MATCHED_REQUIRED_FIELDS - set(record))
        if missing:
            raise ValueError(f"{source}: missing fields: {', '.join(missing)}")
        if record["arm"] != arm:
            raise ValueError(f"{source}: arm must be {arm}")
        key = (record["context"], int(record["session_index"]), int(record["turn"]))
        if key in observed:
            raise ValueError(f"{source}: duplicate cell {key}")
        observed.add(key)
        if record["status"] != "ok" or int(record["status_code"]) != 200:
            raise ValueError(f"{source}: request did not complete successfully")
        for field in ("input_tokens", "output_tokens", "ttfvt_s", "wall_s",
                      "decode_window_s", "tps_turn", "tps_decode"):
            finite_positive(record[field], f"{source}.{field}")
        if float(record["wall_s"]) <= float(record["ttfvt_s"]):
            raise ValueError(f"{source}: wall_s must exceed ttfvt_s")
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"{path}: cell grid mismatch; missing={missing}, extra={extra}")


def matched_arm_summary(records: list[dict]) -> dict:
    contexts = {}
    for context in ("128K", "256K"):
        rows = [row for row in records if row["context"] == context]
        cold = [row for row in rows if int(row["turn"]) == 1]
        warm = [row for row in rows if int(row["turn"]) > 1]
        normalized = [float(row["ttfvt_s"]) + 800.0 / float(row["tps_decode"])
                      for row in warm]
        wall = [float(row["wall_s"]) for row in warm]
        ttfvt = [float(row["ttfvt_s"]) for row in warm]
        decode = [float(row["tps_decode"]) for row in warm]
        output = [float(row["output_tokens"]) for row in warm]
        session_means = []
        for session_index in range(3):
            session_rows = [row for row in warm if int(row["session_index"]) == session_index]
            session_values = [float(row["ttfvt_s"]) + 800.0 / float(row["tps_decode"])
                              for row in session_rows]
            session_means.append({
                "session_index": session_index,
                "n_warm_turns": len(session_values),
                "normalized_800_wall_mean_s": round(mean(session_values), 4),
            })
        contexts[context] = {
            "n_records": len(rows),
            "n_cold_turns": len(cold),
            "n_warm_turns": len(warm),
            "cold_turn1_ttfvt_mean_s": round(mean([float(row["ttfvt_s"]) for row in cold]), 4),
            "cold_turn1_input_tokens": [int(row["input_tokens"]) for row in cold],
            "normalized_800_wall_s": {
                "p50": round(percentile(normalized, 50), 4),
                "mean": round(mean(normalized), 4),
                "range": rounded_range(normalized),
            },
            "raw_wall_s": {
                "p50": round(percentile(wall, 50), 4),
                "mean": round(mean(wall), 4),
                "range": rounded_range(wall),
            },
            "ttfvt_s": {
                "p50": round(percentile(ttfvt, 50), 4),
                "mean": round(mean(ttfvt), 4),
                "range": rounded_range(ttfvt),
            },
            "decode_tokens_per_second": {
                "p50": round(percentile(decode, 50), 2),
                "mean": round(mean(decode), 2),
                "range": rounded_range(decode, 2),
            },
            "output_tokens": {
                "mean": round(mean(output), 1),
                "range": [int(min(output)), int(max(output))],
            },
            "warm_stalls_ttfvt_over_3s": sum(value > 3.0 for value in ttfvt),
            "session_normalized_800_wall": session_means,
        }
    return {
        "n_records": len(records),
        "n_ok": sum(row["status"] == "ok" and int(row["status_code"]) == 200 for row in records),
        "n_errors": sum(row["status"] != "ok" or int(row["status_code"]) != 200 for row in records),
        "contexts": contexts,
    }


def derive_matched(matched_dir: Path = MATCHED_DIR) -> dict:
    spec_path = matched_dir / "raw" / "specoff.json"
    k5_path = matched_dir / "raw" / "k5.json"
    spec_records = load_json(spec_path)
    k5_records = load_json(k5_path)
    validate_matched_records(spec_records, "specoff", spec_path)
    validate_matched_records(k5_records, "k5", k5_path)

    spec_inputs = {
        (row["context"], int(row["session_index"]), int(row["turn"])): int(row["input_tokens"])
        for row in spec_records
    }
    k5_inputs = {
        (row["context"], int(row["session_index"]), int(row["turn"])): int(row["input_tokens"])
        for row in k5_records
    }
    if spec_inputs != k5_inputs:
        raise ValueError("matched records do not have identical input-token counts")

    manifest = load_json(matched_dir / "manifest.json")
    if not isinstance(manifest, dict) or set(manifest) != {"specoff", "k5"}:
        raise ValueError("matched manifest must contain specoff and k5")
    allowed_differences = {"arm", "speculative_config", "kv_capacity_tokens_per_rank"}
    common_keys = set(manifest["specoff"]) | set(manifest["k5"])
    for key in sorted(common_keys - allowed_differences):
        if manifest["specoff"].get(key) != manifest["k5"].get(key):
            raise ValueError(f"matched control differs between arms: {key}")

    arms = {
        "specoff": matched_arm_summary(spec_records),
        "k5": matched_arm_summary(k5_records),
    }
    comparison = {}
    for context in ("128K", "256K"):
        off = arms["specoff"]["contexts"][context]
        k5 = arms["k5"]["contexts"][context]
        off_warm = [row for row in spec_records
                    if row["context"] == context and int(row["turn"]) > 1]
        k5_warm = [row for row in k5_records
                   if row["context"] == context and int(row["turn"]) > 1]
        def raw_delta(baseline: float, candidate: float) -> float:
            return round(100.0 * (candidate - baseline) / baseline, 2)
        off_normalized = [float(row["ttfvt_s"]) + 800.0 / float(row["tps_decode"])
                          for row in off_warm]
        k5_normalized = [float(row["ttfvt_s"]) + 800.0 / float(row["tps_decode"])
                         for row in k5_warm]
        session_deltas = []
        for off_session, k5_session in zip(
            off["session_normalized_800_wall"], k5["session_normalized_800_wall"]
        ):
            baseline = off_session["normalized_800_wall_mean_s"]
            candidate = k5_session["normalized_800_wall_mean_s"]
            session_deltas.append({
                "session_index": off_session["session_index"],
                "delta_pct": round(100.0 * (candidate - baseline) / baseline, 2),
            })
        comparison[context] = {
            "normalized_800_wall_p50_delta_pct": raw_delta(
                percentile(off_normalized, 50), percentile(k5_normalized, 50)),
            "normalized_800_wall_mean_delta_pct": raw_delta(
                mean(off_normalized), mean(k5_normalized)),
            "decode_rate_p50_delta_pct": raw_delta(
                percentile([float(row["tps_decode"]) for row in off_warm], 50),
                percentile([float(row["tps_decode"]) for row in k5_warm], 50)),
            "ttfvt_p50_delta_pct": raw_delta(
                percentile([float(row["ttfvt_s"]) for row in off_warm], 50),
                percentile([float(row["ttfvt_s"]) for row in k5_warm], 50)),
            "session_normalized_800_wall_mean_deltas_pct": session_deltas,
        }

    stalls = [
        {
            "arm": row["arm"],
            "context": row["context"],
            "session_index": int(row["session_index"]),
            "turn": int(row["turn"]),
            "ttfvt_s": round(float(row["ttfvt_s"]), 4),
        }
        for row in spec_records + k5_records
        if int(row["turn"]) > 1 and float(row["ttfvt_s"]) > 3.0
    ]

    spec_engine = load_json(matched_dir / "raw" / "engine-summary-specoff.json")
    k5_engine = load_json(matched_dir / "raw" / "engine-summary-k5.json")
    for expected_arm, engine in (("specoff", spec_engine), ("k5", k5_engine)):
        if engine.get("schema_version") != "dsv4flash-engine-counters/v1":
            raise ValueError(f"{expected_arm} engine summary has the wrong schema")
        if engine.get("arm") != expected_arm:
            raise ValueError(f"{expected_arm} engine summary has the wrong arm")
        if int(engine.get("n_errors", -1)) != 0 or engine.get("errors"):
            raise ValueError(f"{expected_arm} engine summary reports errors")
    if spec_engine.get("spec_decode_counter_delta"):
        raise ValueError("speculation-off engine reports speculative counter movement")
    acceptance = k5_engine.get("acceptance_by_position")
    if not isinstance(acceptance, dict) or set(acceptance) != {"0", "1", "2", "3", "4"}:
        raise ValueError("k5 engine summary lacks five-position acceptance evidence")

    return {
        "schema_version": "dsv4flash-dspark-matched/v1",
        "method": {
            "contexts": ["128K", "256K"],
            "sessions_per_context": 3,
            "turns_per_session": 6,
            "cold_turn": 1,
            "warm_turns": [2, 3, 4, 5, 6],
            "warm_observations_per_arm_and_context": 15,
            "normalized_output_tokens": 800,
            "normalized_wall_formula": "ttfvt_s + 800 / tps_decode",
            "comparison_delta_formula": "100 * (k5 - specoff) / specoff",
            "pooled_warm_values_are_descriptive": True,
            "raw_wall_comparable_between_arms": False,
        },
        "controls": {
            "common": {
                key: manifest["specoff"][key]
                for key in sorted(set(manifest["specoff"]) - allowed_differences)
            },
            "specoff": {
                "speculative_config": manifest["specoff"]["speculative_config"],
                "kv_capacity_tokens_per_rank": manifest["specoff"]["kv_capacity_tokens_per_rank"],
            },
            "k5": {
                "speculative_config": manifest["k5"]["speculative_config"],
                "kv_capacity_tokens_per_rank": manifest["k5"]["kv_capacity_tokens_per_rank"],
            },
        },
        "arms": arms,
        "comparison": comparison,
        "stalls": stalls,
        "k5_acceptance_by_position_tokens": {key: int(value) for key, value in acceptance.items()},
        "speculation_counters": {"specoff_moved": False, "k5_moved": True},
    }


def derive_prefix_affinity(prefix_dir: Path = PREFIX_DIR) -> dict:
    manifest = load_json(prefix_dir / "manifest.json")
    direct_files = [
        prefix_dir / "raw" / f"dp8-rank{rank}-{mode}.json"
        for mode in ("serial", "concurrent")
        for rank in (0, 1)
    ]
    direct = [load_json(path) for path in direct_files]
    for path, record in zip(direct_files, direct):
        if record.get("schema_version") != "dsv4flash-prefix-affinity-records/v1":
            raise ValueError(f"{path}: wrong schema_version")
        if int(record.get("n_errors", -1)) != 0:
            raise ValueError(f"{path}: errors are not allowed")
        rows = record.get("sessions_summary")
        expected_rows = int(record.get("sessions_per_shape", -1)) * 2
        if not isinstance(rows, list) or len(rows) != expected_rows:
            raise ValueError(f"{path}: expected {expected_rows} session summaries")
        if any(int(row.get("n_turns", 0)) != 6 or int(row.get("ok", 0)) != 6 for row in rows):
            raise ValueError(f"{path}: incomplete session")

    gateway_path = prefix_dir / "raw" / "gateway-sessions.json"
    gateway = load_json(gateway_path)
    if gateway.get("schema_version") != "dsv4flash-prefix-affinity-gateway/v1":
        raise ValueError(f"{gateway_path}: wrong schema_version")
    gateway_rows = gateway.get("rows")
    if not isinstance(gateway_rows, list) or len(gateway_rows) != 6:
        raise ValueError(f"{gateway_path}: expected six sessions")
    for row in gateway_rows:
        if not row.get("all_turns_same_rank") or int(row.get("turns_on_serving_rank", 0)) != 6:
            raise ValueError(f"{gateway_path}: session was not pinned for all six turns")
        if not isinstance(row.get("ttfvt_s_by_turn"), list) or len(row["ttfvt_s_by_turn"]) != 6:
            raise ValueError(f"{gateway_path}: expected six per-turn values")

    def direct_context(mode: str, context: str) -> dict:
        rows = [row for item in direct if item["mode"] == mode
                for row in item["sessions_summary"] if row["context"] == context]
        late = [finite_positive(row["late_median_ttft_s"], "late_median_ttft_s") for row in rows]
        warm = [float(row["warm_delta_pct"]) for row in rows]
        return {
            "n_sessions": len(rows),
            "late_median_ttft_s_range": rounded_range(late, 3),
            "warm_delta_pct_range": rounded_range(warm, 1),
        }

    gateway_summary = {}
    for context in ("128K-class", "256K-class"):
        rows = [row for row in gateway_rows if row["context"] == context]
        late = [finite_positive(row["late_median_ttft_s"], "late_median_ttft_s") for row in rows]
        warm = [float(row["warm_delta_pct"]) for row in rows]
        prompts = [int(row["prompt_tokens"]) for row in rows]
        gateway_summary[context] = {
            "n_sessions": len(rows),
            "prompt_tokens_range": [min(prompts), max(prompts)],
            "late_median_ttft_s_range": rounded_range(late, 3),
            "warm_delta_pct_range": rounded_range(warm, 1),
            "all_sessions_pinned_for_all_turns": True,
        }

    return {
        "schema_version": "dsv4flash-prefix-affinity-summary/v1",
        "configuration": manifest,
        "direct_rank_pin": {
            "serial": {context: direct_context("serial", context)
                       for context in ("128K-class", "256K-class")},
            "concurrent": {context: direct_context("concurrent", context)
                           for context in ("128K-class", "256K-class")},
        },
        "consistent_hash_gateway": gateway_summary,
    }


def check_json(expected_path: Path, actual: dict) -> None:
    expected = load_json(expected_path)
    if expected != actual:
        raise ValueError(f"derived result differs from {expected_path}")


def validate_benchmark_summary(path: Path = BENCHMARK_SUMMARY) -> None:
    summary = load_json(path)
    if summary.get("schema_version") != "dsv4flash-benchmark-summary/v1":
        raise ValueError(f"{path}: wrong schema_version")
    required = {"cold_topology", "batched_token_budget", "dspark_widths", "tp4_challenger"}
    if set(summary.get("campaigns", {})) != required:
        raise ValueError(f"{path}: campaign set does not match {sorted(required)}")
    for name, campaign in summary["campaigns"].items():
        if not campaign.get("source_url") and not campaign.get("source_boundary"):
            raise ValueError(f"{path}: campaign {name} lacks a source boundary")


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
    parser.add_argument("--matched-json-out", type=Path,
                        help="write the matched DSpark summary to this explicit path")
    parser.add_argument("--prefix-json-out", type=Path,
                        help="write the prefix-affinity summary to this explicit path")
    parser.add_argument("--check-matched", action="store_true",
                        help="verify results/dspark-matched/summary.json")
    parser.add_argument("--check-prefix-affinity", action="store_true",
                        help="verify results/prefix-affinity/summary.json")
    parser.add_argument("--check-benchmark-summary", action="store_true",
                        help="validate results/benchmark-summary.json")
    parser.add_argument("--check-all", action="store_true",
                        help="verify every bundled result and manifest schema")
    args = parser.parse_args()

    if any((args.matched_json_out, args.prefix_json_out, args.check_matched,
            args.check_prefix_affinity, args.check_benchmark_summary, args.check_all)):
        try:
            if args.matched_json_out or args.check_matched or args.check_all:
                matched = derive_matched()
                if args.matched_json_out:
                    args.matched_json_out.parent.mkdir(parents=True, exist_ok=True)
                    args.matched_json_out.write_text(json.dumps(matched, indent=2) + "\n")
                    print(f"wrote {args.matched_json_out}")
                if args.check_matched or args.check_all:
                    check_json(MATCHED_DIR / "summary.json", matched)
                    print(f"verified {MATCHED_DIR / 'summary.json'}")
            if args.prefix_json_out or args.check_prefix_affinity or args.check_all:
                prefix = derive_prefix_affinity()
                if args.prefix_json_out:
                    args.prefix_json_out.parent.mkdir(parents=True, exist_ok=True)
                    args.prefix_json_out.write_text(json.dumps(prefix, indent=2) + "\n")
                    print(f"wrote {args.prefix_json_out}")
                if args.check_prefix_affinity or args.check_all:
                    check_json(PREFIX_DIR / "summary.json", prefix)
                    print(f"verified {PREFIX_DIR / 'summary.json'}")
            if args.check_benchmark_summary or args.check_all:
                validate_benchmark_summary()
                print(f"validated {BENCHMARK_SUMMARY}")
            if args.check_all:
                legacy_paths = default_paths(DEFAULT_RAW)
                legacy = derive(load_records(legacy_paths), args.expected_sessions,
                                args.expected_turns, args.baseline_leg, args.candidate_leg)
                check_json(ROOT / "results" / "serving-envelope.json", legacy)
                print(f"verified {ROOT / 'results' / 'serving-envelope.json'}")
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0

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
