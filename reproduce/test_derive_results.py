import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("derive-results.py")
SPEC = importlib.util.spec_from_file_location("derive_results", MODULE_PATH)
derive_results = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(derive_results)


class MatchedEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.matched_dir = derive_results.MATCHED_DIR
        self.records = json.loads(
            (self.matched_dir / "raw" / "specoff.json").read_text()
        )

    def test_bundled_matched_result(self):
        result = derive_results.derive_matched()
        self.assertEqual(result["arms"]["specoff"]["n_ok"], 36)
        self.assertEqual(result["arms"]["k5"]["n_ok"], 36)
        self.assertEqual(
            result["comparison"]["128K"]["normalized_800_wall_p50_delta_pct"],
            -55.12,
        )
        self.assertEqual(
            result["comparison"]["256K"]["normalized_800_wall_p50_delta_pct"],
            -52.24,
        )

    def test_duplicate_turn_is_rejected(self):
        records = copy.deepcopy(self.records)
        records[-1] = copy.deepcopy(records[0])
        with self.assertRaisesRegex(ValueError, "duplicate cell"):
            derive_results.validate_matched_records(
                records, "specoff", Path("duplicate.json")
            )

    def test_missing_turn_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "expected 36 records"):
            derive_results.validate_matched_records(
                self.records[:-1], "specoff", Path("missing.json")
            )

    def test_failed_request_is_rejected(self):
        records = copy.deepcopy(self.records)
        records[0]["status"] = "error"
        with self.assertRaisesRegex(ValueError, "did not complete successfully"):
            derive_results.validate_matched_records(
                records, "specoff", Path("failed.json")
            )

    def test_control_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "raw").mkdir()
            for name in (
                "specoff.json",
                "k5.json",
                "engine-summary-specoff.json",
                "engine-summary-k5.json",
            ):
                (root / "raw" / name).write_bytes(
                    (self.matched_dir / "raw" / name).read_bytes()
                )
            manifest = json.loads((self.matched_dir / "manifest.json").read_text())
            manifest["k5"]["max_num_seqs"] = 255
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "matched control differs"):
                derive_results.derive_matched(root)


class OtherEvidenceTests(unittest.TestCase):
    def test_bundled_prefix_affinity_result(self):
        result = derive_results.derive_prefix_affinity()
        self.assertTrue(
            result["consistent_hash_gateway"]["128K-class"][
                "all_sessions_pinned_for_all_turns"
            ]
        )
        self.assertEqual(
            result["direct_rank_pin"]["serial"]["256K-class"][
                "late_median_ttft_s_range"
            ],
            [0.972, 1.037],
        )

    def test_benchmark_summary_schema(self):
        derive_results.validate_benchmark_summary(derive_results.BENCHMARK_SUMMARY)


if __name__ == "__main__":
    unittest.main()
