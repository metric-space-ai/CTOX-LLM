import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_vision_plan import align
from plan_vision_residency import choose_evictions, source_bundles


class VisionResidencyTests(unittest.TestCase):
    def setUp(self):
        self.plan = {
            "format": "ctox.q2q4.quant-plan.v1",
            "total_bytes": 14_000,
            "tensors": [
                {"name": "text.a", "group": "text", "source_shard": "a", "offset": 0, "length": 1_000},
                {"name": "text.a.s_in", "group": "recovery", "source_shard": None, "offset": 1_000, "length": 100},
                {"name": "text.b", "group": "text", "source_shard": "a", "offset": 2_000, "length": 3_000},
                {"name": "mtp.a", "group": "mtp", "source_shard": "b", "offset": 6_000, "length": 4_000},
                {"name": "mtp.a.s_in", "group": "recovery", "source_shard": None, "offset": 10_000, "length": 100},
                {"name": "mtp.b", "group": "mtp", "source_shard": "b", "offset": 11_000, "length": 2_000},
            ],
        }

    def test_alignment_rejects_invalid_values(self):
        self.assertEqual(align(65, 64), 128)
        with self.assertRaises(ValueError):
            align(-1)
        with self.assertRaises(ValueError):
            align(1, 3)

    def test_source_bundle_includes_recovery_and_alignment(self):
        self.assertEqual(
            source_bundles(self.plan, "text"),
            [
                {"name": "text.a", "offset": 0, "bytes": 2_000},
                {"name": "text.b", "offset": 2_000, "bytes": 4_000},
            ],
        )

    def test_evicts_mtp_then_smallest_sufficient_text_bundle(self):
        result = choose_evictions(self.plan, vision_bytes=10_001, steady_total_bytes=20_000, target_bytes=20_000)
        self.assertEqual(result["required_eviction_bytes"], 10_001)
        self.assertEqual(result["selected"][0]["bytes"], 8_000)
        self.assertEqual(result["selected"][1]["name"], "text.b")
        self.assertEqual(result["projected_bytes"], 18_001)


if __name__ == "__main__":
    unittest.main()
