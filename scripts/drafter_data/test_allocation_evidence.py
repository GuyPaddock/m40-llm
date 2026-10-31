"""Serialized optimizer artifacts must survive an independent count audit."""

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from audit_allocation_evidence import audit_evidence
from cp_sat_allocation import allocate
from prepare_v2 import file_hash
from quota_preflight import Component, Requirements


class EvidenceTests(unittest.TestCase):
    """Verify actual outputs and reject corrupted counts and identity bindings."""

    def test_roundtrip_and_corruption_rejection(self) -> None:
        """Independent audit accepts the canonical witness but rejects changed counts."""
        groups = [Component("g0", 0, (0, 1), (2, 0, 0)), Component("g1", 0, (0, 1), (1, 0, 0))]
        required = Requirements(((2, 1, 0),), (0,), (0, 0, 0))
        with tempfile.TemporaryDirectory() as directory:
            problem, result = Path(directory) / "problem.json", Path(directory) / "result.json"
            problem.write_text(json.dumps({"groups": [asdict(g) for g in groups], "requirements": asdict(required)}))
            payload = json.loads(json.dumps({"problem_sha256": file_hash(problem), "result": asdict(allocate(groups, required))}))
            result.write_text(json.dumps(payload))
            audit = audit_evidence(problem, result)
            self.assertTrue(audit["is_witness_feasible"])
            self.assertEqual(audit["canonical_check"], "exhaustive reference")
            payload["result"]["witness"]["counts"][0][0] = 999
            result.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                audit_evidence(problem, result)
            payload["problem_sha256"] = "0" * 64
            result.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "identity"):
                audit_evidence(problem, result)


if __name__ == "__main__":
    unittest.main()
