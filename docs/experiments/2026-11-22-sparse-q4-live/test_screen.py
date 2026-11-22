"""Controls for the predeclared sparse Q4 operator gates."""

import json
import tempfile
import unittest
from pathlib import Path

from screen import ARMS, Layer, Sample, decide, layer_result, paired


class SparseGateTests(unittest.TestCase):
    """Test paired reductions, noise rejection and independent task weighting."""

    def test_gain_must_clear_both_magnitude_and_noise(self) -> None:
        """Even a consistent improvement cannot qualify inside the repeat noise floor."""
        strong = [paired([10.]*5, [8.]*5, name) for name in ("a","b","c")]
        self.assertEqual(decide(strong)["decision"], "PASS")
        small = [paired([10.]*5, [9.5]*5, name) for name in ("a","b","c")]
        self.assertEqual(decide(small)["decision"], "FAIL")
        noisy = [paired([7.,8.,10.,12.,13.], [5.6,6.4,8.,9.6,10.4], name) for name in ("a","b","c")]
        self.assertEqual(decide(noisy)["decision"], "INCONCLUSIVE")

    def test_extra_positions_do_not_multiply_independent_tasks(self) -> None:
        """Eight favorable positions in one task cannot outweigh two losing tasks."""
        samples = [Sample(task="copy",gain=.50,noise=0.)]*8
        self.assertEqual(decide(samples)["decision"], "INCONCLUSIVE")
        samples.extend(Sample(task=task,gain=-.2,noise=0.) for task in ("code","recall"))
        result = decide(samples)
        self.assertEqual(result["tasks"],3)
        self.assertEqual(result["gain"],-.2)
        self.assertLess(result["lower"],0)
        self.assertNotEqual(result["decision"],"PASS")

    def test_synthetic_layer_requires_complete_rotated_pairs(self) -> None:
        """A complete miniature layer screens; a duplicate last interval invalidates it."""
        layer = Layer(layer=0, cutoffs=[0., 1., 2.],
                      rows=[{"task":task,"position":5,"token":7} for task in ("a","b","c")])
        records: list[dict[str, object]] = []
        for row in range(-1,3):
            for level in range(3):
                for index, arm in enumerate(ARMS):
                    records.append({"kind":"check","layer":0,"row":row,"level":level,
                        "arm":arm,"cutoff":float(level),"count":0 if row==-1 else 10-level,
                        "mass_fraction":1.,"matched_max_error":0.,"native_relative_rms":0.,
                        "native_cosine":1.,"layout_relative_rms":0.,"layout_cosine":1.})
                    if row >= 0:
                        for repetition in range(-1,5):
                            time = 8. if arm.startswith("threshold_") else 10.
                            records.append({"kind":"trial","layer":0,"row":row,"level":level,"arm":arm,
                                "repeats":20,"order":(index-row-level-repetition-1)%6,"round":repetition,
                                "event_us":time,"wall_us":time,"submit_us":1.})
        records.append({"kind":"complete","layer":0,"rows":3})
        with tempfile.TemporaryDirectory() as temp:
            report = Path(temp)/"synthetic.jsonl"
            report.write_text("\n".join(map(json.dumps,records)))
            decision = layer_result(layer,report)
            self.assertTrue(decision["comparisons"]["packed"]["passed"])
            self.assertTrue(decision["comparisons"]["zigzag"]["passed"])
            records.insert(-1, records[-2])
            report.write_text("\n".join(map(json.dumps,records)))
            with self.assertRaisesRegex(ValueError,"invalid trial"):
                layer_result(layer,report)
