import unittest

import numpy as np

from drr_pipeline import cohort_balance as CB


def rows(n, age, mfr, seed):
    rng = np.random.default_rng(seed)
    return [{"PatientAge": f"{int(rng.normal(age, 5)):03d}Y", "PatientSex": "F" if rng.random() < .5 else "M", "Manufacturer": mfr,
             "ReconstructionDiameter": "320", "XYSpacing": "[0.65, 0.65]", "ZSpacing": "1", "NumberofSlices": "250"} for _ in range(n)]


class BalanceTests(unittest.TestCase):
    def test_age_parse(self):
        self.assertEqual(CB.age_years("039Y"), 39.0)
        self.assertTrue(np.isnan(CB.age_years("")))

    def test_shortcut_detects_shift_and_balance(self):
        g = {"pos": rows(300, 55, "Philips", 0), "same": rows(300, 55, "Philips", 1), "young": rows(300, 32, "Siemens", 2)}
        _, X = CB.design(g)
        self.assertLess(CB.shortcut_auroc(X["pos"], X["same"]), 0.6)
        self.assertGreater(CB.shortcut_auroc(X["pos"], X["young"]), 0.9)
        self.assertGreater(abs(CB.smd(X["pos"][:, 0], X["young"][:, 0])), 1.0)


if __name__ == "__main__":
    unittest.main()


class MatchTests(unittest.TestCase):
    def test_matching_removes_age_shift(self):
        pos = {f"p{i}": r for i, r in enumerate(rows(100, 57, "Philips", 0))}
        cand = {f"c{i}": r for i, r in enumerate(rows(100, 57, "Philips", 1) + rows(400, 32, "Philips", 2))}
        m, un = CB.match_negatives(pos, cand, seed=0)
        self.assertEqual((len(m), len(un), len(set(m.values()))), (100, 0, 100))
        g = {"pos": list(pos.values()), "neg": [cand[c] for c in m.values()], "rand": list(cand.values())[:100] + list(cand.values())[100:200]}
        _, X = CB.design(g)
        self.assertLess(CB.shortcut_auroc(X["pos"], X["neg"]), 0.65)
        self.assertGreater(CB.shortcut_auroc(X["pos"], X["rand"][:100] if False else X["rand"]), 0.6)

    def test_caliper(self):
        pos = {"p": rows(1, 80, "Philips", 0)[0]}
        cand = {"c": rows(1, 30, "Philips", 1)[0]}
        m, un = CB.match_negatives(pos, cand, age_caliper=5)
        self.assertEqual((m, un), ({}, ["p"]))
