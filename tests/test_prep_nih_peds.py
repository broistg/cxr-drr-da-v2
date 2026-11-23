import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("prep_nih_peds", ROOT / "scripts/prep_nih_peds.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)


class PrepTests(unittest.TestCase):
    def test_groups_and_folds_by_patient(self):
        self.assertEqual(P.sign_group("Effusion|Atelectasis", "Effusion"), "positive")
        self.assertEqual(P.sign_group("No Finding", "Effusion"), "negative")
        self.assertEqual(P.sign_group("Atelectasis", "Effusion"), "other")
        rows = [{"image": f"{p}_{i}.png", "labels": "Pneumothorax" if p % 3 == 0 else "No Finding", "patient_id": str(p),
                 "age": 10, "sex": "M", "view": "PA"} for p in range(30) for i in range(4)] + \
               [{"image": "old.png", "labels": "No Finding", "patient_id": "999", "age": 40, "sex": "M", "view": "PA"}]
        peds = P.select(rows, 18)
        self.assertEqual(len(peds), 120)
        man, folds = P.make_manifest(peds, 5, 0)
        per = {}
        for m in man:
            per.setdefault(m["patient_id"], set()).add(m["fold"])
        self.assertTrue(all(len(v) == 1 for v in per.values()))
        self.assertEqual(sum(len(v) for v in folds.values()), 120)
        self.assertIn("pneumothorax", P.report(man, 5, 18))


if __name__ == "__main__":
    unittest.main()
