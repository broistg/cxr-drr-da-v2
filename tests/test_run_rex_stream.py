import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("run_rex_stream", ROOT / "scripts/run_rex_stream.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


class StreamTests(unittest.TestCase):
    def test_sources_pending_and_cache(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            for stem, sign, cases in [("a_Rm", "a", ["c1", "c2"]), ("a_N", "a", ["c2", "c3"]), ("b_Rm", "b", ["c1"])]:
                (t / f"{stem}.yaml").write_text(yaml.safe_dump({"sign": sign, "roles": {c: "positive" for c in cases}}))
            jobs = S.load_sources(["a_Rm", "a_N", "b_Rm"], t)
            self.assertEqual(jobs, {"c1": ["a", "b"], "c2": ["a"], "c3": ["a"]})
            out = t / "drr"
            self.assertEqual(S.pending(jobs, out, ["moderate"], ["ap"]), jobs)
            full = {f"{k}_{n}": np.zeros((1, 2, 2), np.float32) for k in "M R rot trans".split() for n in ("normal", "positive")}
            f = S.cache_files(out, "a", "c1", ["moderate"], ["ap"])[0]
            f.parent.mkdir(parents=True)
            np.savez(f, **full)
            self.assertTrue(S.cache_ok(f))
            self.assertEqual(S.pending(jobs, out, ["moderate"], ["ap"]), {"c1": ["b"], "c2": ["a"], "c3": ["a"]})
            bad = dict(full, M_normal=np.full((1, 2, 2), np.nan, np.float32))
            np.savez(f, **bad)
            self.assertFalse(S.cache_ok(f))
            self.assertFalse(S.cache_ok(t / "missing.npz"))

    def test_batches(self):
        self.assertEqual(S.batches([1, 2, 3, 4, 5], 2), [[1, 2], [3, 4], [5]])


if __name__ == "__main__":
    unittest.main()
