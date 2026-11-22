import unittest

from drr_pipeline import ctrate_normals as N


def table(zero_all, one_in):
    h = ["VolumeName"] + N.ALL_LABELS
    rows = []
    for name, ones in (("a", []), ("b", ["Hiatal hernia"]), ("c", ["Consolidation"]), ("d", None)):
        r = {c: "0" for c in N.ALL_LABELS}
        r["VolumeName"] = name
        for o in ones or []:
            r[o] = "1"
        if ones is None:
            r["Lung nodule"] = ""
        rows.append(r)
    return rows, h


class NormalsTests(unittest.TestCase):
    def test_tiers(self):
        rows, h = table(0, 0)
        c = N.count_tiers(rows, h)
        self.assertEqual(c["strict"], ["a"])
        self.assertEqual(c["relevant"], ["a", "b"])       # hiatal hernia không liên quan X-quang; ô trống không tính là 0
        self.assertEqual(c["strict_missing_cols"], [])

    def test_missing_columns_not_counted(self):
        rows, h = table(0, 0)
        c = N.count_tiers(rows, [x for x in h if x != "Emphysema"])
        self.assertEqual((c["strict"], c["relevant"]), ([], []))
        self.assertEqual(c["strict_missing_cols"], ["Emphysema"])

    def test_match(self):
        f, miss = N.match_columns(["lung_nodule", "CONSOLIDATION"], ["Lung nodule", "Consolidation", "Emphysema"])
        self.assertEqual(miss, ["Emphysema"])
        self.assertEqual(f["Lung nodule"], "lung_nodule")


if __name__ == "__main__":
    unittest.main()
