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


class ReportTests(unittest.TestCase):
    def test_report_status(self):
        self.assertEqual(N.report_status("consolidation", "Lungs are clear. There is no consolidation."), "negated")
        self.assertEqual(N.report_status("consolidation", "Patchy consolidation in the right lower lobe."), "affirmed")
        self.assertEqual(N.report_status("pleural_effusion", "Heart size normal.\\nLungs clear."), "none")


class CtrateReportTests(unittest.TestCase):
    NEG = ["No pleural effusion was detected in both hemithoraces.",
           "No pleural effusion or thickening was detected in both hemithoraces, and the parenchyma is clear.",
           "Pleural effusion was not observed.", "There is no pleural effusion, pneumothorax or consolidation.",
           "Pleural effusion/thickening is not observed.", "Heart size is normal; no pleural effusion is seen.",
           "No pericardial, pleural effusion or thickening was detected.", "Pericardial, pleural effusion or increase in thickness is not detected.",
           "Minimal pericardial effusion was observed.", "Pericardial effusion is present."]
    NEG2 = ["No significant mass lesion, infiltrative appearance, pneumothorax or pleural effusion were detected at other levels."]
    POS = ["In the previous examination, the effusion between the leaves of the left pleura disappeared, and the increase in bronchial wall thickness regressed.",
           "Minimal pleural effusion is observed in the right hemithorax.",
           "No mass lesion, pleural effusion is seen in the left hemithorax.",
           "Pleural thickening and pleural effusion are present.",
           "There is no pneumothorax but bilateral pleural effusion is present.",
           "Pleural effusion is observed, and no nodule is seen.", "Pleural effusion was observed in the right hemithorax and the nodule is not seen."]

    def test_negated(self):
        for t in self.NEG:
            self.assertEqual(N.report_status_ctrate("pleural_effusion", t), "negated", t)

    def test_affirmed(self):
        for t in self.POS:
            self.assertEqual(N.report_status_ctrate("pleural_effusion", t), "affirmed", t)
        self.assertIsNotNone(N.affirmed_clause("pleural_effusion", "Lungs are clear. Minimal pleural effusion is observed."))
        self.assertIsNone(N.affirmed_clause("pleural_effusion", "No pleural effusion was detected."))

    def test_negated_list_end(self):
        for t in self.NEG2:
            self.assertEqual(N.report_status_ctrate("pleural_effusion", t), "negated", t)

    def test_none(self):
        self.assertEqual(N.report_status_ctrate("pleural_effusion", "Lungs are clear."), "none")


class ConsolidationReportTests(unittest.TestCase):
    NEG = ["When examined in the lung parenchyma window; Ventilation of both lungs is natural and active infiltration, consolidation and space-occupying lesions are not detected in bilateral lungs.",
           "When examined in the lung parenchyma window; Ventilation of both lungs is normal and there is no active infiltration, consolidation or space-occupying lesion.",
           "When examined in the lung parenchyma window: Active infiltration, consolidation or space-occupying lesion was not observed in both lung parenchyma.",
           "In lung parenchyma evaluation; Pneumonic infiltration or consolidation area, suspicious mass or nodular lesion were not detected in the lung parenchyma.",
           "Both lung aerations are normal, and there is no active infiltrative, consolidation or space-occupying lesion area in the bilateral lungs.",
           "Aeration of both lung parenchyma is normal and no nodule, space-occupying lesion, active infiltration or consolidation is detected in the lung parenchyma.",
           "Ventilation is normal, and no space-occupying solid or cystic lesion, active infiltration or consolidation is detected in both lungs.",
           "No active infiltrative, space-occupying lesion with consolidation was detected in both lungs."]
    POS = ["In the case under follow-up, widespread consolidation areas are observed in the merging tendency.",
           "The consolidation area in the lower lobe of the left lung is stable.",
           "In its neighborhood, there is a consolidation area with a cystic necrotic area.",
           "Consolidation in the left lower lobe and the nodule is not seen."]

    def test_neg(self):
        for t in self.NEG:
            self.assertEqual(N.report_status_ctrate("consolidation", t), "negated", t)

    def test_pos(self):
        for t in self.POS:
            self.assertEqual(N.report_status_ctrate("consolidation", t), "affirmed", t)
