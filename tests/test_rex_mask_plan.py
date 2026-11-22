import unittest

from scripts.plan_rex_masks import build_plan, classify_2b, finding_matches


class RexMaskPlanTests(unittest.TestCase):
    def test_category_alone_does_not_decide_finding(self):
        self.assertFalse(finding_matches("2g", "Subpleural bleb"))
        self.assertTrue(finding_matches("2g", "Minimal air in the left pleural space"))
        self.assertFalse(finding_matches("2e", "Pleural thickening adjacent to an effusion"))
        self.assertTrue(finding_matches("2e", "Bilateral pleural effusion persists without significant change"))
        self.assertFalse(finding_matches("2b", "Cavitary lesion"))
        self.assertTrue(finding_matches("2b", "Left lower lobe atelectasis"))

    def test_2b_subtypes_keep_unclear_descriptions_for_review(self):
        cases = {
            "Left lower lobe atelectatic changes": ("atelectasis", None),
            "Collapse of the right middle lobe": ("atelectasis", None),
            "Lobar consolidation with an air bronchogram": ("consolidation", None),
            "Consolidation with compressive atelectasis": (None, "both_subtypes"),
            "Nodule or nodular consolidation": (None, "uncertain_or_alternative"),
            "Loss of volume suggestive of passive atelectasis": (None, "uncertain_or_alternative"),
            "Parenchymal opacity": (None, "no_explicit_subtype"),
        }
        for finding, result in cases.items():
            with self.subTest(finding=finding):
                self.assertEqual(classify_2b(finding), result)

    def test_keep_original_channel_id_and_report_only_heart_evidence(self):
        name = "train_1_a_1.nii.gz"
        data = {"train": [{"name": name, "shape": [3, 4, 5],
                           "categories": {"0": "2g", "1": "2b", "2": "2b", "3": "2b"},
                           "findings": {"0": "Bleb", "1": "Lobar consolidation", "2": "Lobar atelectasis",
                                        "3": "Atelectasis or consolidation"},
                           "pixels": {"0": 20, "1": 10, "2": 11, "3": 12}}]}
        reports = {"train": {"train_1_a.nii.gz": {
            "Findings": {"Heart/Vessels": ["The heart is enlarged."]}}}}
        index = [{"rfilename": f"segmentations/{name}", "size": 120, "lfs": {"sha256": "fixture"}}]
        plan = build_plan(data, reports, index, "fixture")
        self.assertEqual([(c["finding_id"], c["sign"], c["category"]) for c in plan["jobs"][0]["channels"]],
                         [(1, "consolidation", "2b"), (2, "atelectasis", "2b")])
        self.assertEqual(plan["review_findings"][0]["finding_id"], 0)
        self.assertEqual(plan["review_findings"][1]["review_reason"], "both_subtypes")
        self.assertFalse(plan["cardiomegaly_candidates"][0]["mask_available"])
        missing = build_plan(data, reports, [], "fixture")
        self.assertEqual(missing["jobs"], [])
        self.assertEqual(missing["unavailable_masks"][0]["volume_name"], name)
