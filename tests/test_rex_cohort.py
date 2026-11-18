import unittest

from drr_pipeline.rex_cohort import build_rows, isolate_holdout_patients, report_vote, summarize


class RexCohortTests(unittest.TestCase):
    def test_only_explicit_negatives(self):
        self.assertEqual(report_vote("pneumothorax", ["No pneumothorax is detected."])[0], "0")
        self.assertEqual(report_vote("pneumothorax", ["No pleural effusion is detected."])[0], "unknown")
        self.assertEqual(report_vote("pneumothorax", ["No effusion, but a small pneumothorax is present."])[0], "unknown")
        self.assertEqual(report_vote("atelectasis_consolidation", ["No consolidation is seen."])[0], "unknown")

    def test_heart_rules_do_not_label_vessel_enlargement(self):
        self.assertEqual(report_vote("cardiomegaly", ["The heart is enlarged."])[0], "1")
        self.assertEqual(report_vote("cardiomegaly", ["Heart contour and size are normal."])[0], "0")
        self.assertEqual(report_vote("cardiomegaly", ["The aorta is enlarged."])[0], "unknown")
        self.assertEqual(report_vote("cardiomegaly", ["Possible cardiomegaly."])[0], "unknown")
        self.assertEqual(report_vote("cardiomegaly", ["The heart is enlarged.", "Heart size is normal."])[0], "unknown")

    def test_conflict_missing_annotations_and_thickening(self):
        data = {"train": [{"name": "train_1_a_1.nii.gz", "categories": {"0": "2g", "1": "2e"},
                           "findings": {"0": "Right pneumothorax", "1": "Pleural thickening"}}]}
        reports = {"train": {"train_1_a.nii.gz": {"Findings": {
            "Lungs/Airways/Pleura": ["No pneumothorax is detected."], "Heart/Vessels": []}}}}
        row = build_rows(data, reports)[0]
        self.assertEqual(row["pneumothorax"], "needs_review")
        self.assertEqual(row["pneumothorax_review"], "conflict")
        self.assertEqual(row["pleural_effusion"], "unannotated")
        self.assertEqual(row["atelectasis_consolidation"], "unannotated")

    def test_negative_report_never_becomes_exported_negative_label(self):
        data = {"train": [{"name": "train_1_a_1.nii.gz", "categories": {}, "findings": {}}]}
        reports = {"train": {"train_1_a.nii.gz": {"Findings": {
            "Lungs/Airways/Pleura": ["No pneumothorax is detected."],
            "Heart/Vessels": ["Heart size is normal."]}}}}
        row = build_rows(data, reports)[0]
        self.assertEqual(row["pneumothorax"], "unannotated")
        self.assertEqual(row["cardiomegaly"], "unannotated")
        self.assertEqual(row["cardiomegaly_source"], "unknown")
        self.assertFalse(summarize([row])["binary_labels_exported"])

    def test_patient_namespaces_and_recon_grouping(self):
        data = {"train": [{"name": "train_1_a_1.nii.gz", "categories": {}},
                          {"name": "train_1_a_2.nii.gz", "categories": {}}],
                "val": [{"name": "valid_1_a_1.nii.gz", "categories": {}}]}
        rows = build_rows(data, {})
        summary = summarize(rows)
        self.assertEqual(summary["splits"]["train"]["patients"], 1)
        self.assertEqual(summary["patient_overlap"], {"train/val": []})

    def test_holdout_patient_never_moves_into_training(self):
        rows = [{"volume_name": "valid_1_a_1.nii.gz", "patient_id": "valid_1", "source_split": "val", "split": "val"},
                {"volume_name": "valid_1_b_1.nii.gz", "patient_id": "valid_1", "source_split": "test", "split": "test"}]
        output, changes = isolate_holdout_patients(rows)
        self.assertEqual([r["split"] for r in output], ["test", "test"])
        self.assertEqual(len(changes), 1)
        self.assertEqual(output[0]["source_split"], "val")


if __name__ == "__main__":
    unittest.main()
