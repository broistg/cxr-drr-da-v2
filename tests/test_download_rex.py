import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.download_rex import build_download_plan, download_jobs, fetch_metadata, original_ct_path, read_json
from scripts.plan_rex_masks import build_plan


SOURCES = {"rex_repo": "rex/fixture", "rex_revision": "fixture-mask-revision",
           "ct_repo": "ct/fixture", "ct_revision": "fixture-ct-revision"}
MASKED = "valid_1_b_2.nii.gz"
REPORT_ONLY = "train_2_a_1.nii.gz"
MASK_DATA = b"mock mask bytes"
CT_DATA = b"mock CT bytes"


def row(name, source_split):
    return {"volume_name": name, "patient_id": "_".join(name.split("_")[:2]),
            "split": "test", "source_split": source_split}


def file_info(path, body):
    return SimpleNamespace(path=path, size=len(body), lfs=SimpleNamespace(sha256=hashlib.sha256(body).hexdigest()))


def fixtures():
    mask = {**row(MASKED, "val"), "path": f"segmentations/{MASKED}", "size": len(MASK_DATA),
            "sha256": hashlib.sha256(MASK_DATA).hexdigest(), "channels": [{"sign": "consolidation", "finding_id": 3}]}
    return {"repo": SOURCES["rex_repo"], "revision": SOURCES["rex_revision"], "jobs": [mask],
            "cardiomegaly_candidates": [row(REPORT_ONLY, "test")], "unavailable_masks": []}


class FakeHub:
    def __init__(self, files):
        self.files = files
        self.requests = []

    def get_paths_info(self, repo, paths, **options):
        self.requests.append((repo, paths, options))
        return [self.files[path] for path in paths if path in self.files]

    def list_repo_tree(self, repo, **options):
        self.requests.append((repo, options))
        return iter(self.files.values())


class RexDownloadTests(unittest.TestCase):
    def test_match_exact_reconstruction_in_original_collection(self):
        path = f"dataset/valid/valid_1/valid_1_b/{MASKED}"
        duplicate = path.replace("/valid/", "/valid_fixed/")
        api = FakeHub({key: file_info(key, CT_DATA) for key in (path, duplicate)})
        plan = build_download_plan(fixtures(), SOURCES, api)
        self.assertEqual(api.requests, [(SOURCES["ct_repo"], [path],
                                         {"repo_type": "dataset", "revision": SOURCES["ct_revision"]})])
        self.assertEqual([job["ct"]["path"] for job in plan["jobs"]], [path])
        self.assertEqual(plan["jobs"][0]["split"], "test")  # Only the source name determines the CT folder.
        self.assertEqual(plan["missing_ct"], [])
        with self.assertRaises(ValueError):
            original_ct_path("some_folder/" + MASKED)

    def test_report_only_candidates_are_opt_in(self):
        names = (MASKED, REPORT_ONLY)
        api = FakeHub({original_ct_path(name): file_info(original_ct_path(name), CT_DATA) for name in names})
        plan = build_download_plan(fixtures(), SOURCES, api, include_cardiomegaly=True)
        candidates = [job for job in plan["jobs"] if job["volume_name"] == REPORT_ONLY]
        self.assertEqual(len(candidates), 1)
        self.assertIsNone(candidates[0]["mask"])
        self.assertEqual(candidates[0]["ct"]["path"], original_ct_path(REPORT_ONLY))

    def test_missing_original_ct_is_reported_before_transfer(self):
        plan = build_download_plan(fixtures(), SOURCES, FakeHub({}))
        self.assertEqual(plan["missing_ct"][0]["expected_path"], original_ct_path(MASKED))
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, "CT missing"):
            download_jobs(Path(tmp), plan, SOURCES, self.fail)

    def test_download_targets_are_pinned_and_verified_and_resume_skips_files(self):
        ct = original_ct_path(MASKED)
        plan = build_download_plan(fixtures(), SOURCES, FakeHub({ct: file_info(ct, CT_DATA)}))
        calls = []

        def transfer(repo, name, *, revision, local_dir, **options):
            calls.append((repo, name, revision, options["force_download"]))
            path = local_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(CT_DATA if repo == SOURCES["ct_repo"] else MASK_DATA)
            return path

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            download_jobs(root, plan, SOURCES, transfer)
            download_jobs(root, plan, SOURCES, self.fail)
            (root / "raw/ct-rate" / ct).write_bytes(b"wrong bytes!!")
            download_jobs(root, plan, SOURCES, transfer, files="ct")
        self.assertEqual(calls, [(SOURCES["rex_repo"], f"segmentations/{MASKED}", SOURCES["rex_revision"], False),
                                 (SOURCES["ct_repo"], ct, SOURCES["ct_revision"], False),
                                 (SOURCES["ct_repo"], ct, SOURCES["ct_revision"], True)])

    def test_metadata_passes_pinned_file_info_to_mask_manifest(self):
        mask_path = f"segmentations/{MASKED}"
        api = FakeHub({mask_path: file_info(mask_path, MASK_DATA)})
        requests = []

        def transfer(repo, name, **options):
            requests.append((repo, name, options["revision"]))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fetch_metadata(root, SOURCES, api, transfer)
            index = read_json(root / "manifests/rex_files.json")
            sources = read_json(root / "manifests/revisions.json")
        meta = {"val": [{"name": MASKED, "shape": [2, 2, 2], "findings": {"0": "Lobar consolidation"},
                         "categories": {"0": "2b"}, "pixels": {"0": 1}}]}
        mask_plan = build_plan(meta, {}, index, sources["rex_revision"])
        self.assertEqual([job["path"] for job in mask_plan["jobs"]], [mask_path])
        self.assertEqual(requests, [(SOURCES["rex_repo"], name, SOURCES["rex_revision"])
                                    for name in ("dataset.json", "reports_dataset.json")])
