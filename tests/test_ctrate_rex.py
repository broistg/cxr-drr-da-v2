import pytest

from drr_pipeline import ctrate as C
from drr_pipeline import rex as R


def test_parse_volume_name_keeps_prefix():
    v = C.parse_volume_name("train_1741_b_2.nii.gz")
    assert (v.patient, v.scan, v.recon) == ("train_1741", "train_1741_b", "train_1741_b_2")
    assert C.parse_volume_name("valid_1741_a_1").patient == "valid_1741" != v.patient
    with pytest.raises(ValueError):
        C.parse_volume_name("foo.nii.gz")


def test_ptx_filter_negation_and_confounders():
    r = "No pneumothorax is seen. Small apical bleb noted, no pneumothorax. Right pneumothorax with chest tube."
    s = C.pneumothorax_sentences(r)
    assert [x["negated"] for x in s] == [True, True, False]
    assert [x["confounder"] for x in s] == [False, True, True]
    assert not C.is_ptx_candidate(r)
    assert C.is_ptx_candidate("There is a left-sided pneumothorax.")
    assert not C.is_ptx_candidate("")


def _data():
    return {"train": {"train_1_a_1.nii.gz": {"categories": ["2g", "2b"]},
                      "train_2_a_1.nii.gz": {"categories": ["2b", "2b"]}},
            "val": [{"name": "valid_1_a_1.nii.gz", "categories": ["2e"]}],
            "test": [{"name": "valid_9_a_1.nii.gz", "categories": ["2g"]}]}


def test_count_by_category_and_splits():
    c = R.count_by_category(_data())
    assert c["2b"] == {"lesions": 3, "ct": 2, "patients": 2}
    assert c["2g"]["ct"] == 1 and c["2e"]["ct"] == 1   # test không tính
    assert R.patient_overlap(_data()) == {"1"}          # số 1 ở cả train_1 và valid_1 (khác bệnh nhân thật nếu giữ tiền tố)


def test_make_folds_groups_by_patient_and_is_deterministic():
    from drr_pipeline import splits as SP
    cases = [f"train_{p}_{s}_1" for p in range(12) for s in "ab"] + ["valid_3_a_1"]
    f = SP.make_folds(cases, 4, seed=1)
    assert f == SP.make_folds(list(reversed(cases)), 4, seed=1)
    where = {c: i for i, v in f.items() for c in v}
    assert len(where) == len(cases)
    for p in range(12):
        assert where[f"train_{p}_a_1"] == where[f"train_{p}_b_1"]
    assert max(map(len, f.values())) - min(map(len, f.values())) <= 2
