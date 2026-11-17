import numpy as np
from PIL import Image

from drr_pipeline import seg_target as S


def _png(p, arr):
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(arr, np.uint8)).save(p)


def _tree(tmp_path, with_val=False):
    """COVID-QU-Ex-like layout: Infection Segmentation Data/<Split>/COVID-19/{images,lung masks,infection masks}."""
    base = tmp_path / "Infection Segmentation Data"
    for sp, n in (("Train", 6), ("Test", 3)) + ((("Val", 2),) if with_val else ()):
        for i in range(n):
            img = np.random.default_rng(i).integers(0, 255, (40, 50))
            les = np.zeros((40, 50)); les[10:20, 10:25] = 255
            lung = np.zeros((40, 50)); lung[5:35, 5:45] = 255
            _png(base / sp / "COVID-19" / "images" / f"covid_{sp}_{i}.png", img)
            _png(base / sp / "COVID-19" / "infection masks" / f"covid_{sp}_{i}.png", les)
            _png(base / sp / "COVID-19" / "lung masks" / f"covid_{sp}_{i}.png", lung)
        _png(base / sp / "Normal" / "images" / f"normal_{sp}_0.png", np.zeros((40, 50)))     # no infection mask: dropped
    return tmp_path


def test_kind_of_dir():
    assert S.kind_of_dir("images") == "image"
    assert S.kind_of_dir("Images") == "image"
    assert S.kind_of_dir("lung masks") == "lung"
    assert S.kind_of_dir("infection masks") == "lesion"
    assert S.kind_of_dir("Ground-truths") == "lesion"
    assert S.kind_of_dir("masks") == "lesion"
    assert S.kind_of_dir("Lung Segmentation Data") == "lung"          # parent-like name; harmless: it holds no images itself
    assert S.kind_of_dir("COVID-19") == "image"


def test_norm_stem_and_pair():
    assert S.norm_stem("mask_covid_1") == "covid_1" and S.norm_stem("covid_1_mask") == "covid_1" and S.norm_stem("Covid_1") == "covid_1"
    from pathlib import Path
    imgs = [Path("a/covid_1.png"), Path("a/covid_2.png")]
    msk = [Path("m/mask_covid_1.png"), Path("m/covid_3_mask.png")]
    paired, ui, um = S.pair_by_stem(imgs, msk)
    assert paired == {imgs[0]: msk[0]} and ui == [imgs[1]] and um == [msk[1]]


def test_split_group():
    assert S.split_of(["Infection Segmentation Data", "Train", "COVID-19", "images"]) == "train"
    assert S.split_of(["Test Set", "Images"]) == "test" and S.split_of(["Validation"]) == "val" and S.split_of(["x"]) is None
    assert S.group_of(["Train", "COVID-19"]) == "covid" and S.group_of(["Test", "Non-COVID"]) == "non_covid" and S.group_of(["a"]) == ""


def test_build_records_covidqu_layout(tmp_path):
    root = _tree(tmp_path)
    recs, rep = S.build_records(root)
    assert len(recs) == 9 and rep["paired"] == 9                      # normal images have no lesion folder -> not recorded
    assert {r["split"] for r in recs} == {"train", "test"} and all(r["lung"] is not None for r in recs)
    assert len({r["id"] for r in recs}) == 9
    recs2 = S.assign_missing_splits(recs, val_frac=0.5, seed=0)
    assert sum(r["split"] == "val" for r in recs2) == 3 and sum(r["split"] == "test" for r in recs2) == 3
    assert sum(r["split"] == "train" for r in recs2) == 3


def test_build_records_qata_layout(tmp_path):
    for sp, n in (("Train Set", 4), ("Test Set", 2)):
        for i in range(n):
            _png(tmp_path / "QaTa" / sp / "Images" / f"covid_{i}.png", np.ones((8, 8)) * 50)
            _png(tmp_path / "QaTa" / sp / "Ground-truths" / f"mask_covid_{i}.png", np.eye(8) * 255)
    recs, _ = S.build_records(tmp_path)
    assert len(recs) == 6 and all(r["lung"] is None for r in recs) and {r["split"] for r in recs} == {"train", "test"}


def test_assign_missing_splits_none(tmp_path):
    recs = [{"id": str(i), "split": None, "group": "", "image": None, "lesion": None, "lung": None} for i in range(20)]
    out = S.assign_missing_splits(recs, 0.2, 0.2, seed=1)
    c = {s: sum(r["split"] == s for r in out) for s in ("train", "val", "test")}
    assert c == {"train": 12, "val": 4, "test": 4}
    assert out == S.assign_missing_splits(recs, 0.2, 0.2, seed=1)    # deterministic


def test_binarize_crop():
    assert S.binarize(np.array([[0, 1], [1, 0]])).sum() == 2 and S.binarize(np.array([[0, 255], [100, 200]])).sum() == 2
    a = np.arange(16).reshape(4, 4)
    assert S.crop_to_box([a], (1, 3, 0, 2))[0].shape == (2, 2)


def test_kind_of_dir_qata_folders():
    assert S.kind_of_dir("COVID-19_no_infection_sign") == "image"       # QaTa 'Early' folders hold images, not masks
    assert S.kind_of_dir("COVID-19_very_limited_sign") == "image"
    assert S.kind_of_dir("Ground-truths") == "lesion" and S.kind_of_dir("infection masks") == "lesion"
    assert S.kind_of_dir("Control_Group_I") == "image"


def test_release_and_only_and_patient(tmp_path):
    assert S.release_of("QaTa-COV19/QaTa-COV19-v2/Train Set/Images/a.png") == "v2"
    assert S.release_of("QaTa-COV19/QaTa-COV19-v1/Images/a.png") == "v1" and S.release_of("a/b/c.png") == ""
    for rel, n in (("v1", 3), ("v2", 4)):
        for i in range(n):
            _png(tmp_path / f"QaTa-COV19-{rel}" / "Train Set" / "Images" / f"covid_{i}.png", np.ones((8, 8)) * 40)
            _png(tmp_path / f"QaTa-COV19-{rel}" / "Train Set" / "Ground-truths" / f"mask_covid_{i}.png", np.eye(8) * 255)
    allr, _ = S.build_records(tmp_path)
    assert len(allr) == 7 and {r["release"] for r in allr} == {"v1", "v2"}
    only, _ = S.build_records(tmp_path, only=["QaTa-COV19-v2"])
    assert len(only) == 4 and {r["release"] for r in only} == {"v2"}
    assert S.patient_key("train__x__sub-S08047_ses-E16280_run-1_cr") == "S08047" and S.patient_key("covid_12") == "covid_12"


def test_val_carve_keeps_patients_together():
    recs = [{"id": f"train__x__sub-S{p}_ses-{k}", "split": "train", "group": "", "image": None, "lesion": None, "lung": None}
            for p in range(20) for k in range(3)] + [{"id": f"test__x__sub-S99_ses-0", "split": "test", "group": "", "image": None, "lesion": None, "lung": None}]
    out = S.assign_missing_splits(recs, val_frac=0.2, seed=0)
    val_p = {S.patient_key(r["id"]) for r in out if r["split"] == "val"}
    tr_p = {S.patient_key(r["id"]) for r in out if r["split"] == "train"}
    assert val_p and not (val_p & tr_p) and sum(r["split"] == "val" for r in out) >= 12
    assert [r["split"] for r in out if r["id"].startswith("test")] == ["test"]


def test_duplicate_groups():
    hs = {"a": "h1", "b": "h1", "c": "h2", "d": "h3", "e": "h3"}
    grp = {"a": "v1", "b": "v2", "c": "v1", "d": "v2", "e": "v2"}
    d = S.duplicate_groups(hs, lambda i: grp[i])
    assert list(d) == ["h1"] and sorted(d["h1"]) == ["a", "b"]
