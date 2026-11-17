import numpy as np

from drr_pipeline import adult_target as A


def _recs():
    r = []
    for i in range(100):
        r.append({"cls": "Lung_Opacity", "id": f"Lung_Opacity-{i}", "source": "rsna"})
        r.append({"cls": "COVID", "id": f"COVID-{i}", "source": "bimcv"})
    for i in range(300):
        r.append({"cls": "Normal", "id": f"Normal-{i}", "source": "rsna" if i % 4 else "kermany"})
    return r


def test_source_of():
    assert A.source_of("https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data") == "rsna"
    assert A.source_of("[www.kaggle.com/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)") == "kermany"
    assert A.source_of("https://bimcv.cipf.es/bimcv-projects/bimcv-covid19/") == "bimcv"
    assert A.source_of("nan") == "other"


def test_select_excludes_other_sources_and_pediatric():
    s = A.select_records(_recs(), "opacity", 0, 0)
    assert {r["cls"] for r in s} == {"Lung_Opacity", "Normal"}
    assert all(r["source"] == "rsna" for r in s)                    # Normal from the pediatric set is dropped
    assert sum(r["role"] == "positive" for r in s) == 100 and sum(r["role"] == "negative" for r in s) == 225
    c = A.select_records(_recs(), "covid", 0, 0)
    assert {r["cls"] for r in c if r["role"] == "positive"} == {"COVID"}


def test_select_subsample_deterministic_and_order_independent():
    a = A.select_records(_recs(), "opacity", 30, 60, seed=3)
    b = A.select_records(list(reversed(_recs())), "opacity", 30, 60, seed=3)
    assert [r["id"] for r in a] == [r["id"] for r in b]
    assert sum(r["role"] == "positive" for r in a) == 30 and len(a) == 90
    assert [r["id"] for r in a] != [r["id"] for r in A.select_records(_recs(), "opacity", 30, 60, seed=4)]


def test_splits_disjoint_and_stratified():
    s = A.select_records(_recs(), "opacity", 0, 0)
    sp = A.assign_splits(s, 0.2, 0.1, 0)
    assert set(sp) == {r["id"] for r in s}
    for role, n in (("positive", 100), ("negative", 225)):
        ids = [r["id"] for r in s if r["role"] == role]
        n_test = sum(sp[i] == "test" for i in ids)
        assert abs(n_test - 0.2 * n) <= 1
        assert sum(sp[i] == "target_val" for i in ids) >= 1


def test_lung_bbox():
    m = np.zeros((100, 100), np.uint8)
    m[20:60, 30:70] = 255
    y0, y1, x0, x1 = A.lung_bbox(m, 0.05)
    assert (y0, y1, x0, x1) == (18, 62, 28, 72)
    assert A.lung_bbox(np.zeros((10, 12), np.uint8)) == (0, 10, 0, 12)
    assert A.lung_bbox(np.full((10, 12), 255, np.uint8))[0] == 0


def test_frames_helpers():
    assert A.FRAMES == ("full", "crop", "maskout", "center", "maskout_fixed")
    fh, fw = A.box_fractions([((10, 60, 20, 80), (100, 100)), ((0, 40, 0, 60), (100, 100)), ((20, 90, 10, 50), (100, 100))])
    assert abs(fh - 0.5) < 1e-9 and abs(fw - 0.6) < 1e-9
    assert A.center_box(100, 200, 0.5, 0.5) == (25, 75, 50, 150)
    y0, y1, x0, x1 = A.center_box(10, 10, 0.0, 0.0)
    assert y1 - y0 == 1 and x1 - x0 == 1
    img = np.arange(100, dtype=np.uint8).reshape(10, 10) + 1
    box = (2, 6, 3, 9)
    assert A.apply_frame(img, box, "crop").shape == (4, 6) and A.apply_frame(img, box, "center").shape == (4, 6)
    m = A.apply_frame(img, box, "maskout")
    assert m.shape == img.shape and m[2:6, 3:9].tolist() == img[2:6, 3:9].tolist() and m.sum() == img[2:6, 3:9].sum()
    assert A.apply_frame(img, box, "full") is img
    mf = A.apply_frame(img, box, "maskout_fixed")
    assert mf.shape == img.shape and mf.sum() == img[2:6, 3:9].sum()
    bs = [((0, 10 * (i + 1), 0, 10 * (i + 1)), (100, 100)) for i in range(10)]
    assert abs(A.box_fractions(bs, 0.5)[0] - 0.55) < 1e-9 and abs(A.box_fractions(bs, 1.0)[0] - 1.0) < 1e-9


def test_prep_process_frames(tmp_path):
    import importlib.util
    from pathlib import Path as _P
    from PIL import Image
    spec = importlib.util.spec_from_file_location("prep_cov", _P(__file__).resolve().parents[1] / "scripts" / "prep_covid19_radiography.py")
    prep = importlib.util.module_from_spec(spec); spec.loader.exec_module(prep)
    rng = np.random.default_rng(0)
    img = rng.integers(50, 200, (80, 80)).astype(np.uint8)
    img[:10] = 255                                           # bright 'border text' outside the lungs
    mask = np.zeros((80, 80), np.uint8); mask[20:70, 15:65] = 255
    Image.fromarray(img).save(tmp_path / "a.png"); Image.fromarray(mask).save(tmp_path / "m.png")
    r = {"path": tmp_path / "a.png", "mask": tmp_path / "m.png"}
    outs = {}
    for fr in ("full", "crop", "maskout", "center"):
        o = tmp_path / f"{fr}.png"
        prep.process(r, o, 64, fr, (0.5, 0.5))
        outs[fr] = np.asarray(Image.open(o))
        assert outs[fr].shape == (64, 64)
    assert outs["full"][:8].mean() > 200                     # border text visible in the full frame
    assert outs["maskout"][:6].max() < 20 and outs["maskout"][-5:].max() < 20   # zeroed outside the lung box (same scale as full)
    assert outs["crop"][:4].mean() > 20 and outs["crop"].std() > 0                # zoomed: content reaches the top rows
    assert not np.array_equal(outs["center"], outs["crop"])
