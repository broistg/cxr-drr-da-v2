from drr_pipeline import cxr_eda as E
from drr_pipeline.config import load_cxr_dataset

WIDE = ["image_id", "rad_id", "Pneumonia", "No finding"]
WROWS = [
    {"image_id": "a", "rad_id": "R1", "Pneumonia": "1", "No finding": "0"},
    {"image_id": "b", "rad_id": "R1", "Pneumonia": "0", "No finding": "1"},
    {"image_id": "b", "rad_id": "R2", "Pneumonia": "1", "No finding": "0"},
]


def test_wide_labels():
    s = E.summarize_labels(WIDE, WROWS)
    assert s["layout"] == "wide" and s["rows"] == 3 and s["unique_images"] == 2
    assert s["positives_rows"]["Pneumonia"] == 2
    assert s["positive_images"]["Pneumonia"] == 2 and s["positive_images"]["No finding"] == 1


def test_long_labels_and_boxes():
    cols = ["image_id", "rad_id", "class_name", "x_min"]
    rows = [{"image_id": "a", "rad_id": "R1", "class_name": "Consolidation", "x_min": "1"},
            {"image_id": "a", "rad_id": "R1", "class_name": "Consolidation", "x_min": "5"},
            {"image_id": "b", "rad_id": "R1", "class_name": "PE", "x_min": "2"}]
    s = E.summarize_labels(cols, rows)
    assert s["layout"] == "long" and s["positive_images"]["Consolidation"] == 1
    b = E.summarize_boxes(cols, rows)
    assert b["boxes_per_class"]["Consolidation"] == 2 and b["images_per_class"]["Consolidation"] == 1
    assert b["images_with_boxes"] == 2


def test_parse_age():
    assert E.parse_age_months("012M") == 12.0
    assert E.parse_age_months("003Y") == 36.0
    assert abs(E.parse_age_months("014D") - 14 / 30.4375) < 1e-9
    assert E.parse_age_months("") is None and E.parse_age_months("abc") is None


def test_numeric_summary():
    s = E.summarize_numeric([1, 2, 3, 4, 5, None])
    assert s["n"] == 5 and s["min"] == 1 and s["max"] == 5 and s["median"] == 3


def test_cxr_config(tmp_path):
    y = tmp_path / "data.yaml"
    y.write_text("data_root: data/raw\ncxr_datasets:\n  v:\n    root: v\n    train_dir: train\n"
                 "    image_labels: {train: l.csv}\n")
    ds = load_cxr_dataset("v", y, repo_root=tmp_path, environ={})
    assert ds["train_dir"] == tmp_path / "data/raw/v/train"
    assert ds["image_labels"]["train"] == tmp_path / "data/raw/v/l.csv"


def test_boxes_exclude_no_finding():
    cols = ["image_id", "class_name"]
    rows = [{"image_id": "a", "class_name": "No finding"}, {"image_id": "b", "class_name": "Consolidation"},
            {"image_id": "b", "class_name": "Consolidation"}]
    b = E.summarize_boxes(cols, rows)
    assert b["rows_no_finding"] == 1 and b["real_boxes"] == 2 and b["images_with_boxes"] == 1
    assert "No finding" not in b["boxes_per_class"]


def test_label_groups():
    cols = ["image_id", "rad_ID", "No finding", "Pneumonia", "Bronchitis"]
    rows = [{"image_id": "a", "rad_ID": "R", "No finding": "1.0", "Pneumonia": "0.0", "Bronchitis": "0.0"},
            {"image_id": "b", "rad_ID": "R", "No finding": "0.0", "Pneumonia": "1.0", "Bronchitis": "0.0"},
            {"image_id": "c", "rad_ID": "R", "No finding": "0.0", "Pneumonia": "1.0", "Bronchitis": "1.0"},
            {"image_id": "d", "rad_ID": "R", "No finding": "1.0", "Pneumonia": "1.0", "Bronchitis": "0.0"},
            {"image_id": "e", "rad_ID": "R", "No finding": "0.0", "Pneumonia": "0.0", "Bronchitis": "0.0"}]
    g = E.label_groups(cols, rows, {"pn": ["Pneumonia"]})
    assert g["images"] == 5 and g["no_label_at_all"] == 1 and g["no_finding_and_other"] == 1
    assert g["groups"]["pn"] == {"images": 3, "also_no_finding": 1, "exclusive": 2}
    assert g["positive_labels_per_image"] == {0: 2, 1: 2, 2: 1}


def test_match_files():
    m = E.match_files(["a", "b", "c"], ["a", "c", "z"])
    assert m["ids_without_file"] == ["b"] and m["files_without_id"] == ["z"]
