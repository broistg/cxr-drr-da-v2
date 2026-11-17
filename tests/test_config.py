from pathlib import Path

import pytest

from drr_pipeline.config import load_dataset

YAML = """
data_root: data/raw
datasets:
  toy:
    root: toy
    ct_dir: ct
    mask_mode: separate
    lung_dir: lung
    infection_dir: inf
  toy2:
    root: toy2
    ct_dir: ct
    mask_mode: combined
    combined_dir: both
    lung_ids: []
    infection_ids: []
"""


def write(tmp_path):
    p = tmp_path / "data.yaml"
    p.write_text(YAML)
    return p


def test_paths_relative_to_repo_root(tmp_path):
    ds = load_dataset("toy", write(tmp_path), repo_root=tmp_path, environ={})
    assert ds["ct_dir"] == tmp_path / "data/raw/toy/ct"
    assert ds["lung_dir"] == tmp_path / "data/raw/toy/lung"


def test_env_var_overrides_data_root(tmp_path):
    other = tmp_path / "elsewhere"
    ds = load_dataset("toy", write(tmp_path), repo_root=tmp_path, environ={"DRR_DATA_ROOT": str(other)})
    assert ds["ct_dir"] == other / "toy/ct"


def test_unknown_dataset_and_incomplete_combined(tmp_path):
    p = write(tmp_path)
    with pytest.raises(KeyError):
        load_dataset("nope", p, repo_root=tmp_path, environ={})
    with pytest.raises(ValueError):
        load_dataset("toy2", p, repo_root=tmp_path, environ={})
