import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sync_drr_hf", ROOT / "scripts/sync_drr_hf.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


def _repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "configs/signs").mkdir(parents=True)
    (repo / "configs/drr.yaml").write_text("a: 1\n")
    (repo / "configs/signs/consolidation.yaml").write_text("s: 1\n")
    drr = tmp_path / "drr"
    for sign, case in [("consolidation", "c1"), ("consolidation", "c2"), ("pleural_effusion", "c1")]:
        f = drr / sign / "cache/moderate/ap" / f"{case}.npz"
        f.parent.mkdir(parents=True, exist_ok=True)
        np.savez(f, x=np.arange(3))
    return repo, drr


def test_manifest_verify_and_filter(tmp_path):
    repo, drr = _repo(tmp_path)
    m = S.build_manifest(drr, repo, 0)
    assert set(m["files"]) == {"consolidation/cache/moderate/ap/c1.npz", "consolidation/cache/moderate/ap/c2.npz",
                               "pleural_effusion/cache/moderate/ap/c1.npz"}
    assert m["signs"] == ["consolidation", "pleural_effusion"]
    assert S.verify_files(drr, m) == ([], [])
    (drr / "consolidation/cache/moderate/ap/c2.npz").write_bytes(b"corrupt")
    (drr / "pleural_effusion/cache/moderate/ap/c1.npz").unlink()
    missing, bad = S.verify_files(drr, m)
    assert missing == ["pleural_effusion/cache/moderate/ap/c1.npz"] and bad == ["consolidation/cache/moderate/ap/c2.npz"]
    assert S.verify_files(drr, m, signs=["consolidation"]) == ([], ["consolidation/cache/moderate/ap/c2.npz"])
    only = S.build_manifest(drr, repo, 0, valid=lambda p: p.stem == "c1", signs=["consolidation"])
    assert list(only["files"]) == ["consolidation/cache/moderate/ap/c1.npz"]


def test_merge_and_config_mismatch(tmp_path):
    repo, drr = _repo(tmp_path)
    remote = S.build_manifest(drr, repo, 0, valid=lambda p: p.stem == "c1")
    local = S.build_manifest(drr, repo, 0, valid=lambda p: p.stem == "c2", signs=["consolidation"])
    merged = S.merge_manifest(remote, local)
    assert len(merged["files"]) == 3                                   # giữ tệp cũ trên HF, thêm tệp mới
    assert S.merge_manifest(None, local) is local
    (repo / "configs/drr.yaml").write_text("a: 2\n")                   # đổi cấu hình DRR
    changed = S.build_manifest(drr, repo, 0, valid=lambda p: p.stem == "c2", signs=["consolidation"])
    with pytest.raises(SystemExit):
        S.merge_manifest(remote, changed)
    assert S.config_mismatch(remote, S.config_hashes(repo, ["consolidation"])) == ["configs/drr.yaml"]
    with pytest.raises(SystemExit):                                    # seed khác
        S.merge_manifest(remote, dict(local, seed=1))


def test_crlf_config_same_hash(tmp_path):
    repo, _ = _repo(tmp_path)
    h = S.config_hashes(repo, ["consolidation"])
    (repo / "configs/drr.yaml").write_bytes(b"a: 1\r\n")
    assert S.config_hashes(repo, ["consolidation"]) == h
