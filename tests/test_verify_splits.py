import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("verify_splits", ROOT / "scripts/verify_splits.py")
V = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V)


def test_write_verify_crlf_and_clash(tmp_path):
    f = tmp_path / "data/processed/source_splits/a_Rm.yaml"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"folds:\n  0: [a]\n")
    assert V.main(["--root", str(tmp_path), "--write"]) == 0
    assert V.main(["--root", str(tmp_path)]) == 0
    f.write_bytes(b"folds:\r\n  0: [a]\r\n")              # CRLF (Windows) vẫn khớp
    assert V.main(["--root", str(tmp_path)]) == 0
    f.write_bytes(b"folds:\n  0: [b]\n")
    assert V.main(["--root", str(tmp_path)]) == 1          # lệch
    try:
        V.main(["--root", str(tmp_path), "--write"])       # từ chối ghi đè
        raise AssertionError("phải thoát")
    except SystemExit as e:
        assert "chia lại" in str(e)
    f.unlink()
    assert V.main(["--root", str(tmp_path)]) == 1          # thiếu
