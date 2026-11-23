"""Tải các CSV metadata kỹ thuật của CT-RATE (RescaleSlope/Intercept, spacing...) về raw/ct-rate/dataset/metadata/.

    python scripts/fetch_ctrate_metadata.py --root data/rex-drr

Cần đăng nhập Hugging Face và đã chấp nhận điều khoản CT-RATE. Chỉ tải file .csv trong dataset/metadata/ (chưa kiểm tên file cụ thể:
script liệt kê và in ra những gì tìm thấy). Không tải báo cáo/nhãn; nếu thấy CSV tên reports/labels, bỏ qua với --skip.
"""
import argparse
import json
import sys
from pathlib import Path

SKIP_WORDS = ("report", "label", "finding", "impression")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--sources", type=Path, default=Path("configs/rex_sources.json"))
    ap.add_argument("--list-only", action="store_true")
    ap.add_argument("--reports", action="store_true", help="tải CSV báo cáo (dataset/radiology_text_reports) CHỈ để lọc ca bình thường theo D60; không in nội dung")
    ap.add_argument("--labels", action="store_true", help="tải CSV nhãn đa bất thường (dataset/multi_abnormality_labels) thay vì metadata; vẫn không tải báo cáo")
    a = ap.parse_args()
    from huggingface_hub import HfApi, hf_hub_download
    src = json.loads(a.sources.read_text(encoding="utf8"))
    repo, rev = src["ct_repo"], src["ct_revision"]
    entries = [e for e in HfApi().list_repo_tree(repo, path_in_repo="dataset/radiology_text_reports" if a.reports else ("dataset/multi_abnormality_labels" if a.labels else "dataset/metadata"), repo_type="dataset", revision=rev)
               if e.path.endswith(".csv")]
    if not entries:
        print("Không thấy CSV ở đường dẫn đã chọn (kiểm tra quyền truy cập hoặc đường dẫn).", file=sys.stderr)
        return 2
    for e in entries:
        skip = any(w in e.path.lower() for w in (SKIP_WORDS if not (a.labels or a.reports) else ()))
        print(f"{'bỏ qua' if skip else 'tải  '} {e.path} ({e.size / 1e6:.1f} MB)")
        if not skip and not a.list_only:
            hf_hub_download(repo, e.path, repo_type="dataset", revision=rev, local_dir=a.root / "raw/ct-rate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
