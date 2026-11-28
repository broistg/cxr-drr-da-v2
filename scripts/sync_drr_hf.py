"""Chia sẻ cache DRR trong nhóm qua một repo dataset PRIVATE trên Hugging Face (push / pull / verify).

    python scripts/sync_drr_hf.py push   --repo <tổ-chức>/<tên>                        # chỉ người sinh DRR (máy lab)
    python scripts/sync_drr_hf.py pull   --repo <tổ-chức>/<tên> [--signs consolidation]  # mọi thành viên khác
    python scripts/sync_drr_hf.py verify                                                # kiểm sha256 cache cục bộ với drr_manifest.json (không cần mạng)

Chỉ đẩy/kéo cache .npz (data/processed/drr/<dấu hiệu>/cache/<profile>/<view>/<ca>.npz) và drr_manifest.json. KHÔNG đẩy CT, mask, báo cáo, nhãn CT-RATE,
dữ liệu VinDr-PCXR hay tệp chia fold. Cache là dữ liệu phái sinh từ CT-RATE/ReXGroundingCT (CC BY-NC-SA 4.0): repo phải private, chỉ thành viên đã
tự chấp nhận điều khoản của hai dataset gốc được truy cập (docs/DRR_SHARING.md). `push` từ chối ghi vào repo công khai.

drr_manifest.json ghi sha256/kích thước từng tệp, git SHA mã sinh, mã băm các cấu hình ảnh hưởng tới cache (configs/drr.yaml, data.yaml, signs/<dấu hiệu>.yaml,
rex_sources.json), seed/profile/view. `pull` dừng nếu mã băm cấu hình cục bộ khác manifest (cache sinh bằng cấu hình khác không dùng lẫn được);
--force bỏ qua kiểm này (chỉ khi biết chắc). Sau khi kéo luôn kiểm sha256 từng tệp.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
MANIFEST = "drr_manifest.json"
CONFIGS = ["configs/drr.yaml", "configs/data.yaml", "configs/rex_sources.json"]
CARD = """---
license: cc-by-nc-sa-4.0
---
# Cache DRR (nội bộ nhóm HK261-DAGD1-481)

Ảnh chiếu DRR (.npz) sinh từ CT-RATE và ReXGroundingCT bằng `scripts/generate_drr.py` của repo `cxr-drr-da-v2`. Dữ liệu phái sinh, giấy phép CC BY-NC-SA 4.0,
chỉ dùng học thuật/nghiên cứu, không thương mại, ghi nguồn gốc (CT-RATE: Hamamci et al., 2024; ReXGroundingCT: Baharoon et al., 2025).
Repo PRIVATE: chỉ thành viên đã tự chấp nhận điều khoản của CT-RATE và ReXGroundingCT; không chuyển tiếp cho người ngoài nhóm; không có CT thô, mask,
báo cáo hay nhãn. Không cố gắng nhận diện lại bệnh nhân. Chi tiết cấu hình và sha256 từng tệp: `drr_manifest.json`.
"""


def norm_sha(path: Path) -> str:
    """sha256 sau khi đổi CRLF -> LF (cho tệp văn bản cấu hình)."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_files(drr_root: Path, signs=None):
    out = []
    for p in sorted(Path(drr_root).glob("*/cache/*/*/*.npz")):
        if signs is None or p.relative_to(drr_root).parts[0] in signs:
            out.append(p)
    return out


def config_hashes(repo_root: Path, signs) -> dict:
    names = CONFIGS + [f"configs/signs/{s}.yaml" for s in sorted(signs)]
    return {n: norm_sha(repo_root / n) for n in names if (repo_root / n).exists()}


def git_sha(repo_root: Path) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def build_manifest(drr_root: Path, repo_root: Path, seed: int, valid=None, signs=None) -> dict:
    """Manifest cho các tệp cache hợp lệ (valid(path) -> bool; mặc định mọi tệp). Trả về dict; không ghi đĩa."""
    files = {}
    for p in cache_files(drr_root, signs):
        if valid is not None and not valid(p):
            continue
        files[p.relative_to(drr_root).as_posix()] = {"sha256": file_sha(p), "size": p.stat().st_size}
    sg = sorted({k.split("/")[0] for k in files})
    return {"schema": 1, "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "git_sha": git_sha(repo_root),
            "seed": seed, "signs": sg, "configs": config_hashes(repo_root, sg), "files": dict(sorted(files.items()))}


def merge_manifest(remote: dict | None, local: dict) -> dict:
    """Gộp manifest từ xa với cục bộ (tệp cục bộ thắng). Báo lỗi nếu mã băm cấu hình khác nhau."""
    if not remote:
        return local
    bad = config_mismatch(remote, local["configs"])
    if bad:
        raise SystemExit(f"cấu hình cục bộ khác manifest trên HF ở {bad}: cache mới sẽ không dùng lẫn được với cache cũ. "
                         "Không trộn; nếu cố ý đổi cấu hình thì dùng repo/tag mới.")
    if remote.get("seed") != local["seed"]:
        raise SystemExit(f"seed khác manifest trên HF ({remote.get('seed')} so với {local['seed']})")
    files = {**remote["files"], **local["files"]}
    out = dict(local, files=dict(sorted(files.items())), signs=sorted({k.split("/")[0] for k in files}))
    out["configs"] = {**remote["configs"], **local["configs"]}
    return out


def config_mismatch(manifest: dict, local_cfg: dict) -> list:
    """Tên cấu hình có mặt ở cả hai bên mà mã băm khác (tệp chỉ có một bên không tính)."""
    return sorted(k for k, v in manifest["configs"].items() if k in local_cfg and local_cfg[k] != v)


def verify_files(drr_root: Path, manifest: dict, signs=None):
    """(thiếu, lệch) trong số tệp của manifest (lọc theo dấu hiệu nếu có)."""
    missing, bad = [], []
    for rel, meta in manifest["files"].items():
        if signs is not None and rel.split("/")[0] not in signs:
            continue
        p = Path(drr_root) / rel
        if not p.exists():
            missing.append(rel)
        elif p.stat().st_size != meta["size"] or file_sha(p) != meta["sha256"]:
            bad.append(rel)
    return missing, bad


def _api(token=None):
    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("cần huggingface_hub: pip install -e \".[rex]\"")
    return HfApi(token=token)


def cmd_push(a):
    from huggingface_hub import hf_hub_download
    from run_rex_stream import cache_ok
    api = _api()
    try:
        info = api.repo_info(a.repo, repo_type="dataset")
        if not info.private:
            sys.exit(f"{a.repo} là repo CÔNG KHAI: không đẩy cache DRR (dữ liệu phái sinh từ CT-RATE/ReX). Đặt repo private trước.")
    except Exception as e:  # chưa có repo
        if "404" not in str(e) and "not found" not in str(e).lower():
            raise
        api.create_repo(a.repo, repo_type="dataset", private=True)
        print(f"đã tạo repo private {a.repo}")
    local = build_manifest(a.drr_root, ROOT, a.seed, valid=cache_ok, signs=a.signs)
    if not local["files"]:
        sys.exit(f"không có cache hợp lệ trong {a.drr_root}")
    remote = None
    try:
        remote = json.loads(Path(hf_hub_download(a.repo, MANIFEST, repo_type="dataset")).read_text(encoding="utf8"))
    except Exception:
        print("chưa có manifest trên HF (lần đẩy đầu)")
    merged = merge_manifest(remote, local)
    new = [k for k in local["files"] if not remote or remote["files"].get(k, {}).get("sha256") != local["files"][k]["sha256"]]
    print(f"cục bộ {len(local['files'])} tệp hợp lệ; cần đẩy {len(new)} tệp mới/đổi; manifest sau gộp {len(merged['files'])} tệp")
    if a.dry_run:
        print("dry-run: không đẩy gì.")
        return 0
    (a.drr_root / MANIFEST).write_text(json.dumps(merged, indent=1) + "\n", encoding="utf8")
    (a.drr_root / "README.md").write_text(CARD, encoding="utf8")
    api.upload_folder(repo_id=a.repo, repo_type="dataset", folder_path=str(a.drr_root),
                      allow_patterns=["*/cache/*/*/*.npz", MANIFEST, "README.md"],
                      commit_message=f"DRR cache: +{len(new)} tệp, mã {merged['git_sha'][:8]}")
    print("xong.")
    return 0


def cmd_pull(a):
    from huggingface_hub import hf_hub_download, snapshot_download
    mpath = hf_hub_download(a.repo, MANIFEST, repo_type="dataset", revision=a.revision)
    manifest = json.loads(Path(mpath).read_text(encoding="utf8"))
    signs = a.signs or manifest["signs"]
    bad = config_mismatch(manifest, config_hashes(ROOT, signs))
    if bad and not a.force:
        sys.exit(f"cấu hình cục bộ khác lúc sinh cache ở {bad}. Cập nhật mã (git pull / áp zip mới nhất) cho khớp commit {manifest['git_sha'][:8]}; "
                 "không dùng cache khi cấu hình khác. (--force chỉ khi biết chắc)")
    if manifest["git_sha"] != git_sha(ROOT):
        print(f"lưu ý: cache sinh bởi commit {manifest['git_sha'][:8]}, mã cục bộ {git_sha(ROOT)[:8]} (cấu hình khớp nên vẫn dùng được)")
    snapshot_download(a.repo, repo_type="dataset", revision=a.revision, local_dir=str(a.drr_root),
                      allow_patterns=[f"{s}/cache/*/*/*.npz" for s in signs] + [MANIFEST])
    missing, badf = verify_files(a.drr_root, manifest, signs)
    print(f"đã kiểm {sum(1 for k in manifest['files'] if k.split('/')[0] in signs)} tệp: thiếu {len(missing)}, lệch sha256 {len(badf)}")
    for k in (missing + badf)[:10]:
        print("  ", k)
    return 1 if (missing or badf) else 0


def cmd_verify(a):
    mp = a.drr_root / MANIFEST
    if not mp.exists():
        sys.exit(f"chưa có {mp}: chạy pull trước")
    manifest = json.loads(mp.read_text(encoding="utf8"))
    missing, bad = verify_files(a.drr_root, manifest, a.signs)
    print(f"thiếu {len(missing)}, lệch {len(bad)}")
    for k in (missing + bad)[:10]:
        print("  ", k)
    return 1 if (missing or bad) else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["push", "pull", "verify"])
    ap.add_argument("--repo", help="<tổ-chức>/<tên> (dataset)")
    ap.add_argument("--drr-root", type=Path, default=ROOT / "data/processed/drr")
    ap.add_argument("--signs", nargs="+", help="chỉ các dấu hiệu này (mặc định: tất cả)")
    ap.add_argument("--seed", type=int, default=0, help="seed đã dùng khi sinh (run_rex_stream --seed); chỉ để ghi manifest")
    ap.add_argument("--revision", help="pull: commit/tag trên HF")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd in ("push", "pull") and not a.repo:
        ap.error("cần --repo")
    return {"push": cmd_push, "pull": cmd_pull, "verify": cmd_verify}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
