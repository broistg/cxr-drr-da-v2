"""Chạy đường ống CT-RATE/ReX theo lô: tải -> chuẩn hóa -> DRR -> kiểm cache -> xóa CT.

Mỗi lô gồm B volume. Trong khi GPU sinh DRR cho lô i, một luồng nền tải và chuẩn hóa lô i+1 (đỉnh đĩa ~ 2 lô).
Chỉ xóa CT thô, mask thô và CT/nhãn đã chuẩn hóa của ca SAU KHI cache của mọi (dấu hiệu, profile, view) đọc được và hữu hạn.
Chạy lại an toàn: ca đã có cache đầy đủ bị bỏ qua; ca lỗi giữ nguyên CT thô và được ghi vào <out-root>/failed.txt.

Nguồn ca: yaml của make_source_folds.py (data/processed/source_splits/<sign>_<arm>.yaml; khóa "sign", "roles").
Cache của một ca chỉ phụ thuộc (sign, seed, profile, view), nên ca có mặt ở cả nhánh Rm và N chỉ được render một lần.

  python scripts/run_rex_stream.py --root data/rex-drr --sources consolidation_Rm pleural_effusion_Rm --dry-run
  python scripts/run_rex_stream.py --root data/rex-drr --sources consolidation_Rm pleural_effusion_Rm --limit 8
  python scripts/run_rex_stream.py --root data/rex-drr --sources consolidation_Rm pleural_effusion_Rm
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

DEFAULT_SPLITS = ROOT / "data/processed/source_splits"


def load_sources(names, splits_dir=DEFAULT_SPLITS):
    """{volume: [sign, ...]} theo thứ tự xuất hiện; mỗi nguồn là stem yaml trong splits_dir."""
    import yaml
    out = {}
    for n in names:
        d = yaml.safe_load((Path(splits_dir) / f"{n}.yaml").read_text(encoding="utf8"))
        for case in d["roles"]:
            signs = out.setdefault(case, [])
            if d["sign"] not in signs:
                signs.append(d["sign"])
    return out


def sign_out_dir(out_root, sign):
    return Path(out_root) / sign


def cache_files(out_root, sign, case, profiles, views):
    from drr_pipeline import dataset as D
    return [D.cache_path(sign_out_dir(out_root, sign), p, v, case) for p in profiles for v in views]


def pending(jobs, out_root, profiles, views):
    """{volume: [sign cần render]} - bỏ ca-dấu hiệu đã có đủ file cache."""
    todo = {}
    for case, signs in jobs.items():
        need = [s for s in signs if not all(f.exists() for f in cache_files(out_root, s, case, profiles, views))]
        if need:
            todo[case] = need
    return todo


def cache_ok(path):
    """True nếu cache đọc được, có đủ khóa của hai kiểu pose và hữu hạn."""
    import numpy as np
    try:
        with np.load(path) as z:
            for kind in ("normal", "positive"):
                for k in (f"M_{kind}", f"R_{kind}", f"rot_{kind}", f"trans_{kind}"):
                    if k not in z.files or not np.isfinite(z[k]).all():
                        return False
        return True
    except Exception:
        return False


def batches(items, size):
    return [items[i:i + size] for i in range(0, len(items), size)]


def run(cmd, **kw):
    return subprocess.run([str(c) for c in cmd], cwd=ROOT, **kw)


def download(root, names, workers):
    chunks = [names[i::workers] for i in range(workers)]
    procs = [subprocess.Popen([sys.executable, "scripts/download_rex.py", "--root", str(root), "--stage", "download",
                               "--only", *c], cwd=ROOT) for c in chunks if c]
    codes = [p.wait() for p in procs]
    return all(c == 0 for c in codes)


def raw_files(root, name):
    from download_rex import original_ct_path
    return [Path(root) / "raw/ct-rate" / original_ct_path(name + ".nii.gz"),
            Path(root) / "raw/rexgroundingct/segmentations" / (name + ".nii.gz")]


def oriented_files(sign, name, oriented_root):
    base = Path(oriented_root) / sign
    return [base / "ct" / (name + ".nii.gz"), base / "labels" / (name + ".nii.gz")]


def prepare(root, plan, todo, names, oriented_root, workers):
    def one(item):
        name, sign = item
        r = run([sys.executable, "scripts/prepare_ctrate_case.py", "--root", root, "--volume", name, "--sign", sign,
                 "--plan", plan, "--out", Path(oriented_root) / sign, "--overwrite"], capture_output=True, text=True)
        return item, r.returncode == 0, (r.stderr or r.stdout)[-300:]
    work = [(n, s) for n in names for s in todo[n]]
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, work))


def produce(args, todo, names, plan):
    """Luồng nền: tải rồi chuẩn hóa một lô. Trả về {volume: lỗi hoặc None}."""
    t0 = time.time()
    if not download(args.root, names, args.dl_workers):
        return {n: "tải lỗi" for n in names}, 0.0
    t1 = time.time()
    errs = {}
    for (name, sign), ok, msg in prepare(args.root, plan, todo, names, args.oriented_root, args.prep_workers):
        if not ok:
            errs[name] = f"prepare {sign}: {msg}"
    print(f"  lô sẵn sàng: tải {t1 - t0:.0f}s, chuẩn hóa {time.time() - t1:.0f}s, {len(errs)} lỗi", flush=True)
    return errs, time.time() - t0


def render_and_verify(args, todo, names, bad):
    ok_names = [n for n in names if n not in bad]
    by_sign = {}
    for n in ok_names:
        for s in todo[n]:
            by_sign.setdefault(s, []).append(n)
    for sign, cases in by_sign.items():
        r = run([sys.executable, "scripts/generate_drr.py", "project", "--dataset", f"rex_{sign}", "--cases", *cases,
                 "--profiles", *args.profiles, "--views", *args.views, "--out-dir", sign_out_dir(args.out_root, sign),
                 "--seed", args.seed])
        if r.returncode != 0:
            for n in cases:
                bad[n] = "generate_drr lỗi"
    for n in ok_names:
        if n in bad:
            continue
        for s in todo[n]:
            for f in cache_files(args.out_root, s, n, args.profiles, args.views):
                if not cache_ok(f):
                    bad[n] = f"cache hỏng: {f}"
    return bad


def cleanup(args, todo, names, bad):
    for n in names:
        if n in bad:
            continue
        paths = raw_files(args.root, n)
        if not args.keep_oriented:
            paths += [p for s in todo[n] for p in oriented_files(s, n, args.oriented_root)]
        for p in paths:
            p.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--sources", nargs="+", required=True, help="stem yaml trong source_splits, ví dụ consolidation_Rm")
    ap.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS)
    ap.add_argument("--out-root", type=Path, default=Path("data/processed/drr"), help="cache nằm ở <out-root>/<sign>/cache/...")
    ap.add_argument("--oriented-root", type=Path, default=Path("data/processed/oriented"))
    ap.add_argument("--profiles", nargs="+", default=["moderate"])
    ap.add_argument("--views", nargs="+", default=["ap"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--dl-workers", type=int, default=4)
    ap.add_argument("--prep-workers", type=int, default=4)
    ap.add_argument("--limit", type=int, help="chỉ xử lý N volume chưa xong đầu tiên")
    ap.add_argument("--keep-oriented", action="store_true", help="giữ CT/nhãn chuẩn hóa (ví dụ để export hay thử độ phân giải)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sec-per-case-sign", type=float, default=7.0, help="giây DRR/ca/dấu hiệu/profile/view (đo trên A100: 7 s)")
    a = ap.parse_args()
    if a.batch < 1 or a.dl_workers < 1 or a.prep_workers < 1:
        ap.error("--batch, --dl-workers, --prep-workers phải dương")

    plan_path = a.root / "manifests/rex_download_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf8"))
    in_plan = {j["volume_name"].removesuffix(".nii.gz"): j for j in plan["jobs"]}

    jobs = load_sources(a.sources, a.splits_dir)
    missing = [n for n in jobs if n not in in_plan]
    if missing:
        sys.exit(f"{len(missing)} volume không có trong rex_download_plan.json (vd {missing[:3]}); chạy lại download_rex.py --stage plan")
    todo = pending(jobs, a.out_root, a.profiles, a.views)
    names = list(todo)[:a.limit]
    n_pairs = sum(len(todo[n]) for n in names)
    gb = sum((in_plan[n]["ct"] or {}).get("size", 0) + (in_plan[n]["mask"] or {}).get("size", 0) for n in names) / 1e9
    est_h = n_pairs * len(a.profiles) * len(a.views) * a.sec_per_case_sign / 3600
    print(f"nguồn {a.sources}: {len(jobs)} volume, {sum(map(len, jobs.values()))} cặp (volume, dấu hiệu)")
    print(f"còn phải làm: {len(names)} volume, {n_pairs} cặp, tải {gb:.1f} GB, "
          f"DRR ~{est_h:.1f} h (GPU) ở {a.sec_per_case_sign:g} s/cặp/profile/view, {len(batches(names, a.batch))} lô x {a.batch}")
    print(f"đỉnh đĩa ước tính ~{2 * a.batch * 0.2:.0f} GB (hai lô, ~0,2 GB/ca) + cache ~12 MB/cặp/profile/view")
    free = shutil.disk_usage(a.root).free / 1e9
    print(f"đĩa trống: {free:.0f} GB")
    if a.dry_run or not names:
        print("dry-run: không tải, không ghi gì." if a.dry_run else "không còn gì để làm.")
        return

    bs = batches(names, a.batch)
    failed_path = a.out_root / "failed.txt"
    a.out_root.mkdir(parents=True, exist_ok=True)
    t_start, n_done, n_bad = time.time(), 0, 0
    with ThreadPoolExecutor(1) as bg:
        nxt = bg.submit(produce, a, todo, bs[0], plan_path)
        for i, b in enumerate(bs):
            errs, _ = nxt.result()
            if i + 1 < len(bs):
                nxt = bg.submit(produce, a, todo, bs[i + 1], plan_path)
            bad = render_and_verify(a, todo, b, dict(errs))
            cleanup(a, todo, b, bad)
            n_done += len(b) - len(bad)
            n_bad += len(bad)
            if bad:
                with open(failed_path, "a", encoding="utf8") as f:
                    for n, why in bad.items():
                        f.write(f"{n}\t{why.strip()}\n")
            el = time.time() - t_start
            print(f"lô {i + 1}/{len(bs)}: xong {n_done}, lỗi {n_bad}, {el / 60:.1f} phút, "
                  f"còn ~{el / (i + 1) * (len(bs) - i - 1) / 60:.0f} phút", flush=True)
    print(f"kết thúc: {n_done} ca xong, {n_bad} lỗi" + (f" (xem {failed_path})" if n_bad else ""))


if __name__ == "__main__":
    main()
