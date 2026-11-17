"""Inspect a real chest X-ray dataset (default: VinDr-PCXR) and write a report.

    pip install pydicom            # only this script needs it
    python scripts/inspect_cxr.py                       # all CSV statistics + 300 DICOM headers per split
    python scripts/inspect_cxr.py --dicom-sample 0      # CSVs only
    python scripts/inspect_cxr.py --dicom-sample -1     # every DICOM header (slow on 9k files)

Reads only CSV files and DICOM HEADERS (no pixels are decoded, nothing is uploaded or copied).
Output: outputs/eda_cxr/<dataset>_report.md. The column layout was unknown when this was written, so
the script first prints the columns and the first rows of every CSV and states how it interpreted them;
paste that part back if the interpretation is wrong.
"""
import argparse
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drr_pipeline import config, cxr_eda as E  # noqa: E402


def head(path, k=3):
    cols, rows = E.read_csv(path)
    print(f"\n== {path.name}: {len(rows)} rows, columns: {cols}")
    for r in rows[:k]:
        print("  ", dict(list(r.items())[:8]))
    return cols, rows


def dicom_stats(folder: Path, n: int, seed: int):
    import pydicom
    files = sorted(folder.rglob("*.dicom")) + sorted(folder.rglob("*.dcm"))
    if not files:
        files = [f for f in sorted(folder.rglob("*")) if f.is_file()]
    total = len(files)
    if n >= 0 and n < total:
        files = random.Random(seed).sample(files, n)
    ages, rows, cols_, views, photo, sex, pix, bits, bad = [], [], [], Counter(), Counter(), Counter(), [], Counter(), 0
    for f in files:
        try:
            d = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
        except Exception:
            bad += 1
            continue
        a = E.parse_age_months(str(getattr(d, "PatientAge", "")))
        ages.append(a)
        rows.append(int(getattr(d, "Rows", 0) or 0))
        cols_.append(int(getattr(d, "Columns", 0) or 0))
        views[str(getattr(d, "ViewPosition", "?"))] += 1
        photo[str(getattr(d, "PhotometricInterpretation", "?"))] += 1
        sex[str(getattr(d, "PatientSex", "?"))] += 1
        bits[str(getattr(d, "BitsStored", "?"))] += 1
        ps = getattr(d, "PixelSpacing", None) or getattr(d, "ImagerPixelSpacing", None)
        if ps:
            pix.append(float(ps[0]))
    return {"over_120_months": sum(1 for x in ages if x is not None and x > 120), "files_total": total, "read": len(files) - bad, "unreadable": bad,
            "age_months": E.summarize_numeric(ages), "age_missing": sum(a is None for a in ages),
            "rows": E.summarize_numeric(rows), "cols": E.summarize_numeric(cols_),
            "pixel_spacing_mm": E.summarize_numeric(pix), "view": dict(views),
            "photometric": dict(photo), "sex": dict(sex), "bits_stored": dict(bits)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="vindr_pcxr")
    ap.add_argument("--dicom-sample", type=int, default=300, help="headers per split; 0 = none, -1 = all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/eda_cxr"))
    a = ap.parse_args()
    ds = config.load_cxr_dataset(a.dataset)
    lines = [f"# {a.dataset}: report", "", f"root: `{ds['base_dir']}`", ""]
    for split in ("train", "test"):
        lines.append(f"## {split}")
        lp, ap_ = ds["image_labels"][split], ds["annotations"][split]
        for path, kind in ((lp, "labels"), (ap_, "boxes")):
            if not path.exists():
                lines.append(f"- missing {path}")
                continue
            cols, rows = head(path)
            s = E.summarize_labels(cols, rows) if kind == "labels" else E.summarize_boxes(cols, rows)
            if kind == "labels":
                s["groups_report"] = E.label_groups(cols, rows, ds.get("groups", {}))
                d = ds[f"{split}_dir"]
                stems = [f.stem for f in d.rglob("*") if f.is_file()]
                s["file_match"] = E.match_files([r["image_id"] for r in rows], stems)
            lines += [f"### {path.name}", f"columns: {cols}", "", f"```\n{s}\n```", ""]
            print(f"[{split}/{kind}] extra:", {k: s[k] for k in ("groups_report", "file_match") if k in s}) if kind == "labels" else None
            print(f"[{split}/{kind}] interpretation:", {k: v for k, v in s.items() if k not in ("positive_images", "positives_rows", "boxes_per_class", "images_per_class", "rows_per_label")})
        if a.dicom_sample != 0:
            d = ds[f"{split}_dir"]
            try:
                st = dicom_stats(d, a.dicom_sample, a.seed)
                lines += [f"### DICOM headers ({split}, sample {a.dicom_sample})", f"```\n{st}\n```", ""]
                print(f"[{split}] DICOM:", st)
            except ImportError:
                print("pydicom not installed: pip install pydicom (or --dicom-sample 0)")
                lines.append("- pydicom missing, DICOM headers skipped")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    out = a.out_dir / f"{a.dataset}_report.md"
    out.write_text("\n".join(lines), encoding="utf8")
    print("\nreport:", out)


if __name__ == "__main__":
    main()
