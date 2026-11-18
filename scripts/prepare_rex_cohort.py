"""Build local manifests and review indexes before downloading the ReX cohort."""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drr_pipeline.rex_cohort import ANATOMY_ROIS, SIGNS, build_rows, isolate_holdout_patients, summarize


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    raw = args.root / "raw" / "rexgroundingct"
    load = lambda name: json.loads((raw / name).read_text(encoding="utf-8-sig"))
    rows = build_rows(load("dataset.json"), load("reports_dataset.json"))
    original_summary = summarize(rows)
    rows, split_changes = isolate_holdout_patients(rows)
    summary = summarize(rows)
    summary["original_patient_overlap"] = original_summary["patient_overlap"]
    summary["split_policy"] = "Patient isolation: test > val > train; source_split unchanged"
    summary["split_changes"] = split_changes
    if any(summary["patient_overlap"].values()):
        raise ValueError("Patient overlap between official splits; resolve before exporting")
    out = args.root / "manifests"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "ct_candidates.csv", rows, list(rows[0]))
    (out / "cohort_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    selected = [r for r in rows if any(r[s] == "present" for s in SIGNS)]
    volumes = [r["volume_name"] for r in selected]
    (out / "selected_volumes.txt").write_text("\n".join(volumes) + "\n", encoding="utf8")
    write_csv(out / "selected_ct.csv", selected, list(rows[0]))
    (out / "all_cohort_volumes.txt").write_text("\n".join(r["volume_name"] for r in rows) + "\n", encoding="utf8")
    # Retire generated binary indexes from the earlier design without deleting data.
    legacy = args.root / "review_indexes"
    archive = args.root / "archive" / "obsolete_binary_indexes"
    if legacy.exists():
        if legacy.resolve().parent != args.root.resolve() or archive.resolve().parent.parent != args.root.resolve():
            raise ValueError("Archive paths must stay within the data root")
        archive.parent.mkdir(parents=True, exist_ok=True)
        if archive.exists():
            raise ValueError("Legacy archive already exists; inspect it before moving older indexes")
        legacy.rename(archive)
    old_list = out / "positive_candidate_volumes.txt"
    if old_list.exists():
        retired_list = args.root / "archive" / "obsolete_positive_candidate_volumes.txt"
        retired_list.parent.mkdir(parents=True, exist_ok=True)
        if retired_list.exists():
            raise ValueError("Legacy volume-list archive already exists")
        old_list.rename(retired_list)
    # Per-finding membership lists, with no positive/negative subfolders.
    for sign in SIGNS:
        for split in ("train", "val", "test"):
            subset = [r for r in rows if r["split"] == split and r[sign] == "present"]
            fields = ["volume_name", "patient_id", "split", sign, sign + "_source", sign + "_evidence", sign + "_finding_ids", sign + "_review"]
            write_csv(args.root / "indexes" / sign / (split + ".csv"), subset, fields)
        review = [r for r in rows if r[sign] == "needs_review" or r[sign + "_source"] == "report_rule_weak"
                  or (sign == "pleural_effusion" and "2e" in r["annotation_codes"] and r[sign] != "present")]
        write_csv(args.root / "indexes" / sign / "review_required.csv", review, fields)
    (out / "anatomy_rois.json").write_text(json.dumps({"task": "total", "roi_subset": ANATOMY_ROIS}, indent=2), encoding="utf8")
    print(f"Prepared {len(rows)} CT records; selected {len(selected)} with requested findings. No binary labels exported.")
    for split, values in summary["splits"].items():
        print(split, "CT", values["ct"], "patients", values["patients"])
        for sign, counts in values["signs"].items():
            print(" ", sign, counts["annotation_status"], "patients with finding", counts["patients_with_finding"])
    print("Patient overlap:", {k: len(v) for k, v in summary["patient_overlap"].items()})
    print("Report-based labels require review. No DRR images generated.")


if __name__ == "__main__":
    main()
