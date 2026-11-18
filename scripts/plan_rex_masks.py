"""Select ReX channels by category and finding text, using the pinned file index."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drr_pipeline.ctrate import parse_volume_name
from drr_pipeline.rex import iter_entries
from drr_pipeline.rex_cohort import isolate_holdout_patients, report_vote


RULES = {
    "2g": ("pneumothorax", r"\bpneumothora(?:x|ces)\b|\b(?:free )?air\b.{0,40}\b(?:pleural space|hemithorax)\b"),
    "2e": ("pleural_effusion", r"\beffusions?\b|\b(?:pleural fluid|fluid in the pleural space)\b"),
}
UNCERTAIN = re.compile(r"\b(?:no evidence of|no longer|resolved|possible|suspected|cannot exclude|rule out)\b|^\s*(?:no|without)\b", re.I)
SUBTYPE_TERMS = {
    "atelectasis": re.compile(r"\b(?:atelecta\w+|collaps(?:e|ed|ing))\b", re.I),
    "consolidation": re.compile(r"\bconsolidat\w+\b", re.I),
}
SUBTYPE_UNCERTAIN = re.compile(
    r"\b(?:or|versus|vs|suggestive of|probable|likely|questionable|rather than|absent|not (?:seen|detected|present|identified|excluded))\b",
    re.I,
)


def classify_2b(sentence):
    """Assign at most one subtype to a ReX finding and its single mask channel."""
    matches = [sign for sign, terms in SUBTYPE_TERMS.items() if terms.search(sentence)]
    if len(matches) > 1:
        return None, "both_subtypes"
    if UNCERTAIN.search(sentence) or SUBTYPE_UNCERTAIN.search(sentence):
        return None, "uncertain_or_alternative"
    if not matches:
        return None, "no_explicit_subtype"
    return matches[0], None


def finding_matches(code, sentence):
    if code == "2b":
        return classify_2b(sentence)[0] is not None
    if UNCERTAIN.search(sentence) or not re.search(RULES[code][1], sentence, re.I):
        return False
    if code == "2e" and re.search(r"\b(?:thicken\w*|plaque\w*|pericardial)\b", sentence, re.I):
        return False
    return True


def build_plan(data, reports, index, revision):
    indexed = {entry["rfilename"]: entry for entry in index}
    entries = list(iter_entries(data, splits=None))
    identities = [{"volume_name": n, "patient_id": parse_volume_name(n).patient,
                   "source_split": s} for s, n, _ in entries]
    assigned, _ = isolate_holdout_patients(identities)
    jobs, unavailable, review, cardiac = [], [], [], []
    for identity, (_, name, entry) in zip(assigned, entries, strict=True):
        channels = []
        for channel, code in entry["categories"].items():
            if code not in RULES and code != "2b":
                continue
            finding = entry["findings"].get(channel, "")
            sign, reason = classify_2b(finding) if code == "2b" else (RULES[code][0], None)
            item = {"sign": sign, "finding_id": int(channel), "finding": finding,
                    "category": code, "annotated_voxels": entry.get("pixels", {}).get(channel)}
            if item["annotated_voxels"] == 0:
                reason = "empty_mask"
            elif code != "2b" and not finding_matches(code, finding):
                reason = "finding_does_not_match"
            if reason is None:
                channels.append(item)
            else:
                review.append({**identity, **item, "review_reason": reason})

        volume = parse_volume_name(name)
        report = reports.get(identity["source_split"], {}).get(volume.scan + ".nii.gz", {})
        label, evidence = report_vote("cardiomegaly", report.get("Findings", {}).get("Heart/Vessels", []))
        if label == "1":
            cardiac.append({**identity, "finding": evidence, "source": "report_rule_weak",
                            "mask_available": False})

        if channels:
            path = "segmentations/" + name
            row = {**identity, "path": path, "spatial_shape": entry["shape"],
                   "finding_count": len(entry["findings"]), "channels": channels}
            remote = indexed.get(path)
            if remote is None:
                unavailable.append(row)
            else:
                jobs.append({**row, "size": remote["size"], "sha256": remote["lfs"]["sha256"]})

    return {"repo": "rajpurkarlab/ReXGroundingCT", "revision": revision, "jobs": jobs,
            "unavailable_masks": unavailable, "review_findings": review, "cardiomegaly_candidates": cardiac}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Path for the generated JSON manifest")
    args = parser.parse_args()
    manifests = args.root / "manifests"
    metadata = args.root / "raw" / "rexgroundingct"

    def read(path):
        return json.loads(path.read_text(encoding="utf-8-sig"))

    plan = build_plan(read(metadata / "dataset.json"), read(metadata / "reports_dataset.json"),
                      read(manifests / "rex_files.json"), read(manifests / "revisions.json")["rex_revision"])
    target = args.output or manifests / "rex_mask_plan.json"
    partial = target.with_suffix(".json.partial")
    partial.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf8")
    partial.replace(target)
    for split in ("train", "val", "test"):
        counts = Counter(sign for job in plan["jobs"] if job["split"] == split
                         for sign in {channel["sign"] for channel in job["channels"]})
        print(split, dict(counts), "cardiomegaly report candidates", sum(
            row["split"] == split for row in plan["cardiomegaly_candidates"]))
    print("Masks", len(plan["jobs"]), "bytes", sum(j["size"] for j in plan["jobs"]),
          "unavailable", len(plan["unavailable_masks"]), "findings for review", len(plan["review_findings"]))


if __name__ == "__main__":
    main()
