"""Prepare or explicitly download the pinned ReXGroundingCT and CT-RATE pairs."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drr_pipeline.ctrate import parse_volume_name


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".json.partial")
    partial.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    partial.replace(path)


def original_ct_path(name):
    volume = parse_volume_name(name)
    if volume.split not in {"train", "valid"} or name != volume.recon + ".nii.gz":
        raise ValueError(f"Expected an exact CT-RATE NIfTI filename: {name!r}")
    return f"dataset/{volume.split}/{volume.patient}/{volume.scan}/{name}"


def file_record(path, size, sha256):
    if not isinstance(size, int) or size <= 0 or not re.fullmatch(r"[0-9a-f]{64}", sha256 or ""):
        raise ValueError(f"Missing file size or LFS SHA-256: {path}")
    return {"path": path, "size": size, "sha256": sha256}


def fetch_metadata(root, sources, api, download):
    repo = sources["rex_repo"]
    revision = sources["rex_revision"]
    for filename in ("dataset.json", "reports_dataset.json"):
        download(repo, filename, repo_type="dataset", revision=revision,
                 local_dir=root / "raw/rexgroundingct")
    index = []
    for entry in api.list_repo_tree(repo, path_in_repo="segmentations", repo_type="dataset", revision=revision):
        if not entry.path.endswith(".nii.gz"):
            continue
        info = file_record(entry.path, entry.size, getattr(entry.lfs, "sha256", None))
        index.append({"rfilename": info["path"], "size": info["size"], "lfs": {"sha256": info["sha256"]}})
    if not index:
        raise ValueError("ReX segmentation index is empty; check dataset access")
    write_json(root / "manifests/rex_files.json", sorted(index, key=lambda entry: entry["rfilename"]))
    write_json(root / "manifests/revisions.json", sources)
    print(f"ReX metadata ready; indexed {len(index)} mask files.")


def matching_ct_files(api, sources, paths):
    """Check selected full paths, avoiding the truncated full-repo listing."""
    entries = {}
    for start in range(0, len(paths), 100):
        batch = paths[start:start + 100]
        files = api.get_paths_info(sources["ct_repo"], batch, repo_type="dataset", revision=sources["ct_revision"])
        for entry in files:
            if entry.path in batch:
                entries[entry.path] = file_record(entry.path, entry.size, getattr(entry.lfs, "sha256", None))
    return entries


def build_download_plan(mask_plan, sources, api, include_cardiomegaly=False, extra_ct=None):
    if (mask_plan["repo"], mask_plan["revision"]) != (sources["rex_repo"], sources["rex_revision"]):
        raise ValueError("The mask manifest must use the pinned ReX repository and revision")

    heart = {row["volume_name"]: row for row in mask_plan["cardiomegaly_candidates"]}
    jobs = {}
    for row in mask_plan["jobs"]:
        name = row["volume_name"]
        if name in jobs or row["path"] != f"segmentations/{name}":
            raise ValueError(f"Duplicate volume or unexpected ReX mask path: {name}")
        jobs[name] = {"volume_name": name, "patient_id": row["patient_id"], "split": row["split"],
                      "source_split": row["source_split"], "channels": row["channels"],
                      "mask": file_record(row["path"], row["size"], row["sha256"])}
    if include_cardiomegaly:
        for name, row in heart.items():
            jobs.setdefault(name, {"volume_name": name, "patient_id": row["patient_id"], "split": row["split"],
                                   "source_split": row["source_split"], "channels": [], "mask": None})
    # CT không có mask (ca âm, D60): chỉ tải CT; không ghi đè ca đã có mask. Mỗi dòng cần volume_name, patient_id, source_split, split.
    for row in extra_ct or []:
        jobs.setdefault(row["volume_name"], {"volume_name": row["volume_name"], "patient_id": row["patient_id"], "split": row["split"],
                                             "source_split": row["source_split"], "channels": [], "mask": None})
    for name in jobs.keys() & heart.keys():
        jobs[name]["cardiomegaly_candidate"] = heart[name]

    paths = {name: original_ct_path(name) for name in sorted(jobs)}
    available = matching_ct_files(api, sources, list(paths.values()))
    for name, path in paths.items():
        jobs[name]["ct"] = available.get(path)
    missing = [{"volume_name": name, "expected_path": path}
               for name, path in paths.items() if path not in available]
    return {"sources": sources, "jobs": [jobs[name] for name in paths], "missing_ct": missing,
            "unavailable_masks": mask_plan["unavailable_masks"], "includes_report_only_ct": include_cardiomegaly,
            "extra_ct": sorted(r["volume_name"] for r in extra_ct or [])}


def download_jobs(root, plan, sources, download, files="both", limit=None, only=None):
    if plan["sources"] != sources:
        raise ValueError("Download plan revisions differ from the committed source revisions")
    modes = {"both": ("mask", "ct"), "masks": ("mask",), "ct": ("ct",)}
    targets = []
    jobs = [j for j in plan["jobs"] if only is None or j["volume_name"].removesuffix(".nii.gz") in only]
    if only is not None and len(jobs) != len(only):
        raise ValueError(f"--only: {len(only) - len(jobs)} volume không có trong kế hoạch tải")
    for job in jobs[:limit]:
        name = job["volume_name"]
        ct_path = original_ct_path(name)
        for kind in modes[files]:
            entry = job[kind]
            if entry is None:
                if kind == "ct":
                    raise ValueError(f"CT missing at the pinned original path: {ct_path}")
                continue  # A report-only cardiomegaly candidate has no ReX mask.
            repo_key = "rex" if kind == "mask" else "ct"
            folder = "rexgroundingct" if kind == "mask" else "ct-rate"
            expected = f"segmentations/{name}" if kind == "mask" else ct_path
            if entry["path"] != expected:
                raise ValueError(f"Unexpected {kind} path for {name}: {entry['path']!r}")
            file_record(expected, entry["size"], entry["sha256"])
            dest = root / "raw" / folder
            targets.append((entry, dest, repo_key))
    if not targets:
        raise ValueError("No files selected for download")

    def verified(path, entry):
        if not path.is_file() or path.stat().st_size != entry["size"]:
            return False
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest() == entry["sha256"]

    pending = [(entry, dest, key) for entry, dest, key in targets if not verified(dest / entry["path"], entry)]
    required = sum(entry["size"] for entry, _, _ in pending)
    if required > shutil.disk_usage(root).free:
        raise ValueError(f"Insufficient disk space: downloads need {required / 1e9:.2f} GB")
    for number, (entry, dest, key) in enumerate(pending, 1):
        path = Path(download(sources[f"{key}_repo"], entry["path"], repo_type="dataset",
                             revision=sources[f"{key}_revision"], local_dir=dest,
                             force_download=(dest / entry["path"]).exists()))
        if not verified(path, entry):
            raise ValueError(f"Downloaded size or SHA-256 mismatch: {entry['path']}")
        print(f"Verified download {number}/{len(pending)}", flush=True)
    print(f"Verified {len(targets)} files ({len(targets) - len(pending)} already present).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=("metadata", "plan", "download"))
    parser.add_argument("--mask-plan", type=Path, help="Override the local mask manifest for the plan stage")
    parser.add_argument("--include-cardiomegaly", action="store_true", help="Include CT for report-only candidates in the plan")
    parser.add_argument("--extra-ct-list", type=Path, nargs="+", help="một hoặc nhiều CSV (volume_name,patient_id,source_split,split) của CT cần tải mà không có mask, ví dụ ca âm từ make_source_folds.py; ca trùng chỉ tính một lần")
    parser.add_argument("--files", choices=("both", "masks", "ct"), default="both")
    parser.add_argument("--only", nargs="+", metavar="VOLUME", help="Chỉ tải các volume này (tên có hoặc không có .nii.gz); phải nằm trong kế hoạch")
    parser.add_argument("--limit", type=int, help="Download only the first N volumes from the prepared plan")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError
    from httpx import TransportError

    sources = read_json(ROOT / "configs/rex_sources.json")
    manifests = args.root / "manifests"
    target = manifests / "rex_download_plan.json"
    try:
        if args.stage == "metadata":
            fetch_metadata(args.root, sources, HfApi(), hf_hub_download)
        elif args.stage == "plan":
            masks = read_json(args.mask_plan or manifests / "rex_mask_plan.json")
            extra = None
            if args.extra_ct_list:
                import csv
                extra = {}
                for path in args.extra_ct_list:
                    with open(path, encoding="utf8", newline="") as f:
                        for row in csv.DictReader(f):
                            extra.setdefault(row["volume_name"], row)
                extra = list(extra.values())
            plan = build_download_plan(masks, sources, HfApi(), args.include_cardiomegaly, extra)
            write_json(target, plan)
            ct_bytes = sum(job["ct"]["size"] for job in plan["jobs"] if job["ct"])
            mask_bytes = sum(job["mask"]["size"] for job in plan["jobs"] if job["mask"])
            print(f"Plan ready: {len(plan['jobs'])} volumes, {ct_bytes / 1e9:.2f} GB CT and {mask_bytes / 1e9:.2f} GB masks.")
            if plan["missing_ct"]:
                print(f"Missing {len(plan['missing_ct'])} original CT paths; see {target.name}.", file=sys.stderr)
                return 2
        else:
            download_jobs(args.root, read_json(target), sources, hf_hub_download, args.files, args.limit,
                          None if not args.only else {v.removesuffix(".nii.gz") for v in args.only})
    except (HfHubHTTPError, TransportError) as exc:
        response = getattr(exc, "response", None)
        status = response.status_code if response is not None else "no HTTP response"
        print(f"Hugging Face request failed ({status}). Check network, login and dataset access.", file=sys.stderr)
        return 2
    except (OSError, TypeError, ValueError, KeyError) as exc:
        print(f"ReX pipeline error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
