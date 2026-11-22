"""Export only the selected ReX finding channels in their original voxel grid."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from shutil import copyfileobj
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile

import nibabel as nib
import numpy as np


def metadata_for(job, revision):
    return {"revision": revision, "source_sha256": job["sha256"], "channels": job["channels"]}


def refresh_from_export(job, previous, target, revision):
    """Reuse previously validated arrays when an updated plan only relabels or drops channels."""
    meta = metadata_for(job, revision)
    members = [f"finding_{item['finding_id']}.npy" for item in job["channels"]]
    if not previous.is_file():
        return False
    with ZipFile(previous) as source:
        with source.open("metadata.npy") as stream:
            old = json.loads(np.lib.format.read_array(stream, allow_pickle=False).item())
        if old["source_sha256"] != meta["source_sha256"] or old["revision"] != meta["revision"]:
            return False
        validated = {item["finding_id"]: item["annotated_voxels"] for item in old["channels"]}
        if any(item["finding_id"] not in validated or validated[item["finding_id"]] != item["annotated_voxels"]
               for item in job["channels"]):
            return False
        if not set(members).issubset(source.namelist()):
            return False
        if previous == target and old == meta and set(source.namelist()) == {*members, "metadata.npy"}:
            return True
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".npz.partial")
        with ZipFile(partial, "w", compression=ZIP_DEFLATED, compresslevel=1) as output:
            with output.open("metadata.npy", "w") as stream:
                np.lib.format.write_array(stream, np.asarray(json.dumps(meta)), allow_pickle=False)
            for member in members:
                with source.open(member) as reader, output.open(member, "w", force_zip64=True) as writer:
                    copyfileobj(reader, writer, length=1024 * 1024)
    partial.replace(target)
    return True


def extract(job, source, target, revision):
    shape = tuple(job["spatial_shape"])
    selected = job["channels"]
    keys = [f"finding_{item['finding_id']}" for item in selected]
    available = nib.load(source)
    if available.shape != (job["finding_count"], *shape):
        raise ValueError(f"Unexpected annotation shape {available.shape}; expected {(job['finding_count'], *shape)}")
    masks = np.asanyarray(available.dataobj)
    channels = {}
    for item, key in zip(selected, keys, strict=True):
        channel = np.asfortranarray(masks[item["finding_id"]])
        voxels = int(np.count_nonzero(channel))
        if voxels == 0 or (item["annotated_voxels"] is not None and voxels != item["annotated_voxels"]):
            raise ValueError(f"{key}: found {voxels} voxels; metadata lists {item['annotated_voxels']}")
        channels[key] = channel
    # The source NIfTI has a finding-first array. Do not assign its affine to a
    # derived 3D volume without checking the paired CT's spatial transform.
    metadata = json.dumps(metadata_for(job, revision))
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".npz.partial")
    with ZipFile(partial, "w", compression=ZIP_DEFLATED, compresslevel=1) as archive:
        for key, data in {"metadata": np.asarray(metadata), **channels}.items():
            with archive.open(key + ".npy", "w", force_zip64=True) as stream:
                np.lib.format.write_array(stream, data, allow_pickle=False)
    partial.replace(target)


def extract_one(args):
    job, raw, output, reuse, revision = args
    source = raw / job["path"]
    relative = Path(job["split"]) / (job["volume_name"].removesuffix(".nii.gz") + ".npz")
    target = output / relative
    try:
        for previous in (target, reuse / relative) if reuse else (target,):
            try:
                if refresh_from_export(job, previous, target, revision):
                    return "done", None
            except (BadZipFile, KeyError, ValueError):
                continue
        if not source.is_file():
            return "missing", None
        extract(job, source, target, revision)
        return "done", None
    except (ValueError, OSError, IndexError) as error:
        return "error", {"volume_name": job["volume_name"], "error": str(error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=1)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reuse", type=Path, help="Earlier NPZ export with validated channels for this revision")
    args = parser.parse_args()
    plan_path = args.plan or args.root / "manifests/rex_mask_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))
    output = args.output or args.root / "processed" / "rex_findings"
    raw = args.root / "raw" / "rexgroundingct"
    done, missing, errors = 0, 0, []
    tasks = ((job, raw, output, args.reuse, plan["revision"]) for job in plan["jobs"])
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for status, error in pool.map(extract_one, tasks):
            if status == "missing":
                missing += 1
            elif error:
                errors.append(error)
            else:
                done += 1
                if done == 1 or done % 50 == 0:
                    print(f"Extracted {done} ReX volumes", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / "extraction_status.json").write_text(json.dumps(
        {"extracted": done, "missing_source": missing, "errors": errors}, indent=2), encoding="utf8")
    print(f"Extracted {done}; missing source {missing}; extraction errors {len(errors)}", flush=True)
    return bool(missing or errors)


if __name__ == "__main__":
    raise SystemExit(main())
