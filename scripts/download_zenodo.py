"""Download a Zenodo record into data/raw/<name>, verifying MD5.

    python scripts/download_zenodo.py 3757476 data/raw/covid19_ctseg
    python scripts/download_zenodo.py 3757476 data/raw/covid19_ctseg --only COVID-19-CT-Seg_20cases.zip Lung_Mask.zip Infection_Mask.zip
    python scripts/download_zenodo.py 3757476 data/raw/covid19_ctseg --extract

With --extract every .zip is unpacked into <out_dir>/<zip name without .zip>/ (skipped when that
folder already has files). Without it, nothing is unpacked.

Standard library only. NOT yet tested against the live Zenodo API (written without network
access): if the JSON layout differs, adjust `list_files`.
"""
import argparse
import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

API = "https://zenodo.org/api/records/{rid}"


def list_files(rid: str):
    with urllib.request.urlopen(API.format(rid=rid)) as r:
        rec = json.load(r)
    files = rec.get("files") or rec.get("entries") or []
    out = []
    for f in files:
        name = f.get("key") or f.get("filename")
        url = (f.get("links") or {}).get("self") or (f.get("links") or {}).get("download")
        md5 = (f.get("checksum") or "").replace("md5:", "") or None
        out.append((name, url, md5, f.get("size")))
    return out


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_all(out_dir: Path):
    for z in sorted(out_dir.glob("*.zip")):
        target = out_dir / z.stem
        if target.exists() and any(target.iterdir()):
            print(f"already extracted: {z.name}")
            continue
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(target)  # zipfile drops absolute paths and '..' components
        print(f"extracted: {z.name} -> {target}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("record_id")
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--only", nargs="*", help="file names to fetch (default: all)")
    ap.add_argument("--extract", action="store_true", help="unpack every .zip in out_dir after downloading")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for name, url, md5, size in list_files(a.record_id):
        if a.only and name not in a.only:
            continue
        dest = a.out_dir / name
        if dest.exists() and md5 and md5sum(dest) == md5:
            print(f"ok (already downloaded): {name}")
            continue
        print(f"downloading {name} ({size} bytes)")
        urllib.request.urlretrieve(url, dest)
        if md5 and md5sum(dest) != md5:
            sys.exit(f"MD5 mismatch for {name}; delete it and retry")
        print(f"verified: {name}")
    if a.extract:
        extract_all(a.out_dir)


if __name__ == "__main__":
    main()
