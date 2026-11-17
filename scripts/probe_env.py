"""In thông tin môi trường (Python, hệ điều hành, GPU/VRAM, thư viện, ổ đĩa) để chọn cấu hình tính toán.

    python scripts/probe_env.py                       # in JSON ra màn hình
    python scripts/probe_env.py --out env.json        # ghi thêm ra tệp
    python scripts/probe_env.py --data-root data      # kiểm tra dung lượng trống ở thư mục dữ liệu

Không bắt buộc torch/diffdrr: thiếu thì ghi "không cài". Dán đầu ra vào docs/COMPUTE.md (bảng "Máy đã đo").
"""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from importlib import metadata


def version_of(pkg):
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return "không cài"


def gpu_info():
    out = {"torch_cuda_available": None, "devices": [], "nvidia_smi": None}
    try:
        import torch  # noqa: WPS433

        out["torch_cuda_available"] = bool(torch.cuda.is_available())
        out["torch_cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                out["devices"].append({"index": i, "name": p.name, "total_mib": round(p.total_memory / 2**20),
                                       "capability": f"{p.major}.{p.minor}"})
    except Exception as e:  # noqa: BLE001
        out["torch_error"] = repr(e)
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=15)
        out["nvidia_smi"] = r.stdout.strip() or r.stderr.strip()
    except Exception as e:  # noqa: BLE001
        out["nvidia_smi"] = f"không chạy được: {e!r}"
    return out


def ram_mib():
    try:
        if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names:
            return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**20)
    except Exception:  # noqa: BLE001
        pass
    try:
        import psutil  # noqa: WPS433

        return round(psutil.virtual_memory().total / 2**20)
    except Exception:  # noqa: BLE001
        return None


def collect(data_root):
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "ram_mib": ram_mib(),
        "packages": {p: version_of(p) for p in ("torch", "torchvision", "diffdrr", "torchio", "numpy", "scipy", "nibabel", "pydicom", "pandas")},
        "gpu": gpu_info(),
    }
    root = data_root if os.path.isdir(data_root) else "."
    du = shutil.disk_usage(root)
    info["disk_free_gib"] = round(du.free / 2**30, 1)
    info["disk_checked_path"] = os.path.abspath(root)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--data-root", default="data")
    a = ap.parse_args()
    info = collect(a.data_root)
    text = json.dumps(info, indent=2, ensure_ascii=False)
    print(text)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
