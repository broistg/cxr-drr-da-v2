"""Một điểm vào cho các kiểm tra sau huấn luyện (không huấn luyện lại; dùng best.pt/predictions của run đã có).

    python scripts/diagnose.py confounds         --runs RUN [RUN ...]   # điểm có bị giải thích bởi yếu tố ngoài tổn thương (tuổi, độ sáng...)?
    python scripts/diagnose.py gradcam           --runs RUN [RUN ...]   # CAM có chỉ vào tổn thương (hộp/mask)?
    python scripts/diagnose.py occlusion         --runs RUN [RUN ...]   # che vùng tổn thương/vùng ảnh: điểm giảm bao nhiêu?
    python scripts/diagnose.py label-sensitivity DIR [--arm a0]         # độ nhạy theo định nghĩa nhãn (không cần torch)
Cờ còn lại chuyển nguyên cho script con (xem --help của từng lệnh: `diagnose.py gradcam --help`). Mã ở scripts/diag/.
"""
import runpy
import sys
from pathlib import Path

CMDS = {"confounds": "confounds", "gradcam": "gradcam", "occlusion": "occlusion", "label-sensitivity": "label_sensitivity"}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in CMDS:
        sys.exit(__doc__)
    name = CMDS[sys.argv[1]]
    script = Path(__file__).resolve().parent / "diag" / f"{name}.py"
    sys.argv = [f"diagnose.py {sys.argv[1]}"] + sys.argv[2:]
    runpy.run_path(str(script), run_name="__main__")
