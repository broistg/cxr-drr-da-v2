"""Run a whole experiment suite from one YAML file (see src/drr_pipeline/suite.py for the format).

    python scripts/run_suite.py configs/experiments/e14_seg.yaml              # train every variant x seed, then diagnostics + summary
    python scripts/run_suite.py configs/experiments/e14_seg.yaml --dry-run    # print the commands only
    python scripts/run_suite.py configs/experiments/e14_seg.yaml --only a0_seg1
    python scripts/run_suite.py ... --force                                   # retrain even if outputs exist
    python scripts/run_suite.py ... --no-diagnostics --no-summary

Runs that already have outputs/exp1/<run>/metrics.json are skipped (train_cls.py --skip-if-done), so an interrupted suite can simply be
started again. A failed run does not stop the suite; it is listed at the end and the exit code is 1.
Everything printed by the diagnostics and the summary is also collected in ONE file, outputs/exp1/_suite_reports/<name>/REPORT.txt:
send that file (or paste it) when you want the results looked at. Per-run training logs go to outputs/exp1/_suite_logs/.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drr_pipeline import suite as S  # noqa: E402


def run_cmd(cmd, log_path=None, echo=True):
    """Run, stream stdout+stderr to the console and a log file; return (returncode, full text)."""
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}   # unbuffered: per-epoch lines of the child appear at once
    chunks = []
    f = open(log_path, "w", encoding="utf8") if log_path else None
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf8", errors="replace",
                             bufsize=1, env=env)
        for line in p.stdout:
            chunks.append(line)
            if f:
                f.write(line)
            if echo:
                sys.stdout.write(line)
        p.wait()
    finally:
        if f:
            f.close()
    return p.returncode, "".join(chunks)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("suite", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", nargs="+", default=None, help="run only these variant names")
    ap.add_argument("--force", action="store_true", help="do not skip runs that already have metrics.json")
    ap.add_argument("--no-diagnostics", action="store_true")
    ap.add_argument("--no-summary", action="store_true")
    ap.add_argument("--train-script", default=None, help="overrides the suite file's train-script (default scripts/train_cls.py)")
    ap.add_argument("--out-root", type=Path, default=Path("outputs/exp1"))
    a = ap.parse_args()

    suite = S.load_suite(a.suite)
    jobs = S.expand(suite, a.only)
    a.train_script = a.train_script or suite.get("train-script") or "scripts/train_cls.py"
    py = sys.executable
    print(f"suite '{suite['name']}': {len(jobs)} training runs; diagnostics {suite['diagnostics'] or 'none'}; summary {suite['summary']}")
    if a.dry_run:
        for j in jobs:
            print(" ".join([py, a.train_script, *j["flags"], *([] if a.force else ["--skip-if-done"])]))
        for v in dict.fromkeys(j["variant"] for j in jobs):
            for d in ([] if a.no_diagnostics else suite["diagnostics"]):
                print(" ".join([py, *S.diagnostic_cmd(d, [f"<runs of {v}>"], suite["diagnostic_args"].get(d))]))
        return

    log_dir = a.out_root / "_suite_logs"; rep_dir = a.out_root / "_suite_reports" / suite["name"]
    log_dir.mkdir(parents=True, exist_ok=True); rep_dir.mkdir(parents=True, exist_ok=True)
    report = [f"SUITE {suite['name']}  started {dt.datetime.now():%Y-%m-%d %H:%M:%S}", f"file {a.suite}", suite.get("description", "") or ""]
    runs_of, status = {}, []
    for k, j in enumerate(jobs, 1):
        cmd = [py, a.train_script, *j["flags"], *([] if a.force else ["--skip-if-done"])]
        print(f"\n[{k}/{len(jobs)}] {j['variant']} seed {j['seed']}\n  " + " ".join(cmd))
        t0 = time.time()
        rc, text = run_cmd(cmd, log_dir / f"{suite['name']}__{j['variant']}__s{j['seed']}.log", echo=True)
        run = S.parse_run_name(text)
        status.append((j["variant"], j["seed"], run, "ok" if rc == 0 and run else f"FAILED (exit {rc})", round(time.time() - t0)))
        if rc == 0 and run:
            runs_of.setdefault(j["variant"], []).append(run)
    report += ["", "RUNS (variant, seed, run name, status, seconds)"] + ["  " + " | ".join(map(str, s)) for s in status]

    if not a.no_diagnostics:
        for v, runs in runs_of.items():
            for d in suite["diagnostics"]:
                cmd = [py, *S.diagnostic_cmd(d, runs, suite["diagnostic_args"].get(d))]
                print(f"\n[diagnostic] {d} on {v}")
                rc, text = run_cmd(cmd, echo=True)
                report += ["", f"### {d} :: {v}" + ("" if rc == 0 else f"  (FAILED, exit {rc})"), text.rstrip()]
    if suite["summary"] and not a.no_summary:
        for flag in ("--per-run", "--md"):
            rc, text = run_cmd([py, "scripts/summarize_exp1.py", flag, *S.to_flags(suite["summary_args"])], echo=False)
            report += ["", f"### summarize_exp1.py {flag}" + ("" if rc == 0 else f"  (FAILED, exit {rc})"), text.rstrip()]
    (rep_dir / "REPORT.txt").write_text("\n".join(report) + "\n", encoding="utf8")
    failed = [s for s in status if not s[3].startswith("ok")]
    print(f"\nDONE. Report: {rep_dir / 'REPORT.txt'}  | runs {len(status)}, failed {len(failed)}")
    for s in failed:
        print("  FAILED:", s[:3])
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
