import subprocess
import sys
from pathlib import Path

import pytest

from drr_pipeline import suite as S

ROOT = Path(__file__).resolve().parents[1]


def test_to_flags():
    assert S.to_flags({"arm": "a0", "gamma": 1, "views": ["ap", "lateral"], "shuffle-labels": True, "x": False, "y": None}) == \
        ["--arm", "a0", "--gamma", "1", "--views", "ap", "lateral", "--shuffle-labels"]


def _write(tmp_path, text):
    p = tmp_path / "s.yaml"; p.write_text(text, encoding="utf8"); return p


def test_load_and_expand(tmp_path):
    p = _write(tmp_path, """
name: t
defaults: {arm: a1, views: [ap]}
seeds: [0, 1]
variants:
  - {name: v1, args: {gamma: 1}}
  - {name: v2, args: {gamma: 0}, seeds: [5]}
diagnostics: [confounds]
""")
    s = S.load_suite(p)
    jobs = S.expand(s)
    assert [(j["variant"], j["seed"]) for j in jobs] == [("v1", 0), ("v1", 1), ("v2", 5)]
    assert jobs[0]["flags"] == ["--arm", "a1", "--views", "ap", "--gamma", "1", "--seed", "0"]
    assert [j["variant"] for j in S.expand(s, only=["v2"])] == ["v2"]


def test_validation(tmp_path):
    bodies = [
        "name: t\nvariants: [{name: a, args: {arm: a0}}, {name: a, args: {arm: a0}}]",       # duplicate names
        "name: t\nvariants: [{name: a, args: {arm: a0, seed: 3}}]",                           # seed in args
        "name: t\nvariants: [{name: a, args: {}}]",                                           # no arm
        "name: t\nvariants: [{name: a, args: {arm: a0}}]\ndiagnostics: [nonsense]",           # unknown diagnostic
    ]
    for body in bodies:
        with pytest.raises(ValueError):
            S.load_suite(_write(tmp_path, body))


def test_parse_run_name_and_diag_cmd():
    assert S.parse_run_name("x\nRUN_NAME=a0_moderate_ap_f0_s0\nmore\nRUN_NAME=b\n") == "b"
    assert S.parse_run_name("nothing") is None
    assert S.diagnostic_cmd("gradcam", ["r1", "r2"], {"annotations": "a.csv"}) == ["scripts/diagnose.py", "gradcam", "--runs", "r1", "r2", "--annotations", "a.csv"]


def test_shipped_suites_load():
    for f in sorted((ROOT / "configs" / "experiments").glob("*.yaml")):
        s = S.load_suite(f)
        assert S.expand(s), f


def test_runner_end_to_end_with_fake_trainer(tmp_path):
    suite = tmp_path / "s.yaml"
    suite.write_text("name: fake\ndefaults: {arm: a0}\nseeds: [0, 1]\nvariants:\n  - {name: v, args: {gamma: 2}}\nsummary: false\n", encoding="utf8")
    fake = tmp_path / "fake_train.py"
    fake.write_text("import sys\nseed = sys.argv[sys.argv.index('--seed') + 1]\nprint('hello')\nprint(f'RUN_NAME=fake_s{seed}')\n", encoding="utf8")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_suite.py"), str(suite), "--train-script", str(fake),
                        "--out-root", str(tmp_path / "out"), "--no-diagnostics", "--no-summary"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    rep = (tmp_path / "out" / "_suite_reports" / "fake" / "REPORT.txt").read_text(encoding="utf8")
    assert "fake_s0" in rep and "fake_s1" in rep and "ok" in rep
    assert (tmp_path / "out" / "_suite_logs" / "fake__v__s1.log").exists()


def test_summary_args_default_and_flags(tmp_path):
    p = _write(tmp_path, "name: t\ndefaults: {arm: a0}\nvariants: [{name: v}]\nsummary_args: {dir: outputs/e10a}\n")
    s = S.load_suite(p)
    assert S.to_flags(s["summary_args"]) == ["--dir", "outputs/e10a"]
    assert S.load_suite(_write(tmp_path, "name: t\ndefaults: {arm: a0}\nvariants: [{name: v}]\n"))["summary_args"] == {}


def test_suite_file_selects_train_script(tmp_path):
    """a suite file with train-script must use that script without a command-line flag; --train-script still overrides."""
    suite = tmp_path / "s.yaml"
    suite.write_text("name: s\ntrain-script: scripts/train_alt.py\ndefaults: {arm: drr}\nseeds: [0]\nvariants: [{name: v}]\nsummary: false\n", encoding="utf8")
    cmd = [sys.executable, str(ROOT / "scripts" / "run_suite.py"), str(suite), "--dry-run"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0 and "scripts/train_alt.py --arm drr" in r.stdout, r.stdout + r.stderr
    r = subprocess.run(cmd + ["--train-script", "scripts/other.py"], capture_output=True, text=True, cwd=ROOT)
    assert "scripts/other.py --arm drr" in r.stdout
    assert S.load_suite(suite)["train-script"] == "scripts/train_alt.py"
