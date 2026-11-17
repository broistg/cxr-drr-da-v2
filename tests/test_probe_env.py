import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("probe_env", Path(__file__).resolve().parents[1] / "scripts" / "probe_env.py")
probe_env = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe_env)


def test_collect_has_keys():
    info = probe_env.collect("data")
    for k in ("python", "platform", "cpu_count", "packages", "gpu", "disk_free_gib"):
        assert k in info
    assert "torch" in info["packages"]
    assert "devices" in info["gpu"]
