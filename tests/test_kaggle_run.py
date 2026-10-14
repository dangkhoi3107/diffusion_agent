"""scripts/kaggle_run.py without Kaggle: waiting through an offline machine, time-bounded download retries,
truncated zips, and the stale camo-code check injected into the notebook.

Run: python -m pytest tests/test_kaggle_run.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import zipfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import kaggle_run  # noqa: E402


@pytest.fixture
def offline_then_online(tmp_path, monkeypatch):
    """Fake `kaggle`: status calls fail (offline) twice, then RUNNING, then COMPLETE; output calls fail twice,
    leave a truncated zip once, then deliver a valid run zip."""
    calls = {"status": 0, "output": 0}
    monkeypatch.setattr(kaggle_run, "ROOT", str(tmp_path))
    monkeypatch.setattr(kaggle_run.shutil, "which", lambda name: "/usr/bin/kaggle")
    monkeypatch.setattr(kaggle_run.time, "sleep", lambda s: None)

    def fake_sh(*cmd, check=True, timeout=900):
        if cmd[1:3] == ("kernels", "status"):
            calls["status"] += 1
            return ["ConnectionError: offline", "ConnectionError: offline",
                    'u/camo-pilot-run has status "KernelWorkerStatus.RUNNING"'][calls["status"] - 1] \
                if calls["status"] <= 3 else 'u/camo-pilot-run has status "KernelWorkerStatus.COMPLETE"'
        if cmd[1:3] == ("kernels", "output"):
            calls["output"] += 1
            out = cmd[cmd.index("-p") + 1]
            zpath = os.path.join(out, "runx.zip")
            if calls["output"] <= 2:
                return "ConnectionError: offline"
            if calls["output"] == 3:
                open(zpath, "wb").write(b"PK\x03\x04 truncated")
                return "IncompleteRead"
            src = tmp_path / "build" / "runx" / "runs" / "P01" / "C00"
            src.mkdir(parents=True)
            (src / "trial.json").write_text(json.dumps({"status": "ok"}))
            shutil.make_archive(os.path.join(out, "runx"), "zip", root_dir=tmp_path / "build", base_dir="runx")
            return "done"
        raise AssertionError(f"unexpected command {cmd}")

    monkeypatch.setattr(kaggle_run, "sh", fake_sh)
    return calls


def test_wait_kernel_treats_failed_status_calls_as_still_running(offline_then_online):
    assert kaggle_run.wait_kernel("u/camo-pilot-run") == "complete"
    assert offline_then_online["status"] == 4


def test_download_only_waits_then_retries_through_an_outage(offline_then_online, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["kaggle_run.py", "--run", "runx", "--user", "u", "--download-only"])
    assert kaggle_run.main() == 0
    assert offline_then_online == {"status": 4, "output": 4}  # 2 offline + 1 truncated zip, then the real one
    assert zipfile.is_zipfile(tmp_path / "outputs" / "runx.zip")
    assert (tmp_path / "outputs" / "runx" / "runs" / "P01" / "C00" / "trial.json").exists()


def test_download_gives_up_after_the_deadline(offline_then_online, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["kaggle_run.py", "--run", "runx", "--user", "u", "--download-only",
                                      "--download-hours", "0"])
    with pytest.raises(SystemExit):
        kaggle_run.main()
    assert offline_then_online["output"] == 1


def test_download_only_stops_at_once_when_the_kernel_failed(offline_then_online, monkeypatch):
    # 13/10: a kernel that died while mounting its datasets left the runner downloading nothing for 7 hours
    def failed(*cmd, check=True, timeout=900):
        if cmd[1:3] == ("kernels", "status"):
            return 'u/camo-pilot-run has status "KernelWorkerStatus.ERROR"'
        raise AssertionError(f"no download expected after a failed kernel: {cmd}")

    monkeypatch.setattr(kaggle_run, "sh", failed)
    monkeypatch.setattr(sys, "argv", ["kaggle_run.py", "--run", "runx", "--user", "u", "--download-only"])
    with pytest.raises(SystemExit, match="no output to download"):
        kaggle_run.main()


def test_frozen_runs_mount_only_the_code(tmp_path, monkeypatch):
    monkeypatch.setattr(kaggle_run, "ROOT", str(tmp_path))
    (tmp_path / "pilot" / "labelset" / "b1").mkdir(parents=True)
    (tmp_path / "pilot" / "labelset" / "b1" / "inputs.yaml").write_text("subjects: {}\n")
    assert kaggle_run.dataset_sources("u", "audit", "pilot/labelset/b1") == ["u/camo-code"]
    assert kaggle_run.AISEGMENT in kaggle_run.dataset_sources("u", "fetch-inputs", "pilot/labelset/b1")
    assert kaggle_run.AISEGMENT in kaggle_run.dataset_sources("u", "audit", "pilot/no_inputs_yet")


def test_notebook_carries_the_stale_code_check(tmp_path, monkeypatch):
    expect = {"pilot/labelset/grid.yaml": "abc"}
    nb = kaggle_run.notebook("runx", "pilot/labelset/grid.yaml", {}, "audit", "pilot/labelset/b1", True, expect)
    src = "".join(nb["cells"][1]["source"])
    assert "STALE_CAMO_CODE" in src and json.dumps(expect) in src
    assert "STALE_CAMO_CODE" not in "".join(kaggle_run.notebook("runx", "", {}, "audit")["cells"][1]["source"])
