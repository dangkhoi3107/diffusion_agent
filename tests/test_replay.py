"""Replay of search agents on a labelled grid (scripts/replay_agents.py) and the label server item scan.

Run: python -m pytest tests/test_replay.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import label_server  # noqa: E402
import replay_agents  # noqa: E402
from camo.agent import ACTION_SPACE, all_configs, grid_id  # noqa: E402


LEVELS = ACTION_SPACE["subject.strength"]  # faint -> strong
BGS = ACTION_SPACE["sampling.bg_strength"]


def make_grid(out, good_level="a008", identical=("a020_b52",), unlabeled=()):
    """One pair P01: C00 + the full action space. Pixel value encodes the config so configs differ, except
    `identical` which copies C00 (should inherit its label). Labels: stronger than `good_level` -> too_visible,
    weaker -> missing."""
    run = out / "grid_test"
    trials, labels = [], []
    cfgs = {grid_id(c): c for c in all_configs()}
    for cid in ["C00"] + list(cfgs):
        d = run / "runs" / "P01" / cid
        d.mkdir(parents=True)
        level = LEVELS.index(cfgs[cid]["subject.strength"]) if cid != "C00" else None
        value = 100 if cid in ("C00", *identical) else 20 + 30 * level + 5 * BGS.index(cfgs[cid]["sampling.bg_strength"])
        Image.fromarray(np.full((32, 32, 3), value, np.uint8)).save(d / "output.png")
        trials.append({"trial_id": f"P01_{cid}", "pair_id": "P01", "config_id": cid, "status": "ok", "subject_label": "cat",
                       "category": "animal", "subject_id": "O01", "background_id": "B02", "overrides": {}})
        if cid in identical or cid in unlabeled:
            continue
        if cid == "C00":
            label = "too_visible"
        else:
            good = LEVELS.index(good_level)
            label = "good" if level == good else ("too_visible" if level > good else "missing")
        labels.append({"rater": "tac_gia", "uid": f"grid_test/P01_{cid}", "label": label})
    (run / "trials_latest.json").write_text(json.dumps(trials))
    (out / "labels.jsonl").write_text("\n".join(json.dumps(r) for r in labels) + "\n")
    return run


def test_replay_inherits_identical_and_finds_good(tmp_path, monkeypatch):
    make_grid(tmp_path)
    monkeypatch.setattr(label_server, "OUT", str(tmp_path))
    monkeypatch.setattr(replay_agents, "OUT", str(tmp_path))
    table, meta = replay_agents.grid_labels("grid_test", "tac_gia")
    assert table[("P01", "a020_b52")] == ("too_visible", "inherited_from_C00")
    result = replay_agents.replay("P01", replay_agents.RulePolicy(), 5, table)
    assert result["status"] == "done" and result["selected"].startswith("a008")
    assert [c for c, _ in result["path"]] == ["a020_b52", "a012_b52", "a008_b52"]  # START (= C00 look), then weaker


def test_replay_stops_on_missing_label(tmp_path, monkeypatch):
    make_grid(tmp_path, unlabeled=("a012_b52",))
    monkeypatch.setattr(label_server, "OUT", str(tmp_path))
    monkeypatch.setattr(replay_agents, "OUT", str(tmp_path))
    table, _ = replay_agents.grid_labels("grid_test", "tac_gia")
    result = replay_agents.replay("P01", replay_agents.RulePolicy(), 5, table)
    assert result["status"].startswith("incomplete") and "a012_b52" in result["status"] and result["selected"] is None


def test_main_writes_report_and_page(tmp_path, monkeypatch):
    make_grid(tmp_path)
    monkeypatch.setattr(label_server, "OUT", str(tmp_path))
    monkeypatch.setattr(replay_agents, "OUT", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["x", "--run", "grid_test", "--seeds", "2"])
    assert replay_agents.main() == 0
    report = json.loads((tmp_path / "compare" / "report.json").read_text())
    assert report["before_default"] == {"good": 0, "labelled": 1}
    assert report["policies"]["rule"]["success"] == 1 and report["policies"]["random"]["runs"] == 2
    page = (tmp_path / "compare" / "index.html").read_text()
    assert "/files/grid_test/runs/P01/C00/output.png" in page


def test_label_server_hides_near_duplicates(tmp_path, monkeypatch):
    make_grid(tmp_path)
    monkeypatch.setattr(label_server, "OUT", str(tmp_path))
    shown = {i["config_id"] for i in label_server.collect(2.0, False)}
    assert "C00" in shown and "a020_b52" not in shown and "a008_b44" in shown


def test_eval_observer_fake_writes_labels_and_report(tmp_path, monkeypatch):
    import eval_observer

    make_grid(tmp_path)
    (tmp_path / "grid_test" / "inputs_normalized").mkdir()
    for name in ("O01", "B02"):
        Image.new("RGB", (8, 8)).save(tmp_path / "grid_test" / "inputs_normalized" / f"{name}.png")
    monkeypatch.setattr(label_server, "OUT", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["x", "--fake", "--limit", "4", "--few-shot", "1"])
    assert eval_observer.main() == 0
    report = json.loads((tmp_path / "observer_eval.json").read_text())
    assert report["n"] == 4 and 0 <= report["exact_agreement"] <= 1
    recs = [json.loads(line) for line in (tmp_path / "labels.jsonl").read_text().splitlines() if line.strip()]
    assert sum(r["rater"] == "observer_fake" for r in recs) == 4
    assert all(r["rater"] == "tac_gia" for r in recs[:-4])  # human labels untouched


def test_label_server_skips_set_aside_runs(tmp_path, monkeypatch):
    import shutil

    make_grid(tmp_path)
    for name in ("stale_grid_test", "oldgrid_grid_test"):
        shutil.copytree(tmp_path / "grid_test", tmp_path / name)
    monkeypatch.setattr(label_server, "OUT", str(tmp_path))
    assert {i["run"] for i in label_server.collect(2.0, False)} == {"grid_test"}


def test_label_server_hides_retired_runs_from_the_page_only(tmp_path, monkeypatch):
    make_grid(tmp_path)
    (tmp_path / "grid_test" / "RETIRED").write_text("old grid")
    monkeypatch.setattr(label_server, "OUT", str(tmp_path))
    assert label_server.collect(2.0, False) == []  # the label page
    assert {i["run"] for i in label_server.collect(0.0, True, runs={"grid_test"})} == {"grid_test"}  # training


def test_labelset_report_counts_boards_with_a_good_config(tmp_path, monkeypatch):
    import labelset

    make_grid(tmp_path)
    monkeypatch.setattr(label_server, "OUT", str(tmp_path))
    monkeypatch.setattr(replay_agents, "OUT", str(tmp_path))
    res = labelset.report("grid_test")
    assert (res["pairs"], res["labelled_pairs"], res["pairs_with_good"]) == (1, 1, 1)
    assert res["per_config"]["a008_b44"] == {"good": 1} and res["per_config"]["a020_b52"] == {"too_visible": 1}
    assert res["rule"] == {"complete": 1, "success": 1, "mean_renders": 3.0}
    (tmp_path / "grid_test" / "RETIRED").write_text("old")
    assert labelset.report("grid_test")["pairs"] == 1  # a retired run still reports
