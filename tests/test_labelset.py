"""Label set v2 helpers: shape-group re-pairing and DTD fill (scripts/labelset.py), the box filter of the SAM gap fill
(scripts/fetch_subjects.py) and the even-crop chooser (scripts/fetch_backgrounds.py). CPU only, no Kaggle.

Run: python -m pytest tests/test_labelset.py
"""
from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter, defaultdict

import numpy as np
import pytest
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import fetch_backgrounds  # noqa: E402
import fetch_subjects  # noqa: E402
import labelset  # noqa: E402
from camo.sources import homogeneity  # noqa: E402

SHAPES = ["quadruped_compact", "quadruped_tall", "streamlined_fins", "bird_upright", "round_shelled", "wings_spread"]


@pytest.fixture
def labelset_dir(tmp_path, monkeypatch):
    """Tiny label set: b1 rendered (v1 subjects), b2/b3 held; 6 shape groups of picks, one with a single image."""
    img = tmp_path / "img.png"
    Image.new("RGBA", (8, 8), (10, 20, 30, 255)).save(img)
    bg_dir = tmp_path / "bg_dtd"
    bg_dir.mkdir()
    for name in ("striped_0001.jpg", "striped_0002.jpg", "zigzagged_0001.jpg", "grid_0001.jpg", "dotted_0001.jpg"):
        Image.new("RGB", (8, 8)).save(bg_dir / name)
    subjects = {f"S{i:02d}": {"label": c, "category": "animal" if c != "human" else "human_portrait", "class": c,
                              "origin": f"O0{i}_0", "path": f"candidates/S{i:02d}.png"}
                for i, c in enumerate(["cat", "fish", "human"], 1)}
    backgrounds = {f"D{i:02d}": {"material": "dtd:striped", "group": "regular_pattern_2d", "file": "striped_0001.jpg",
                                 "split": "test" if i == 4 else "train", "path": f"candidates/D{i:02d}.jpg"}
                   for i in range(1, 7)}
    pairs = {f"D{b:02d}S0{s}": [f"S0{s}", f"D{b:02d}"] for b in range(1, 7) for s in (1, 2)}
    batches = [{"run": f"label_b{k}", "folder": f"pilot/labelset/b{k}",
                "pairs": [p for p, v in pairs.items() if int(v[1][1:]) in (2 * k - 1, 2 * k)]} for k in (1, 2, 3)]
    for b in batches[1:]:
        b["hold"] = "test"
    lset = tmp_path / "labelset"
    lset.mkdir()
    (lset / "manifest.json").write_text(json.dumps({"subjects": subjects, "backgrounds": backgrounds, "pairs": pairs,
                                                    "batches": batches, "configs": ["C00"]}))
    picks = {g: [{"id": f"{g}_{k}", "species": f"{g}_sp{k % 2}", "src": str(img)} for k in range(1 if g == "wings_spread" else 4)]
             for g in SHAPES}
    picks["streamlined_fins"][0]["species"] = "Rabbit"  # a TEST_SPECIES outside the test shape
    (tmp_path / "picks.json").write_text(json.dumps({"picks": picks}))
    cand = tmp_path / "cand" / "candidates"
    cand.mkdir(parents=True)
    for s in subjects.values():
        Image.new("RGBA", (8, 8)).save(cand / f"{s['origin']}.png")
    monkeypatch.setattr(labelset, "SET", str(lset))
    monkeypatch.setattr(labelset, "BG_DIR", str(bg_dir))
    monkeypatch.setattr(labelset, "OUT", str(tmp_path / "outputs"))
    monkeypatch.setattr(labelset, "CANDIDATES", str(tmp_path / "cand"))
    monkeypatch.setattr(labelset, "SUBJECT_PICKS", str(tmp_path / "picks.json"))
    monkeypatch.setattr(labelset, "used_by_bgmap", lambda: set())
    return lset


def test_rebuild_subjects_balances_shape_groups(labelset_dir):
    res = labelset.rebuild_subjects(1, release=False)
    man = json.loads((labelset_dir / "manifest.json").read_text())
    subj = man["subjects"]
    per_bg = defaultdict(list)
    for sid, bid in res["pairs"].values():
        per_bg[bid].append(subj[sid]["shape"])
    assert sorted(per_bg) == ["D03", "D04", "D05", "D06"]  # b1 (D01, D02) keeps its v1 pairs
    assert all(len(v) == len(set(v)) == labelset.SUBJECTS_PER_BG for v in per_bg.values())
    assert sum(v.count("human") for v in per_bg.values()) == 2  # every 3rd background of 4: index 0 and 3
    groups = Counter(g for v in per_bg.values() for g in v if g != "human")
    assert max(groups.values()) - min(groups.values()) <= 1
    uses = Counter(sid for sid, _ in res["pairs"].values() if subj[sid].get("version") == 2)
    assert max(uses.values()) <= 3  # the one-image group repeats, the others spread over their instances
    assert {subj[s]["split"] for s in subj if subj[s].get("shape") == "round_shelled"} == {"test"}
    assert [subj[s]["split"] for s in subj if subj[s].get("species") == "Rabbit"] == ["test"]
    assert {p for p, v in man["pairs"].items() if v[1] in ("D01", "D02")} == set(man["batches"][0]["pairs"])
    assert all(b.get("hold") for b in man["batches"][1:])
    assert (labelset_dir / "b2" / "inputs.yaml").exists()


def test_rebuild_refuses_rendered_batch(labelset_dir):
    os.makedirs(os.path.join(labelset.OUT, "label_b2"))
    open(os.path.join(labelset.OUT, "label_b2", "trials_latest.json"), "w").close()
    with pytest.raises(SystemExit):
        labelset.rebuild_subjects(1, release=False)


def test_fill_dtd_swaps_duplicates_in_held_batches_only(labelset_dir):
    changes = labelset.fill_dtd(1)
    assert [c[2] for c in changes] == ["dtd:zigzagged", "dtd:grid", "dtd:dotted"]  # DTD_FILL_FIRST order
    assert all(bid not in ("D01", "D02") for bid, _, _ in changes)
    man = json.loads((labelset_dir / "manifest.json").read_text())
    mats = Counter(m["material"] for m in man["backgrounds"].values())
    assert mats["dtd:striped"] == 3 and man["backgrounds"]["D04"]["split"] == "test"


def test_box_candidates_keeps_one_clear_instance():
    base = dict(LabelName="/m/x", XMin="0.2", XMax="0.6", YMin="0.2", YMax="0.7", IsOccluded="0", IsTruncated="0",
                IsGroupOf="0", IsDepiction="0", IsInside="0")
    rows = [dict(base, ImageID="a"), dict(base, ImageID="b", IsOccluded="1"), dict(base, ImageID="c"),
            dict(base, ImageID="c"), dict(base, ImageID="d", XMin="0.0"), dict(base, ImageID="e", XMax="0.25", YMax="0.25"),
            dict(base, ImageID="f", LabelName="/m/y")]
    assert [r["ImageID"] for r in fetch_subjects._box_candidates(rows, "/m/x", random.Random(0))] == ["a"]


def test_best_square_avoids_the_busy_corner():
    rng = np.random.RandomState(0)
    a = np.full((1000, 1000, 3), 120, np.uint8)
    a += rng.randint(0, 6, a.shape).astype(np.uint8)  # even texture
    a[:300, :300] = rng.randint(0, 255, (300, 300, 3))  # busy corner
    box, _ = fetch_backgrounds.best_square(Image.fromarray(a), homogeneity)
    assert box[0] >= 300 or box[1] >= 300


@pytest.mark.parametrize("types", [10, 5])
def test_nature_split_keeps_a_train_image_of_every_test_type(types):
    split = {(i, j): labelset.nature_split(i, j) for i in range(labelset.PER_NATURE) for j in range(types)}
    test_types = {j for (i, j), s in split.items() if s == "test"}
    assert sum(s == "test" for s in split.values()) == labelset.PER_NATURE * types // labelset.NATURE_TEST_EVERY
    assert all(any(split[(i, j)] == "train" for i in range(labelset.PER_NATURE)) for j in test_types)


def _render_batch(out, run, pairs, configs, status="ok"):
    for pair in pairs:
        for cid in configs:
            d = out / run / "runs" / pair / cid
            d.mkdir(parents=True, exist_ok=True)
            (d / "trial.json").write_text(json.dumps({"status": status}))
    (out / run / "trials_latest.json").write_text("[]")


def test_check_batch_catches_a_stale_grid_and_failed_renders(tmp_path, monkeypatch):
    monkeypatch.setattr(labelset, "OUT", str(tmp_path))
    pairs, grid = ["D01S01", "D01S02"], ["C00", "a000_b40", "a004_b40"]
    _render_batch(tmp_path, "good", pairs, grid)
    _render_batch(tmp_path, "stale", pairs, ["C00", "a020_b45"])
    _render_batch(tmp_path, "failed", pairs, grid, status="error")
    assert labelset.check_batch("good", pairs, grid) == ""
    assert "missing" in labelset.check_batch("stale", pairs, grid)
    assert "renders ok" in labelset.check_batch("failed", pairs, grid)


@pytest.fixture
def one_batch(tmp_path, monkeypatch):
    """Manifest with one batch; subprocess.run (the kaggle_run call) is replaced by `script` entries."""
    lset, out = tmp_path / "labelset", tmp_path / "outputs"
    lset.mkdir()
    out.mkdir()
    pairs, grid = ["D01S01", "D01S02"], ["C00", "a000_b40"]
    (lset / "manifest.json").write_text(json.dumps({"configs": grid, "pairs": {}, "subjects": {}, "backgrounds": {},
                                                    "batches": [{"run": "label_b1", "folder": "pilot/labelset/b1",
                                                                 "pairs": pairs}]}))
    monkeypatch.setattr(labelset, "SET", str(lset))
    monkeypatch.setattr(labelset, "OUT", str(out))
    monkeypatch.setattr(labelset, "STATUS", str(out / ".labelset_status.json"))
    monkeypatch.setattr(labelset, "kernel_running", lambda: False)
    monkeypatch.setattr(labelset.time, "sleep", lambda s: None)
    calls, script = [], []

    class Done:
        def __init__(self, code):
            self.returncode = code

    def fake_run(cmd, cwd=None):
        calls.append(cmd)
        action = script.pop(0)
        if action == "stale":
            _render_batch(out, "label_b1", pairs, ["C00", "a020_b45"])
        elif action == "good":
            _render_batch(out, "label_b1", pairs, grid)
        return Done(0 if action in ("stale", "good") else 1)

    monkeypatch.setattr(labelset.subprocess, "run", fake_run)
    return out, calls, script


def test_run_sets_a_stale_batch_aside_and_renders_it_again(one_batch):
    out, calls, script = one_batch
    script += ["stale", "good"]
    assert labelset.run(argparse_ns()) == 0
    assert "--download-only" not in calls[1] and "--skip-upload" in calls[1]  # rendered again, not re-downloaded
    assert [p.name for p in out.iterdir() if p.name.startswith("stale_label_b1_")]
    assert labelset.check_batch("label_b1", ["D01S01", "D01S02"], ["C00", "a000_b40"]) == ""
    assert json.loads((out / ".labelset_status.json").read_text())["state"] == "done"


def test_run_retries_a_failed_batch_then_gives_up(one_batch):
    out, calls, script = one_batch
    script += ["fail"] * 5
    assert labelset.run(argparse_ns()) == 1
    # push, download again, push, download again, push: two extra renders at most
    assert ["--download-only" in c for c in calls] == [False, True, False, True, False]
    assert json.loads((out / ".labelset_status.json").read_text())["state"] == "error"


def test_run_takes_a_finished_batch_on_the_second_download(one_batch):
    out, calls, script = one_batch
    script += ["fail", "good"]
    assert labelset.run(argparse_ns()) == 0
    assert "--download-only" in calls[1] and len(calls) == 2


def argparse_ns():
    import argparse

    return argparse.Namespace(skip_upload=False)


def test_new_grid_retires_rendered_runs_and_renames_batches(labelset_dir):
    import yaml

    out = os.path.join(labelset.OUT, "label_b1")
    os.makedirs(out)
    open(os.path.join(out, "trials_latest.json"), "w").write("[]")  # batch 1 was rendered on the old grid
    res = labelset.new_grid()
    man = json.loads((labelset_dir / "manifest.json").read_text())
    assert res["retired"] == ["label_b1"] and os.path.exists(os.path.join(out, "RETIRED"))
    assert [b["run"] for b in man["batches"]] == ["label_b1v3", "label_b2v3", "label_b3v3"]
    assert all(b["hold"] for b in man["batches"]) and man["grid_version"] == labelset.GRID_VERSION
    grid = yaml.safe_load((labelset_dir / "grid.yaml").read_text())
    assert grid["seed_per_pair"] is True and list(grid["configs"]) == man["configs"] and len(grid["configs"]) == 26
    assert grid["configs"]["g50_b60"]["sampling.guidance_sub"] == 5.0 and grid["configs"]["a008_b44"]["blend.alpha_end"] == 0.08
    assert grid["configs"]["C00_s2"] == {"sampling.seed": 98765} and grid["configs"]["a008_b44_s2"]["sampling.seed"] == 98765


def test_rebatch_with_a_trial_batch_keeps_every_pair_once(labelset_dir):
    labelset.new_grid()
    res = labelset.rebatch(1, trial=2)
    man = json.loads((labelset_dir / "manifest.json").read_text())
    runs = [b["run"] for b in man["batches"]]
    assert runs == ["label_b0v3", "label_b1v3", "label_b2v3", "label_b3v3"]
    trial = man["batches"][0]
    assert len(trial["pairs"]) == 2 and not trial.get("hold") and all(b.get("hold") for b in man["batches"][1:])
    every = [p for b in man["batches"] for p in b["pairs"]]
    assert sorted(every) == sorted(man["pairs"]) and len(every) == len(set(every))
    for b in man["batches"]:  # one pair per background in a batch
        bgs = [man["pairs"][p][1] for p in b["pairs"]]
        assert len(bgs) == len(set(bgs))
    assert (labelset_dir / "b0" / "inputs.yaml").exists() and res["order"]


def test_new_grid_keep_names_only_rewrites_the_grid(labelset_dir):
    import yaml

    before = [b["run"] for b in json.loads((labelset_dir / "manifest.json").read_text())["batches"]]
    labelset.new_grid(rename=False)
    man = json.loads((labelset_dir / "manifest.json").read_text())
    assert [b["run"] for b in man["batches"]] == before and man["grid_version"] == labelset.GRID_VERSION
    assert yaml.safe_load((labelset_dir / "grid.yaml").read_text())["seed_per_pair"] is True


def _run_with(out, run, configs, value=100):
    for cid in configs:
        d = out / run / "runs" / "D01S01" / cid
        d.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16), (value, value, value)).save(d / "output.png")
        (d / "trial.json").write_text(json.dumps({"trial_id": f"D01S01_{cid}", "config_id": cid, "status": "ok"}))


def test_merge_run_moves_missing_configs_after_the_seed_check(tmp_path, monkeypatch):
    out = tmp_path / "outputs"
    monkeypatch.setattr(labelset, "OUT", str(out))
    _run_with(out, "trial", ["C00", "a008_b44"])
    _run_with(out, "addon", ["C00", "a004_b44", "a012_b44"])
    res = labelset.merge_run("addon", "trial")
    assert res["moved"] == 2 and res["check_diff"] == {"D01S01": 0.0}
    latest = json.loads((out / "trial" / "trials_latest.json").read_text())
    assert sorted(t["config_id"] for t in latest) == ["C00", "a004_b44", "a008_b44", "a012_b44"]
    assert (out / "addon" / "RETIRED").exists()


def test_merge_run_refuses_when_the_check_render_differs(tmp_path, monkeypatch):
    out = tmp_path / "outputs"
    monkeypatch.setattr(labelset, "OUT", str(out))
    _run_with(out, "trial", ["C00"])
    _run_with(out, "addon", ["C00"], value=10)
    _run_with(out, "addon", ["a004_b44"])
    with pytest.raises(SystemExit):
        labelset.merge_run("addon", "trial")
    assert not (out / "trial" / "runs" / "D01S01" / "a004_b44").exists()
