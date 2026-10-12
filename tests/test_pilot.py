"""CPU tests for the M0/M1 pilot: input normalization, pair matrix, trial bookkeeping/resume.

Run: python -m pytest tests/test_pilot.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camo import pilot  # noqa: E402
from camo.config import Config  # noqa: E402


def test_pair_matrix_is_balanced():
    subjects = Counter(s for s, _ in pilot.DEFAULT_PAIRS.values())
    backgrounds = Counter(b for _, b in pilot.DEFAULT_PAIRS.values())
    assert len(pilot.DEFAULT_PAIRS) == 18
    assert set(subjects.values()) == {3} and len(subjects) == 6
    assert set(backgrounds.values()) == {3} and len(backgrounds) == 6
    assert pilot.DEFAULT_PAIRS["P02"] == ("O01", "B02")
    assert pilot.DEFAULT_PAIRS["P13"] == ("O05", "B01")
    assert pilot.DEFAULT_PAIRS["P16"] == ("O06", "B03")


def test_audit_configs_match_baseline():
    cfg = Config()
    for key, value in pilot.C00_EXPECTED.items():
        section, name = key.split(".")
        assert getattr(getattr(cfg, section), name) == value
    assert len(pilot.AUDIT_CONFIGS) == 7
    assert all(len(o) <= 1 for o in pilot.AUDIT_CONFIGS.values())  # one-at-a-time perturbations


def test_hidden_rgb_under_alpha_does_not_leak(tmp_path):
    rgba = np.zeros((40, 80, 4), np.uint8)
    rgba[..., :3] = (255, 0, 0)  # hidden red everywhere
    rgba[10:30, 30:50] = (0, 0, 255, 255)  # opaque blue subject
    path = tmp_path / "s.png"
    Image.fromarray(rgba, "RGBA").save(path)
    img, alpha, meta = pilot.normalize_image(str(path), 64, "pad")
    arr = np.asarray(img)
    assert meta["has_alpha"] and alpha is not None
    assert img.size == (64, 64)
    assert tuple(arr[2, 2]) == pilot.GRAY  # pad area
    assert tuple(arr[20, 5]) == pilot.GRAY  # transparent area inside the source: gray, not red
    assert arr[..., 0].max() < 200  # no hidden red anywhere
    x0, y0, x1, y1 = meta["transform"]["placed_box"]
    assert (x1 - x0, y1 - y0) == (64, 32)  # 2:1 aspect kept, no stretching


def test_background_crop_keeps_aspect(tmp_path):
    path = tmp_path / "b.jpg"
    Image.new("RGB", (300, 100), (10, 120, 30)).save(path)
    img, alpha, meta = pilot.normalize_image(str(path), 64, "crop")
    assert img.size == (64, 64) and alpha is None and meta["transform"]["mode"] == "crop"


def _inputs(tmp_path, with_b06=True):
    for item in ["O01_cat.png", "O05_face.jpg", "O06_face34.jpg", "B01_wall.jpg", "B02_sand.jpg", "B03_leaf.jpg"]:
        Image.new("RGB", (120, 90), (90, 90, 90)).save(tmp_path / item)
    if with_b06:
        pilot.make_pattern(str(tmp_path / "B06_pattern.png"), size=128)
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pilot", "inputs.yaml")
    target = tmp_path / "inputs.yaml"
    target.write_text(open(src, encoding="utf-8").read(), encoding="utf-8")
    return str(target)


def test_load_inputs_finds_files_by_id_and_reports_gaps(tmp_path):
    data = pilot.load_inputs(_inputs(tmp_path))
    assert data["subjects"]["O01"]["path"].endswith("O01_cat.png")
    assert data["backgrounds"]["B06"]["path"].endswith("B06_pattern.png")
    assert sorted(data["missing"]) == ["B04", "B05", "O02", "O03", "O04"]
    assert pilot.pair_status(data, "P02") is None
    assert "O02" in pilot.pair_status(data, "P04")


def test_trials_write_manifest_and_resume(tmp_path, monkeypatch):
    data = pilot.load_inputs(_inputs(tmp_path))
    out = str(tmp_path / "out")
    inv = pilot.normalize(data, out, 64)
    calls = []

    class FakeSession:
        def __init__(self, cfg):
            pass

    def fake_run_pair(session, cfg, subject, background, seed, out_dir=None):
        calls.append((os.path.basename(subject), cfg.sampling.bg_strength, cfg.blend.alpha_end, cfg.prompt.subject_name))
        os.makedirs(out_dir, exist_ok=True)
        Image.new("RGB", (64, 64)).save(os.path.join(out_dir, "output.png"))
        return SimpleNamespace(prompt=f"a {cfg.prompt.subject_name}", steps_run=29, timings={"render": 1.0}, metrics={"SSIM_bg": 0.9},
                               prepared=SimpleNamespace(softedge_source="hed"), seconds=1.0)

    import camo.runner as runner

    monkeypatch.setattr(runner, "Session", FakeSession)
    monkeypatch.setattr(runner, "run_pair", fake_run_pair)
    monkeypatch.setattr(pilot, "environment", lambda cfg: {"stub": True})
    base = Config()
    plan = [(p, c) for p in data["audit_pairs"] for c in pilot.AUDIT_CONFIGS]
    counts = pilot.run_trials(base, data, inv, out, plan, "audit")
    assert counts == {"ok": 21, "skipped": 0, "error": 0, "not_run": 0}
    trial = json.load(open(os.path.join(out, "runs", "P13", "C03", "trial.json"), encoding="utf-8"))
    assert trial["status"] == "ok" and trial["steps_run"] == 29
    assert trial["params"]["blend.alpha_end"] == 0.48 and trial["subject_label"] == "human face, frontal portrait"
    assert trial["inputs"]["O05"]["normalized_sha256"] == inv["items"]["O05"]["normalized_sha256"]
    assert all(c[3] for c in calls)  # prompt label always set from the manifest

    again = pilot.run_trials(base, data, inv, out, plan, "audit")
    assert again["skipped"] == 21 and again["ok"] == 0
    pilot.contact_sheets(out, data["audit_pairs"])
    assert os.path.exists(os.path.join(out, "runs", "P13", "contact_sheet.png"))
    latest = json.load(open(pilot.rebuild_index(out), encoding="utf-8"))
    assert len(latest) == 21



def test_oi_candidates_keep_single_untruncated_medium_instances():
    import random

    from camo.sources import _oi_candidates

    def row(img, x0, x1, y0, y1, mid="/m/cat"):
        return {"ImageID": img, "LabelName": mid, "BoxXMin": x0, "BoxXMax": x1, "BoxYMin": y0, "BoxYMax": y1}

    rows = [row("ok", 0.2, 0.8, 0.2, 0.8), row("edge", 0.0, 0.7, 0.2, 0.8), row("tiny", 0.4, 0.5, 0.4, 0.5),
            row("two", 0.2, 0.8, 0.2, 0.8), row("two", 0.1, 0.3, 0.1, 0.3), row("dog", 0.2, 0.8, 0.2, 0.8, "/m/dog")]
    assert [r["ImageID"] for r in _oi_candidates(rows, "/m/cat", random.Random(0))] == ["ok"]


def test_portraits_from_aisegment_layout(tmp_path):
    from camo.sources import fetch_portraits

    for i in range(3):
        clip = tmp_path / "ds" / "clip_img" / "1803" / "clip_00000000"
        matting = tmp_path / "ds" / "matting" / "1803" / "matting_00000000"
        clip.mkdir(parents=True, exist_ok=True)
        matting.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (60, 80), (200, 150, 120)).save(clip / f"p{i}.jpg")
        rgba = Image.new("RGBA", (60, 80), (0, 0, 0, 0))
        rgba.paste((255, 255, 255, 255), (15, 10, 45, 80))
        rgba.save(matting / f"p{i}.png")
    out = tmp_path / "cand"
    out.mkdir()
    found = fetch_portraits(str(out), str(tmp_path / "ds"), 1, 0)
    assert len(found["O05"]) == 1 and len(found["O06"]) == 1
    cut = Image.open(found["O05"][0])
    assert cut.mode == "RGBA" and cut.getchannel("A").getextrema() == (0, 255)
    assert json.load(open(found["O05"][0][:-4] + ".json"))["pose"].startswith("unknown")



def test_empty_audit_pairs_means_default_only(tmp_path):
    path = tmp_path / "inputs.yaml"
    Image.new("RGB", (32, 32)).save(tmp_path / "S1.png")
    Image.new("RGB", (32, 32)).save(tmp_path / "G1.png")
    path.write_text("subjects: {S1: {label: cat, category: animal}}\nbackgrounds: {G1: {}}\n"
                    "pairs: {G1S1: [S1, G1]}\naudit_pairs: []\n", encoding="utf-8")
    data = pilot.load_inputs(str(path))
    assert data["audit_pairs"] == [] and list(data["pairs"]) == ["G1S1"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
