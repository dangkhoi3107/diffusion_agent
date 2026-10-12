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
