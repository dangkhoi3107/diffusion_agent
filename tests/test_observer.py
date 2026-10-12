"""Model 1 stage A: CORAL helpers, metrics, label-set rows (inheritance, split, test-input leak), feature cache, fit.

Run: python -m pytest tests/test_observer.py      (the fit test needs torch: ~/.venvs/camo-ml/bin/python -m pytest ...)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import label_server  # noqa: E402
import train_observer  # noqa: E402
from camo import observer as obs  # noqa: E402
from camo.backbone import FeatureCache  # noqa: E402


def test_coral_probs_are_rank_consistent_and_sum_to_one():
    p_beyond = np.array([[0.9, 0.6, 0.1], [0.2, 0.1, 0.05], [0.99, 0.95, 0.9]])
    p_broken = np.array([0.1, 0.0, 0.7])
    P = obs.class_probs(p_beyond, p_broken)
    assert np.allclose(P.sum(1), 1)
    assert obs.decide(p_beyond, p_broken) == ["good", "missing", "broken"]
    assert obs.coral_targets(np.array([0, 2, 3])).tolist() == [[0, 0, 0], [1, 1, 0], [1, 1, 1]]
    with pytest.raises(ValueError):
        obs.encode(["good", "unsure"])


def test_metrics_counts_ordinal_and_broken():
    m = obs.metrics(["good", "missing", "broken", "too_visible"], ["good", "subtle", "broken", "good"])
    assert m["exact"] == 0.5 and m["good_vs_not"] == 0.75
    assert m["ordinal_mae"] == pytest.approx(2 / 3, abs=1e-3) and m["broken_f1"] == 1.0


def _png(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(np.full((16, 16, 3), value, np.uint8)).save(path)


def make_outputs(tmp_path):
    """label_b1: pair D01T01 (train), pair D02T01 (background D02 is test); old run bgmap: pair G1 reuses D02."""
    out = tmp_path / "outputs"
    trials, labels = [], []
    for run, pairs in (("label_b1", [("D01", "T01"), ("D02", "T01")]), ("bgmap", [("G1", "O1")])):
        for bg, subj in pairs:
            _png(out / run / "inputs_normalized" / f"{bg}.png", 10)
            _png(out / run / "inputs_normalized" / f"{subj}.png", 20)
            for cid, value in (("C00", 100), ("a000_b40", 100), ("a020_b55", 180)):  # a000_b40 == C00 pixels
                _png(out / run / "runs" / f"{bg}{subj}" / cid / "output.png", value)
                trials.append({"trial_id": f"{bg}{subj}_{cid}", "pair_id": f"{bg}{subj}", "config_id": cid,
                               "status": "ok", "subject_id": subj, "background_id": bg, "run": run})
        (out / run / "trials_latest.json").write_text(json.dumps([t for t in trials if t["run"] == run]))
    for uid, label in (("label_b1/D01T01_C00", "missing"), ("label_b1/D01T01_a020_b55", "good"),
                       ("label_b1/D02T01_C00", "unsure"), ("label_b1/D02T01_a020_b55", "too_visible"),
                       ("bgmap/G1O1_C00", "subtle")):
        labels.append({"rater": "tac_gia", "uid": uid, "label": label})
    labels.append({"rater": "someone_else", "uid": "label_b1/D01T01_C00", "label": "good"})
    (out / "labels.jsonl").write_text("\n".join(json.dumps(r) for r in labels) + "\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"subjects": {"T01": {"split": "train"}},
                                    "backgrounds": {"D01": {"split": "train"}, "D02": {"split": "test"}}}))
    return out, manifest


def test_rows_inherit_split_and_keep_test_inputs_out(tmp_path, monkeypatch):
    out, manifest = make_outputs(tmp_path)
    monkeypatch.setattr(label_server, "OUT", str(out))
    monkeypatch.setattr(train_observer, "MANIFEST", str(manifest))
    data = {r["uid"]: r for r in train_observer.rows("tac_gia")}
    assert data["label_b1/D01T01_a000_b40"]["label"] == "missing"  # inherited from C00 (identical pixels)
    assert data["label_b1/D01T01_a000_b40"]["source"] == "inherited_from_C00"
    assert "label_b1/D02T01_C00" not in data and "label_b1/D02T01_a000_b40" not in data  # unsure, nothing to inherit
    assert data["label_b1/D02T01_a020_b55"]["split"] == "test" and data["label_b1/D01T01_C00"]["split"] == "train"
    rows = list(data.values())
    keys = sorted({r[k] for r in rows for k in ("subj", "bg")})
    # bgmap's G1 is the same picture as the test background D02 (identical feature), T01 and O1 differ
    vec = {"bg": np.array([1.0, 0, 0]), "subj": np.array([0, 1.0, 0])}
    X = np.stack([vec["bg"] if "/D" in k or "/G" in k else vec["subj"] for k in keys])
    X[[i for i, k in enumerate(keys) if k.endswith("D01.png")]] = [0.5, 0, 0.9]
    X[[i for i, k in enumerate(keys) if k.endswith("O1.png")]] = [0, 0.2, 1.0]
    canon = train_observer.same_picture(keys, X)
    assert canon["bgmap/inputs_normalized/G1.png"] == "bgmap/inputs_normalized/G1.png"  # sorted first: bgmap < label_b1
    assert canon["label_b1/inputs_normalized/D02.png"] == "bgmap/inputs_normalized/G1.png"
    leaked = train_observer.attach_groups(rows, canon)
    # both bgmap rows leave training: G1O1 C00 and its identical a000_b40 (inherited label)
    assert leaked == {"bgmap": 2} and data["bgmap/G1O1_C00"]["split"] == "excluded"


class FakeBackbone:
    def __init__(self):
        self.calls = 0

    def embed(self, paths, log=None):
        self.calls += len(paths)
        return np.stack([np.full(4, float(np.asarray(Image.open(p)).mean()), np.float32) for p in paths])


def test_feature_cache_only_embeds_new_or_changed(tmp_path):
    files = {f"r/{i}.png": str(tmp_path / f"{i}.png") for i in range(3)}
    for i, p in enumerate(files.values()):
        _png(p, 10 * i)
    cache, bb = FeatureCache(str(tmp_path / "c" / "feats.npz")), FakeBackbone()
    assert cache.update(bb, files) == 3
    again = FeatureCache(str(tmp_path / "c" / "feats.npz"))
    assert again.update(bb, files) == 0 and bb.calls == 3
    _png(files["r/1.png"], 200)
    os.utime(files["r/1.png"], (1, 1))  # mtime changes even within the same second
    assert again.update(bb, files) == 1 and again.get(["r/1.png"])[0, 0] == 200


def test_fit_learns_an_ordinal_signal():
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    n = 600
    level = rng.integers(0, 4, n)
    X = rng.normal(0, 1, (n, 16)).astype(np.float32)
    X[:, 0] += level * 2.0  # visibility along one direction
    broken = rng.random(n) < 0.08
    X[broken, 1] += 4.0
    labels = ["broken" if b else obs.VIS[k] for k, b in zip(level, broken)]
    bundle = obs.fit(X[:450], labels[:450], obs.FitConfig(hidden=32))
    pred, P = obs.predict(bundle, X[450:])
    assert np.allclose(P.sum(1), 1, atol=1e-5)
    m = obs.metrics(labels[450:], pred)
    assert m["within_one"] > 0.95 and m["exact"] > 0.65 and m["broken_f1"] > 0.6  # best possible exact ~0.76
