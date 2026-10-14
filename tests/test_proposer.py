"""Model 2 (camo/proposer.py, scripts/train_proposer.py): simulator rules, baselines, boards from label rows, and on a
synthetic task where the good config depends on the background, the GRU must learn it (needs torch: run with
~/.venvs/camo-ml/bin/python -m pytest tests/test_proposer.py; the torch tests skip elsewhere).

Run: python -m pytest tests/test_proposer.py
"""
from __future__ import annotations

import os
import random
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from camo import proposer as P  # noqa: E402
from camo.agent import START, Controller, FixedDefault, RandomPolicy, RulePolicy, grid_id  # noqa: E402

from camo.agent import ACTION_SPACE  # noqa: E402

LEVELS = list(ACTION_SPACE["subject.strength"])  # faint -> strong


TEXTURE = np.random.RandomState(7).randn(4, 768)  # 4 directions shared by all backgrounds, like texture types


def synthetic_board(name: str, good_alpha: int, rng: random.Random, dim: int = 8) -> P.Board:
    """Visibility rises with the strength level: below the good level -> missing, above -> too_visible. The background feature
    carries the good alpha in dimension 0 (plus noise), as a texture type would; dim=768 spreads it over a few
    shared directions plus per-image noise, the shape of real DINOv2 features."""
    labels = {}
    for cid in P.CONFIG_IDS:
        a = LEVELS.index(cid.split("_")[0])
        labels[cid] = "good" if a == good_alpha else ("missing" if a < good_alpha else "too_visible")
    if dim == 768:
        coef = np.array([good_alpha - 2.0] + [rng.gauss(0, 1) for _ in range(3)])
        bg = (coef @ TEXTURE + np.array([rng.gauss(0, 0.5) for _ in range(768)])).astype(np.float32)
    else:
        bg = np.array([good_alpha - 2.0] + [rng.gauss(0, 1) for _ in range(dim - 1)], np.float32)
    return P.Board(pair=f"label_bX/{name}", labels=labels, bg=bg)


def test_rollout_stops_on_good_and_scores_it():
    b = synthetic_board("D01S01", 2, random.Random(0))
    order = iter(["a008_b44", "a020_b52_s2", "g50_b60"])
    ep = P.rollout(b, lambda h: next(order))
    assert ep.path == ["a008_b44", "a020_b52_s2"] and ep.success
    assert ep.reward == pytest.approx(P.REWARD_GOOD - 2 * P.COST_RENDER)


def test_rollout_budget_and_false_good_fail():
    b = synthetic_board("D01S01", len(LEVELS) - 1, random.Random(0))
    ep = P.rollout(b, lambda h: P.CONFIG_IDS[len(h)])  # the weakest levels only: never good
    assert ep.renders == P.BUDGET and not ep.success
    assert ep.reward == pytest.approx(P.REWARD_FAIL - P.BUDGET * P.COST_RENDER)
    b.probs = {"a008_b44": P.one_hot("good")}  # Model 1 wrongly calls it good
    ep = P.rollout(b, lambda h: "a008_b44", noisy=True)
    assert ep.renders == 1 and not ep.success  # stopped on a false good: a failure
    assert P.rollout(b, lambda h: "a008_b44", noisy=False).renders == 1  # repeating a config ends the search


def test_baselines_on_the_simulator():
    rng = random.Random(1)
    boards = [synthetic_board(f"D{i:02d}S01", i % len(LEVELS), rng) for i in range(20)]
    for pol in (RulePolicy(), RandomPolicy(seed=3), FixedDefault()):
        eps = [P.rollout(b, P.agent_chooser(pol, b.pair)) for b in boards]
        assert all(len(set(e.path)) == e.renders for e in eps)  # never renders a config twice
    fixed = P.summarize([P.rollout(b, P.agent_chooser(FixedDefault(), b.pair)) for b in boards], boards)
    assert fixed["mean_renders"] == 1 and fixed["oracle"] == 1.0
    assert fixed["success"] == pytest.approx(sum(b.labels[grid_id(START)] == "good" for b in boards) / 20)
    rule = P.summarize([P.rollout(b, P.agent_chooser(RulePolicy(), b.pair)) for b in boards], boards)
    assert rule["success"] == 1.0  # visibility is monotone in alpha here: the coordinate rule always gets there


def test_boards_from_rows_keeps_complete_train_and_test_boards():
    import train_proposer

    def rows_for(pair, split, skip=()):
        return [{"run": "label_b1", "pair_id": pair, "config_id": c, "label": "good" if c == "g50_b52" else "missing",
                 "split": split, "uid": f"label_b1/{pair}_{c}", "bg": f"label_b1/inputs_normalized/{pair[:3]}.png",
                 "bg_group": "granular"} for c in P.CONFIG_IDS + ["C00"] if c not in skip]
    rows = rows_for("D01S01", "train") + rows_for("D02S01", "test") + rows_for("D03S01", "train", skip=("a008_b44",)) \
        + rows_for("D04S01", "excluded")
    feats = {f"label_b1/inputs_normalized/D0{i}.png": np.full(4, i, np.float32) for i in range(1, 5)}
    probs = {"label_b1/D01S01_g50_b52": P.one_hot("good")}
    boards = train_proposer.boards_from_rows(rows, feats, probs)
    assert [(b.pair, b.split) for b in boards] == [("label_b1/D01S01", "train"), ("label_b1/D02S01", "test")]
    assert set(boards[0].probs) == {"g50_b52"} and boards[0].goods() == ["g50_b52"]
    folds = train_proposer.group_folds(boards + boards, 2, 0)
    assert folds[0] == folds[2] and folds[1] == folds[3]  # same background, same fold


try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
needs_torch = pytest.mark.skipif(not HAS_TORCH, reason="needs torch: ~/.venvs/camo-ml/bin/python -m pytest")


@pytest.fixture(scope="module")
def trained():
    rng = random.Random(0)
    train = [synthetic_board(f"D{i:02d}S01", i % len(LEVELS), rng, 768) for i in range(60)]
    test = [synthetic_board(f"T{i:02d}S01", i % len(LEVELS), rng, 768) for i in range(25)]
    bundle = P.fit(train, P.FitConfig(rl_epochs=40, seed=0))
    return bundle, test


@needs_torch
def test_net_learns_the_background_cue(trained):
    bundle, test = trained
    net = P.summarize([P.rollout(b, P.chooser(bundle, b.bg)) for b in test], test)
    rnd = P.summarize([P.rollout(b, P.agent_chooser(RandomPolicy(seed=s), b.pair)) for s in range(5) for b in test], test)
    rule = P.summarize([P.rollout(b, P.agent_chooser(RulePolicy(), b.pair)) for b in test], test)
    # unseen backgrounds: always finds a good one, sooner than the rule, often first try (random: 6/24). With 4 strength
    # levels the rule starts one step from most goods (about 2 renders), so the margin is half a render
    assert net["success"] >= 0.95 and net["first_try_good"] >= 0.5 > rnd["first_try_good"]
    assert net["mean_renders"] <= rule["mean_renders"] - 0.5 and net["success"] > rnd["success"]
    assert bundle["W"].shape == (768, P.FitConfig().pca)  # the 768 numbers reach the net as 16 components


@needs_torch
def test_net_policy_in_the_controller_and_save_load(trained, tmp_path):
    bundle, test = trained
    board = test[0]
    pol = P.NetPolicy(bundle, background=lambda pair: board.bg)
    ctl = Controller(str(tmp_path / "events.jsonl"))
    search = ctl.start(board.pair, 5, pol)
    ctl.run(search, pol, lambda cfg: grid_id(cfg), lambda trial: board.labels[grid_id(trial.config)])
    assert search.selected() is not None
    assert len({grid_id(t.config) for t in search.trials}) == len(search.trials)
    P.save(bundle, str(tmp_path / "p.pt"))
    again = P.load(str(tmp_path / "p.pt"))
    hist = [("a008_b44", P.one_hot("missing"), "missing")]
    assert np.allclose(P.scores(bundle, board.bg, hist), P.scores(again, board.bg, hist), atol=1e-6)
    assert P.scores(bundle, board.bg, hist)[P.CONFIG_IDS.index("a008_b44")] == 0  # tried configs are masked


@needs_torch
def test_noisy_judge_training_runs():
    rng = random.Random(5)
    boards = [synthetic_board(f"N{i:02d}S01", i % len(LEVELS), rng) for i in range(10)]
    for b in boards:
        b.probs = {c: 0.8 * P.one_hot(lab) + 0.05 for c, lab in b.labels.items()}
    bundle = P.fit(boards, P.FitConfig(bc_epochs=5, rl_epochs=5, noisy_share=1.0))
    assert bundle["W"].shape == (8, 8)  # fewer components than requested: limited by the 8 numbers
    eps = [P.rollout(b, P.chooser(bundle, b.bg), noisy=True) for b in boards]
    assert all(1 <= e.renders <= P.BUDGET for e in eps)


@needs_torch
def test_train_proposer_main_end_to_end(tmp_path, monkeypatch):
    """CV, fit, test scoring and the saved model, on synthetic rows standing in for labelled label_b pairs."""
    import json

    import label_server
    import train_proposer

    rng = random.Random(2)
    rows, feats = [], {}
    for i in range(30):
        board = synthetic_board(f"D{i:02d}T{i:03d}", i % len(LEVELS), rng, 768)
        pair, bg = board.pair.split("/")[1], f"label_b1/inputs_normalized/D{i:02d}.png"
        feats[bg] = board.bg
        split = "test" if i % 6 == 0 else "train"
        rows += [{"run": "label_b1", "pair_id": pair, "config_id": c, "label": lab, "split": split,
                  "uid": f"label_b1/{pair}_{c}", "bg": bg, "bg_group": "granular"} for c, lab in board.labels.items()]
    monkeypatch.setattr(label_server, "OUT", str(tmp_path))
    monkeypatch.setattr(train_proposer.train_observer, "rows", lambda rater: rows)
    monkeypatch.setattr(train_proposer, "background_features", lambda r: feats)
    monkeypatch.setattr(sys, "argv", ["train_proposer.py", "--quick", "--folds", "3"])
    assert train_proposer.main() == 0
    report = json.loads((tmp_path / "proposer" / "report.json").read_text())
    assert report["boards"] == {"train": 25, "test": 5}
    assert set(report["cv"]) >= {"net", "rule", "random", "fixed_start"} and report["cv"]["net"]["n"] == 25
    assert report["test"]["net"]["n"] == 5 and report["test"]["rule"]["oracle"] == 1.0
    bundle = P.load(str(tmp_path / "proposer" / "proposer.pt"))
    assert len(P.scores(bundle, feats["label_b1/inputs_normalized/D01.png"], [])) == len(P.CONFIG_IDS)
