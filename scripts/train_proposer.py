"""Model 2 (Proposer): train the GRU policy on the labelled label-set grid and compare it with the baselines.

    ~/.venvs/camo-ml/bin/python scripts/train_proposer.py            # CV on train pairs, fit, score test pairs, save
    ~/.venvs/camo-ml/bin/python scripts/train_proposer.py --quick    # fewer epochs (smoke run)

Boards = label-set pairs (outputs/label_b*) whose 15 grid configs all have a label of yours (own, or inherited from
C00 for near-identical renders, as in scripts/train_observer.py; "unsure" leaves the board incomplete). Background
feature = cached DINOv2 vector of the pair's background (outputs/observer/feats_dinov2.npz; new images are embedded).
Model 1's out-of-fold probabilities (outputs/observer/cv_predictions.jsonl) feed the noisy-judge self-play.
  cv    5 folds grouped by background (an unseen background in every fold), train pairs only
  test  fit on all train pairs, score the label-set test pairs (manifest split)
Policies on the same boards: net (greedy), rule, random (5 seeds), fixed START, oracle (a good exists in the grid).
Output: outputs/proposer/report.json, proposer.pt.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import label_server  # noqa: E402
import train_observer  # noqa: E402
from camo import proposer as P  # noqa: E402
from camo.agent import FixedDefault, RandomPolicy, RulePolicy  # noqa: E402


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def model1_probs() -> Dict[str, np.ndarray]:
    path = os.path.join(train_observer.out_dir(), "cv_predictions.jsonl")
    out: Dict[str, np.ndarray] = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r.get("probs"):
                        out[r["uid"]] = np.array([r["probs"].get(k, 0.0) for k in P.OBS], np.float32)
    return out


def boards_from_rows(rows: List[dict], bg_features: Dict[str, np.ndarray], probs: Dict[str, np.ndarray]) -> List[P.Board]:
    """One board per label-set pair with all grid configs labelled; split excluded -> dropped."""
    by_pair = defaultdict(list)
    for r in rows:
        if r["run"].startswith("label_b") and r["config_id"] in P.CONFIG_IDS:
            by_pair[(r["run"], r["pair_id"])].append(r)
    boards = []
    for (run, pair), rs in sorted(by_pair.items()):
        labels = {r["config_id"]: r["label"] for r in rs}
        splits = {r["split"] for r in rs}
        if set(labels) != set(P.CONFIG_IDS) or "excluded" in splits or rs[0]["bg"] not in bg_features:
            continue
        boards.append(P.Board(pair=f"{run}/{pair}", labels=labels, bg=bg_features[rs[0]["bg"]],
                              probs={r["config_id"]: probs[r["uid"]] for r in rs if r["uid"] in probs},
                              split="test" if "test" in splits else "train", bg_group=rs[0].get("bg_group")))
    return boards


def background_features(rows: List[dict]) -> Dict[str, np.ndarray]:
    from camo.backbone import Backbone, FeatureCache

    cache = FeatureCache(os.path.join(train_observer.out_dir(), "feats_dinov2.npz"))
    keys = sorted({r["bg"] for r in rows if r["run"].startswith("label_b")})
    files = {k: os.path.join(label_server.OUT, k) for k in keys}
    if cache.missing(files):
        log(f"dinov2: embedding {len(cache.missing(files))} backgrounds")
        cache.update(Backbone("dinov2"), files, log=log)
    return dict(zip(keys, cache.get(keys))) if keys else {}


def baselines(boards: List[P.Board], seeds: int = 5) -> Dict[str, dict]:
    res = {"rule": P.summarize([P.rollout(b, P.agent_chooser(RulePolicy(), b.pair)) for b in boards], boards),
           "fixed_start": P.summarize([P.rollout(b, P.agent_chooser(FixedDefault(), b.pair)) for b in boards], boards)}
    eps = [P.rollout(b, P.agent_chooser(RandomPolicy(seed=s), b.pair)) for s in range(seeds) for b in boards]
    res["random"] = P.summarize(eps, boards)
    return res


def net_eval(bundle, boards: List[P.Board], noisy: bool = False) -> dict:
    return P.summarize([P.rollout(b, P.chooser(bundle, b.bg), noisy=noisy) for b in boards], boards)


def group_folds(boards: List[P.Board], k: int, seed: int) -> List[int]:
    bgs = sorted({b.pair.split("/")[1][:3] for b in boards})  # pair id = <background id><subject id>, e.g. D03T151
    random.Random(seed).shuffle(bgs)
    fold_of = {g: i % k for i, g in enumerate(bgs)}
    return [fold_of[b.pair.split("/")[1][:3]] for b in boards]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rater", default="tac_gia")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="few epochs: check the pipeline end to end")
    ap.add_argument("--min-boards", type=int, default=10)
    args = ap.parse_args()

    rows = train_observer.rows(args.rater)
    boards = boards_from_rows(rows, background_features(rows), model1_probs())
    train = [b for b in boards if b.split == "train"]
    test = [b for b in boards if b.split == "test"]
    log(f"{len(boards)} complete boards ({len(train)} train, {len(test)} test); "
        f"{sum(bool(b.probs) for b in boards)} with Model 1 probabilities")
    if len(train) < args.min_boards:
        sys.exit(f"only {len(train)} fully labelled train pairs: label more of outputs/label_b* (need {args.min_boards})")
    cfg = P.FitConfig(seed=args.seed, **({"bc_epochs": 10, "rl_epochs": 10} if args.quick else {}))

    report = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "rater": args.rater,
              "boards": {"train": len(train), "test": len(test)}, "budget": P.BUDGET, "fit": cfg.__dict__,
              "reward": {"good": P.REWARD_GOOD, "fail": P.REWARD_FAIL, "per_render": -P.COST_RENDER}}
    fold = group_folds(train, args.folds, args.seed)
    cv_eps, cv_noisy = [], []
    for f in range(args.folds):
        fit_b = [b for b, g in zip(train, fold) if g != f]
        val_b = [b for b, g in zip(train, fold) if g == f]
        if not fit_b or not val_b:
            continue
        bundle = P.fit(fit_b, cfg)
        cv_eps += [P.rollout(b, P.chooser(bundle, b.bg)) for b in val_b]
        cv_noisy += [P.rollout(b, P.chooser(bundle, b.bg), noisy=True) for b in val_b if b.probs]
        log(f"fold {f + 1}/{args.folds}: {len(val_b)} boards, success "
            f"{sum(e.success for e in cv_eps[-len(val_b):]) / len(val_b):.2f}")
    report["cv"] = {"net": P.summarize(cv_eps, train), **baselines(train)}
    if cv_noisy:
        report["cv"]["net_model1_judge"] = P.summarize(cv_noisy, [b for b in train if b.probs])

    bundle = P.fit(train, cfg, log=log)
    if test:
        report["test"] = {"net": net_eval(bundle, test), **baselines(test)}
    else:
        report["test"] = {"n": 0, "note": "no fully labelled test pair yet"}
    out = os.path.join(label_server.OUT, "proposer")
    os.makedirs(out, exist_ok=True)
    P.save(bundle, os.path.join(out, "proposer.pt"), boards=len(train), created_utc=report["created_utc"])
    with open(os.path.join(out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    for part in ("cv", "test"):
        for name, s in report[part].items():
            if isinstance(s, dict) and "success" in s:
                log(f"{part:4s} {name:17s} success {s['success']:.2f}  renders {s['mean_renders']:.2f}  "
                    f"first-try {s['first_try_good']:.2f}  (oracle {s['oracle']:.2f}, n={s['n']})")
    log(f"saved {os.path.relpath(out, ROOT)}/proposer.pt and report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
