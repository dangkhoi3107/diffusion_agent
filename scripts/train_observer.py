"""Model 1 stage A: frozen-backbone features (cached) -> heads A (visibility, CORAL) + B (broken), evaluated on your labels.

    ~/.venvs/camo-ml/bin/python scripts/train_observer.py                  # features for new renders, CV, test, save
    ~/.venvs/camo-ml/bin/python scripts/train_observer.py --features-only  # just extend the feature cache

Data: every render you labelled (rater tac_gia, outputs/labels.jsonl) in any run, plus the label set batches
outputs/label_b*. A config that looks identical to its pair's C00 (mean |delta| < 2/255, hidden on the label page)
inherits C00's label, as in scripts/replay_agents.py. "unsure" is dropped.
Scores are on your own labels only: inherited near-duplicates are trained on, never scored.
  cv        5 folds grouped by pair (subject x background, matched by content across runs), train pairs only
  bg_group  leave one background group out (an unseen kind of background)
  test      fit on all train pairs, score the label-set test pairs (manifest split: test background or subject);
            old-run renders that reuse a test background or subject are kept out of training altogether
Variants: dinov2 pair MLP (the model), dinov2 render only, dinov2 linear probe, clip pair MLP, majority class;
GPT vision (outputs/observer_eval*.json) for reference.
Output in outputs/observer/: report.json, cv_predictions.jsonl (out-of-fold, for Model 2's noisy-label practice),
observer_dinov2.pt (main variant fit on all train pairs), feats_<backbone>.npz (feature cache).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import label_server  # noqa: E402
from camo import observer as obs  # noqa: E402

MANIFEST = os.path.join(ROOT, "pilot", "labelset", "manifest.json")
VARIANTS = {  # name: (backbone, mode, hidden)
    "dinov2_pair_mlp": ("dinov2", "pair", 256),
    "dinov2_render_mlp": ("dinov2", "render", 256),
    "dinov2_pair_linear": ("dinov2", "pair", 0),
    "clip_pair_mlp": ("clip", "pair", 256),
}
MAIN = "dinov2_pair_mlp"
SAME_IMAGE = 0.98  # cosine of DINOv2 features above which two input images are the same picture


def out_dir() -> str:
    return os.path.join(label_server.OUT, "observer")


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def latest_labels(rater: str) -> Dict[str, str]:
    latest: Dict[str, str] = {}
    path = os.path.join(label_server.OUT, "labels.jsonl")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r.get("rater") == rater:
                        latest[r["uid"]] = r["label"]
    return latest


def labelset_runs() -> set:
    """Runs of the current label set (manifest batches). Older label_b* runs (grid v2, before the backgrounds were
    re-picked) reuse ids such as D01 for other pictures, so the manifest split must not be applied to them."""
    if not os.path.exists(MANIFEST):
        return set()
    with open(MANIFEST, encoding="utf-8") as f:
        m = json.load(f)
    return {b["run"] for b in m.get("batches", [])}


def manifest_split() -> Dict[str, str]:
    """subject or background id of the label set -> train | test."""
    if not os.path.exists(MANIFEST):
        return {}
    with open(MANIFEST, encoding="utf-8") as f:
        m = json.load(f)
    return {k: v.get("split", "train") for part in ("subjects", "backgrounds") for k, v in m[part].items()}


def rows(rater: str = "tac_gia", inherit_below: float = 2.0) -> List[dict]:
    """One row per labelled render (own or inherited label). Paths are relative to outputs/."""
    labels = latest_labels(rater)
    runs = {u.split("/")[0] for u in labels} | {os.path.basename(p) for p in glob.glob(os.path.join(label_server.OUT, "label_b*"))}
    items = label_server.collect(0.0, True, runs=runs)
    split_of = manifest_split()
    current = labelset_runs()
    by_config = {(i["run"], i["pair_id"], i["config_id"]): labels.get(i["uid"]) for i in items}
    out = []
    for i in items:
        label, source = labels.get(i["uid"]), "own"
        if label is None and i["diff_vs_c00"] is not None and i["diff_vs_c00"] < inherit_below:
            label, source = by_config.get((i["run"], i["pair_id"], i.get("base_id", "C00"))), "inherited_from_C00"
        if label is None or label not in obs.CLASSES:  # unlabelled or unsure
            continue
        if i["run"] in current:
            split = "test" if "test" in (split_of.get(i["subject_id"]), split_of.get(i["background_id"])) else "train"
        else:
            split = "train"
        out.append({"uid": i["uid"], "run": i["run"], "pair_id": i["pair_id"], "config_id": i["config_id"],
                    "subject_id": i["subject_id"], "background_id": i["background_id"], "bg_group": i.get("bg_group"),
                    "label": label, "source": source, "split": split,
                    "out": i["out"][len("files/"):], "bg": i["bg"][len("files/"):], "subj": i["subj"][len("files/"):]})
    return out


def same_picture(keys: List[str], X: np.ndarray, threshold: float = SAME_IMAGE) -> Dict[str, str]:
    """Input images shown under different ids in different runs -> one canonical key (first seen, sorted)."""
    U = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    canon: Dict[str, str] = {}
    reps: List[int] = []
    for i in np.argsort(keys, kind="stable"):
        hit = next((r for r in reps if float(U[i] @ U[r]) >= threshold), None)
        if hit is None:
            reps.append(i)
        canon[keys[i]] = keys[i if hit is None else hit]
    return canon


def attach_groups(data: List[dict], canon: Dict[str, str]) -> Dict[str, int]:
    """pair group = canonical subject x canonical background; renders that share a test input leave training."""
    split_of = manifest_split()
    current = labelset_runs()
    test_inputs = set()
    for r in data:
        r["subj_c"], r["bg_c"] = canon[r["subj"]], canon[r["bg"]]
        r["group"] = f"{r['subj_c']}|{r['bg_c']}"
        if r["run"] in current:
            test_inputs |= {c for c, i in ((r["subj_c"], r["subject_id"]), (r["bg_c"], r["background_id"]))
                            if split_of.get(i) == "test"}
    leaked = Counter()
    for r in data:
        if r["split"] == "train" and (r["subj_c"] in test_inputs or r["bg_c"] in test_inputs):
            r["split"] = "excluded"
            leaked[r["run"]] += 1
    return dict(leaked)


def features(data: List[dict], backbones: List[str]) -> Dict[str, "FeatureCache"]:
    from camo.backbone import Backbone, FeatureCache

    caches = {}
    for name in backbones:
        cache = FeatureCache(os.path.join(out_dir(), f"feats_{name}.npz"))
        keys = {r[k] for r in data for k in ("out", "bg")} | ({r["subj"] for r in data} if name == "dinov2" else set())
        files = {k: os.path.join(label_server.OUT, k) for k in keys}
        todo = cache.missing(files)
        if todo:
            log(f"{name}: {len(todo)} new images to embed ({len(files) - len(todo)} cached)")
            cache.update(Backbone(name), files, log=log)
        caches[name] = cache
    return caches


def matrix(data: List[dict], cache, mode: str) -> np.ndarray:
    return obs.inputs(cache.get([r["out"] for r in data]), cache.get([r["bg"] for r in data]), mode)


def folds(data: List[dict], k: int, seed: int) -> List[int]:
    """Fold per row, by pair group; -1 for rows outside train."""
    groups = sorted({r["group"] for r in data if r["split"] == "train"})
    random.Random(seed).shuffle(groups)
    fold_of = {g: n % k for n, g in enumerate(groups)}
    return [fold_of[r["group"]] if r["split"] == "train" else -1 for r in data]


def cross_validate(data, X, cfg, fold) -> List[Optional[tuple]]:
    """Out-of-fold (label, probs) per row (None outside train)."""
    labels = np.array([r["label"] for r in data])
    fold = np.array(fold)
    out: List[Optional[tuple]] = [None] * len(data)
    for f in sorted(set(fold) - {-1}):
        tr, va = np.where((fold != f) & (fold >= 0))[0], np.where(fold == f)[0]
        bundle = obs.fit(X[tr], labels[tr], cfg)
        pred, P = obs.predict(bundle, X[va])
        for i, p, pr in zip(va, pred, P):
            out[i] = (p, pr)
        log(f"  fold {f + 1}/{fold.max() + 1} done")
    return out


def scored(data, preds, keep=lambda r: True) -> dict:
    idx = [i for i, r in enumerate(data) if preds[i] is not None and r["source"] == "own" and keep(r)]
    return obs.metrics([data[i]["label"] for i in idx], [preds[i][0] for i in idx])


def gpt_reference() -> List[dict]:
    out = []
    for path in sorted(glob.glob(os.path.join(label_server.OUT, "observer_eval_*.json"))):
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        if r.get("observer") == "observer_llm":
            out.append({"file": os.path.basename(path), "n": r["n"], "exact": r["exact_agreement"],
                        "good_vs_not": r["good_vs_not_agreement"]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rater", default="tac_gia")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--features-only", action="store_true")
    ap.add_argument("--no-bg-group", action="store_true", help="skip the leave-one-background-group-out runs")
    args = ap.parse_args()

    data = rows(args.rater)
    if not data:
        sys.exit("no labelled renders yet")
    log(f"{len(data)} labelled renders ({Counter(r['source'] for r in data)}) from {len({r['run'] for r in data})} runs")
    backbones = sorted({VARIANTS[v][0] for v in args.variants} | {"dinov2"})
    caches = features(data, backbones)
    if args.features_only:
        return 0

    inputs_keys = sorted({r[k] for r in data for k in ("subj", "bg")})
    canon = same_picture(inputs_keys, caches["dinov2"].get(inputs_keys))
    leaked = attach_groups(data, canon)
    split_counts = Counter((r["split"], r["source"]) for r in data)
    log(f"splits {dict(split_counts)}; {len({r['group'] for r in data})} pair groups; kept out (test input reused): {leaked}")
    fold = folds(data, args.folds, args.seed)
    is_train = [r["split"] == "train" for r in data]
    labels = np.array([r["label"] for r in data])

    report = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "rater": args.rater,
              "rows": len(data), "splits": {f"{a}/{b}": n for (a, b), n in split_counts.items()},
              "label_counts": {s: dict(Counter(r["label"] for r in data if r["split"] == s and r["source"] == "own"))
                               for s in ("train", "test")},
              "runs": dict(Counter(r["run"] for r in data)), "kept_out_test_inputs": leaked,
              "folds": args.folds, "variants": {}, "gpt_vision_reference": gpt_reference()}
    majority = Counter(labels[is_train]).most_common(1)[0][0]
    report["variants"]["majority_class"] = {"predicts": majority,
                                            "cv": scored(data, [(majority, None) if t else None for t in is_train])}
    oof_main = None
    for name in args.variants:
        backbone, mode, hidden = VARIANTS[name]
        cfg = obs.FitConfig(hidden=hidden, seed=args.seed)
        X = matrix(data, caches[backbone], mode)
        t0 = time.time()
        oof = cross_validate(data, X, cfg, fold)
        res = {"backbone": backbone, "mode": mode, "hidden": hidden, "cv": scored(data, oof),
               "cv_label_set_only": scored(data, oof, lambda r: r["run"].startswith("label_b"))}
        tr = np.where(is_train)[0]
        te = [i for i, r in enumerate(data) if r["split"] == "test"]
        if te:
            bundle = obs.fit(X[tr], labels[tr], cfg)
            pred, P = obs.predict(bundle, X[te])
            test_preds: List[Optional[tuple]] = [None] * len(data)
            for i, p, pr in zip(te, pred, P):
                test_preds[i] = (p, pr)
            res["test"] = scored(data, test_preds)
        else:
            res["test"] = {"n": 0, "note": "no labelled test pair yet"}
        if name == MAIN and not args.no_bg_group:
            per = {}
            for g in sorted({r["bg_group"] for r in data if r["bg_group"] and r["split"] == "train"}):
                held = np.array([r["bg_group"] == g and t for r, t in zip(data, is_train)])
                rest = np.array(is_train) & ~held
                if held.sum() < 5 or rest.sum() < 20:
                    continue
                bundle = obs.fit(X[rest], labels[rest], cfg)
                log(f"  background group out: {g} done")
                pred, _ = obs.predict(bundle, X[held])
                gp: List[Optional[tuple]] = [None] * len(data)
                for i, p in zip(np.where(held)[0], pred):
                    gp[i] = (p, None)
                per[g] = scored(data, gp)
            res["bg_group_out"] = {g: {k: m[k] for k in ("n", "exact", "good_vs_not", "ordinal_mae")} for g, m in per.items()}
        res["seconds"] = round(time.time() - t0, 1)
        report["variants"][name] = res
        log(f"{name}: cv exact {res['cv'].get('exact')} good/not {res['cv'].get('good_vs_not')} "
            f"(n={res['cv']['n']}), test n={res['test']['n']} ({res['seconds']}s)")
        if name == MAIN:
            oof_main = oof
            final = obs.fit(X[tr], labels[tr], cfg)
            obs.save(final, os.path.join(out_dir(), "observer_dinov2.pt"), backbone=backbone, mode=mode, rater=args.rater,
                     created_utc=report["created_utc"], cv=res["cv"])

    os.makedirs(out_dir(), exist_ok=True)
    with open(os.path.join(out_dir(), "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    if oof_main is not None:
        with open(os.path.join(out_dir(), "cv_predictions.jsonl"), "w", encoding="utf-8") as f:
            for r, p in zip(data, oof_main):
                if p is not None:
                    f.write(json.dumps({"uid": r["uid"], "label": r["label"], "source": r["source"], "pred": p[0],
                                        "probs": dict(zip(obs.CLASSES, np.round(p[1], 4).tolist()))}) + "\n")
    print_table(report)
    return 0


def print_table(report: dict) -> None:
    print(f"\n{'variant':<22}{'part':<6}{'n':>6}{'exact':>8}{'MAE':>7}{'macroR':>8}{'goodP':>7}{'goodR':>7}{'brokenF1':>10}")
    for name, v in report["variants"].items():
        for part in ("cv", "test"):
            m = v.get(part) or {}
            if m.get("n"):
                print(f"{name:<22}{part:<6}{m['n']:>6}{m.get('exact', '-')!s:>8}{m.get('ordinal_mae', '-')!s:>7}"
                      f"{m.get('macro_recall', '-')!s:>8}{m.get('good_precision', '-')!s:>7}{m.get('good_recall', '-')!s:>7}"
                      f"{m.get('broken_f1', '-')!s:>10}")
    for g in report["gpt_vision_reference"]:
        print(f"{'gpt ' + g['file'][13:-5]:<22}{g['n']:>6}{g['exact']!s:>8}{g['good_vs_not']!s:>10}")
    main_v = report["variants"].get(MAIN, {})
    if main_v.get("bg_group_out"):
        print("\nleave one background group out (dinov2_pair_mlp):")
        for g, m in main_v["bg_group_out"].items():
            print(f"  {g:<24} n={m['n']:<5} exact {m['exact']}  good/not {m['good_vs_not']}")


if __name__ == "__main__":
    sys.exit(main())
