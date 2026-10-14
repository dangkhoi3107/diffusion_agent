"""Label set for the self-trained agents (docs/AGENT_PLAN.html §4): 300 subject/background pairs x the config grid,
rendered on Kaggle in 6 batches and pulled into outputs/ as each batch finishes, so labelling starts after batch 1.

    python3 scripts/labelset.py build     # pick 60 DTD backgrounds + 18 subjects -> pilot/labelset/b1..b6
    python3 scripts/labelset.py build-nature  # append natural backgrounds as batches b7, b8 (needs bg_nature_raw/)
    python3 scripts/labelset.py fill-dtd          # held batches: swap duplicate DTD categories for missing ones
    python3 scripts/labelset.py build-nature2     # append batch b9: 5 new natural types (pilot/labelset_nature2_picks.json)
    python3 scripts/labelset.py rebuild-subjects [--release]  # held batches -> shape-group subjects v2
    python3 scripts/labelset.py run       # render batch by batch on Kaggle, unzip each into outputs/label_bK
    python3 scripts/labelset.py status

5 classes (cat, lizard, butterfly, fish, human) x 60 backgrounds (10 per DTD group of camo/sources.py, categories that
often contain animals excluded). Each pair renders C00 + every ACTION_SPACE config (pilot/labelset/grid.yaml; v2 since
2026-10-12: 5 alpha x 3 bg_strength = 15, the v1 grid barely changed the image, see camo/agent.py).
Every batch mixes all 6 groups. 2 backgrounds per group (60 pairs) are marked "test" in manifest.json: never train on them.
The runner writes outputs/.labelset_status.json, which the label page shows; it resumes where it stopped.

Subjects v2 (2026-10-12, batches 2-8): cutouts picked by eye into pilot/labelset_subject_picks.json, grouped by
silhouette shape (scripts/fetch_subjects.py SHAPES). Every background gets 5 subjects from 5 different shape groups,
the least used group / instance / species first; every 3rd background swaps one slot for a human face. Subjects of
TEST_SHAPES (unseen shape) and TEST_SPECIES (unseen species, familiar shape) are "test": a pair is test when its
background or its subject is.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

import yaml
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from camo.agent import ACTION_SPACE, SEEDS, _product, grid_id, render_overrides  # noqa: E402
from camo.sources import DTD_CONTAINS_ANIMAL, DTD_GROUPS  # noqa: E402

SET = os.path.join(ROOT, "pilot", "labelset")
OUT = os.path.join(ROOT, "outputs")
STATUS = os.path.join(OUT, ".labelset_status.json")
BG_DIR = os.path.join(ROOT, "bg_dtd")
CANDIDATES = os.path.join(OUT, "pilot", "pilot_inputs")  # the 4 fetched candidates of each pilot subject
KAGGLE_USER = "dangkhoi3107"
BATCHES, PER_GROUP = 6, 10
TEST_ROWS = (1, 6)  # background rows (one per group each) held out as test; they land in batches 1-2 and 4-5
# Pilot candidates kept as subjects. Dropped: O02_3 (newt, not a lizard), O04_3 (sea slug, not a fish), O05_1 and
# O06_3 (hand / microphone over the face), O05_2 (printed text), O06_1 (a child).
SUBJECTS = {"cat": ["O01_0", "O01_1", "O01_2", "O01_3"], "lizard": ["O02_0", "O02_1", "O02_2"],
            "butterfly": ["O03_0", "O03_1", "O03_2", "O03_3"], "fish": ["O04_0", "O04_1", "O04_2"],
            "human": ["O05_0", "O05_3", "O06_0", "O06_2"]}


# Natural backgrounds (added 2026-10-12, appended as extra batches b7, b8): 2 images per type. Sources in bg_nature_raw/
# (Kaggle): BarkVN-50 (CC BY 4.0), Flickr Material Database (mask coverage >= 95% only), GTOS-mobile (ground terrain).
SUBJECT_PICKS = os.path.join(ROOT, "pilot", "labelset_subject_picks.json")  # {"picks": {shape: [{id, species, src}]}}
V1_SHAPES = {"cat": "quadruped_compact", "lizard": "low_elongated_legged", "butterfly": "wings_spread",
             "fish": "streamlined_fins", "human": "human"}  # v1 subjects join the v2 shape pools
TEST_SHAPES = ("round_shelled",)
TEST_SPECIES = ("Rabbit", "Horse", "Penguin")
SUBJECTS_PER_BG, HUMAN_EVERY = 5, 3

NATURE_GROUPS = {"nature_plant": ["bark", "foliage", "wood", "grass", "moss"],
                 "nature_ground": ["stone", "water", "sand", "soil", "gravel"]}
NATURE_RAW = os.path.join(ROOT, "bg_nature_raw")
NATURE_PICKS = os.path.join(ROOT, "pilot", "labelset_nature_picks.json")  # {"picks": {type: [path, ...]}}
FMD_TYPES = ("foliage", "stone", "water", "wood")
GTOS_CLASSES = {"grass": ("grass",), "moss": ("moss",), "sand": ("sand",), "soil": ("soil",), "gravel": ("pebble",)}
PER_NATURE, NATURE_TEST_EVERY = 2, 5  # 1 natural background in 5 is test


def nature_split(i: int, j: int) -> str:
    """Round i (one image of every type), position j: the test slot shifts by 2 each round, so a test type keeps a
    train image of the same type (a plain every-5th rule put both moss and both gravel images in test)."""
    return "test" if (j + 2 * i) % NATURE_TEST_EVERY == NATURE_TEST_EVERY - 1 else "train"
# Second natural batch (2026-10-12): candidates from scripts/fetch_backgrounds.py (Open Images, cut to the most even
# square), picked by eye. Type -> existing group; the material field keeps the type apart.
NATURE2_TYPES = {"leaf_litter": "nature_plant", "snow": "nature_ground", "seabed": "nature_ground",
                 "rock": "nature_ground", "mud": "nature_ground"}
NATURE2_PICKS = os.path.join(ROOT, "pilot", "labelset_nature2_picks.json")  # {"picks": {type: [path, ...]}}
# fill-dtd order when a group has fewer duplicate slots than missing categories: zigzagged failed 6/6 in the pilot;
# lined ~ striped, studded ~ polka-dotted and waffled ~ grid come last
DTD_FILL_FIRST = ["zigzagged", "grid", "dotted", "banded", "braided", "grooved"]


GRID_VERSION = 5
RUN_SUFFIX = "v3"  # label-set generation 3 (clean backgrounds, label_b<k>v3); grids v4, v5 since the trial batch


def run_name(k: int) -> str:
    return f"label_b{k}{RUN_SUFFIX}"


def grid_configs() -> dict:
    configs = {"C00": {}}
    for k, seed in SEEDS.items():  # the thesis default once per seed: base of the "identical to C00" check
        if seed is not None:
            configs[f"C00_s{k}"] = {"sampling.seed": seed}
    for values in _product([ACTION_SPACE[k] for k in ACTION_SPACE]):
        cfg = dict(zip(ACTION_SPACE, values))
        configs[grid_id(cfg)] = render_overrides(cfg)
    return configs


def write_grid() -> str:
    """pilot/labelset/grid.yaml: C00 + every ACTION_SPACE config, one seed per pair (pilot.pair_seed)."""
    path = os.path.join(SET, "grid.yaml")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Label set grid v{GRID_VERSION}: C00 (once per seed) + every ACTION_SPACE config of camo/agent.py (generated by\n"
                "# scripts/labelset.py). seed_per_pair: each pair renders with its own seed, the same for all its configs.\n")
        yaml.safe_dump({"seed_per_pair": True, "configs": grid_configs()}, f, sort_keys=False)
    return path


def used_by_bgmap() -> set:
    path = os.path.join(ROOT, "pilot", "bgmap_inputs", "inputs.yaml")
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        bgs = (yaml.safe_load(f) or {}).get("backgrounds") or {}
    return {str(b.get("source_and_license", "")).split("/")[-1].split(" ")[0] for b in bgs.values()}


def pick_backgrounds(rng: random.Random) -> dict:
    """{group: [file, ...]}: PER_GROUP images per group, categories taken round-robin so each group stays varied."""
    files = sorted(os.listdir(BG_DIR))
    skip = used_by_bgmap()
    by_cat = {}
    for name in files:
        cat = name.rsplit("_", 1)[0]
        if name.lower().endswith((".jpg", ".png")) and name not in skip:
            by_cat.setdefault(cat, []).append(name)
    picked = {}
    for group, cats in DTD_GROUPS.items():
        cats = [c for c in cats if c in by_cat and c not in DTD_CONTAINS_ANIMAL]
        rng.shuffle(cats)
        pools = {c: rng.sample(by_cat[c], len(by_cat[c])) for c in cats}
        chosen = []
        while len(chosen) < PER_GROUP:
            for c in cats:
                if len(chosen) < PER_GROUP and pools[c]:
                    chosen.append(pools[c].pop())
        picked[group] = chosen
    return picked


def build(seed: int) -> dict:
    rng = random.Random(seed)
    with open(os.path.join(CANDIDATES, "inputs.yaml"), encoding="utf-8") as f:
        pilot_subjects = yaml.safe_load(f)["subjects"]
    subjects = {}
    for cls, cands in SUBJECTS.items():
        for cand in cands:
            sid = f"S{len(subjects) + 1:02d}"
            meta = pilot_subjects[cand.split("_")[0]]
            subjects[sid] = {"label": meta["label"], "category": meta["category"], "class": cls, "origin": cand,
                             "representation": "photograph", "shape_tags": meta.get("shape_tags") or [],
                             "source_and_license": meta.get("source_and_license", ""), "path": f"candidates/{sid}.png"}
    # interleave groups so that batch k holds backgrounds of every group
    picked = pick_backgrounds(rng)
    groups = list(DTD_GROUPS)
    backgrounds = {}
    for i in range(PER_GROUP):
        for g in groups:
            name = picked[g][i]
            cat = name.rsplit("_", 1)[0]
            bid = f"D{len(backgrounds) + 1:02d}"
            backgrounds[bid] = {"material": f"dtd:{cat}", "group": g, "representation": "photograph",
                                "contains_animal_risk": False, "file": name,
                                "split": "test" if i in TEST_ROWS else "train",
                                "source_and_license": f"DTD r1.0.1/{cat}/{name} (DTD: research use)",
                                "path": f"candidates/{bid}{os.path.splitext(name)[1]}"}
    pairs = assign_pairs(rng, subjects, list(backgrounds))
    if os.path.isdir(SET):
        shutil.rmtree(SET)
    os.makedirs(SET)
    write_grid()
    bg_ids = list(backgrounds)
    per_batch = len(bg_ids) // BATCHES
    batches = [write_batch(k, f"Label set batch {k}/{BATCHES}", bg_ids[(k - 1) * per_batch:k * per_batch],
                           pairs, subjects, backgrounds, seed) for k in range(1, BATCHES + 1)]
    manifest = {"seed": seed, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "configs": list(grid_configs()), "subjects": subjects, "backgrounds": backgrounds,
                "pairs": pairs, "batches": batches}
    with open(os.path.join(SET, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    return manifest


def assign_pairs(rng: random.Random, subjects: dict, bids: list) -> dict:
    """One subject per class on every background; instances cycle so each is used about equally often."""
    cycles = {cls: [] for cls in SUBJECTS}
    pairs = {}
    for bid in bids:
        for cls in SUBJECTS:
            if not cycles[cls]:
                cycles[cls] = rng.sample([s for s, m in subjects.items() if m["class"] == cls], len(SUBJECTS[cls]))
            sid = cycles[cls].pop()
            pairs[f"{bid}{sid}"] = [sid, bid]
    return pairs


def write_batch(k: int, note: str, bids: list, pairs: dict, subjects: dict, backgrounds: dict, seed: int) -> dict:
    """pilot/labelset/b<k>/: inputs.yaml (every pair is an audit pair, so all grid configs render) + its images."""
    bpairs = {p: v for p, v in pairs.items() if v[1] in bids}
    sids = sorted({v[0] for v in bpairs.values()})
    folder = os.path.join(SET, f"b{k}")
    os.makedirs(os.path.join(folder, "candidates"))
    for sid in sids:
        src = os.path.join(ROOT, subjects[sid]["src"]) if subjects[sid].get("src") else \
            os.path.join(CANDIDATES, "candidates", f"{subjects[sid]['origin']}.png")
        shutil.copy2(src, os.path.join(folder, subjects[sid]["path"]))
        lic = os.path.splitext(src)[0] + ".json"
        if os.path.exists(lic):
            shutil.copy2(lic, os.path.join(folder, "candidates", f"{sid}.json"))
    for bid in bids:
        meta = backgrounds[bid]
        src = os.path.join(ROOT, meta["src"]) if meta.get("src") else os.path.join(BG_DIR, meta["file"])
        if meta.get("crop"):  # natural photos: keep only the region the material fills
            with Image.open(src) as img:
                img.convert("RGB").crop(tuple(meta["crop"])).save(os.path.join(folder, meta["path"]), quality=95)
        else:
            shutil.copy2(src, os.path.join(folder, meta["path"]))
    doc = {"version": run_name(k),
           "note": f"{note}: {len(bpairs)} pairs x grid.yaml (scripts/labelset.py, seed {seed}).",
           "subjects": {s: {k2: v for k2, v in subjects[s].items() if k2 not in ("class", "src", "split", "version")}
                        for s in sids},
           "backgrounds": {b: {k2: v for k2, v in backgrounds[b].items() if k2 not in ("file", "src", "crop", "split")}
                           for b in bids},
           "pairs": bpairs, "audit_pairs": list(bpairs)}
    with open(os.path.join(folder, "inputs.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True)
    return {"run": run_name(k), "folder": os.path.relpath(folder, ROOT), "pairs": list(bpairs)}


def nature_candidates() -> dict:
    """{type: [(path relative to ROOT, license), ...]} from the three downloaded datasets."""
    import glob

    import numpy as np

    out = {}
    bark = sorted(glob.glob(os.path.join(NATURE_RAW, "barkvn50", "*", "*", "*", "*.[jJ][pP][gG]")))
    out["bark"] = [(p, "BarkVN-50 (Mendeley), CC BY 4.0") for p in bark]
    fmd = os.path.join(NATURE_RAW, "fmd")
    for t in FMD_TYPES:
        keep = []
        for name in sorted(os.listdir(os.path.join(fmd, "image", t))):
            mask = os.path.join(fmd, "mask", t, name)
            if name.endswith(".jpg") and os.path.exists(mask):
                with Image.open(mask) as m:
                    if (np.asarray(m.convert("L")) > 127).mean() >= 0.95:
                        keep.append((os.path.join(fmd, "image", t, name), "Flickr Material Database (Flickr CC images; research use)"))
        out[t] = keep
    gtos_dirs = [d for d in glob.glob(os.path.join(NATURE_RAW, "gtos", "**", "*"), recursive=True) if os.path.isdir(d)]
    for t, names in GTOS_CLASSES.items():
        dirs = [d for d in gtos_dirs if os.path.basename(d) in names]  # exact: "sand" must not match "sandPaper"
        videos = {}
        for d in dirs:  # crops/frames of one scene look alike (256_sample054_01_i00h, IMG_0169_frame3): one per scene
            for f in sorted(glob.glob(os.path.join(d, "*.jpg"))):
                scene = re.search(r"(sample\d+|IMG_\d+)", os.path.basename(f))
                videos.setdefault((os.path.basename(d), scene.group(1) if scene else f), []).append(f)
        out[t] = [(v[len(v) // 2], f"GTOS-mobile ({os.path.basename(os.path.dirname(v[0]))}; research use)")
                  for v in videos.values()]
    for t in [t for ts in NATURE_GROUPS.values() for t in ts]:
        if not out.get(t):
            sys.exit(f"no source images for natural type '{t}': is bg_nature_raw/ complete?")
    return {t: [(os.path.relpath(p, ROOT), lic) for p, lic in v] for t, v in out.items()}


def build_nature(seed: int) -> dict:
    """Append natural-background batches to the existing label set without touching batches already built."""
    man = load_manifest()
    if any(b.get("group", "").startswith("nature_") for b in man["backgrounds"].values()):
        sys.exit("natural batches already in manifest.json")
    rng = random.Random(seed + 1)
    cands = nature_candidates()
    picked = {t: rng.sample(cands[t], PER_NATURE) for ts in NATURE_GROUPS.values() for t in ts}
    if os.path.exists(NATURE_PICKS):  # hand-picked images win over the random sample
        with open(NATURE_PICKS, encoding="utf-8") as f:
            for t, paths in json.load(f)["picks"].items():
                lic = dict(cands[t])
                picked[t] = [(p, lic[p]) for p in paths]
    backgrounds = {}
    n0 = len(man["backgrounds"])
    for i in range(PER_NATURE):
        for j, (group, t) in enumerate((g, t) for g, ts in NATURE_GROUPS.items() for t in ts):
            path, lic = picked[t][i]
            bid = f"D{n0 + len(backgrounds) + 1:02d}"
            with Image.open(os.path.join(ROOT, path)) as img:
                w, h = img.size
            side = min(w, h)  # square centre crop at the source: no stretched textures
            crop = [(w - side) // 2, (h - side) // 2, (w - side) // 2 + side, (h - side) // 2 + side]
            backgrounds[bid] = {"material": f"nature:{t}", "group": group, "representation": "photograph",
                                "contains_animal_risk": False, "src": path, "crop": crop,
                                "split": nature_split(i, j),
                                "source_and_license": f"{lic}: {os.path.relpath(path, 'bg_nature_raw')}",
                                "path": f"candidates/{bid}.jpg"}
    pairs = assign_pairs(rng, man["subjects"], list(backgrounds))
    bg_ids, k0 = list(backgrounds), len(man["batches"])
    per_batch = len(man["batches"][0]["pairs"]) // len(SUBJECTS)
    n_new = -(-len(bg_ids) // per_batch)
    new = [write_batch(k0 + j, f"Label set natural batch {j}/{n_new}", bg_ids[(j - 1) * per_batch:j * per_batch],
                       pairs, man["subjects"], backgrounds, seed) for j in range(1, n_new + 1)]
    man["backgrounds"].update(backgrounds)
    man["pairs"].update(pairs)
    man["batches"].extend(new)
    with open(os.path.join(SET, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=1)
    return {"backgrounds": backgrounds, "pairs": pairs, "batches": new}


def fill_dtd(seed: int) -> list:
    """Held batches only: a background whose DTD category appears twice in its group takes a category the group lacks
    (bg_dtd/ was missing 10 categories when the set was built). Background id, group, split and batch stay."""
    from collections import Counter

    man = load_manifest()
    held = {man["pairs"][p][1] for b in man["batches"] if b.get("hold") for p in b["pairs"]}
    available = {}
    for name in sorted(os.listdir(BG_DIR)):
        if name.lower().endswith((".jpg", ".png")):
            available.setdefault(name.rsplit("_", 1)[0], []).append(name)
    skip = used_by_bgmap()
    rng = random.Random(seed + 3)
    changes = []
    for group, cats in DTD_GROUPS.items():
        bids = sorted(b for b, m in man["backgrounds"].items() if m["group"] == group)
        count = Counter(man["backgrounds"][b]["material"].split(":", 1)[1] for b in bids)
        missing = [c for c in cats if c not in count and c not in DTD_CONTAINS_ANIMAL and available.get(c)]
        missing.sort(key=lambda c: DTD_FILL_FIRST.index(c) if c in DTD_FILL_FIRST else len(DTD_FILL_FIRST))
        for cat in missing:
            slots = [b for b in bids if b in held and count[man["backgrounds"][b]["material"].split(":", 1)[1]] > 1]
            if not slots:
                print(f"{group}: no duplicate slot left for {cat}")
                continue
            # a train slot first: a category with one image should be seen in training
            bid = max(slots, key=lambda b: (man["backgrounds"][b]["split"] == "train",
                                            count[man["backgrounds"][b]["material"].split(":", 1)[1]], b))
            meta = man["backgrounds"][bid]
            name = rng.choice([n for n in available[cat] if n not in skip] or available[cat])
            changes.append((bid, meta["material"], f"dtd:{cat}"))
            count[meta["material"].split(":", 1)[1]] -= 1
            count[cat] += 1
            meta.update(material=f"dtd:{cat}", file=name, path=f"candidates/{bid}{os.path.splitext(name)[1]}",
                        source_and_license=f"DTD r1.0.1/{cat}/{name} (DTD: research use)")
    with open(os.path.join(SET, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=1)
    return changes


def build_nature2(seed: int) -> dict:
    """Append one held batch of the new natural types; rebuild-subjects then pairs it with subjects v2."""
    man = load_manifest()
    if any(m["material"].split(":", 1)[1] in NATURE2_TYPES for m in man["backgrounds"].values()):
        sys.exit("new natural types already in manifest.json")
    with open(NATURE2_PICKS, encoding="utf-8") as f:
        picks = json.load(f)["picks"]
    short = [t for t in NATURE2_TYPES if len(picks.get(t, [])) < PER_NATURE]
    if short:
        sys.exit(f"pick {PER_NATURE} images for: {', '.join(short)}")
    backgrounds, n0 = {}, len(man["backgrounds"])
    for i in range(PER_NATURE):
        for j, (t, group) in enumerate(NATURE2_TYPES.items()):
            path = picks[t][i]
            bid = f"D{n0 + len(backgrounds) + 1:02d}"
            meta = {"material": f"nature:{t}", "group": group, "representation": "photograph",
                    "contains_animal_risk": False, "src": path,
                    "split": nature_split(i, j),
                    "path": f"candidates/{bid}.jpg"}
            info_path = os.path.join(ROOT, os.path.splitext(path)[0] + ".json")
            if os.path.exists(info_path):  # fetch_backgrounds.py output: already cut to its most even square
                with open(info_path, encoding="utf-8") as f:
                    info = json.load(f)
                meta["source_and_license"] = (f"Open Images V7 {info['image_id']} ({info['license']}, "
                                              f"{info.get('author') or 'unknown author'}), crop {info['crop']}")
            else:  # a raw dataset image (GTOS-mobile fallback): square centre crop, as in build-nature
                with Image.open(os.path.join(ROOT, path)) as img:
                    w, h = img.size
                side = min(w, h)
                meta["crop"] = [(w - side) // 2, (h - side) // 2, (w - side) // 2 + side, (h - side) // 2 + side]
                meta["source_and_license"] = f"GTOS-mobile (research use): {os.path.relpath(path, 'bg_nature_raw')}"
            backgrounds[bid] = meta
    rng = random.Random(seed + 4)
    pairs = assign_pairs(rng, man["subjects"], list(backgrounds))  # placeholder v1 pairs until rebuild-subjects
    k = len(man["batches"]) + 1
    batch = write_batch(k, "Label set natural batch 3/3 (new types)", list(backgrounds), pairs, man["subjects"],
                        backgrounds, seed)
    batch["hold"] = "new natural types: run rebuild-subjects"
    man["backgrounds"].update(backgrounds)
    man["pairs"].update(pairs)
    man["batches"].append(batch)
    with open(os.path.join(SET, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=1)
    return {"backgrounds": backgrounds, "batch": batch["run"]}


def rebuild_subjects(seed: int, release: bool) -> dict:
    """Re-pair the held batches with the shape-group subjects v2; backgrounds and batch names stay."""
    from collections import Counter

    man = load_manifest()
    with open(SUBJECT_PICKS, encoding="utf-8") as f:
        picks = json.load(f)["picks"]
    targets = [k for k, b in enumerate(man["batches"]) if b.get("hold")]
    if not targets:
        sys.exit("no held batch to rebuild")
    for k in targets:
        if downloaded(man["batches"][k]["run"]):
            sys.exit(f"{man['batches'][k]['run']} is already rendered: un-hold it instead of rebuilding")
    target_bids = [man["pairs"][p][1] for k in targets for p in man["batches"][k]["pairs"]]
    target_bids = list(dict.fromkeys(target_bids))
    kept_pairs = {p: v for p, v in man["pairs"].items() if v[1] not in target_bids}
    used = {v[0] for v in kept_pairs.values()}
    subjects = {sid: m for sid, m in man["subjects"].items() if m.get("version") != 2 or sid in used}
    for m in subjects.values():  # v1 subjects: tag shape/species so the pools and the analysis can use them
        if m.get("version") != 2:
            m.update(shape=V1_SHAPES[m["class"]], species=m["class"].capitalize(), split="train")
    n = max([int(sid[1:]) for sid in subjects if sid.startswith("T")] or [0])
    for shape, entries in picks.items():
        for e in entries:
            n += 1
            sid = f"T{n:03d}"
            label = re.sub(r"\s*\(.*\)", "", e["species"]).lower()
            test = shape in TEST_SHAPES or e["species"] in TEST_SPECIES
            subjects[sid] = {"label": label, "category": "animal", "class": label, "origin": e["id"], "version": 2,
                             "shape": shape, "species": e["species"], "split": "test" if test else "train",
                             "representation": "photograph", "shape_tags": [shape],
                             "source_and_license": "see candidates/<ID>.json", "src": e["src"],
                             "path": f"candidates/{sid}.png"}
    pools = {}
    for sid, m in subjects.items():
        pools.setdefault(m["shape"], []).append(sid)
    animal_groups = sorted(g for g in pools if g != "human")
    if len(animal_groups) < SUBJECTS_PER_BG:
        sys.exit(f"only {len(animal_groups)} shape groups with subjects, need {SUBJECTS_PER_BG}")
    rng = random.Random(seed + 2)
    use_group, use_species, use_subject = Counter(), Counter(), Counter()
    pairs = {}
    for i, bid in enumerate(target_bids):
        human = i % HUMAN_EVERY == 0 and "human" in pools
        groups = sorted(animal_groups, key=lambda g: (use_group[g], rng.random()))[:SUBJECTS_PER_BG - human]
        for g in groups + (["human"] if human else []):
            # least used instance first (one-image species must not repeat), then the least used species
            sid = min(pools[g], key=lambda s: (use_subject[s], use_species[subjects[s]["species"]], rng.random()))
            use_group[g] += 1
            use_species[subjects[sid]["species"]] += 1
            use_subject[sid] += 1
            pairs[f"{bid}{sid}"] = [sid, bid]
    for k in targets:
        old = man["batches"][k]
        bids = list(dict.fromkeys(man["pairs"][p][1] for p in old["pairs"]))
        shutil.rmtree(os.path.join(SET, f"b{k + 1}"), ignore_errors=True)  # the folder write_batch recreates
        batch = write_batch(k + 1, f"Label set batch {k + 1}/{len(man['batches'])} (subjects v2, shape groups)", bids,
                            pairs, subjects, man["backgrounds"], seed)
        if not release:
            batch["hold"] = "subjects v2 built: check batch 1 renders, then `rebuild-subjects --release`"
        man["batches"][k] = batch
    man["subjects"] = subjects
    man["pairs"] = {**kept_pairs, **pairs}
    man["subject_split_rule"] = "a pair is test when its background or its subject has split test"
    with open(os.path.join(SET, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=1)
    return {"pairs": pairs, "subjects": subjects, "groups": use_group, "species": use_species,
            "batches": [man["batches"][k]["run"] for k in targets]}


def repick_dtd(exclude: list) -> list:
    """Held batches only: each DTD background becomes the most homogeneous image of its category (camo.sources
    .homogeneity, the bgmap_v1 rule) that bgmap did not use. The random picks of `build` included objects, animals
    and scenes (a shield bug for "striped", a cat for "smeared", a handbag, a pumpkin). Id, group, split stay.
    `exclude`: file names rejected by eye, skipped on a rerun."""
    from camo.sources import homogeneity

    man = load_manifest()
    held = {man["pairs"][p][1] for b in man["batches"] if b.get("hold") and not downloaded(b["run"]) for p in b["pairs"]}
    skip = used_by_bgmap() | set(exclude)
    by_cat = {}
    for name in sorted(os.listdir(BG_DIR)):
        if name.lower().endswith((".jpg", ".png")) and name not in skip:
            by_cat.setdefault(name.rsplit("_", 1)[0], []).append(name)
    ranked, changes = {}, []
    taken = {m["file"] for b, m in man["backgrounds"].items() if m.get("file") and b not in held}
    for bid in sorted(b for b, m in man["backgrounds"].items() if m["material"].startswith("dtd:") and b in held):
        meta = man["backgrounds"][bid]
        cat = meta["material"].split(":", 1)[1]
        if cat not in ranked:
            scored = []
            for name in by_cat.get(cat, []):
                with Image.open(os.path.join(BG_DIR, name)) as img:
                    scored.append((homogeneity(img), name))
            ranked[cat] = [n for _h, n in sorted(scored)]
        name = next((n for n in ranked[cat] if n not in taken), None)
        if name is None:
            print(f"{bid}: no image of {cat} left")
            continue
        taken.add(name)
        if name != meta["file"]:
            changes.append((bid, meta["file"], name))
        meta.update(file=name, path=f"candidates/{bid}{os.path.splitext(name)[1]}",
                    source_and_license=f"DTD r1.0.1/{cat}/{name} (DTD: research use)", picked="most homogeneous")
    with open(os.path.join(SET, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=1)
    return changes


def new_grid(rename: bool = True) -> dict:
    """Switch the label set to the current ACTION_SPACE: rewrite grid.yaml, rename every batch to the RUN_SUFFIX name
    and hold it. Rendered runs of the old grid keep their names (their labels stay valid) and get a RETIRED marker,
    which hides their unlabelled images from the label page."""
    man = load_manifest()
    retired = []
    for k, b in enumerate(man["batches"], 1):
        if not rename:  # same batches, new grid only (the rendered trial batch gets the missing configs added)
            if not downloaded(b["run"]):
                b["hold"] = b.get("hold") or f"grid v{GRID_VERSION}"
            continue
        if downloaded(b["run"]):
            open(os.path.join(OUT, b["run"], "RETIRED"), "w").write(f"grid v{man.get('grid_version')} batch; see manifest\n")
            retired.append(b["run"])
        b["run"] = run_name(k)
        b["hold"] = f"grid v{GRID_VERSION}: rebuild (repick-dtd, rebatch) and render the trial batch first"
    man["configs"] = list(grid_configs())
    man["grid_version"] = GRID_VERSION
    man.setdefault("retired_runs", []).extend(r for r in retired if r not in man.get("retired_runs", []))
    write_grid()
    with open(os.path.join(SET, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=1)
    return {"retired": retired, "configs": man["configs"]}


def rebatch(seed: int, trial: int = 0) -> dict:
    """Spread every batch over many backgrounds (user, 2026-10-13: batch 1 showed only 10): backgrounds are ordered
    round-robin over their groups and each batch takes the next 50 (background, slot) items, slot k = the k-th of a
    background's 5 pairs. A batch never holds two pairs of one background and every batch mixes all groups, DTD and
    natural. Same pairs, same batch names; every batch must be held and not rendered yet."""
    man = load_manifest()
    for b in man["batches"]:
        if not b.get("hold") or downloaded(b["run"]):
            sys.exit(f"{b['run']} is not a held, unrendered batch: set it aside first")
    per_batch = len(man["batches"][0]["pairs"])
    rng = random.Random(seed + 5)
    by_bg = {}
    for pid, (sid, bid) in sorted(man["pairs"].items()):
        by_bg.setdefault(bid, []).append(pid)
    groups = {}
    for bid in sorted(by_bg):
        groups.setdefault(man["backgrounds"][bid]["group"], []).append(bid)
    for g in groups.values():
        rng.shuffle(g)
    order = []  # round-robin over groups, so any window of the order covers every group
    while any(groups.values()):
        for g in sorted(groups, key=lambda g: -len(groups[g])):
            if groups[g]:
                order.append(groups[g].pop())
    for bid in order:
        rng.shuffle(by_bg[bid])
    items = [by_bg[bid][slot] for slot in range(max(len(v) for v in by_bg.values())) for bid in order
             if slot < len(by_bg[bid])]
    if per_batch > len(order):
        sys.exit(f"{per_batch} pairs per batch but only {len(order)} backgrounds")
    old = [b for b in man["batches"] if b["run"] != run_name(0)]  # a previous trial batch is folded back in
    new = []
    if trial:  # batch 0: a few pairs (distinct backgrounds, every group) to try the grid on before the rest
        chunk, items = items[:trial], items[trial:]
        bids = list(dict.fromkeys(man["pairs"][p][1] for p in chunk))
        shutil.rmtree(os.path.join(SET, "b0"), ignore_errors=True)
        new.append(write_batch(0, f"Label set trial batch ({len(chunk)} pairs, {len(bids)} backgrounds)", bids,
                               {p: man["pairs"][p] for p in chunk}, man["subjects"], man["backgrounds"],
                               man.get("seed", seed)))
        per_batch = -(-len(items) // len(old))
    for k in range(1, len(old) + 1):
        chunk = items[(k - 1) * per_batch:k * per_batch]
        bids = list(dict.fromkeys(man["pairs"][p][1] for p in chunk))
        shutil.rmtree(os.path.join(SET, f"b{k}"), ignore_errors=True)
        batch = write_batch(k, f"Label set batch {k}/{len(old)} ({len(bids)} backgrounds, one pair each)", bids,
                            {p: man["pairs"][p] for p in chunk}, man["subjects"], man["backgrounds"], man.get("seed", seed))
        batch["hold"] = old[k - 1]["hold"] if not trial else f"waiting for the trial batch {run_name(0)}"
        new.append(batch)
    man["batches"] = new
    man["batching"] = "rebatch: one pair per background per batch, groups round-robin"
    with open(os.path.join(SET, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=1)
    return {"batches": new, "order": order}


def report(run: str, rater: str = "tac_gia") -> dict:
    """How usable a labelled batch is: pairs with at least one good config (a board Model 2 can learn from), label
    counts per config, and the rule policy replayed on the labels. Near-identical renders inherit C00's label."""
    from collections import Counter

    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import replay_agents

    table, meta = replay_agents.grid_labels(run, rater)
    pairs = sorted(meta)
    per_config = {}
    for (pair, cid), (label, _src) in table.items():
        per_config.setdefault(cid, Counter())[label or "none"] += 1
    labelled = [p for p in pairs if any(table.get((p, c), (None,))[0] for c in [*grid_configs()])]
    any_good = [p for p in pairs if any(table.get((p, c), (None,))[0] == "good" for c in grid_configs())]
    rule = [replay_agents.replay(p, replay_agents.RulePolicy(), 5, table) for p in labelled]
    done = [r for r in rule if r["status"] == "done"]
    return {"run": run, "pairs": len(pairs), "labelled_pairs": len(labelled), "pairs_with_good": len(any_good),
            "per_config": {c: dict(per_config.get(c, {})) for c in grid_configs()},
            "rule": {"complete": len(done), "success": sum(r["selected"] is not None for r in done),
                     "mean_renders": round(sum(r["renders"] for r in done) / max(len(done), 1), 2)}}


def merge_run(src: str, into: str, check: str = "C00", max_diff: float = 3.0, force: bool = False) -> dict:
    """Move the renders of run `src` that run `into` lacks into it (an add-on render of extra configs for the same
    pairs), so one run holds each pair's whole grid and its labels. Same pair + same per-pair seed + same config must
    give the same image: `check` (rendered in both) must match within `max_diff`/255 or nothing moves."""
    import numpy as np

    from camo.pilot import rebuild_index

    s_runs, d_runs = os.path.join(OUT, src, "runs"), os.path.join(OUT, into, "runs")
    diffs, todo = {}, []
    for pair in sorted(os.listdir(s_runs)):
        if not os.path.isdir(os.path.join(d_runs, pair)):
            sys.exit(f"{pair} is not in {into}")
        for cid in sorted(os.listdir(os.path.join(s_runs, pair))):
            sdir, ddir = os.path.join(s_runs, pair, cid), os.path.join(d_runs, pair, cid)
            if not os.path.isdir(sdir):
                continue
            if not os.path.exists(ddir):
                todo.append((sdir, ddir))
            elif cid == check:
                a, b = (np.asarray(Image.open(os.path.join(d, "output.png")).convert("RGB"), np.float32) for d in (sdir, ddir))
                diffs[pair] = round(float(np.abs(a - b).mean()), 3)
    worst = max(diffs.values(), default=0.0)
    if worst > max_diff and not force:
        sys.exit(f"{check} differs between {src} and {into} by up to {worst}/255: seeds do not reproduce, not merging")
    for sdir, ddir in todo:
        shutil.move(sdir, ddir)
    rebuild_index(os.path.join(OUT, into))
    with open(os.path.join(OUT, src, "RETIRED"), "w", encoding="utf-8") as f:
        f.write(f"merged into {into}\n")
    return {"moved": len(todo), "check": check, "check_diff": diffs}


def build_sweep(name: str, batch: int, pairs: list) -> str:
    """pilot/labelset/<name>/: a few pairs of a built batch (inputs.yaml + their images), to try knobs on before
    rendering a new grid on every batch (`kaggle_run.py --frozen pilot/labelset/<name> --configs <grid>`)."""
    src = os.path.join(SET, f"b{batch}")
    with open(os.path.join(src, "inputs.yaml"), encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    unknown = [p for p in pairs if p not in doc["pairs"]]
    if unknown:
        sys.exit(f"not in b{batch}: {', '.join(unknown)}")
    keep = {p: doc["pairs"][p] for p in pairs}
    sids, bids = sorted({v[0] for v in keep.values()}), sorted({v[1] for v in keep.values()})
    folder = os.path.join(SET, name)
    shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(os.path.join(folder, "candidates"))
    for item, meta in [(i, doc["subjects"][i]) for i in sids] + [(i, doc["backgrounds"][i]) for i in bids]:
        shutil.copy2(os.path.join(src, meta["path"]), os.path.join(folder, meta["path"]))
        lic = os.path.join(src, "candidates", f"{item}.json")
        if os.path.exists(lic):
            shutil.copy2(lic, os.path.join(folder, "candidates", f"{item}.json"))
    out = {"version": name, "note": f"Knob sweep on {len(keep)} pairs of b{batch} (scripts/labelset.py sweep).",
           "subjects": {i: doc["subjects"][i] for i in sids}, "backgrounds": {i: doc["backgrounds"][i] for i in bids},
           "pairs": keep, "audit_pairs": list(keep)}
    with open(os.path.join(folder, "inputs.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(out, f, sort_keys=False, allow_unicode=True)
    return folder


def load_manifest() -> dict:
    path = os.path.join(SET, "manifest.json")
    if not os.path.exists(path):
        sys.exit("Run `python3 scripts/labelset.py build` first")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


BATCH_RETRIES = 2  # extra renders of one batch before the runner gives up


def check_batch(run: str, pairs: list, configs: list) -> str:
    """"" when the downloaded batch is the intended one: every pair rendered with exactly the grid's configs and at
    least 95% of the trials ok. Catches a kernel that ran stale inputs (2026-10-12: the v1 grid came back)."""
    want, ok = set(configs), 0
    for pair in pairs:
        pdir = os.path.join(OUT, run, "runs", pair)
        got = set(os.listdir(pdir)) if os.path.isdir(pdir) else set()
        got = {c for c in got if os.path.isdir(os.path.join(pdir, c))}
        if got != want:
            return f"{pair}: configs {sorted(got - want)[:3]} rendered, {sorted(want - got)[:3]} missing"
        for cid in want:
            trial = os.path.join(pdir, cid, "trial.json")
            if os.path.isfile(trial):
                with open(trial, encoding="utf-8") as f:
                    ok += json.load(f).get("status") == "ok"
    if ok < 0.95 * len(pairs) * len(want):
        return f"only {ok} of {len(pairs) * len(want)} renders ok"
    return ""


def downloaded(run: str) -> bool:
    return os.path.exists(os.path.join(OUT, run, "trials_latest.json"))


def write_status(**fields) -> None:
    os.makedirs(OUT, exist_ok=True)
    fields["updated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    tmp = STATUS + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(fields, f, ensure_ascii=False)
    os.replace(tmp, STATUS)


def kernel_running() -> bool:
    try:
        out = subprocess.run(["kaggle", "kernels", "status", f"{KAGGLE_USER}/camo-pilot-run"], capture_output=True,
                             text=True, timeout=120)
        text = (out.stdout + out.stderr).lower()
    except subprocess.TimeoutExpired:
        return True
    if "has status" not in text:  # network error (e.g. wifi change): assume it still runs, ask again next minute
        return True
    return any(s in text for s in ("running", "queued"))


def run(args) -> int:
    previous = {}
    if os.path.exists(STATUS):
        with open(STATUS, encoding="utf-8") as f:
            previous = json.load(f)
    uploaded = set(previous.get("uploaded", []))  # batch folders already inside the camo-code dataset on Kaggle
    while True:
        man = load_manifest()  # re-read every batch: batches appended meanwhile (build-nature) are picked up
        runs = [b["run"] for b in man["batches"]]
        pending = [(k, b) for k, b in enumerate(man["batches"], 1) if not downloaded(b["run"]) and not b.get("hold")]
        if not pending:
            break
        k, batch = pending[0]
        name = batch["run"]
        done = [r for r in runs if downloaded(r)]
        started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        cmd = [sys.executable, os.path.join(ROOT, "scripts", "kaggle_run.py"), "--run", name, "--user", KAGGLE_USER,
               "--configs", "pilot/labelset/grid.yaml", "--frozen", batch["folder"], "--step", "audit",
               "--pick", os.path.join(ROOT, "pilot", "pick_none.json"), "--slim"]
        # resumed while this batch was pushed (still rendering, or finished while no runner was watching): wait for it
        # and download, never re-push. "paused" counts too: the kernel keeps running on Kaggle while the runner is off.
        if previous.get("current") == name and previous.get("state") in ("rendering", "paused") and \
                (kernel_running() or previous.get("pushed")):
            cmd.append("--download-only")
            started = previous.get("started_utc", started)
            while kernel_running():
                write_status(state="rendering", current=name, batch=k, batches=len(runs), done=done, started_utc=started,
                             uploaded=sorted(uploaded))
                time.sleep(60)
        elif args.skip_upload or name in uploaded:
            cmd.append("--skip-upload")
        else:
            uploaded = set(runs)  # kaggle_run uploads the whole pilot/ folder: every batch built so far
        previous = {}
        write_status(state="rendering", current=name, batch=k, batches=len(runs), done=done, started_utc=started,
                     pairs=len(batch["pairs"]), images=len(batch["pairs"]) * len(man["configs"]), uploaded=sorted(uploaded),
                     pushed=True)
        print(f"== batch {k}/{len(runs)}: {name}", flush=True)
        pushes = 0
        while True:  # unattended overnight: a failed or wrong batch is retried, not the end of the run
            res = subprocess.run(cmd, cwd=ROOT)
            problem = "Kaggle run failed" if res.returncode != 0 or not downloaded(name) else \
                check_batch(name, batch["pairs"], man["configs"])
            if not problem:
                break
            if downloaded(name):  # keep the wrong renders for the record, out of the label page
                shutil.move(os.path.join(OUT, name), os.path.join(OUT, f"stale_{name}_{int(time.time())}"))
            if pushes >= BATCH_RETRIES:
                write_status(state="error", current=name, batch=k, batches=len(runs), done=done, started_utc=started,
                             uploaded=sorted(uploaded),
                             message=f"{problem}: https://www.kaggle.com/code/{KAGGLE_USER}/camo-pilot-run")
                return 1
            # first try only the download (the kernel may have finished fine), then render the batch again
            again = "--download-only" not in cmd and problem == "Kaggle run failed" and not kernel_running()
            cmd = [c for c in cmd if c != "--download-only"]
            if again:
                cmd.append("--download-only")
            else:
                pushes += 1
                if "--skip-upload" not in cmd:
                    cmd.append("--skip-upload")  # kaggle_run still uploads when Kaggle's copy is not current
            print(f"{name}: {problem}; {'downloading again' if again else f'rendering again ({pushes}/{BATCH_RETRIES})'}"
                  " in 5 min", flush=True)
            write_status(state="retrying", current=name, batch=k, batches=len(runs), done=done, started_utc=started,
                         uploaded=sorted(uploaded), message=f"{problem}: thử lại sau 5 phút")
            time.sleep(300)
    man = load_manifest()
    runs = [b["run"] for b in man["batches"]]
    held = [b["run"] for b in man["batches"] if b.get("hold")]
    if held:  # batches on hold (e.g. waiting for new subjects): stop here, the label page says so
        write_status(state="paused", batches=len(runs), done=[r for r in runs if downloaded(r)], uploaded=sorted(uploaded),
                     message=f"{len(held)} đợt đang tạm giữ (chờ dựng lại subject)")
        print(f"Held batches, not rendered: {', '.join(held)}")
        return 0
    write_status(state="done", batches=len(runs), done=runs, uploaded=sorted(uploaded))
    print("All batches downloaded.")
    return 0


def status() -> int:
    man = load_manifest()
    for b in man["batches"]:
        state = "downloaded" if downloaded(b["run"]) else ("HOLD: " + b["hold"] if b.get("hold") else "pending")
        print(f"{b['run']}: {len(b['pairs'])} pairs, {state}")
    if os.path.exists(STATUS):
        with open(STATUS, encoding="utf-8") as f:
            print(f.read())
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--seed", type=int, default=20261012)
    r = sub.add_parser("run")
    r.add_argument("--skip-upload", action="store_true", help="camo-code on Kaggle already holds pilot/labelset")
    n = sub.add_parser("build-nature", help="append natural-background batches (needs bg_nature_raw/)")
    n.add_argument("--seed", type=int, default=20261012)
    fd = sub.add_parser("fill-dtd", help="held batches: swap duplicate DTD categories for the missing ones")
    fd.add_argument("--seed", type=int, default=20261012)
    n2 = sub.add_parser("build-nature2", help="append a held batch of the new natural types")
    n2.add_argument("--seed", type=int, default=20261012)
    rb = sub.add_parser("rebuild-subjects", help="re-pair held batches with pilot/labelset_subject_picks.json")
    rb.add_argument("--seed", type=int, default=20261012)
    rb.add_argument("--release", action="store_true", help="also lift the hold so `run` renders them")
    rp = sub.add_parser("repick-dtd", help="held batches: DTD backgrounds -> most homogeneous image of the category")
    rp.add_argument("--exclude", nargs="*", default=[], help="DTD file names rejected by eye")
    mr = sub.add_parser("merge-run", help="move an add-on render of extra configs into the run of the same pairs")
    mr.add_argument("--src", required=True)
    mr.add_argument("--into", required=True)
    mr.add_argument("--force", action="store_true", help="merge even if the check config does not reproduce")
    rpt = sub.add_parser("report", help="after labelling: pairs with a good config, labels per config, rule replay")
    rpt.add_argument("--run", required=True)
    rpt.add_argument("--rater", default="tac_gia")
    ng = sub.add_parser("new-grid", help="switch to the current ACTION_SPACE: grid.yaml, batch names, hold everything")
    ng.add_argument("--keep-names", action="store_true", help="only rewrite grid.yaml/configs (no rename, no retire)")
    rbt = sub.add_parser("rebatch", help="spread every held batch over ~50 backgrounds, one pair each")
    rbt.add_argument("--seed", type=int, default=20261012)
    rbt.add_argument("--trial", type=int, default=0, help="first N pairs as an unheld trial batch 0")
    sw = sub.add_parser("sweep", help="copy a few pairs of a built batch into pilot/labelset/<name> for a knob sweep")
    sw.add_argument("--name", required=True)
    sw.add_argument("--batch", type=int, required=True)
    sw.add_argument("--pairs", nargs="+", required=True)
    sub.add_parser("status")
    args = ap.parse_args()
    if args.cmd == "build":
        man = build(args.seed)
        test = sum(1 for p in man["pairs"].values() if man["backgrounds"][p[1]]["split"] == "test")
        print(f"{len(man['pairs'])} pairs ({test} test) x {len(man['configs'])} configs in {len(man['batches'])} batches "
              f"-> {os.path.relpath(SET, ROOT)}")
        return 0
    if args.cmd == "build-nature":
        add = build_nature(args.seed)
        test = sum(1 for b in add["backgrounds"].values() if b["split"] == "test")
        print(f"+{len(add['backgrounds'])} natural backgrounds ({test} test), +{len(add['pairs'])} pairs in "
              f"{', '.join(b['run'] for b in add['batches'])}")
        return 0
    if args.cmd == "repick-dtd":
        changes = repick_dtd(args.exclude)
        for bid, was, now in changes:
            print(f"{bid}: {was} -> {now}")
        print(f"{len(changes)} backgrounds changed (rebuild-subjects / rebatch rewrites the batch folders)")
        return 0
    if args.cmd == "merge-run":
        res = merge_run(args.src, args.into, force=args.force)
        print(f"moved {res['moved']} renders into {args.into}; {res['check']} reproduces within "
              f"{max(res['check_diff'].values(), default=0)}/255 ({len(res['check_diff'])} pairs)")
        return 0
    if args.cmd == "report":
        res = report(args.run, args.rater)
        print(f"{res['run']}: {res['labelled_pairs']}/{res['pairs']} pairs labelled, "
              f"{res['pairs_with_good']} with at least one good config")
        for cid, counts in res["per_config"].items():
            print(f"  {cid:9s} " + "  ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])))
        r = res["rule"]
        print(f"rule policy (budget 5): {r['success']}/{r['complete']} complete searches end on good, "
              f"{r['mean_renders']} renders on average")
        return 0
    if args.cmd == "new-grid":
        res = new_grid(rename=not args.keep_names)
        print(f"grid v{GRID_VERSION}: {len(res['configs'])} configs; retired runs: {', '.join(res['retired']) or 'none'}")
        return 0
    if args.cmd == "rebatch":
        res = rebatch(args.seed, args.trial)
        man = load_manifest()
        for b in res["batches"]:
            bids = {man["pairs"][p][1] for p in b["pairs"]}
            groups = {man["backgrounds"][x]["group"] for x in bids}
            test = sum(1 for p in b["pairs"] if "test" in (man["subjects"][man["pairs"][p][0]].get("split"),
                                                            man["backgrounds"][man["pairs"][p][1]]["split"]))
            print(f"{b['run']}: {len(b['pairs'])} pairs, {len(bids)} backgrounds, {len(groups)} groups, {test} test")
        return 0
    if args.cmd == "sweep":
        print(os.path.relpath(build_sweep(args.name, args.batch, args.pairs), ROOT))
        return 0
    if args.cmd == "fill-dtd":
        for bid, was, now in fill_dtd(args.seed):
            print(f"{bid}: {was} -> {now}")
        return 0
    if args.cmd == "build-nature2":
        add = build_nature2(args.seed)
        test = sum(1 for b in add["backgrounds"].values() if b["split"] == "test")
        print(f"+{len(add['backgrounds'])} natural backgrounds ({test} test) in {add['batch']} (held)")
        return 0
    if args.cmd == "rebuild-subjects":
        res = rebuild_subjects(args.seed, args.release)
        subj = res["subjects"]
        test = sum(1 for sid, _ in res["pairs"].values() if subj[sid]["split"] == "test")
        print(f"{len(res['pairs'])} pairs in {', '.join(res['batches'])}; {test} with a test subject")
        print("pairs per shape group:", dict(sorted(res["groups"].items())))
        print("subjects used:", len({sid for sid, _ in res["pairs"].values()}), "of", len(subj))
        return 0
    return run(args) if args.cmd == "run" else status()


if __name__ == "__main__":
    sys.exit(main())
