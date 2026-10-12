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
from camo.agent import ACTION_SPACE, FIXED, _product, grid_id  # noqa: E402
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


def run_name(k: int) -> str:
    return f"label_b{k}"


def grid_configs() -> dict:
    configs = {"C00": {}}
    for values in _product([ACTION_SPACE[k] for k in ACTION_SPACE]):
        cfg = dict(zip(ACTION_SPACE, values))
        configs[grid_id(cfg)] = {**FIXED, **cfg}
    return configs


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
    with open(os.path.join(SET, "grid.yaml"), "w", encoding="utf-8") as f:
        f.write("# Label set grid: C00 + every ACTION_SPACE config (generated by scripts/labelset.py).\n")
        yaml.safe_dump({"configs": grid_configs()}, f, sort_keys=False)
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


def load_manifest() -> dict:
    path = os.path.join(SET, "manifest.json")
    if not os.path.exists(path):
        sys.exit("Run `python3 scripts/labelset.py build` first")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


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
        res = subprocess.run(cmd, cwd=ROOT)
        if res.returncode != 0 or not downloaded(name):
            write_status(state="error", current=name, batch=k, batches=len(runs), done=done, started_utc=started,
                         uploaded=sorted(uploaded),
                         message=f"Kaggle run failed: https://www.kaggle.com/code/{KAGGLE_USER}/camo-pilot-run")
            return 1
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
