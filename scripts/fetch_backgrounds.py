"""More background types for the label set (2026-10-12), on a Kaggle CPU kernel:

1. the 9 DTD categories missing from the local bg_dtd/ (the earlier download was incomplete; freckled stays out,
   it is human skin), all 120 images each, same <category>_NNNN.jpg names;
2. candidates for 5 new natural types (leaf litter, snow/ice, seabed/coral, rock, mud) from Open Images
   image-level labels (human-verified positives, images with people/animals/vehicles/text skipped). Each photo is
   cut to its most even square region (camo.sources.homogeneity over a few crops), the 20 most even per type are
   kept with provenance json, and sheet_<type>.png is for picking by eye.

    python3 scripts/fetch_backgrounds.py --push   # local: run on Kaggle, unpack outputs/backgrounds_v2/
    python3 fetch_backgrounds.py                  # on Kaggle (what --push runs)

Then: copy dtd/*.jpg into bg_dtd/, pick natural images into pilot/labelset_nature2_picks.json and run
`labelset.py fill-dtd` / `labelset.py build-nature2`.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
import tarfile

KAGGLE_USER = "dangkhoi3107"
KERNEL = f"{KAGGLE_USER}/camo-fetch-backgrounds"
RUN = "backgrounds_v2"
DTD_MISSING = ["banded", "braided", "dotted", "grid", "grooved", "lined", "studded", "waffled", "zigzagged"]
# new natural type -> Open Images image-level classes (any of them, human-verified positive)
NEW_TYPES = {
    "leaf_litter": ["Leaf", "Autumn", "Deciduous"],
    "snow": ["Snow", "Ice", "Frost"],
    "seabed": ["Coral reef", "Reef", "Underwater", "Seabed"],
    "rock": ["Rock", "Bedrock", "Boulder", "Outcrop", "Cliff"],
    "mud": ["Mud"],
}
# a positive label for any of these means the photo is about something else
EXCLUDE = ["Person", "Human", "Man", "Woman", "Boy", "Girl", "Child", "Human face", "Animal", "Mammal", "Bird", "Fish",
           "Dog", "Cat", "Insect", "Vehicle", "Car", "Land vehicle", "Building", "Text", "Skiing", "Snowboarding",
           "Boat", "Sky"]
LABEL_URLS = {
    "validation": ["https://storage.googleapis.com/openimages/v7/oidv7-val-annotations-human-imagelabels.csv",
                   "https://storage.googleapis.com/openimages/v5/validation-annotations-human-imagelabels.csv"],
    "test": ["https://storage.googleapis.com/openimages/v7/oidv7-test-annotations-human-imagelabels.csv",
             "https://storage.googleapis.com/openimages/v5/test-annotations-human-imagelabels.csv"],
}
DOWNLOAD, KEEP, MIN_SIDE = 70, 20, 448


def _labels(sources, split: str, cache: str) -> str:
    for url in LABEL_URLS[split]:
        try:
            return sources._download(url, os.path.join(cache, f"oi_{split}_imagelabels.csv"), deadline_s=1800)
        except Exception as exc:
            print(f"[bg] {url}: {exc}")
    raise SystemExit(f"no Open Images image labels for {split}")


def best_square(img, homogeneity):
    """(crop box, score) of the most even square: the full centre square or a 70% square on a 3x3 grid."""
    w, h = img.size
    side = min(w, h)
    boxes = [((w - side) // 2, (h - side) // 2, (w - side) // 2 + side, (h - side) // 2 + side)]
    small = int(side * 0.7)
    for fx in (0.0, 0.5, 1.0):
        for fy in (0.0, 0.5, 1.0):
            x0, y0 = int((w - small) * fx), int((h - small) * fy)
            boxes.append((x0, y0, x0 + small, y0 + small))
    scored = [(homogeneity(img.crop(b)), b) for b in boxes if b[2] - b[0] >= MIN_SIDE]
    return min(scored)[::-1] if scored else (None, None)


def _attach_code() -> None:
    import glob

    code = next((os.path.dirname(p) for p in sorted(glob.glob("/kaggle/input/**/camo/sources.py", recursive=True))), None)
    if not code:
        raise SystemExit("camo-code dataset not attached")
    sys.path.insert(0, os.path.dirname(code))


def fetch() -> int:
    _attach_code()
    from camo import sources
    from camo.imageio import labeled_grid, save_json
    from PIL import Image

    out, cache = f"/kaggle/working/{RUN}", "/kaggle/working/_cache"
    os.makedirs(os.path.join(out, "dtd"), exist_ok=True)
    counts = {}

    tar = sources._download(sources.DTD_URL, os.path.join(cache, "dtd.tar.gz"), deadline_s=2400)
    with tarfile.open(tar) as t:
        for m in t:
            parts = m.name.split("/")
            if m.isfile() and len(parts) >= 3 and parts[-2] in DTD_MISSING and parts[-1].endswith(".jpg"):
                with t.extractfile(m) as src, open(os.path.join(out, "dtd", parts[-1]), "wb") as dst:
                    shutil.copyfileobj(src, dst)
                counts[f"dtd:{parts[-2]}"] = counts.get(f"dtd:{parts[-2]}", 0) + 1
    os.remove(tar)

    with open(sources._download(sources.OI_FILES["classes"], os.path.join(cache, "oi_classes.csv")), encoding="utf-8") as f:
        mids = {row[1]: row[0] for row in csv.reader(f) if len(row) >= 2}
    want = {mids[c]: t for t, cs in NEW_TYPES.items() for c in cs if c in mids}
    bad = {mids[c] for c in EXCLUDE if c in mids}
    print("[bg] classes not in Open Images:", [c for cs in NEW_TYPES.values() for c in cs + EXCLUDE if c not in mids])
    found = {t: [] for t in NEW_TYPES}
    for split in ("validation", "test"):
        positives, excluded = {t: set() for t in NEW_TYPES}, set()
        with open(_labels(sources, split, cache), encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["Confidence"] != "1":
                    continue
                if r["LabelName"] in want:
                    positives[want[r["LabelName"]]].add(r["ImageID"])
                elif r["LabelName"] in bad:
                    excluded.add(r["ImageID"])
        with open(sources._download(sources.OI_FILES[split][2], os.path.join(cache, f"oi_{split}_meta.csv")),
                  encoding="utf-8") as f:
            meta = {r["ImageID"]: r for r in csv.DictReader(f)}
        for t, ids in positives.items():
            ids = sorted(ids - excluded)
            random.Random(f"20261012-{t}-{split}").shuffle(ids)
            print(f"[bg] {split} {t}: {len(ids)} clean positives")
            for image_id in ids[:max(0, DOWNLOAD - len(found[t]))]:
                url = sources.OI_IMAGE.format(split=split, image_id=image_id)
                try:
                    img = Image.open(sources._download(url, os.path.join(cache, "oi_images", f"{image_id}.jpg"))).convert("RGB")
                except Exception as exc:
                    print(f"[bg] {t}: skip {image_id} ({type(exc).__name__})")
                    continue
                box, score = best_square(img, sources.homogeneity)
                if box:
                    found[t].append((score, image_id, split, box, img.crop(box), meta.get(image_id, {})))
    for t, items in found.items():
        tdir = os.path.join(out, "nature", t)
        os.makedirs(tdir, exist_ok=True)
        tiles = []
        for k, (score, image_id, split, box, crop, info) in enumerate(sorted(items, key=lambda x: x[0])[:KEEP]):
            name = f"{t}_{k:02d}"
            crop.save(os.path.join(tdir, name + ".jpg"), quality=95)
            save_json(os.path.join(tdir, name + ".json"), {
                "source": "Open Images V7 (image-level label)", "type": t, "split": split, "image_id": image_id,
                "crop": list(box), "homogeneity": round(score, 4), "image_url": sources.OI_IMAGE.format(split=split, image_id=image_id),
                "original_url": info.get("OriginalURL"), "landing_url": info.get("OriginalLandingURL"),
                "author": info.get("Author"), "license": info.get("License") or "CC BY 2.0 (Open Images images)"})
            tiles.append((crop, f"{name} h={score:.2f}"))
        counts[f"nature:{t}"] = len(tiles)
        if tiles:
            labeled_grid(tiles, columns=5, tile=200).save(os.path.join(out, f"sheet_{t}.png"))
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"counts": counts, "dtd_missing": DTD_MISSING, "new_types": NEW_TYPES, "exclude": EXCLUDE}, f, indent=1)
    shutil.rmtree(cache, ignore_errors=True)
    shutil.make_archive(f"/kaggle/working/{RUN}", "zip", root_dir="/kaggle/working", base_dir=RUN)
    print(json.dumps(counts))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--push", action="store_true")
    a, _ = ap.parse_known_args()
    if a.push:  # local side only: the kernel itself is this single file
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from fetch_subjects import push_kernel

        with open(os.path.abspath(__file__), encoding="utf-8") as f:
            sys.exit(push_kernel(f.read(), KERNEL, RUN))
    sys.exit(fetch())
