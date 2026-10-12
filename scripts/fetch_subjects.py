"""More animal subjects for the label set (batches 2-8), organised by SILHOUETTE SHAPE rather than by species:
cat and dog share a shape, cat and lizard do not, and the agents must cover shapes. Open Images cutouts
(segmentation mask as alpha), a few representative species per shape group, several poses each.

    python3 scripts/fetch_subjects.py --push            # local: run on a Kaggle CPU kernel, download outputs/subjects_v2/
    python3 scripts/fetch_subjects.py --push --extra    # gap fill -> outputs/subjects_v2b/ (separate kernel)
    python3 fetch_subjects.py                           # on Kaggle (what --push runs)

Reuses camo.sources.fetch_animals (one clearly framed, untruncated instance per image, 12-75% of the frame) with a
longer species list. Species without an Open Images segmentation class are reported as gaps. Every cutout gets a
provenance json (licence, author, URL). Pick by eye from sheet_<species>.png before using them.

--extra fills the shape groups Open Images has no masks for: butterflies from the Leeds dataset (A23) and the
many_legs group from Open Images *boxes* (not occluded/truncated/group/depiction), cut out with SAM ViT-B prompted
by the box and kept only when the mask is confident and in one piece.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile

KAGGLE_USER = "dangkhoi3107"
KERNEL = f"{KAGGLE_USER}/camo-fetch-subjects"
PER_SPECIES = 8
# shape group -> Open Images class names (segmentation subset). Gaps are reported, the other species fill the group.
SHAPES = {
    "quadruped_compact": ["Cat", "Dog", "Fox", "Rabbit", "Squirrel"],      # four legs, short body, tail
    "quadruped_tall": ["Horse", "Deer", "Goat", "Giraffe"],                  # long legs, raised body
    "quadruped_heavy": ["Elephant", "Bear", "Sheep", "Cattle"],              # bulky mass, short legs
    "low_elongated_legged": ["Lizard", "Crocodile", "Otter"],                # long low body, splayed legs
    "legless_elongated": ["Snake", "Caterpillar"],                           # line or coil, no limbs
    "streamlined_fins": ["Fish", "Shark", "Dolphin", "Goldfish"],            # torpedo body, fins
    "wings_spread": ["Butterfly", "Moths and butterflies", "Bat (Animal)"],  # flat, symmetric wings
    "bird_upright": ["Owl", "Parrot", "Penguin", "Duck"],                    # perched or standing bird
    "round_shelled": ["Turtle", "Snail", "Frog", "Tortoise"],                # dome or blob, short limbs
    "many_legs": ["Spider", "Crab", "Insect", "Beetle"],                     # small body, radiating legs
}
SPECIES = [s for group in SHAPES.values() for s in group]
# --extra: species ids continue the A## numbering (Butterfly = A23, Spider = A34, ...); new many_legs species append.
KERNEL_MODE = "segmented"  # push() rewrites this line to "extra" for the gap-fill kernel
EXTRA_KERNEL = f"{KAGGLE_USER}/camo-fetch-subjects-extra"
MANY_LEGS = SHAPES["many_legs"] + ["Scorpion", "Centipede", "Ladybug", "Lobster"]
PER_SPECIES_EXTRA = 12
OI_BOXES = {"validation": "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
            "test": "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv"}
SAM_MODEL = "facebook/sam-vit-base"


def species_id(name: str) -> str:
    names = SPECIES + [s for s in MANY_LEGS if s not in SPECIES]
    return f"A{names.index(name) + 1:02d}"


def push(extra: bool = False) -> int:
    kernel, run = (EXTRA_KERNEL, "subjects_v2b") if extra else (KERNEL, "subjects_v2")
    with open(os.path.abspath(__file__), encoding="utf-8") as f:
        src = f.read()
    if extra:  # a script kernel takes no arguments: bake the mode into the uploaded copy
        assert 'KERNEL_MODE = "segmented"' in src
        src = src.replace('KERNEL_MODE = "segmented"', 'KERNEL_MODE = "extra"', 1)
    return push_kernel(src, kernel, run)


def push_kernel(src: str, kernel: str, run: str) -> int:
    """Run ``src`` as a private Kaggle CPU script kernel (camo-code attached, internet on), wait for it, then
    download and unpack outputs/<run>.zip. Shared by the fetch_* scripts."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    kdir = os.path.join(root, "dist", "kernel_" + kernel.split("/")[1].replace("-", "_"))
    code_file = kernel.split("/")[1].replace("camo-", "").replace("-", "_") + ".py"
    shutil.rmtree(kdir, ignore_errors=True)
    os.makedirs(kdir)
    with open(os.path.join(kdir, code_file), "w", encoding="utf-8") as f:
        f.write(src)
    with open(os.path.join(kdir, "kernel-metadata.json"), "w", encoding="utf-8") as f:
        json.dump({"id": kernel, "title": kernel.split("/")[1], "code_file": code_file, "language": "python",
                   "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_internet": True,
                   "dataset_sources": [f"{KAGGLE_USER}/camo-code"], "competition_sources": [], "kernel_sources": []}, f, indent=1)
    subprocess.run(["kaggle", "kernels", "push", "-p", kdir], check=True)
    print(f"Running on Kaggle CPU: https://www.kaggle.com/code/{kernel}", flush=True)
    while True:
        time.sleep(60)
        try:
            out = subprocess.run(["kaggle", "kernels", "status", kernel], capture_output=True, text=True, timeout=120)
            status = (out.stdout + out.stderr).lower()
        except subprocess.TimeoutExpired:
            continue
        print(time.strftime("%H:%M"), status.strip().splitlines()[-1] if status.strip() else "?", flush=True)
        if "kernelworkerstatus.complete" in status or "kernelworkerstatus.error" in status or "kernelworkerstatus.cancel" in status:
            break
    dest = os.path.join(root, "outputs")
    zpath = os.path.join(dest, f"{run}.zip")
    for _ in range(10):
        subprocess.run(["kaggle", "kernels", "output", kernel, "-p", dest, "--file-pattern", rf"^{run}\.zip$"], check=False)
        if zipfile.is_zipfile(zpath):
            break
        if os.path.exists(zpath):  # a dropped connection leaves a truncated (or empty) zip
            os.remove(zpath)
        time.sleep(60)
    if not zipfile.is_zipfile(zpath):
        sys.exit(f"{run}.zip not downloaded: see https://www.kaggle.com/code/{kernel}")
    shutil.unpack_archive(zpath, dest)
    with open(os.path.join(dest, run, "summary.json"), encoding="utf-8") as f:
        print(json.dumps(json.load(f)["counts"], indent=1))
    return 0


def _attach_code() -> None:
    code = next((os.path.dirname(p) for p in sorted(glob.glob("/kaggle/input/**/camo/sources.py", recursive=True))), None)
    if not code:
        raise SystemExit("camo-code dataset not attached")
    sys.path.insert(0, os.path.dirname(code))


def _sheet(out: str, item_id: str, name: str, paths: list) -> None:
    from camo.imageio import labeled_grid
    from PIL import Image

    tiles = []
    for p in paths:
        img = Image.open(p).convert("RGBA")
        bg = Image.new("RGBA", img.size, (128, 128, 128, 255))
        bg.alpha_composite(img)
        tiles.append((bg.convert("RGB"), os.path.basename(p)))
    labeled_grid(tiles, columns=5, tile=180).save(os.path.join(out, f"sheet_{item_id}_{name}.png"))


def fetch() -> int:
    _attach_code()
    from camo import sources

    out = "/kaggle/working/subjects_v2"
    cand, cache = os.path.join(out, "candidates"), "/kaggle/working/_cache"
    os.makedirs(cand, exist_ok=True)
    sources.ANIMALS = {f"A{i + 1:02d}": name for i, name in enumerate(SPECIES)}
    found = sources.fetch_animals(cand, cache, PER_SPECIES, seed=20261012)
    counts = {}
    for item_id, paths in found.items():
        name = sources.ANIMALS[item_id]
        counts[name] = len(paths)
        if paths:
            _sheet(out, item_id, name, paths)
    missing = [s for s in SPECIES if s not in counts]
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        by_shape = {g: {s: counts.get(s, 0) for s in sp} for g, sp in SHAPES.items()}
        json.dump({"species": sources.ANIMALS, "shapes": SHAPES, "counts": counts, "by_shape": by_shape,
                   "no_segmentation_class": missing, "per_species": PER_SPECIES}, f, indent=1)
    shutil.rmtree(cache, ignore_errors=True)
    shutil.make_archive("/kaggle/working/subjects_v2", "zip", root_dir="/kaggle/working", base_dir="subjects_v2")
    print(json.dumps(counts), "missing:", missing)
    return 0


def _box_candidates(rows: list, mid: str, rng) -> list:
    """Open Images boxes of one clearly visible instance: not occluded/truncated/group/depiction/inside, alone in its
    image, 8-75% of the frame (bugs are small), away from the borders."""
    from collections import Counter

    per_image = Counter(r["ImageID"] for r in rows if r["LabelName"] == mid)
    good = []
    for r in rows:
        if r["LabelName"] != mid or per_image[r["ImageID"]] != 1:
            continue
        if any(r.get(k) == "1" for k in ("IsOccluded", "IsTruncated", "IsGroupOf", "IsDepiction", "IsInside")):
            continue
        x0, x1, y0, y1 = (float(r[k]) for k in ("XMin", "XMax", "YMin", "YMax"))
        if 0.08 <= (x1 - x0) * (y1 - y0) <= 0.75 and min(x0, y0) > 0.02 and max(x1, y1) < 0.98:
            good.append(r)
    rng.shuffle(good)
    return good


def _sam_mask(sam, proc, rgb, box):
    """Best of SAM's 3 box-prompted masks -> (mask HxW bool, iou score, share of the mask in its largest piece)."""
    import numpy as np
    import torch
    from scipy import ndimage

    inputs = proc(rgb, input_boxes=[[list(box)]], return_tensors="pt")
    with torch.no_grad():
        res = sam(**inputs)
    masks = proc.image_processor.post_process_masks(res.pred_masks, inputs["original_sizes"],
                                                    inputs["reshaped_input_sizes"])[0][0]
    scores = res.iou_scores[0, 0]
    best = int(scores.argmax())
    mask = masks[best].numpy().astype(bool)
    x0, y0, x1, y1 = box  # nothing outside the box belongs to the animal
    keep = np.zeros_like(mask)
    keep[y0:y1, x0:x1] = True
    mask &= keep
    lab, n = ndimage.label(mask)
    largest = np.bincount(lab.ravel())[1:].max() / mask.sum() if n else 0.0
    return mask, float(scores[best]), float(largest)


def fetch_extra() -> int:
    """Gap fill on Kaggle: Leeds butterflies (A23) + many_legs species cut out from Open Images boxes with SAM."""
    import csv
    import random

    import numpy as np
    import torch
    from PIL import Image
    from transformers import SamModel, SamProcessor

    _attach_code()
    from camo import sources
    from camo.imageio import save_json

    out = "/kaggle/working/subjects_v2b"
    cand, cache = os.path.join(out, "candidates"), "/kaggle/working/_cache"
    os.makedirs(cand, exist_ok=True)
    found, rejected = {}, {}

    tmp = os.path.join(cache, "leeds")
    os.makedirs(tmp, exist_ok=True)
    bid = species_id("Butterfly")
    found["Butterfly"] = []
    try:  # Zenodo sometimes stalls: lose the butterflies, keep the SAM part
        leeds = sources.fetch_butterflies(tmp, cache, PER_SPECIES_EXTRA, seed=20261012)["O03"]
    except Exception as exc:
        print(f"[extra] Leeds butterflies failed: {type(exc).__name__}: {exc}")
        leeds = []
    for k, p in enumerate(leeds):
        dst = os.path.join(cand, f"{bid}_{k}.png")
        shutil.move(p, dst)
        shutil.move(os.path.splitext(p)[0] + ".json", os.path.splitext(dst)[0] + ".json")
        found["Butterfly"].append(dst)

    with open(sources._download(sources.OI_FILES["classes"], os.path.join(cache, "oi_classes.csv")), encoding="utf-8") as f:
        mids = {row[1]: row[0] for row in csv.reader(f) if len(row) >= 2}
    torch.set_num_threads(os.cpu_count() or 4)
    proc, sam = SamProcessor.from_pretrained(SAM_MODEL), SamModel.from_pretrained(SAM_MODEL).eval()
    for split, url in OI_BOXES.items():
        with open(sources._download(url, os.path.join(cache, f"oi_{split}_bbox.csv")), encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        with open(sources._download(sources.OI_FILES[split][2], os.path.join(cache, f"oi_{split}_meta.csv")),
                  encoding="utf-8") as f:
            meta = {r["ImageID"]: r for r in csv.DictReader(f)}
        for name in MANY_LEGS:
            got, item_id = found.setdefault(name, []), species_id(name)
            if name not in mids:
                print(f"[extra] no Open Images class {name!r}")
                continue
            for r in _box_candidates(rows, mids[name], random.Random(f"20261012-{name}-{split}")):
                if len(got) >= PER_SPECIES_EXTRA:
                    break
                image_id = r["ImageID"]
                try:
                    url_img = sources.OI_IMAGE.format(split=split, image_id=image_id)
                    rgb = Image.open(sources._download(url_img, os.path.join(cache, "oi_images", f"{image_id}.jpg"))).convert("RGB")
                except Exception as exc:
                    print(f"[extra] {name}: skip {image_id} ({type(exc).__name__}: {exc})")
                    continue
                w, h = rgb.size
                box = (int(float(r["XMin"]) * w), int(float(r["YMin"]) * h), int(float(r["XMax"]) * w), int(float(r["YMax"]) * h))
                mask, iou, largest = _sam_mask(sam, proc, rgb, box)
                fill = mask.sum() / max(1, (box[2] - box[0]) * (box[3] - box[1]))
                # thin legs give a low fill; below 10% SAM mostly kept a fragment, above 95% it kept the whole box
                if iou < 0.85 or largest < 0.9 or not 0.10 <= fill <= 0.95:
                    rejected[name] = rejected.get(name, 0) + 1
                    print(f"[extra] {name}: reject {image_id} (iou {iou:.2f}, piece {largest:.2f}, fill {fill:.2f})")
                    continue
                dst = os.path.join(cand, f"{item_id}_{len(got)}.png")
                sources._rgba_cutout(rgb, Image.fromarray((mask * 255).astype(np.uint8)), box).save(dst)
                info = meta.get(image_id, {})
                save_json(os.path.splitext(dst)[0] + ".json", {
                    "source": "Open Images V7 (box)", "split": split, "image_id": image_id, "class": name,
                    "mask": f"SAM {SAM_MODEL}, box prompt", "sam_iou": round(iou, 3), "box_fill": round(float(fill), 3),
                    "image_url": url_img, "original_url": info.get("OriginalURL"), "landing_url": info.get("OriginalLandingURL"),
                    "author": info.get("Author"), "license": info.get("License") or "CC BY 2.0 (Open Images images)",
                    "annotation_license": "CC BY 4.0 (Open Images annotations)", "rotation": info.get("Rotation"),
                    "note": "cutout = SAM mask as alpha, cropped to box + 8% margin; check EXIF rotation if misaligned"})
                got.append(dst)
                print(f"[extra] {name} <- Open Images {image_id} (iou {iou:.2f}, fill {fill:.2f})", flush=True)
    counts = {name: len(paths) for name, paths in found.items()}
    for name, paths in found.items():
        if paths:
            _sheet(out, species_id(name), name, paths)
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"species": {species_id(n): n for n in found}, "shapes": {"wings_spread": ["Butterfly"], "many_legs": MANY_LEGS},
                   "counts": counts, "sam_rejected": rejected, "per_species": PER_SPECIES_EXTRA}, f, indent=1)
    shutil.rmtree(cache, ignore_errors=True)
    shutil.make_archive("/kaggle/working/subjects_v2b", "zip", root_dir="/kaggle/working", base_dir="subjects_v2b")
    print(json.dumps(counts), "rejected:", rejected)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--extra", action="store_true", help="gap fill: Leeds butterflies + many_legs cut out with SAM")
    a, _ = ap.parse_known_args()
    if a.push:
        sys.exit(push(a.extra))
    sys.exit(fetch_extra() if a.extra or KERNEL_MODE == "extra" else fetch())
