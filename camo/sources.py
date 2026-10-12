"""Pilot inputs from public datasets (no generated images).

- Animals O01/O02/O04 (cat, lizard, fish): Open Images V7 instance masks (images CC BY 2.0, author/URL kept) -> RGBA.
- Butterfly O03: Leeds Butterfly Dataset (CC BY 4.0, segmentation masks; Open Images has no butterfly masks).
- Portraits O05-O06: AISegment Matting Human (Kaggle dataset, alpha mattes) -> RGBA cutouts, random picks.
- Backgrounds B01-B06: DTD (Describable Textures Dataset), one attribute category per ID.

Each ID gets ``candidates`` images in ``<target>/candidates/<ID>_<k>.png`` with a provenance json, a contact sheet
``candidates.png``, and an ``inputs.yaml`` pointing at candidate 0. Swap a pick by editing its ``path``.
"""
from __future__ import annotations

import csv
import glob
import io
import os
import random
import tarfile
import time
import urllib.request
import zipfile
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from PIL import Image

from .imageio import labeled_grid, save_json

OI = "https://storage.googleapis.com/openimages"
OI_FILES = {
    "classes": f"{OI}/v7/oidv7-class-descriptions.csv",
    "seg_classes": f"{OI}/v7/oidv7-classes-segmentation.txt",
    "validation": (f"{OI}/v5/validation-annotations-object-segmentation.csv",
                   f"{OI}/v5/validation-masks/validation-masks-{{}}.zip",
                   f"{OI}/2018_04/validation/validation-images-with-rotation.csv"),
    "test": (f"{OI}/v5/test-annotations-object-segmentation.csv",
             f"{OI}/v5/test-masks/test-masks-{{}}.zip",
             f"{OI}/2018_04/test/test-images-with-rotation.csv"),
}
OI_IMAGE = "https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg"
ANIMALS = {"O01": "Cat", "O02": "Lizard", "O04": "Fish"}
LEEDS_URL = "https://zenodo.org/records/7559420/files/leedsbutterfly_dataset_v1.1.zip"
DTD_URL = "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"
# DTD has attribute categories, not materials ("a|b" = use b when a is absent locally): closest stand-ins for the B01-B06 questions in INPUT_SELECTION_PLAN §3.1.
DTD_MAP = {
    "B01": ("marbled|stained", "low", "smooth low-contrast surface (wall/stone-like, sticker risk)"),
    "B02": ("porous|flecked", "medium", "fine granular holes/specks (sand-like)"),
    "B03": ("fibrous|matted", "high", "dense overlapping strands (straw/yarn, foliage-like clutter)"),
    "B04": ("stratified|grooved", "medium", "directional layers (rock strata / wood-like)"),
    "B05": ("cracked|bumpy", "high", "relief with shading (3D cue, still a photograph)"),
    "B06": ("zigzagged|chequered", "medium", "regular 2D repeating pattern"),
}


# All 47 DTD categories grouped by the property that plausibly matters for blending (planning tags, not measured).
DTD_GROUPS = {
    "regular_pattern_2d": ["chequered", "zigzagged", "striped", "polka-dotted", "grid", "honeycombed", "dotted",
                           "waffled", "studded", "paisley", "crosshatched", "lined", "banded"],
    "granular": ["porous", "pitted", "flecked", "sprinkled", "freckled", "perforated"],
    "relief_shading": ["bumpy", "cracked", "potholed", "crystalline", "bubbly", "scaly"],
    "fibrous_overlapping": ["fibrous", "matted", "cobwebbed", "woven", "knitted", "braided", "interlaced", "meshed",
                            "lacelike", "gauzy", "frilly"],
    "directional_flow": ["grooved", "stratified", "wrinkled", "pleated", "swirly", "spiralled", "veined"],
    "low_detail_blotchy": ["blotchy", "stained", "smeared", "marbled"],
}
# Categories whose images often contain animals/people themselves: flag, do not mix into "clean" results.
DTD_CONTAINS_ANIMAL = {"scaly", "cobwebbed", "freckled"}  # snake/fish skin, spiders on webs, human skin


def homogeneity(img: Image.Image) -> float:
    """Lower = texture spread evenly over the frame (no single dominant object). Std over a 4x4 tile grid of the
    tile mean luminance and the tile gradient energy, each relative to the image average."""
    g = np.asarray(img.convert("L").resize((256, 256)), dtype=np.float32)
    gy, gx = np.gradient(g)
    energy = np.hypot(gx, gy)
    tiles = [(g[i:i + 64, j:j + 64].mean(), energy[i:i + 64, j:j + 64].mean()) for i in range(0, 256, 64) for j in range(0, 256, 64)]
    lum, en = np.array(tiles).T
    return float(lum.std() / (lum.mean() + 1e-3) + en.std() / (en.mean() + 1e-3))


def build_bg_catalog(images_dir: str, out_dir: str, per_category: int = 1) -> List[Dict[str, Any]]:
    """Pick the most homogeneous DTD image(s) of every category -> <out_dir>/BG_<category>_<k>.png + metadata."""
    os.makedirs(out_dir, exist_ok=True)
    group_of = {c: g for g, cats in DTD_GROUPS.items() for c in cats}
    catalog = []
    for category in sorted(group_of):
        files = _dtd_files(images_dir, category)
        scored = sorted((homogeneity(Image.open(f)), f) for f in files)
        for k, (score, src) in enumerate(scored[:per_category]):
            name = f"BG_{category.replace('-', '')}_{k}.png"
            Image.open(src).convert("RGB").save(os.path.join(out_dir, name))
            catalog.append({"file": name, "category": category, "group": group_of[category], "homogeneity": round(score, 4),
                            "source": f"DTD r1.0.1/{os.path.relpath(src, images_dir)}", "license": "DTD: research use",
                            "contains_animal_risk": category in DTD_CONTAINS_ANIMAL, "of": len(files)})
    return catalog


def _download(url: str, path: str, attempts: int = 3, deadline_s: float = 900.0) -> str:
    """Retrying download; a server that keeps the socket open but stalls (seen with Zenodo) hits the deadline."""
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for attempt in range(1, attempts + 1):
        print(f"[fetch] {url}" + (f" (attempt {attempt})" if attempt > 1 else ""), flush=True)
        started, done = time.time(), 0
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "camo-pilot/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r, open(path + ".part", "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    if time.time() - started > deadline_s:
                        raise TimeoutError(f"{done >> 20} MB after {deadline_s:.0f}s")
            os.replace(path + ".part", path)
            return path
        except Exception as exc:
            print(f"[fetch] failed: {type(exc).__name__}: {exc}", flush=True)
    raise RuntimeError(f"download failed after {attempts} attempts: {url}")


def _shallow_dirs(top: str = "/kaggle/input", max_depth: int = 6) -> List[str]:
    """Directories under ``top`` without descending into huge image folders (a full ** glob over
    AISegment's ~68k files on Kaggle's network mount takes minutes)."""
    out: List[str] = []
    for root, dirs, _files in os.walk(top):
        depth = root[len(top):].count(os.sep)
        out.append(root)
        if depth >= max_depth or os.path.basename(root) in ("clip_img", "matting", "images", "candidates"):
            dirs[:] = []
        else:
            dirs[:] = [d for d in dirs if not d.startswith(".")]
    return out


def _rgba_cutout(rgb: Image.Image, mask: Image.Image, box: Tuple[int, int, int, int], margin: float = 0.08) -> Image.Image:
    w, h = rgb.size
    x0, y0, x1, y1 = box
    mx, my = int((x1 - x0) * margin), int((y1 - y0) * margin)
    crop = (max(0, x0 - mx), max(0, y0 - my), min(w, x1 + mx), min(h, y1 + my))
    rgba = rgb.convert("RGB").copy()
    rgba.putalpha(mask.convert("L").resize(rgb.size, Image.BILINEAR))
    return rgba.crop(crop)


# ------------------------------------------------------------------
# Open Images animals
# ------------------------------------------------------------------
def _oi_mids(cache: str) -> Dict[str, str]:
    with open(_download(OI_FILES["classes"], os.path.join(cache, "oi_classes.csv")), encoding="utf-8") as f:
        by_name = {row[1]: row[0] for row in csv.reader(f) if len(row) >= 2}
    with open(_download(OI_FILES["seg_classes"], os.path.join(cache, "oi_seg_classes.txt")), encoding="utf-8") as f:
        seg = {line.strip() for line in f if line.strip()}
    mids = {}
    for item_id, name in ANIMALS.items():
        mid = by_name.get(name)
        if mid in seg:
            mids[item_id] = mid
        else:
            print(f"[fetch] {item_id}: Open Images has no segmentation class {name!r} -> coverage gap")
    return mids


def _oi_candidates(rows: List[Dict[str, str]], mid: str, rng: random.Random) -> List[Dict[str, str]]:
    """One clearly framed instance: medium size, box away from the borders (not truncated), alone in its image."""
    per_image = defaultdict(int)
    for row in rows:
        if row["LabelName"] == mid:
            per_image[row["ImageID"]] += 1
    good = []
    for row in rows:
        if row["LabelName"] != mid or per_image[row["ImageID"]] != 1:
            continue
        x0, x1, y0, y1 = (float(row[k]) for k in ("BoxXMin", "BoxXMax", "BoxYMin", "BoxYMax"))
        area = (x1 - x0) * (y1 - y0)
        if 0.12 <= area <= 0.75 and min(x0, y0) > 0.02 and max(x1, y1) < 0.98:
            good.append(row)
    rng.shuffle(good)
    return good


def fetch_animals(cand_dir: str, cache: str, candidates: int, seed: int) -> Dict[str, List[str]]:
    mids = _oi_mids(cache)
    found: Dict[str, List[str]] = {i: [] for i in mids}
    for split in ("validation", "test"):
        if all(len(v) >= candidates for v in found.values()):
            break
        ann_url, mask_url, meta_url = OI_FILES[split]
        with open(_download(ann_url, os.path.join(cache, f"oi_{split}_seg.csv")), encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        with open(_download(meta_url, os.path.join(cache, f"oi_{split}_meta.csv")), encoding="utf-8") as f:
            meta = {r["ImageID"]: r for r in csv.DictReader(f)}
        for item_id, mid in mids.items():
            rng = random.Random(f"{seed}-{item_id}-{split}")
            for row in _oi_candidates(rows, mid, rng):
                k = len(found[item_id])
                if k >= candidates:
                    break
                image_id = row["ImageID"]
                try:
                    zpath = _download(mask_url.format(image_id[0].upper()), os.path.join(cache, f"oi_{split}_masks_{image_id[0].upper()}.zip"))
                    with zipfile.ZipFile(zpath) as z:
                        mask = Image.open(io.BytesIO(z.read(row["MaskPath"]))).convert("L")
                    info = meta.get(image_id, {})
                    url = OI_IMAGE.format(split=split, image_id=image_id)
                    raw = _download(url, os.path.join(cache, "oi_images", f"{image_id}.jpg"))
                    rgb = Image.open(raw).convert("RGB")
                except Exception as exc:
                    print(f"[fetch] {item_id}: skip {image_id} ({type(exc).__name__}: {exc})")
                    continue
                w, h = rgb.size
                box = (int(float(row["BoxXMin"]) * w), int(float(row["BoxYMin"]) * h),
                       int(float(row["BoxXMax"]) * w), int(float(row["BoxYMax"]) * h))
                out = os.path.join(cand_dir, f"{item_id}_{k}.png")
                _rgba_cutout(rgb, mask, box).save(out)
                save_json(os.path.splitext(out)[0] + ".json", {
                    "source": "Open Images V7", "split": split, "image_id": image_id, "mask_path": row["MaskPath"],
                    "class": ANIMALS[item_id], "image_url": url, "original_url": info.get("OriginalURL"),
                    "landing_url": info.get("OriginalLandingURL"), "author": info.get("Author"),
                    "license": info.get("License") or "CC BY 2.0 (Open Images images)",
                    "annotation_license": "CC BY 4.0 (Open Images annotations)", "rotation": info.get("Rotation"),
                    "note": "cutout = mask as alpha, cropped to box + 8% margin; check EXIF rotation if the cutout looks misaligned",
                })
                found[item_id].append(out)
                print(f"[fetch] {item_id} <- Open Images {image_id}")
    return found


# ------------------------------------------------------------------
# Leeds butterflies
# ------------------------------------------------------------------
def fetch_butterflies(cand_dir: str, cache: str, candidates: int, seed: int, local: Optional[str] = None) -> Dict[str, List[str]]:
    zpath = local or _download(LEEDS_URL, os.path.join(cache, "leedsbutterfly_dataset_v1.1.zip"))
    found: List[str] = []
    with zipfile.ZipFile(zpath) as z:
        masks = sorted(n for n in z.namelist() if n.endswith("_seg0.png"))
        random.Random(f"{seed}-O03").shuffle(masks)
        for name in masks:
            if len(found) >= candidates:
                break
            stem = os.path.basename(name)[: -len("_seg0.png")]
            labels = np.asarray(Image.open(io.BytesIO(z.read(name))))
            mask = Image.fromarray(((labels == 1) * 255).astype(np.uint8))  # 2/3 only mark the image border
            box = mask.getbbox()
            w, h = mask.size
            if not box or min(w, h) < 380:  # skip tiny source images
                continue
            area = (box[2] - box[0]) * (box[3] - box[1]) / (w * h)
            if not (0.12 <= area <= 0.75 and box[0] > 0.02 * w and box[1] > 0.02 * h and box[2] < 0.98 * w and box[3] < 0.98 * h):
                continue
            rgb = Image.open(io.BytesIO(z.read(f"leedsbutterfly/images/{stem}.png"))).convert("RGB")
            out = os.path.join(cand_dir, f"O03_{len(found)}.png")
            _rgba_cutout(rgb, mask, box).save(out)
            save_json(os.path.splitext(out)[0] + ".json", {
                "source": "Leeds Butterfly Dataset v1.1", "image": f"images/{stem}.png", "mask": name,
                "species_category": stem[:3], "license": "CC BY 4.0", "url": LEEDS_URL})
            found.append(out)
            print(f"[fetch] O03 <- Leeds {stem}")
    return {"O03": found}


# ------------------------------------------------------------------
# AISegment portraits
# ------------------------------------------------------------------
def _find_aisegment(root: Optional[str]) -> Optional[str]:
    bases = [root] if root else [d for d in _shallow_dirs() if os.path.isdir(os.path.join(d, "matting"))]
    for base in bases:
        if os.path.isdir(os.path.join(base, "matting")) and os.path.isdir(os.path.join(base, "clip_img")):
            return base
    return None


def fetch_portraits(cand_dir: str, root: Optional[str], candidates: int, seed: int) -> Dict[str, List[str]]:
    base = _find_aisegment(root)
    if not base:
        print("[fetch] O05/O06: AISegment not found (Kaggle: add dataset 'aisegmentcom-matting-human-datasets' or pass "
              "--portraits <dir with clip_img/ and matting/>) -> coverage gap")
        return {}
    mattes = sorted(glob.glob(os.path.join(base, "matting", "*", "*", "*.png")))
    rng = random.Random(f"{seed}-portraits")
    rng.shuffle(mattes)
    found: Dict[str, List[str]] = {"O05": [], "O06": []}
    order = [i for _ in range(candidates) for i in ("O05", "O06")]
    for matte_path in mattes:
        if not order:
            break
        group, sub, name = matte_path.split(os.sep)[-3:]
        photo = os.path.join(base, "clip_img", group, sub.replace("matting_", "clip_"), os.path.splitext(name)[0] + ".jpg")
        if not os.path.exists(photo):
            continue
        matte = Image.open(matte_path)
        alpha = matte.getchannel("A") if matte.mode == "RGBA" else matte.convert("L")
        if np.asarray(alpha).mean() < 20:  # empty/near-empty matte
            continue
        item_id = order.pop(0)
        k = len(found[item_id])
        rgb = Image.open(photo).convert("RGB")
        out = os.path.join(cand_dir, f"{item_id}_{k}.png")
        bbox = alpha.point(lambda v: 255 if v > 128 else 0).getbbox() or (0, 0, *rgb.size)
        _rgba_cutout(rgb, alpha, bbox, margin=0.04).save(out)
        save_json(os.path.splitext(out)[0] + ".json", {
            "source": "AISegment Matting Human (Kaggle)", "photo": os.path.relpath(photo, base), "matte": os.path.relpath(matte_path, base),
            "license": "unclear (web-collected photos of real people): pilot testing only, do not publish",
            "pose": "unknown (random pick): check frontal vs three-quarter",
        })
        found[item_id].append(out)
        print(f"[fetch] {item_id} <- AISegment {os.path.relpath(photo, base)}")
    return found


# ------------------------------------------------------------------
# DTD backgrounds
# ------------------------------------------------------------------
def _dtd_files(folder: str, category: str) -> List[str]:
    """Official layout images/<category>/*.jpg, or a flat folder of <category>_NNNN.jpg."""
    nested = glob.glob(os.path.join(folder, category, "*.jpg"))
    return sorted(nested or glob.glob(os.path.join(folder, f"{category}_*.jpg")))


def _find_dtd(root: Optional[str], cache: str) -> str:
    if root:
        return root
    for hit in _shallow_dirs():
        if _dtd_files(hit, "chequered"):
            return hit
    local = os.path.join(cache, "dtd", "images")
    if not os.path.isdir(local):
        tar = _download(DTD_URL, os.path.join(cache, "dtd-r1.0.1.tar.gz"))
        print("[fetch] extracting DTD ...")
        with tarfile.open(tar) as t:
            t.extractall(cache)
    return local


def fetch_backgrounds(cand_dir: str, cache: str, root: Optional[str], candidates: int, seed: int,
                      mapping: Dict[str, Tuple[str, str, str]]) -> Dict[str, List[str]]:
    images = _find_dtd(root, cache)
    found: Dict[str, List[str]] = {}
    for item_id, (choices, density, why) in mapping.items():
        category, files = next(((c, f) for c in choices.split("|") if (f := _dtd_files(images, c))), (choices.split("|")[0], []))
        mapping[item_id] = (category, density, why)  # record the category actually used
        random.Random(f"{seed}-{item_id}").shuffle(files)
        found[item_id] = []
        for k, src in enumerate(files[:candidates]):
            out = os.path.join(cand_dir, f"{item_id}_{k}.png")
            Image.open(src).convert("RGB").save(out)
            save_json(os.path.splitext(out)[0] + ".json", {
                "source": "DTD r1.0.1", "category": category, "file": os.path.relpath(src, images),
                "license": "DTD: research use (images collected from the web)", "why": why})
            found[item_id].append(out)
        print(f"[fetch] {item_id} <- DTD/{category}: {len(found[item_id])} candidates")
    return found


# ------------------------------------------------------------------
def fetch_inputs(target: str, candidates: int = 4, seed: int = 0, portraits: Optional[str] = None,
                 dtd: Optional[str] = None, dtd_map: Optional[Dict[str, str]] = None, template: Optional[str] = None) -> str:
    cand_dir, cache = os.path.join(target, "candidates"), os.path.join(target, "_cache")
    os.makedirs(cand_dir, exist_ok=True)
    mapping = {k: (dtd_map.get(k, v[0]) if dtd_map else v[0], v[1], v[2]) for k, v in DTD_MAP.items()}
    found: Dict[str, List[str]] = {}
    found.update(fetch_animals(cand_dir, cache, candidates, seed))
    found.update(fetch_butterflies(cand_dir, cache, candidates, seed))
    found.update(fetch_portraits(cand_dir, portraits, candidates, seed))
    found.update(fetch_backgrounds(cand_dir, cache, dtd, 2 * candidates, seed, mapping))  # DTD is noisy: more to pick from

    sheet = [(Image.open(p), os.path.basename(p)[:-4]) for item in sorted(found) for p in found[item]]
    if sheet:
        flat = [(_on_gray(img), cap) for img, cap in sheet]
        labeled_grid(flat, columns=2 * candidates, tile=160).save(os.path.join(target, "candidates.png"))

    src = template or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pilot", "inputs.yaml")
    with open(src, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    for kind in ("subjects", "backgrounds"):
        for item_id, meta in data[kind].items():
            paths = found.get(item_id) or []
            meta["path"] = os.path.relpath(paths[0], target) if paths else ""
            meta["candidates"] = len(paths)
    for item_id in ANIMALS:
        data["subjects"][item_id].update(representation="photograph", source_and_license="Open Images V7, CC BY 2.0; see candidates/<ID>_<k>.json")
    data["subjects"]["O03"].update(representation="photograph", source_and_license="Leeds Butterfly Dataset, CC BY 4.0")
    for item_id in ("O05", "O06"):
        data["subjects"][item_id].update(label="human face, portrait", representation="photograph", pose="unknown",
                                         source_and_license="AISegment Matting Human; license unclear, pilot only")
    for item_id, (category, density, why) in mapping.items():
        data["backgrounds"][item_id].update(material=f"dtd:{category}", representation="photograph", texture_density=density,
                                            source_and_license="DTD r1.0.1, research use", note=why)
    data["backgrounds"]["B05"]["depth_cues"] = "strong"
    data["sources_note"] = ("Animals: Open Images masks (butterfly: Leeds masks); portraits: AISegment random picks (pose unknown); backgrounds: DTD "
                            "attribute categories, not the material list of INPUT_SELECTION_PLAN. No rendered_3d coverage.")
    out_yaml = os.path.join(target, "inputs.yaml")
    with open(out_yaml, "w", encoding="utf-8") as f:
        f.write("# Generated by `python -m camo.pilot fetch-inputs`. Look at candidates.png; to use another candidate change\n"
                "# e.g. O05 path to candidates/O05_2.png, then rerun normalize/preview/audit. Empty path = coverage gap.\n")
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    return out_yaml


def _on_gray(img: Image.Image) -> Image.Image:
    if img.mode != "RGBA":
        return img.convert("RGB")
    flat = Image.new("RGB", img.size, (128, 128, 128))
    flat.paste(img.convert("RGB"), mask=img.getchannel("A"))
    return flat
