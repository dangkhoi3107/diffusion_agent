"""Pilot M0/M1: normalize the O01-O06 / B01-B06 inputs, QA previews, then the 21 audit + 15 default renders.

    python -m camo.pilot normalize --inputs pilot/inputs.yaml --out OUT
    python -m camo.pilot preview   --inputs pilot/inputs.yaml --out OUT      # hints/mask/QA, no diffusion
    python -m camo.pilot audit     --inputs pilot/inputs.yaml --out OUT      # P02/P13/P16 x C00-C06
    python -m camo.pilot default   --inputs pilot/inputs.yaml --out OUT      # C00 on the other 15 pairs
    python -m camo.pilot all       --inputs pilot/inputs.yaml --out OUT      # the four steps in order
    python -m camo.pilot make-pattern --save pilot/B06_pattern.png           # procedural B06 + params json
    python -m camo.pilot fetch-inputs --save /kaggle/working/pilot_inputs    # Open Images animals, AISegment portraits, DTD

Every trial writes ``OUT/runs/<pair>/<config>/trial.json`` (re-running skips trials already ``ok``) and
appends to ``OUT/trials.jsonl``. Sampling code is not changed: inputs are normalized beforehand.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import platform
import shutil
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from PIL import Image, ImageOps

from .config import IS_KAGGLE, Config, apply_overrides, load_config, save_config
from .imageio import labeled_grid, save_json, to_pil

PREPROCESS_VERSION = "norm_v1"  # alpha over RGB(128) + aspect-preserving pad (subjects) / center crop (backgrounds)
GRAY = (128, 128, 128)

# INPUT_SELECTION_PLAN.md §5: each subject meets 3 backgrounds, each background 3 subjects.
DEFAULT_PAIRS: Dict[str, Tuple[str, str]] = {
    "P01": ("O01", "B06"), "P02": ("O01", "B02"), "P03": ("O01", "B04"),
    "P04": ("O02", "B02"), "P05": ("O02", "B03"), "P06": ("O02", "B05"),
    "P07": ("O03", "B03"), "P08": ("O03", "B04"), "P09": ("O03", "B06"),
    "P10": ("O04", "B01"), "P11": ("O04", "B04"), "P12": ("O04", "B05"),
    "P13": ("O05", "B01"), "P14": ("O05", "B02"), "P15": ("O05", "B05"),
    "P16": ("O06", "B03"), "P17": ("O06", "B01"), "P18": ("O06", "B06"),
}
DEFAULT_AUDIT_PAIRS = ["P02", "P13", "P16"]
# GOLDEN_SET_GUIDE.md §7: one-at-a-time perturbations around the default C00.
AUDIT_CONFIGS: Dict[str, Dict[str, float]] = {
    "C00": {},
    "C01": {"sampling.bg_strength": 0.47},
    "C02": {"sampling.bg_strength": 0.57},
    "C03": {"blend.alpha_end": 0.48},
    "C04": {"blend.alpha_end": 0.60},
    "C05": {"control.softedge_scale": 1.30},
    "C06": {"control.softedge_scale": 1.60},
}
WATCHDOG_FIRST_S, WATCHDOG_S = 900.0, 240.0  # per-trial limits; a render takes ~10 s on a T4
C00_EXPECTED = {"sampling.bg_strength": 0.52, "blend.alpha_end": 0.54, "control.softedge_scale": 1.50}


# ------------------------------------------------------------------
# Inputs manifest
# ------------------------------------------------------------------
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_by_id(folder: str, item_id: str) -> Optional[str]:
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    hits = sorted(p for p in glob.glob(os.path.join(folder, "**", f"{item_id}*"), recursive=True)
                  if p.lower().endswith(exts) and "_alpha" not in os.path.basename(p))
    return hits[0] if hits else None


def load_inputs(path: str) -> Dict[str, Any]:
    """Read inputs.yaml; empty ``path`` fields resolve to a file named ``<ID>*`` next to the yaml."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    root = os.path.dirname(os.path.abspath(path))
    missing: List[str] = []
    for kind in ("subjects", "backgrounds"):
        for item_id, meta in (data.get(kind) or {}).items():
            raw = (meta.get("path") or "").strip()
            resolved = (raw if os.path.isabs(raw) else os.path.join(root, raw)) if raw else _find_by_id(root, item_id)
            if resolved and os.path.isfile(resolved):
                meta["path"] = resolved
            else:
                meta["path"] = None
                missing.append(item_id)
    for item_id, meta in (data.get("subjects") or {}).items():
        if not str(meta.get("label") or "").strip():
            raise SystemExit(f"[pilot] {item_id}: `label` is empty (it becomes the prompt subject name)")
        if meta.get("category") not in ("animal", "human_portrait"):
            raise SystemExit(f"[pilot] {item_id}: category must be animal | human_portrait")
    pairs = {pid: tuple(v) for pid, v in (data.get("pairs") or DEFAULT_PAIRS).items()}
    data["pairs"] = pairs
    # An explicit empty list means "no audit pairs" (default-only runs such as the background map).
    data["audit_pairs"] = list(data["audit_pairs"] if data.get("audit_pairs") is not None else DEFAULT_AUDIT_PAIRS)
    unknown = [p for p in data["audit_pairs"] if p not in pairs]
    if unknown:
        raise SystemExit(f"[pilot] audit_pairs not in pairs: {unknown}")
    data["missing"] = missing
    data["inputs_file"] = os.path.abspath(path)
    return data


def pair_status(data: Dict[str, Any], pair_id: str) -> Optional[str]:
    """None if both inputs exist, else a reason."""
    s, b = data["pairs"][pair_id]
    gaps = [i for i in (s, b) if i in data["missing"] or i not in {**data["subjects"], **data["backgrounds"]}]
    return f"missing input {', '.join(gaps)}" if gaps else None


# ------------------------------------------------------------------
# Normalization (M0.8)
# ------------------------------------------------------------------
def _split_alpha(img: Image.Image) -> Tuple[Image.Image, Optional[Image.Image]]:
    """Composite onto gray so hidden RGB under transparent pixels cannot leak into hints."""
    if img.mode == "P" and "transparency" in img.info:
        img = img.convert("RGBA")
    if img.mode in ("RGBA", "LA", "PA"):
        rgba = img.convert("RGBA")
        alpha = rgba.getchannel("A")
        if alpha.getextrema() == (255, 255):
            return rgba.convert("RGB"), None
        flat = Image.new("RGB", rgba.size, GRAY)
        flat.paste(rgba.convert("RGB"), mask=alpha)
        return flat, alpha
    return img.convert("RGB"), None


def normalize_image(path: str, size: int, mode: str) -> Tuple[Image.Image, Optional[Image.Image], Dict[str, Any]]:
    """mode='pad': fit inside size x size, pad with gray (keeps every body part, no distortion).
    mode='crop': cover then center crop (backgrounds: no gray bars)."""
    with Image.open(path) as raw:
        src = ImageOps.exif_transpose(raw)
        src.load()
    rgb, alpha = _split_alpha(src)
    w, h = rgb.size
    if mode == "pad":
        scale = size / max(w, h)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        x0, y0 = (size - nw) // 2, (size - nh) // 2
        out = Image.new("RGB", (size, size), GRAY)
        out.paste(rgb.resize((nw, nh), Image.LANCZOS), (x0, y0))
        out_alpha = None
        if alpha is not None:
            out_alpha = Image.new("L", (size, size), 0)
            out_alpha.paste(alpha.resize((nw, nh), Image.LANCZOS), (x0, y0))
        transform = {"mode": "pad", "scale": scale, "placed_box": [x0, y0, x0 + nw, y0 + nh]}
    else:
        scale = size / min(w, h)
        nw, nh = max(size, round(w * scale)), max(size, round(h * scale))
        x0, y0 = (nw - size) // 2, (nh - size) // 2
        box = (x0, y0, x0 + size, y0 + size)
        out = rgb.resize((nw, nh), Image.LANCZOS).crop(box)
        out_alpha = alpha.resize((nw, nh), Image.LANCZOS).crop(box) if alpha is not None else None
        transform = {"mode": "crop", "scale": scale, "crop_box_resized": list(box)}
    meta = {
        "original_size": [w, h],
        "original_mode": src.mode,
        "has_alpha": alpha is not None,
        "transform": transform,
    }
    return out, out_alpha, meta


def normalize(data: Dict[str, Any], out: str, size: int = 512) -> Dict[str, Any]:
    norm_dir = os.path.join(out, "inputs_normalized")
    os.makedirs(norm_dir, exist_ok=True)
    inventory: Dict[str, Any] = {"preprocess_version": PREPROCESS_VERSION, "size": size,
                                 "inputs_file": data["inputs_file"], "items": {}, "missing": data["missing"]}
    previews: List[Tuple[Image.Image, str]] = []
    for kind, mode in (("subjects", "pad"), ("backgrounds", "crop")):
        for item_id, meta in (data.get(kind) or {}).items():
            if not meta.get("path"):
                print(f"[normalize] {item_id}: no image found -> coverage gap")
                continue
            img, alpha, info = normalize_image(meta["path"], size, mode)
            target = os.path.join(norm_dir, f"{item_id}.png")
            img.save(target)
            entry = {**{k: v for k, v in meta.items() if k != "path"}, **info, "kind": kind[:-1],
                     "original_path": meta["path"], "original_sha256": sha256_file(meta["path"]),
                     "normalized_path": target, "normalized_sha256": sha256_file(target)}
            if alpha is not None:
                alpha_path = os.path.join(norm_dir, f"{item_id}_alpha.png")
                alpha.save(alpha_path)
                entry["alpha_path"] = alpha_path
            inventory["items"][item_id] = entry
            previews.append((img, f"{item_id} {meta.get('label') or meta.get('material') or ''}".strip()))
            print(f"[normalize] {item_id}: {info['original_size']} {info['original_mode']} -> {mode} "
                  f"{'(alpha kept)' if alpha is not None else ''}")
    if previews:
        labeled_grid(previews, columns=6, tile=192).save(os.path.join(out, "inputs_overview.png"))
    save_json(os.path.join(out, "inventory.json"), inventory)
    return inventory


def load_inventory(out: str) -> Dict[str, Any]:
    path = os.path.join(out, "inventory.json")
    if not os.path.exists(path):
        raise SystemExit(f"[pilot] {path} not found: run the `normalize` step first")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------
# Environment / contract (M0.1, M0.6, M0.10)
# ------------------------------------------------------------------
def code_hash() -> str:
    pkg = os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha256()
    for path in sorted(glob.glob(os.path.join(pkg, "*.py"))):
        h.update(os.path.basename(path).encode())
        with open(path, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:16]


def environment(cfg: Config) -> Dict[str, Any]:
    import importlib.metadata as md

    env: Dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_hash": code_hash(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "kaggle": IS_KAGGLE,
        "packages": {},
        "models": {k: getattr(cfg.models, k) for k in ("base", "controlnet_canny", "controlnet_softedge", "hed_repo", "clip")},
        "model_revisions": {},
    }
    for pkg in ("torch", "diffusers", "transformers", "accelerate", "huggingface_hub", "safetensors",
                "numpy", "opencv-python", "opencv-python-headless", "pillow", "gradio"):
        try:
            env["packages"][pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            pass
    try:
        import torch

        env["cuda"] = torch.version.cuda
        env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    try:
        from huggingface_hub import model_info

        for key, repo in env["models"].items():
            try:
                env["model_revisions"][key] = model_info(repo).sha
            except Exception as exc:
                env["model_revisions"][key] = f"unresolved: {type(exc).__name__}"
    except ImportError:
        pass
    return env


def prompt_version(cfg: Config) -> str:
    text = json.dumps([cfg.prompt.subject_template, cfg.prompt.background, cfg.prompt.negative])
    return "prompt_" + hashlib.sha256(text.encode()).hexdigest()[:10]


def check_baseline(cfg: Config) -> None:
    """The C01-C06 table assumes C00 equals these values (GOLDEN_SET_GUIDE §7)."""
    from .config import get_value

    diff = {k: get_value(cfg, k) for k, v in C00_EXPECTED.items() if abs(float(get_value(cfg, k)) - v) > 1e-9}
    if diff:
        print(f"[pilot] WARNING: baseline differs from the audit table C00 {C00_EXPECTED}: {diff}")
    if cfg.sampling.legacy_unipc_corrector:
        print("[pilot] WARNING: legacy_unipc_corrector=true (legacy behavior, not the corrected baseline)")


def pair_seed(pair_id: str, seed: int) -> int:
    """Seed of one pair: stable (from the pair id), different between pairs, the same for every config of the pair
    so configs stay comparable. An explicit sampling.seed override shifts it."""
    return (int(seed) + int(hashlib.sha256(pair_id.encode("utf-8")).hexdigest()[:8], 16)) % 2**31


def pair_config(base: Config, inv: Dict[str, Any], subject_id: str, overrides: Dict[str, Any]) -> Config:
    cfg = base.copy()
    cfg.prompt.subject_name = inv["items"][subject_id]["label"]
    cfg.sampling.seed_mode = "fixed"
    cfg.output.make_zip = False
    apply_overrides(cfg, [f"{k}={v}" for k, v in overrides.items()])
    return cfg


# ------------------------------------------------------------------
# Preview / QA (M1.6, M1.11)
# ------------------------------------------------------------------
def preview(base: Config, data: Dict[str, Any], inv: Dict[str, Any], out: str, pair_ids: List[str]) -> List[Dict[str, Any]]:
    import cv2

    from .preprocess import Preprocessor
    from .runner import stage_images

    qa_dir = os.path.join(out, "qa")
    os.makedirs(qa_dir, exist_ok=True)
    pre = Preprocessor(base)
    rows = []
    for pid in pair_ids:
        reason = pair_status(data, pid)
        if reason:
            rows.append({"pair_id": pid, "status": reason})
            continue
        s, b = data["pairs"][pid]
        cfg = pair_config(base, inv, s, {})
        prepared = pre.prepare(cfg, inv["items"][s]["normalized_path"], inv["items"][b]["normalized_path"], cfg.sampling.seed)
        mask = prepared.mask.mask
        row = {"pair_id": pid, "subject": s, "background": b, "softedge_source": prepared.softedge_source,
               "mask_mean": round(float(mask.mean()), 4), "mask_active_frac": round(float((mask > 0.5).mean()), 4)}
        alpha_path = inv["items"][s].get("alpha_path")
        if alpha_path and abs(cfg.layout.scale - 1.0) < 1e-9 and not (cfg.layout.offset_x or cfg.layout.offset_y or cfg.layout.rotation_deg):
            alpha = np.asarray(Image.open(alpha_path).resize((mask.shape[1], mask.shape[0])), dtype=np.float32) / 255.0
            support = cv2.dilate((alpha > 0.5).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
            row["mask_outside_alpha_mass"] = round(float(mask[~support].sum() / max(mask.sum(), 1e-6)), 4)
        stages = stage_images(prepared)
        labeled_grid([(img, name) for name, img in stages], columns=4, tile=256).save(os.path.join(qa_dir, f"{pid}_{s}_{b}.png"))
        rows.append({**row, "status": "previewed"})
        print(f"[preview] {pid} {s}+{b}: {row}")
    save_json(os.path.join(qa_dir, "qa_preview.json"), rows)
    return rows


# ------------------------------------------------------------------
# Trials (M1.7, M1.9)
# ------------------------------------------------------------------
def run_trials(base: Config, data: Dict[str, Any], inv: Dict[str, Any], out: str,
               plan: List[Tuple[str, str]], stage: str, configs: Optional[Dict[str, Dict[str, Any]]] = None,
               seed_per_pair: bool = False) -> Dict[str, Any]:
    from .config import get_value
    from .runner import Session, run_pair

    configs = configs or AUDIT_CONFIGS

    env_path = os.path.join(out, "environment.json")
    if not os.path.exists(env_path):
        save_json(env_path, environment(base))
    session: Optional[Session] = None
    counts = {"ok": 0, "skipped": 0, "error": 0, "not_run": 0}
    for pid, cid in plan:
        trial_dir = os.path.join(out, "runs", pid, cid)
        trial_json = os.path.join(trial_dir, "trial.json")
        if os.path.exists(trial_json):
            with open(trial_json, encoding="utf-8") as f:
                if json.load(f).get("status") == "ok":
                    counts["skipped"] += 1
                    continue
        reason = pair_status(data, pid)
        s, b = data["pairs"][pid]
        record: Dict[str, Any] = {
            "trial_id": f"{pid}_{cid}", "pair_id": pid, "config_id": cid, "stage": stage, "split": "dev",
            "subject_id": s, "background_id": b,
            "subject_label": inv["items"].get(s, {}).get("label"), "category": inv["items"].get(s, {}).get("category"),
            "overrides": configs[cid], "preprocess_version": PREPROCESS_VERSION,
            "code_hash": code_hash(), "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        if reason:
            counts["not_run"] += 1
            print(f"[{stage}] {pid}/{cid}: {reason}")
            continue
        os.makedirs(trial_dir, exist_ok=True)
        timeouts_path = os.path.join(trial_dir, "timeouts.txt")
        timeouts = int(open(timeouts_path).read() or 0) if os.path.exists(timeouts_path) else 0
        if timeouts >= 2:  # hung twice: record and move on instead of blocking the whole run
            record.update({"status": "timeout", "error": f"hung {timeouts} times (watchdog)"})
            save_json(trial_json, record)
            counts["error"] += 1
            print(f"[{stage}] {pid}/{cid}: skipped after {timeouts} watchdog timeouts", flush=True)
            continue
        cfg = pair_config(base, inv, s, configs[cid])
        if seed_per_pair:
            cfg.sampling.seed = pair_seed(pid, cfg.sampling.seed)
        record.update({"seed": cfg.sampling.seed, "steps": cfg.sampling.steps, "prompt_version": prompt_version(cfg),
                       "negative_prompt": cfg.prompt.negative,
                       "params": {k: get_value(cfg, k) for k in dict.fromkeys([*C00_EXPECTED, "layout.scale", *configs[cid]])},
                       "inputs": {k: {"normalized_sha256": inv["items"][k]["normalized_sha256"],
                                      "original_sha256": inv["items"][k]["original_sha256"]} for k in (s, b)}})
        print(f"[{stage}] {pid}/{cid} {s}+{b} label={record['subject_label']!r} {configs[cid] or '(default)'}")
        started = time.perf_counter()
        limit = WATCHDOG_FIRST_S if session is None else WATCHDOG_S  # first trial also downloads/loads the models

        def on_hang(path=timeouts_path, n=timeouts, label=f"{pid}/{cid}", limit=limit):
            # A stuck CUDA call cannot be interrupted from Python: count it and exit so the caller reruns (resume skips ok trials).
            with open(path, "w") as f:
                f.write(str(n + 1))
            print(f"[{stage}] {label}: no result after {limit:.0f}s -> exiting for a restart (watchdog)", flush=True)
            os._exit(75)

        watchdog = threading.Timer(limit, on_hang)
        watchdog.daemon = True
        watchdog.start()
        try:
            if session is None:
                session = Session(base)
            result = run_pair(session, cfg, inv["items"][s]["normalized_path"], inv["items"][b]["normalized_path"],
                              cfg.sampling.seed, out_dir=trial_dir)
            record.update({"status": "ok", "prompt": result.prompt, "steps_run": result.steps_run,
                           "timings": {k: round(v, 3) for k, v in result.timings.items()},
                           "metrics": result.metrics, "softedge_source": result.prepared.softedge_source,
                           "output": os.path.relpath(os.path.join(trial_dir, "output.png"), out)})
            counts["ok"] += 1
            print(f"    ok {result.seconds:.1f}s render, steps_run={result.steps_run}")
        except Exception as exc:
            with open(os.path.join(trial_dir, "error.txt"), "w", encoding="utf-8") as f:
                f.write(traceback.format_exc())
            record.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
            counts["error"] += 1
            print(f"    FAILED: {record['error']}")
        finally:
            watchdog.cancel()
        record["wall_seconds"] = round(time.perf_counter() - started, 3)
        save_json(trial_json, record)
        with open(os.path.join(out, "trials.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    print(f"[{stage}] {counts}")
    return counts


def contact_sheets(out: str, pair_ids: List[str], configs: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
    """Diagnosis only: these sheets show config labels, never use them for first-view recognition."""
    for pid in pair_ids:
        items = []
        for cid, overrides in (configs or AUDIT_CONFIGS).items():
            path = os.path.join(out, "runs", pid, cid, "output.png")
            if os.path.exists(path):
                label = cid + (" " + " ".join(f"{k.split('.')[-1]}={v}" for k, v in overrides.items()) if overrides else " default")
                items.append((Image.open(path), label))
        if items:
            labeled_grid(items, columns=len(items), tile=256).save(os.path.join(out, "runs", pid, "contact_sheet.png"))


def rebuild_index(out: str) -> str:
    """trials.jsonl is append-only; trials_latest.json keeps the newest record per trial for tools."""
    latest: Dict[str, Any] = {}
    for path in sorted(glob.glob(os.path.join(out, "runs", "*", "*", "trial.json"))):
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
        latest[rec["trial_id"]] = rec
    return save_json(os.path.join(out, "trials_latest.json"), list(latest.values()))


# ------------------------------------------------------------------
# Procedural B06 (INPUT_SELECTION_PLAN §12: save generation params)
# ------------------------------------------------------------------
def make_pattern(path: str, size: int = 1024, seed: int = 7, cell: int = 64) -> str:
    from PIL import ImageDraw

    rng = np.random.default_rng(seed)
    palette = [tuple(int(c) for c in rng.integers(40, 220, 3)) for _ in range(3)]
    img = Image.new("RGB", (size, size), palette[0])
    draw = ImageDraw.Draw(img)
    for gy in range(0, size, cell):
        for gx in range(0, size, cell):
            cx, cy, r = gx + cell / 2, gy + cell / 2, cell * 0.36
            if ((gx // cell) + (gy // cell)) % 2 == 0:
                draw.polygon([(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)], fill=palette[1])
            else:
                draw.ellipse([cx - r * 0.6, cy - r * 0.6, cx + r * 0.6, cy + r * 0.6], fill=palette[2])
    img.save(path)
    save_json(os.path.splitext(path)[0] + "_params.json",
              {"generator": "camo.pilot.make_pattern", "size": size, "seed": seed, "cell": cell, "palette": palette,
               "representation": "procedural_pattern_2d", "repetition": "regular"})
    return path


# ------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m camo.pilot", description=__doc__.split("\n\n")[0])
    parser.add_argument("step", choices=["normalize", "preview", "audit", "default", "all", "index", "make-pattern", "fetch-inputs"])
    parser.add_argument("--inputs", default="pilot/inputs.yaml", help="inputs.yaml (images named <ID>* next to it)")
    parser.add_argument("--out", default="/kaggle/working/pilot" if IS_KAGGLE else "./outputs/pilot")
    default_cfg = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "default.yaml")
    parser.add_argument("--config", "-c", default=default_cfg if os.path.exists(default_cfg) else None,
                        help="Base YAML config (default: configs/default.yaml)")
    parser.add_argument("--set", "-s", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--configs", help="YAML {base?: {key: value}, audit_pairs?: [...], configs: {C00: {}, C01: {key: value}}} replacing the audit table")
    parser.add_argument("--pairs", help="Comma-separated pair IDs to restrict audit/default/preview")
    parser.add_argument("--zip", action="store_true", help="Zip OUT at the end (download from Kaggle Output)")
    parser.add_argument("--save", default=None, help="make-pattern: target file | fetch-inputs: target folder")
    parser.add_argument("--candidates", type=int, default=4, help="fetch-inputs: images per ID")
    parser.add_argument("--portraits", help="fetch-inputs: AISegment root (clip_img/ + matting/); default: search /kaggle/input")
    parser.add_argument("--dtd", help="fetch-inputs: DTD images/ folder or flat folder of <category>_NNNN.jpg; default: search /kaggle/input, else download")
    parser.add_argument("--dtd-map", help="fetch-inputs: override categories, e.g. B03=veined,B04=stratified")
    args = parser.parse_args(argv)

    if args.step == "make-pattern":
        print(f"[pilot] wrote {make_pattern(args.save or 'pilot/B06_pattern.png')}")
        return 0
    if args.step == "fetch-inputs":
        from .sources import fetch_inputs

        target = args.save or ("/kaggle/working/pilot_inputs" if IS_KAGGLE else "./outputs/pilot_inputs")
        dtd_map = dict(item.split("=", 1) for item in args.dtd_map.split(",")) if args.dtd_map else None
        print(f"[pilot] wrote {fetch_inputs(target, args.candidates, portraits=args.portraits, dtd=args.dtd, dtd_map=dtd_map)}")
        return 0

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    if args.step == "index":
        print(rebuild_index(out))
        return 0

    base = load_config(args.config, args.overrides)
    check_baseline(base)
    data = load_inputs(args.inputs)
    shutil.copy2(args.inputs, os.path.join(out, "inputs.yaml"))
    save_config(base, os.path.join(out, "base_config.yaml"))
    configs, seed_per_pair = AUDIT_CONFIGS, False
    if args.configs:
        with open(args.configs, encoding="utf-8") as f:
            spec = yaml.safe_load(f)
        configs = {str(k): dict(v or {}) for k, v in spec["configs"].items()}
        seed_per_pair = bool(spec.get("seed_per_pair"))  # one seed per pair (not 1234 for every image)
        if spec.get("base"):  # shared overrides (e.g. layout) applied before every config
            apply_overrides(base, [f"{k}={v}" for k, v in spec["base"].items()])
            save_config(base, os.path.join(out, "base_config.yaml"))
        if spec.get("audit_pairs"):
            data["audit_pairs"] = list(spec["audit_pairs"])
        if "C00" not in configs:
            raise SystemExit("[pilot] --configs needs a C00 (default) entry: it is reused for the default pairs")
        shutil.copy2(args.configs, os.path.join(out, "audit_configs.yaml"))
    only = [p.strip() for p in args.pairs.split(",")] if args.pairs else None
    audit_pairs = [p for p in data["audit_pairs"] if not only or p in only]
    default_pairs = [p for p in data["pairs"] if p not in data["audit_pairs"] and (not only or p in only)]
    if data["missing"]:
        print(f"[pilot] missing images (coverage gaps): {', '.join(data['missing'])}")

    steps = ["normalize", "preview", "audit", "default"] if args.step == "all" else [args.step]
    for step in steps:
        if step == "normalize":
            normalize(data, out, base.sampling.width)
            continue
        inv = load_inventory(out)
        if step == "preview":
            preview(base, data, inv, out, audit_pairs + default_pairs)
        elif step == "audit":
            run_trials(base, data, inv, out, [(p, c) for p in audit_pairs for c in configs], "audit", configs,
                       seed_per_pair)
            contact_sheets(out, audit_pairs, configs)
        elif step == "default":
            run_trials(base, data, inv, out, [(p, "C00") for p in default_pairs], "default", configs, seed_per_pair)
    rebuild_index(out)
    if args.zip:
        print(f"[pilot] zip: {shutil.make_archive(out, 'zip', root_dir=os.path.dirname(out), base_dir=os.path.basename(out))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
