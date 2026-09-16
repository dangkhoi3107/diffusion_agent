"""Orchestration: one pair, batches of pairs, and variant comparisons (ablations / sweeps)."""
from __future__ import annotations

import csv
import itertools
import json
import os
import shutil
import threading
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .config import Config, save_config
from .imageio import comparison_sheet, labeled_grid, list_images, overlay_mask, save_json, slugify, subject_name_from_path, to_pil
from .metrics import compute_metrics
from .preprocess import Prepared, Preprocessor
from .presets import Variant, apply_variant, grid_to_variants, variants_to_text

StatusFn = Callable[[int, int, str], None]
Pair = Tuple[str, str]
SUMMARY_METRICS = ("SSIM_bg", "PSNR_bg", "SSIM_subject", "S_edge", "hidden_score", "clip_text_gain", "clip_image_gain", "seconds")


class RunCancelled(Exception):
    """Raised between denoising steps after Session.request_stop()."""


@dataclass
class PairResult:
    output: Image.Image
    prepared: Prepared
    subject_name: str
    prompt: str
    seed: int
    seconds: float
    metrics: Dict[str, Any]

    def record(self) -> Dict[str, Any]:
        return {
            "subject": self.prepared.subject_path,
            "background": self.prepared.background_path,
            "subject_name": self.subject_name,
            "prompt": self.prompt,
            "seed": self.seed,
            "seconds": round(self.seconds, 2),
            "softedge_source": self.prepared.softedge_source,
            "metrics": self.metrics,
        }


class Session:
    """Owns the heavy models. Configs are passed per call so UI edits and sweeps apply immediately."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.preprocessor = Preprocessor(cfg)
        self._generator = None
        self._clip = None
        self._clip_failed = False
        self._stop = threading.Event()

    @property
    def generator(self):
        if self._generator is None:
            from .generator import CamoGenerator

            print("[session] Loading Stable Diffusion v1.5 + 2 ControlNets ...")
            self._generator = CamoGenerator.load(self.cfg)
        return self._generator

    def clip(self, cfg: Config):
        if not cfg.metrics.clip or self._clip_failed:
            return None
        if self._clip is None:
            try:
                from .clip_score import ClipScorer

                self._clip = ClipScorer(cfg.models.clip, self.generator.device)
            except Exception as exc:
                self._clip_failed = True
                print(f"[metrics] CLIP disabled -> {type(exc).__name__}: {exc}")
        return self._clip

    def request_stop(self) -> None:
        self._stop.set()

    def reset_stop(self) -> None:
        self._stop.clear()

    def check_stop(self) -> None:
        if self._stop.is_set():
            raise RunCancelled("Stopped by user")


def pick_seed(cfg: Config, index: int) -> int:
    if cfg.sampling.seed_mode == "random":
        return int.from_bytes(os.urandom(4), "big") % 2_147_483_647 or 1
    if cfg.sampling.seed_mode == "increment":
        return cfg.sampling.seed + index
    return cfg.sampling.seed


def new_run_dir(cfg: Config, label: str, name: Optional[str] = None) -> str:
    path = os.path.join(cfg.paths.out_dir, name or f"{label}_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(path, exist_ok=True)
    return path


def all_pairs(cfg: Config) -> List[Pair]:
    subjects, backgrounds = list_images(cfg.paths.subjects), list_images(cfg.paths.backgrounds)
    if not subjects:
        raise FileNotFoundError(f"No subject images found at: {cfg.paths.subjects}")
    if not backgrounds:
        raise FileNotFoundError(f"No background images found at: {cfg.paths.backgrounds}")
    return list(itertools.product(subjects, backgrounds))


def _stem(path: str) -> str:
    return slugify(os.path.splitext(os.path.basename(path))[0])[:40]


def pair_caption(subject: str, background: str) -> str:
    return f"{subject_name_from_path(subject)} x {_stem(background)}"


# ------------------------------------------------------------------
# Single pair
# ------------------------------------------------------------------
def run_pair(
    session: Session,
    cfg: Config,
    subject_path: str,
    background_path: str,
    seed: int,
    out_dir: Optional[str] = None,
    subject_name: Optional[str] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> PairResult:
    session.check_stop()
    name = subject_name or cfg.prompt.subject_name or subject_name_from_path(subject_path)
    prepared = session.preprocessor.prepare(cfg, subject_path, background_path, seed)

    def on_step(i: int, n: int) -> None:
        session.check_stop()
        if progress is not None:
            progress(i, n)

    gen = session.generator.generate(cfg, prepared, seed, name, on_step)

    metrics: Dict[str, Any] = {}
    if cfg.metrics.enabled:
        metrics.update(compute_metrics(np.array(gen.image), prepared.background, prepared.canny_clean, prepared.mask.mask))
        scorer = session.clip(cfg)
        if scorer is not None:
            metrics.update(scorer.score(gen.image, to_pil(prepared.background), to_pil(prepared.subject), name))

    result = PairResult(gen.image, prepared, name, gen.prompt, gen.seed, gen.seconds, metrics)
    if out_dir:
        save_pair(result, cfg, out_dir)
    return result


def stage_images(prepared: Prepared, output: Optional[Image.Image] = None) -> List[Tuple[str, Image.Image]]:
    items = [
        ("subject", to_pil(prepared.subject)),
        ("background", to_pil(prepared.background)),
        ("canny", to_pil(prepared.canny)),
        ("softedge", to_pil(prepared.softedge)),
        ("mask_1_smoothed", to_pil(prepared.mask.smoothed)),
        ("mask_2_remapped", to_pil(prepared.mask.remapped)),
        ("mask_3_final", to_pil(prepared.mask.mask)),
        ("mask_overlay", to_pil(overlay_mask(prepared.background, prepared.mask.mask))),
    ]
    if output is not None:
        items.append(("output", output))
    return items


def save_pair(result: PairResult, cfg: Config, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    result.output.save(os.path.join(out_dir, "output.png"))
    if cfg.output.save_debug:
        stages = dict(stage_images(result.prepared, result.output))
        for name, img in stages.items():
            if name != "output":
                img.save(os.path.join(out_dir, f"{name}.png"))
        panel_names = ("subject", "background", "canny", "softedge", "mask_overlay", "output")
        labeled_grid([(stages[n], n) for n in panel_names], columns=len(panel_names)).save(os.path.join(out_dir, "panel.png"))
    save_json(os.path.join(out_dir, "result.json"), result.record())
    save_config(cfg, os.path.join(out_dir, "config.yaml"))


# ------------------------------------------------------------------
# Job execution shared by batch and variant runs
# ------------------------------------------------------------------
@dataclass
class Job:
    cfg: Config
    subject: str
    background: str
    seed: int
    out_dir: str
    caption: str
    columns: Dict[str, Any] = field(default_factory=dict)  # extra CSV columns
    cell: Tuple[int, int] = (0, 0)  # (row, column) in a comparison sheet

    @property
    def output_path(self) -> str:
        return os.path.join(self.out_dir, "output.png")


def _flat_row(index: int, job: Job, record: Dict[str, Any]) -> Dict[str, Any]:
    row = {"index": index, **job.columns, "pair_dir": job.out_dir}
    row.update({k: v for k, v in record.items() if k != "metrics"})
    row.update(record.get("metrics", {}))
    return row


def execute_jobs(session: Session, jobs: Sequence[Job], label: str, status: Optional[StatusFn] = None) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    done: List[Job] = []
    stopped = False
    for index, job in enumerate(jobs):
        result_json = os.path.join(job.out_dir, "result.json")
        if job.cfg.output.skip_existing and os.path.exists(result_json):
            with open(result_json, encoding="utf-8") as f:
                rows.append(_flat_row(index, job, json.load(f)))
            done.append(job)
            continue

        message = f"[{label} {index + 1}/{len(jobs)}] {job.caption} seed={job.seed}"
        print(message)
        if status is not None:
            status(index, len(jobs), message)
        try:
            result = run_pair(session, job.cfg, job.subject, job.background, job.seed, job.out_dir)
        except RunCancelled:
            stopped = True
            print(f"[{label}] stopped by user after {len(done)} images")
            break
        except Exception as exc:
            os.makedirs(job.out_dir, exist_ok=True)
            with open(os.path.join(job.out_dir, "error.txt"), "w", encoding="utf-8") as f:
                f.write(traceback.format_exc())
            failures.append({"index": index, "subject": job.subject, "background": job.background, "error": f"{type(exc).__name__}: {exc}"})
            print(f"    FAILED: {type(exc).__name__}: {exc}")
            _free_cuda()
            continue
        rows.append(_flat_row(index, job, result.record()))
        done.append(job)
        m = result.metrics
        if m:
            clip = f" clip_text_gain={m['clip_text_gain']:+.3f}" if "clip_text_gain" in m else ""
            print(f"    {result.seconds:.1f}s | SSIM_bg={m['SSIM_bg']:.3f} S_edge={m['S_edge']:.3f}{clip}")
    return {"rows": rows, "failures": failures, "done": done, "stopped": stopped}


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> Optional[str]:
    if not rows:
        return None
    fieldnames: List[str] = []
    for row in rows:
        fieldnames.extend(k for k in row if k not in fieldnames)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def summarize(rows: Sequence[Dict[str, Any]], group_key: str) -> List[Dict[str, Any]]:
    """Mean of the headline metrics per group (e.g. per variant), in first-seen order."""
    groups: Dict[Any, List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row.get(group_key, ""), []).append(row)
    out = []
    for key, items in groups.items():
        entry: Dict[str, Any] = {group_key: key, "n": len(items)}
        for metric in SUMMARY_METRICS:
            vals = [float(r[metric]) for r in items if isinstance(r.get(metric), (int, float)) and np.isfinite(r[metric])]
            if vals:
                entry[metric] = round(float(np.mean(vals)), 4)
        out.append(entry)
    return out


def _grid_image(cfg: Config, jobs: Sequence[Job], path: str) -> Optional[str]:
    items = [(Image.open(j.output_path), j.caption) for j in jobs[: cfg.output.max_grid_items] if os.path.exists(j.output_path)]
    if not items:
        return None
    labeled_grid(items, columns=cfg.output.grid_columns).save(path)
    return path


def _zip(run_dir: str) -> str:
    return shutil.make_archive(run_dir, "zip", root_dir=os.path.dirname(run_dir), base_dir=os.path.basename(run_dir))


def _free_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ------------------------------------------------------------------
# Batch: every pair with the same settings
# ------------------------------------------------------------------
def run_batch(
    session: Session,
    cfg: Config,
    pairs: Optional[Sequence[Pair]] = None,
    run_dir: Optional[str] = None,
    status: Optional[StatusFn] = None,
) -> Dict[str, Any]:
    pairs = list(pairs or all_pairs(cfg))
    run_dir = run_dir or new_run_dir(cfg, "batch")
    save_config(cfg, os.path.join(run_dir, "config.yaml"))
    print(f"[batch] {len(pairs)} pairs -> {run_dir}")
    jobs = [
        Job(cfg, s, b, pick_seed(cfg, i), os.path.join(run_dir, f"{i:04d}_{_stem(s)}__{_stem(b)}"), pair_caption(s, b))
        for i, (s, b) in enumerate(pairs)
    ]
    out = execute_jobs(session, jobs, "batch", status)
    summary = {
        "run_dir": run_dir,
        "pairs": len(pairs),
        "succeeded": len(out["rows"]),
        "failed": len(out["failures"]),
        "stopped": out["stopped"],
        "failures": out["failures"],
        "csv": write_csv(os.path.join(run_dir, "metrics.csv"), out["rows"]),
        "grid": _grid_image(cfg, out["done"], os.path.join(run_dir, "grid.png")),
    }
    save_json(os.path.join(run_dir, "manifest.json"), summary)
    if cfg.output.make_zip and out["rows"]:
        summary["zip"] = _zip(run_dir)
    summary["gallery"] = [(j.output_path, j.caption) for j in out["done"]]
    summary["rows"] = out["rows"]
    print(f"[batch] done: {summary['succeeded']}/{len(pairs)} ok, {summary['failed']} failed" + (" (stopped)" if out["stopped"] else ""))
    return summary


# ------------------------------------------------------------------
# Variants: the same pairs under several settings (ablations, sweeps, seeds)
# ------------------------------------------------------------------
def run_variants(
    session: Session,
    cfg: Config,
    variants: Sequence[Variant],
    pairs: Optional[Sequence[Pair]] = None,
    run_dir: Optional[str] = None,
    status: Optional[StatusFn] = None,
    label: str = "compare",
) -> Dict[str, Any]:
    pairs = list(pairs or all_pairs(cfg))
    resolved = [(name, overrides, apply_variant(cfg, overrides)) for name, overrides in variants]  # fail fast on bad keys
    run_dir = run_dir or new_run_dir(cfg, label)
    save_config(cfg, os.path.join(run_dir, "config.yaml"))
    with open(os.path.join(run_dir, "variants.txt"), "w", encoding="utf-8") as f:
        f.write(variants_to_text(variants) + "\n")
    print(f"[{label}] {len(variants)} variants x {len(pairs)} pairs -> {run_dir}")

    jobs: List[Job] = []
    for pi, (subject, background) in enumerate(pairs):
        pair_seed = pick_seed(cfg, pi)  # shared by all variants of this pair (fair comparison)
        pair_dir = os.path.join(run_dir, f"{pi:03d}_{_stem(subject)}__{_stem(background)}")
        for vi, (name, overrides, vcfg) in enumerate(resolved):
            jobs.append(Job(
                cfg=vcfg,
                subject=subject,
                background=background,
                seed=pair_seed + (vcfg.sampling.seed - cfg.sampling.seed),
                out_dir=os.path.join(pair_dir, f"v{vi:02d}_{slugify(name)[:40]}"),
                caption=name,
                columns={"pair": pair_caption(subject, background), "variant": name,
                         **{f"param:{k}": v for k, v in overrides.items()}},
                cell=(pi, vi),
            ))

    out = execute_jobs(session, jobs, label, status)
    sheet_path = os.path.join(run_dir, "comparison.png")
    cells = {j.cell: j.output_path for j in out["done"] if os.path.exists(j.output_path)}
    if cells:
        comparison_sheet(cells, [pair_caption(s, b) for s, b in pairs], [name for name, _, _ in resolved]).save(sheet_path)
    summary_rows = summarize(out["rows"], "variant")
    result = {
        "run_dir": run_dir,
        "variants": len(variants),
        "pairs": len(pairs),
        "succeeded": len(out["rows"]),
        "failed": len(out["failures"]),
        "stopped": out["stopped"],
        "failures": out["failures"],
        "csv": write_csv(os.path.join(run_dir, "variants.csv"), out["rows"]),
        "summary_csv": write_csv(os.path.join(run_dir, "summary.csv"), summary_rows),
        "comparison": sheet_path if cells else None,
    }
    save_json(os.path.join(run_dir, "manifest.json"), result)
    if cfg.output.make_zip and out["rows"]:
        result["zip"] = _zip(run_dir)
    result["summary"] = summary_rows
    result["gallery"] = [(j.output_path, f"{j.columns['pair']} | {j.caption}") for j in out["done"]]
    return result


def run_sweep(session: Session, cfg: Config, grid: Optional[Dict[str, Any]] = None, pairs: Optional[Sequence[Pair]] = None,
              run_dir: Optional[str] = None, status: Optional[StatusFn] = None) -> Dict[str, Any]:
    grid = grid or cfg.sweep
    if not grid:
        raise ValueError("Empty sweep grid: set `sweep:` in the YAML or pass --grid key=v1,v2")
    return run_variants(session, cfg, grid_to_variants(grid), pairs, run_dir, status, label="sweep")
