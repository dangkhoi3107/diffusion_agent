"""Stages 1-2: load inputs, extract hints, apply layout, build the soft mask."""
from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .config import Config
from .hints import apply_layout, break_edges, canny_edges, softedge_map
from .imageio import load_rgb
from .masks import SoftMaskStages, build_soft_mask


@dataclass
class Prepared:
    subject_path: str
    background_path: str
    subject: np.ndarray  # HxWx3 uint8, resized, before layout
    background: np.ndarray  # HxWx3 uint8, resized
    canny_clean: np.ndarray  # after layout, before broken edges (used by metrics)
    canny: np.ndarray  # ControlNet input
    softedge: np.ndarray  # ControlNet input
    mask: SoftMaskStages
    softedge_source: str  # "hed" | "canny-fallback"


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class Preprocessor:
    """Stateless w.r.t. the config; caches the (expensive) raw HED map per subject."""

    def __init__(self, cfg: Config, cache_size: int = 64):
        self.hed_repo = cfg.models.hed_repo
        self.device = resolve_device(cfg.models.device)
        self._hed = None
        self._hed_error: Optional[str] = None
        self._cache: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
        self._cache_size = cache_size

    def _detector(self):
        if self._hed is None and self._hed_error is None:
            try:
                from .hed import HEDDetector

                self._hed = HEDDetector.from_pretrained(self.hed_repo, self.device)
            except Exception as exc:  # network, missing torch, bad weights
                self._hed_error = f"{type(exc).__name__}: {exc}"
                print(f"[preprocess] HED unavailable -> {self._hed_error}")
        return self._hed

    def _softedge_raw(self, subject_path: str, subject: np.ndarray, cfg: Config, canny: np.ndarray):
        key = (os.path.abspath(subject_path), subject.shape)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key], "hed"
        soft, source = softedge_map(subject, self._detector(), cfg.hints, canny)
        if source == "hed":
            self._cache[key] = soft
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return soft, source

    def prepare(self, cfg: Config, subject_path: str, background_path: str, seed: int) -> Prepared:
        size = (cfg.sampling.width, cfg.sampling.height)
        subject = load_rgb(subject_path, size)
        background = load_rgb(background_path, size)

        canny = canny_edges(subject, cfg.hints)
        soft, source = self._softedge_raw(subject_path, subject, cfg, canny)

        canny = apply_layout(canny, cfg.layout, interp=cv2.INTER_NEAREST)
        soft = apply_layout(soft, cfg.layout, interp=cv2.INTER_LINEAR)
        canny_control = break_edges(canny, cfg.hints.broken_drop_prob, seed) if cfg.hints.broken_edges else canny

        return Prepared(
            subject_path=subject_path,
            background_path=background_path,
            subject=subject,
            background=background,
            canny_clean=canny,
            canny=canny_control,
            softedge=soft,
            mask=build_soft_mask(soft, cfg.mask),
            softedge_source=source,
        )
