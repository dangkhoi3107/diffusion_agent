"""Stage 2: soft latent mask  M = clip(R(B(H_s)), 0, 1)."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import MaskConfig


@dataclass
class SoftMaskStages:
    """Every intermediate of the mask construction, all HxW float32 in [0, 1]."""

    gray: np.ndarray  # H_s
    smoothed: np.ndarray  # B(H_s)
    remapped: np.ndarray  # R(B(H_s)) (invert + gamma)
    mask: np.ndarray  # after floor/ceiling normalisation and clipping


def build_soft_mask(softedge_rgb: np.ndarray, cfg: MaskConfig) -> SoftMaskStages:
    gray = cv2.cvtColor(softedge_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY) if softedge_rgb.ndim == 3 else softedge_rgb
    gray = gray.astype(np.float32) / 255.0

    smoothed = gray
    if cfg.blur > 1:
        k = cfg.blur if cfg.blur % 2 == 1 else cfg.blur + 1
        smoothed = cv2.GaussianBlur(gray, (k, k), 0)

    remapped = 1.0 - smoothed if cfg.invert else smoothed
    remapped = np.clip(remapped, 0.0, 1.0)
    if cfg.gamma != 1.0:
        remapped = remapped ** float(cfg.gamma)

    floor = float(np.clip(cfg.floor, 0.0, 1.0))
    ceiling = float(np.clip(cfg.ceiling, 0.0, 1.0))
    mask = (remapped - floor) / (ceiling - floor) if ceiling > floor else remapped
    return SoftMaskStages(
        gray=gray.astype(np.float32),
        smoothed=np.clip(smoothed, 0.0, 1.0).astype(np.float32),
        remapped=remapped.astype(np.float32),
        mask=np.clip(mask, 0.0, 1.0).astype(np.float32),
    )
