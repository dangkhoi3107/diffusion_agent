"""Stage 1: structural hints (Canny, SoftEdge/HED) and the layout transform."""
from __future__ import annotations

import math
from typing import Callable, Optional

import cv2
import numpy as np

from .config import HintsConfig, LayoutConfig


def _odd(k: int) -> int:
    k = max(int(k), 1)
    return k if k % 2 == 1 else k + 1


def gray3(gray: np.ndarray) -> np.ndarray:
    return np.repeat(gray[..., None], 3, axis=2)


def normalize_u8(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-6:
        return np.zeros(x.shape, dtype=np.uint8)
    return ((x - lo) / (hi - lo) * 255.0).clip(0, 255).astype(np.uint8)


def canny_edges(rgb: np.ndarray, cfg: HintsConfig) -> np.ndarray:
    """HxWx3 uint8 Canny map of the subject."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if cfg.canny_blur > 1:
        k = _odd(cfg.canny_blur)
        gray = cv2.GaussianBlur(gray, (k, k), 0)
    return gray3(cv2.Canny(gray, int(cfg.canny_low), int(cfg.canny_high)))


def break_edges(hint: np.ndarray, drop_prob: float, seed: int) -> np.ndarray:
    """Randomly drop edge pixels so the outline is less sticker-like."""
    rng = np.random.default_rng(seed)
    keep = rng.random(hint.shape[:2]) > float(drop_prob)
    return gray3(((hint[..., 0] > 0) & keep).astype(np.uint8) * 255)


# ------------------------------------------------------------------
# Layout: scale -> rotate -> translate, around the image centre.
# Matches the interactive canvas (ctx.translate; ctx.rotate; drawImage).
# ------------------------------------------------------------------
def is_identity_layout(layout: LayoutConfig) -> bool:
    return (
        abs(layout.scale - 1.0) < 1e-6
        and abs(layout.offset_x) < 1e-6
        and abs(layout.offset_y) < 1e-6
        and abs(layout.rotation_deg) < 1e-6
    )


def layout_matrix(width: int, height: int, layout: LayoutConfig) -> np.ndarray:
    """Forward 2x3 affine: dst = c + t + s * R(theta) * (src - c)."""
    s = max(float(layout.scale), 1e-3)
    theta = math.radians(float(layout.rotation_deg))
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    cx, cy = (width - 1) * 0.5, (height - 1) * 0.5
    tx, ty = float(layout.offset_x) * width, float(layout.offset_y) * height
    a, b = s * cos_t, -s * sin_t
    c, d = s * sin_t, s * cos_t
    return np.array(
        [[a, b, cx + tx - a * cx - b * cy], [c, d, cy + ty - c * cx - d * cy]],
        dtype=np.float64,
    )


def apply_layout(arr: np.ndarray, layout: LayoutConfig, interp: int = cv2.INTER_LINEAR) -> np.ndarray:
    if is_identity_layout(layout):
        return arr
    h, w = arr.shape[:2]
    return cv2.warpAffine(
        arr, layout_matrix(w, h, layout), (w, h), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )


# ------------------------------------------------------------------
# SoftEdge
# ------------------------------------------------------------------
SoftEdgeFn = Callable[[np.ndarray], np.ndarray]


def softedge_map(rgb: np.ndarray, detector: Optional[SoftEdgeFn], cfg: HintsConfig, canny: np.ndarray) -> tuple[np.ndarray, str]:
    """Return (HxWx3 uint8 soft-edge map, source name)."""
    if detector is not None:
        return gray3(normalize_u8(detector(rgb))), "hed"
    if not cfg.softedge_fallback_to_canny:
        raise RuntimeError(
            "HED soft-edge detector is unavailable. Fix the model download or set "
            "hints.softedge_fallback_to_canny=true (results will differ from the thesis)."
        )
    return canny.copy(), "canny-fallback"
