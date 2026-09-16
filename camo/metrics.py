"""Proxy metrics for hidden-subject images (numpy/OpenCV, no GPU needed).

All region metrics are *weighted* by the soft mask M (subject region) or 1-M
(background region) instead of zeroing pixels, which inflated the old SSIM_bg.
"""
from __future__ import annotations

from typing import Dict

import cv2
import numpy as np


def _gray01(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0


def ssim_map(a: np.ndarray, b: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Per-pixel SSIM with a Gaussian window (Wang et al. 2004), inputs in [0, 1]."""
    c1, c2 = 0.01**2, 0.03**2
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), sigma)  # noqa: E731
    mu_a, mu_b = blur(a), blur(b)
    var_a = blur(a * a) - mu_a**2
    var_b = blur(b * b) - mu_b**2
    cov = blur(a * b) - mu_a * mu_b
    return ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2))


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(weights.sum())
    return float((values * weights).sum() / total) if total > 1e-8 else float("nan")


def canny01(rgb: np.ndarray, low: int = 80, high: int = 160) -> np.ndarray:
    return cv2.Canny(cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY), low, high).astype(np.float32) / 255.0


def compute_metrics(output_rgb: np.ndarray, background_rgb: np.ndarray, subject_edges: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """
    SSIM_bg / PSNR_bg   background preservation outside the subject (higher = better preserved)
    SSIM_subject        similarity to the background inside the mask (lower = subject changed more)
    S_edge              cosine similarity of subject edges vs output edges inside the mask
    edge_density_*      Canny density of the output inside / outside the mask
    hidden_score        -|edge_in - edge_out|; 0 means the subject region is as "busy" as the rest
    """
    out_g, bg_g = _gray01(output_rgb), _gray01(background_rgb)
    m = np.clip(mask.astype(np.float32), 0.0, 1.0)
    inv = 1.0 - m

    ssim = ssim_map(out_g, bg_g)
    mse_bg = weighted_mean((out_g - bg_g) ** 2, inv)
    psnr_bg = float(10.0 * np.log10(1.0 / max(mse_bg, 1e-10))) if np.isfinite(mse_bg) else float("nan")

    e_out = canny01(output_rgb)
    e_sub = (subject_edges[..., 0] if subject_edges.ndim == 3 else subject_edges).astype(np.float32)
    e_sub = e_sub / 255.0 if e_sub.max() > 1.0 else e_sub
    soft = lambda x: cv2.GaussianBlur(x, (5, 5), 1.0)  # noqa: E731  (tolerate 1-2 px misalignment)
    a, b = (soft(e_sub) * m).ravel(), (soft(e_out) * m).ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))

    active = m > 0.25
    edge_in = float(e_out[active].mean()) if active.any() else 0.0
    edge_out = float(e_out[~active].mean()) if (~active).any() else 0.0

    return {
        "SSIM_bg": weighted_mean(ssim, inv),
        "PSNR_bg": psnr_bg,
        "SSIM_subject": weighted_mean(ssim, m),
        "S_edge": float(np.dot(a, b) / denom) if denom > 1e-8 else 0.0,
        "edge_density_in_mask": edge_in,
        "edge_density_outside": edge_out,
        "hidden_score": -abs(edge_in - edge_out),
        "mask_mean": float(m.mean()),
        "mask_active_ratio": float(active.mean()),
    }
