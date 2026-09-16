"""Time schedules: ControlNet gate g_t, blend coefficient alpha_t, per-layer weights."""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np


def _window(total: int, start_frac: float, end_frac: float) -> Tuple[int, int]:
    return int(total * float(start_frac)), int(total * float(end_frac))


def control_gate(step: int, total: int, start_frac: float, end_frac: float, kind: str = "cosine") -> float:
    """ControlNet strength multiplier; 0 outside [start, end)."""
    s0, s1 = _window(total, start_frac, end_frac)
    if step < s0 or step >= s1:
        return 0.0
    x = (step - s0) / max(1, s1 - s0)
    if kind == "constant":
        return 1.0
    if kind == "linear":
        return 1.0 - x
    return math.cos(0.5 * math.pi * x)


def blend_alpha(
    step: int, total: int, start_frac: float, end_frac: float, alpha_end: float, profile: str = "decay"
) -> float:
    """alpha_t for z_out = (1 - alpha_t M) z_bg + alpha_t M z_sub.

    decay: 1 until start, linear down to alpha_end at end, then alpha_end.
    ramp:  0 until start, linear up to alpha_end at end, then alpha_end.
    constant: alpha_end at every step.
    """
    if profile == "constant":
        return float(alpha_end)
    s0, s1 = _window(total, start_frac, end_frac)
    s0 = max(0, min(s0, total - 1))
    s1 = max(0, min(s1, total))
    progress = (step - s0) / max(1, s1 - s0)
    if profile == "ramp":
        if step <= s0:
            return 0.0
        return float(alpha_end) if step >= s1 else progress * float(alpha_end)
    if step <= s0:
        return 1.0
    return float(alpha_end) if step >= s1 else 1.0 - progress * (1.0 - float(alpha_end))


def layer_weights(preset: str, n_down: int) -> Tuple[List[float], float]:
    """Per down-block residual weights and the mid-block weight for ControlNet."""
    if preset == "uniform":
        return [1.0] * n_down, 1.0
    if preset == "B":
        weights = np.linspace(0.25, 0.65, n_down)
        if n_down >= 12:
            weights[4:9] = np.minimum(0.70, weights[4:9] + 0.08)
        return [float(w) for w in weights], 0.55
    weights = np.array([0, 0, 0, 0, 0.1, 0.1, 0.2, 0.2, 0.3, 0.6, 0.6, 0.6], dtype=np.float32)
    return [float(w) for w in np.resize(weights, n_down)], 0.9
