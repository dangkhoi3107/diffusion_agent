"""Named config presets and variant sets (ablations / sweeps), shared by the CLI and the UI.

A *variant* is ``(label, {config_key: value})`` applied on top of a base config.
Relative numeric changes are written as ``"+=1"`` / ``"-=0.1"`` (e.g. extra seeds).
Text form, one variant per line:  ``label: key=value, key+=1``
"""
from __future__ import annotations

import itertools
import re
from typing import Any, Dict, List, Sequence, Tuple

import yaml

from .config import Config, apply_overrides, flatten, get_value, set_value, validate

Variant = Tuple[str, Dict[str, Any]]

# Whole-config presets (applied on top of the launch config; paths, subject name and layout are kept).
CONFIG_PRESETS: Dict[str, List[str]] = {
    "Thesis default": [],
    "Reproduce thesis images (legacy UniPC corrector)": ["sampling.legacy_unipc_corrector=true"],
    "Fast draft": ["sampling.steps=40", "sampling.guidance_bg=1.0", "metrics.clip=false"],
    "Subtler subject": ["blend.alpha_end=0.40", "mask.gamma=0.70", "control.softedge_scale=1.2"],
    "Stronger subject": ["blend.alpha_end=0.68", "mask.gamma=0.40", "sampling.guidance_sub=3.5"],
    "Textured subject photo (cut mask leakage)": ["mask.floor=0.20", "mask.gamma=0.60"],
}


def _grid(axes: Dict[str, Sequence[Any]]) -> List[Variant]:
    keys = list(axes)
    return [
        (", ".join(f"{k.split('.')[-1]}={v}" for k, v in zip(keys, combo)), dict(zip(keys, combo)))
        for combo in itertools.product(*axes.values())
    ]


VARIANT_PRESETS: Dict[str, Tuple[str, List[Variant]]] = {
    "structure": ("Structural guidance (Fig 4.9)", [
        ("no ControlNet", {"control.canny_scale": 0.0, "control.softedge_scale": 0.0}),
        ("Canny only", {"control.canny_scale": 0.3, "control.softedge_scale": 0.0}),
        ("SoftEdge only", {"control.canny_scale": 0.0}),
        ("Canny + SoftEdge", {"control.canny_scale": 0.3}),
        ("current settings", {}),
    ]),
    "mask": ("Mask strength (Fig 4.10)", [
        ("weak mask", {"mask.gamma": 1.5, "mask.floor": 0.15}),
        ("medium (current)", {}),
        ("strong mask", {"mask.gamma": 0.3, "mask.floor": 0.0, "mask.ceiling": 0.7}),
    ]),
    "alpha": ("Blend strength alpha_end (Fig 4.7)", [(f"alpha_end={a}", {"blend.alpha_end": a}) for a in (0.3, 0.45, 0.6, 0.75)]),
    "schedule": ("Blend schedule (Fig 4.11)", [(f"profile={p}", {"blend.profile": p}) for p in ("decay", "ramp", "constant")]),
    "layout": ("Layout scale and position (Fig 4.8)", [
        ("scale 0.5", {"layout.scale": 0.5, "layout.offset_x": 0.0}),
        ("scale 0.75", {"layout.scale": 0.75, "layout.offset_x": 0.0}),
        ("scale 1.0", {"layout.scale": 1.0, "layout.offset_x": 0.0}),
        ("0.75 left", {"layout.scale": 0.75, "layout.offset_x": -0.2}),
        ("0.75 right", {"layout.scale": 0.75, "layout.offset_x": 0.2}),
    ]),
    "bg_strength": ("Background strength", [(f"bg_strength={b}", {"sampling.bg_strength": b}) for b in (0.4, 0.5, 0.6, 0.7)]),
    "guidance": ("Subject guidance scale", [(f"guidance_sub={g}", {"sampling.guidance_sub": g}) for g in (1.5, 2.3, 3.5, 5.0)]),
    "seeds": ("Seeds x4 (same settings)", [("seed", {})] + [(f"seed+{i}", {"sampling.seed": f"+={i}"}) for i in (1, 2, 3)]),
    "scheduler": ("Scheduler / legacy corrector", [
        ("unipc (fixed)", {"sampling.scheduler": "unipc", "sampling.legacy_unipc_corrector": False}),
        ("unipc legacy (thesis)", {"sampling.scheduler": "unipc", "sampling.legacy_unipc_corrector": True}),
        ("dpmpp_2m", {"sampling.scheduler": "dpmpp_2m"}),
        ("ddim", {"sampling.scheduler": "ddim"}),
    ]),
    "tuning": ("Tuning grid: alpha_end x mask.gamma x bg_strength (12)", _grid({
        "blend.alpha_end": [0.40, 0.55, 0.70], "mask.gamma": [0.40, 0.60], "sampling.bg_strength": [0.45, 0.60],
    })),
}

TUNABLE_SECTIONS = ("sampling", "hints", "control", "mask", "blend", "layout")
TUNABLE_KEYS = [k for k, v in flatten(Config()).items() if k.split(".")[0] in TUNABLE_SECTIONS and not isinstance(v, dict)]


def grid_to_variants(grid: Dict[str, Any]) -> List[Variant]:
    return _grid({k: (v if isinstance(v, (list, tuple)) else [v]) for k, v in grid.items()})


def apply_config_preset(base: Config, current: Config, name: str) -> Config:
    cfg = apply_overrides(base.copy(), CONFIG_PRESETS[name])
    cfg.paths = current.paths
    cfg.layout = current.layout
    cfg.prompt.subject_name = current.prompt.subject_name
    return validate(cfg)


def apply_variant(cfg: Config, overrides: Dict[str, Any]) -> Config:
    out = cfg.copy()
    for key, value in overrides.items():
        if isinstance(value, str) and value[:2] in ("+=", "-="):
            delta = float(value[2:]) * (1 if value[0] == "+" else -1)
            value = get_value(out, key) + delta
        set_value(out, key, value)
    return validate(out)


def _fmt(value: Any) -> str:
    return str(value).lower() if isinstance(value, bool) else str(value)


def variants_to_text(variants: Sequence[Variant]) -> str:
    lines = []
    for label, overrides in variants:
        parts = [f"{k}{v}" if isinstance(v, str) and v[:2] in ("+=", "-=") else f"{k}={_fmt(v)}" for k, v in overrides.items()]
        lines.append(f"{label}: {', '.join(parts)}")
    return "\n".join(lines)


_OVERRIDE_PART = re.compile(r"^\s*[A-Za-z_]+\.[A-Za-z_]+\s*[+-]?=")


def _is_override_list(text: str) -> bool:
    parts = [p for p in text.split(",") if p.strip()]
    return bool(parts) and all(_OVERRIDE_PART.match(p) for p in parts)


def parse_variants(text: str) -> List[Variant]:
    variants: List[Variant] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        head, sep, rest = line.partition(":")
        if not sep or _is_override_list(head):  # no label, e.g. "blend.alpha_end=0.5, mask.gamma=0.4"
            head, rest = "", line
        overrides: Dict[str, Any] = {}
        for part in filter(None, (p.strip() for p in rest.split(","))):
            if "=" not in part:
                raise ValueError(f"Expected key=value in {raw!r}, got {part!r}")
            key, _, value = part.partition("=")
            key = key.strip()
            if key.endswith(("+", "-")):
                overrides[key[:-1].strip()] = f"{key[-1]}={value.strip()}"
            else:
                overrides[key] = yaml.safe_load(value) if value.strip() else ""
        label = head.strip() or ", ".join(f"{k.split('.')[-1]}={v}" for k, v in overrides.items()) or "current"
        variants.append((label, overrides))
    if not variants:
        raise ValueError("No variants given")
    return variants
