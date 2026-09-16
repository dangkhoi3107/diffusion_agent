"""Typed configuration for the camouflage pipeline.

A config is a tree of dataclasses that round-trips to YAML. Any leaf can be
overridden with a dotted key, e.g. ``blend.alpha_end=0.6`` (CLI ``--set``),
and old Gradio ``ui_config_*.json`` files can be imported.
"""
from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Dict, Iterable, List, Tuple

import yaml

IS_KAGGLE = os.path.isdir("/kaggle/working")


@dataclass
class PathsConfig:
    subjects: str = "/kaggle/input/datasets/dangkhoi3107/thesis/animals"
    backgrounds: str = "/kaggle/input/datasets/dangkhoi3107/thesis/bg"
    out_dir: str = "/kaggle/working/outputs" if IS_KAGGLE else "./outputs"


@dataclass
class ModelsConfig:
    base: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    controlnet_canny: str = "lllyasviel/sd-controlnet-canny"
    controlnet_softedge: str = "lllyasviel/control_v11p_sd15_softedge"
    hed_repo: str = "lllyasviel/Annotators"
    clip: str = "openai/clip-vit-base-patch32"
    device: str = "auto"  # auto | cuda | cpu
    dtype: str = "auto"  # auto | float16 | float32
    attention_slicing: bool = False
    batch_branches: bool = True  # one UNet call for both branches (faster, more VRAM)


@dataclass
class PromptConfig:
    subject_name: str = ""  # empty -> derived from the subject file name
    subject_template: str = (
        "a hidden {animal} silhouette seamlessly integrated into the background, "
        "subtle facial cues, optical illusion, camouflage"
    )
    background: str = ""
    negative: str = (
        "sticker, pasted object, sharp outline, cartoon, text, watermark, logo, "
        "extra limbs, deformed, low quality, high contrast foreground object"
    )


@dataclass
class SamplingConfig:
    width: int = 512
    height: int = 512
    steps: int = 57
    seed: int = 1234
    seed_mode: str = "fixed"  # fixed | increment | random
    scheduler: str = "unipc"  # unipc | dpmpp_2m | ddim
    legacy_unipc_corrector: bool = False  # True reproduces the old notebook's (buggy) blending
    bg_strength: float = 0.52
    sub_init_noise: float = 0.0
    guidance_bg: float = 1.2
    guidance_sub: float = 2.3
    couple_mode: str = "bg_only"  # none | bg_only | both


@dataclass
class HintsConfig:
    canny_low: int = 60
    canny_high: int = 160
    canny_blur: int = 5
    broken_edges: bool = True
    broken_drop_prob: float = 0.30
    softedge_fallback_to_canny: bool = False


@dataclass
class ControlConfig:
    canny_scale: float = 0.02
    softedge_scale: float = 1.5
    start_frac: float = 0.0
    end_frac: float = 0.84
    gate: str = "cosine"  # cosine | linear | constant
    layer_weights: str = "A"  # A | B | uniform
    guess_mode: bool = True


@dataclass
class MaskConfig:
    blur: int = 11
    gamma: float = 0.5
    invert: bool = False
    floor: float = 0.05
    ceiling: float = 0.90


@dataclass
class BlendConfig:
    profile: str = "decay"  # decay | ramp | constant
    start_frac: float = 0.05
    end_frac: float = 0.56
    alpha_end: float = 0.54


@dataclass
class LayoutConfig:
    scale: float = 1.0  # < 1 shrinks the subject
    offset_x: float = 0.0  # fraction of width, + = right
    offset_y: float = 0.0  # fraction of height, + = down
    rotation_deg: float = 0.0  # + = clockwise on screen


@dataclass
class OutputConfig:
    save_debug: bool = True
    skip_existing: bool = False
    make_zip: bool = True
    grid_columns: int = 4
    max_grid_items: int = 64


@dataclass
class MetricsConfig:
    enabled: bool = True
    clip: bool = True


@dataclass
class UIConfig:
    share: bool = True
    port: int = 7860


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    hints: HintsConfig = field(default_factory=HintsConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    blend: BlendConfig = field(default_factory=BlendConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    # Parameter grid for `sweep`: {"blend.alpha_end": [0.4, 0.55, 0.7], ...}
    sweep: Dict[str, List[Any]] = field(default_factory=dict)

    def copy(self) -> "Config":
        return copy.deepcopy(self)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


CHOICES: Dict[str, Tuple[str, ...]] = {
    "models.device": ("auto", "cuda", "cpu"),
    "models.dtype": ("auto", "float16", "float32"),
    "sampling.seed_mode": ("fixed", "increment", "random"),
    "sampling.scheduler": ("unipc", "dpmpp_2m", "ddim"),
    "sampling.couple_mode": ("none", "bg_only", "both"),
    "control.gate": ("cosine", "linear", "constant"),
    "control.layer_weights": ("A", "B", "uniform"),
    "blend.profile": ("decay", "ramp", "constant"),
}

UNIT_INTERVAL_KEYS = (
    "sampling.bg_strength",
    "hints.broken_drop_prob",
    "control.start_frac",
    "control.end_frac",
    "mask.floor",
    "mask.ceiling",
    "blend.start_frac",
    "blend.end_frac",
    "blend.alpha_end",
)


# ------------------------------------------------------------------
# Dotted access
# ------------------------------------------------------------------
def _resolve(cfg: Config, key: str) -> Tuple[Any, str]:
    parts = key.split(".")
    node: Any = cfg
    for part in parts[:-1]:
        if not is_dataclass(node) or not hasattr(node, part):
            raise KeyError(f"Unknown config key: {key}")
        node = getattr(node, part)
    leaf = parts[-1]
    if not is_dataclass(node) or leaf not in {f.name for f in fields(node)}:
        raise KeyError(f"Unknown config key: {key}")
    return node, leaf


def get_value(cfg: Config, key: str) -> Any:
    node, leaf = _resolve(cfg, key)
    return getattr(node, leaf)


def _coerce(value: Any, current: Any, key: str) -> Any:
    if isinstance(current, bool):
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)
    if isinstance(current, int):
        return int(round(float(value)))
    if isinstance(current, float):
        return float(value)
    if isinstance(current, str):
        return "" if value is None else str(value)
    if isinstance(current, dict):
        if not isinstance(value, dict):
            raise TypeError(f"{key} expects a mapping")
        return dict(value)
    return value


def set_value(cfg: Config, key: str, value: Any) -> None:
    node, leaf = _resolve(cfg, key)
    setattr(node, leaf, _coerce(value, getattr(node, leaf), key))


def flatten(cfg: Config) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    def walk(node: Any, prefix: str) -> None:
        for f in fields(node):
            value = getattr(node, f.name)
            key = f"{prefix}{f.name}"
            if is_dataclass(value):
                walk(value, key + ".")
            else:
                out[key] = value

    walk(cfg, "")
    return out


def parse_override(text: str) -> Tuple[str, Any]:
    if "=" not in text:
        raise ValueError(f"Override must look like key=value, got: {text!r}")
    key, raw = text.split("=", 1)
    return key.strip(), yaml.safe_load(raw) if raw.strip() else ""


def apply_overrides(cfg: Config, overrides: Iterable[str]) -> Config:
    for text in overrides or []:
        key, value = parse_override(text)
        set_value(cfg, key, value)
    return cfg


def merge_dict(cfg: Config, data: Dict[str, Any], prefix: str = "") -> Config:
    for name, value in (data or {}).items():
        key = f"{prefix}{name}"
        if key == "sweep":
            cfg.sweep = dict(value or {})
            continue
        current = get_value(cfg, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise TypeError(f"Config section {key!r} must be a mapping")
            merge_dict(cfg, value, prefix=key + ".")
        else:
            set_value(cfg, key, value)
    return cfg


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------
def validate(cfg: Config) -> Config:
    errors: List[str] = []
    flat = flatten(cfg)
    for key, allowed in CHOICES.items():
        if flat[key] not in allowed:
            errors.append(f"{key}={flat[key]!r} (allowed: {', '.join(allowed)})")
    for key in UNIT_INTERVAL_KEYS:
        if not 0.0 <= float(flat[key]) <= 1.0:
            errors.append(f"{key}={flat[key]} must be in [0, 1]")
    for key in ("sampling.width", "sampling.height"):
        if flat[key] % 8 != 0 or flat[key] < 64:
            errors.append(f"{key}={flat[key]} must be a multiple of 8 and >= 64")
    if cfg.sampling.steps < 1:
        errors.append("sampling.steps must be >= 1")
    if cfg.mask.ceiling <= cfg.mask.floor:
        errors.append("mask.ceiling must be greater than mask.floor")
    if cfg.layout.scale <= 0:
        errors.append("layout.scale must be > 0")
    if cfg.hints.canny_low > cfg.hints.canny_high:
        errors.append("hints.canny_low must be <= hints.canny_high")
    for key in cfg.sweep:
        try:
            get_value(cfg, key)
        except KeyError:
            errors.append(f"sweep key {key!r} is not a config key")
    if errors:
        raise ValueError("Invalid config:\n  - " + "\n  - ".join(errors))
    return cfg


# ------------------------------------------------------------------
# Load / save
# ------------------------------------------------------------------
def load_config(path: str | None = None, overrides: Iterable[str] = ()) -> Config:
    cfg = Config()
    if path:
        if path.lower().endswith(".json"):
            cfg = import_legacy_ui_json(path)
        else:
            with open(path, "r", encoding="utf-8") as f:
                merge_dict(cfg, yaml.safe_load(f) or {})
    apply_overrides(cfg, overrides)
    return validate(cfg)


def save_config(cfg: Config, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False, allow_unicode=True)
    return path


# ------------------------------------------------------------------
# Legacy notebook config import (ui_config_*.json)
# ------------------------------------------------------------------
LEGACY_FIELD_ORDER = [
    "path_animal", "path_bg", "animal_name", "prompt_bg", "prompt_sub", "negative_prompt",
    "steps", "seed", "randomize_seed_each_generate", "bg_strength", "guidance_bg", "guidance_sub",
    "sub_init_noise", "couple_mode", "canny_low", "canny_high", "canny_blur_ks", "use_broken_edges",
    "broken_drop_prob", "mask_mode", "mask_dilate", "mask_close_ks", "mask_blur", "mask_ring_k",
    "mask_ring_blur", "mask_gamma", "mask_auto_invert", "outer_weight", "feature_weight", "fill_weight",
    "feature_thresh", "feature_open_ks", "feature_blur", "fill_blur", "use_soft_grad_mask",
    "soft_grad_dilation_k", "soft_grad_blur_k", "soft_grad_kernel_shape", "soft_grad_iterations",
    "soft_grad_use_skeleton", "soft_grad_skeleton_stage", "soft_grad_post_skel_dilate_k",
    "soft_mask_blur", "soft_mask_gamma", "soft_mask_invert", "soft_mask_floor", "soft_mask_ceiling",
    "canny_scale", "soft_scale", "start_frac", "end_frac", "gate_kind", "blend_profile", "blend_start",
    "blend_end", "alpha_end", "auto_attention_compensation", "attention_scale_ref", "attention_gamma_ref",
    "attention_ceiling_ref", "attention_blur_ref", "blend_mask_scale", "blend_mask_offset_x",
    "blend_mask_offset_y", "blend_mask_rotation_deg",
]

LEGACY_TO_KEY = {
    "path_animal": "paths.subjects",
    "path_bg": "paths.backgrounds",
    "animal_name": "prompt.subject_name",
    "prompt_bg": "prompt.background",
    "prompt_sub": "prompt.subject_template",
    "negative_prompt": "prompt.negative",
    "steps": "sampling.steps",
    "seed": "sampling.seed",
    "bg_strength": "sampling.bg_strength",
    "guidance_bg": "sampling.guidance_bg",
    "guidance_sub": "sampling.guidance_sub",
    "sub_init_noise": "sampling.sub_init_noise",
    "couple_mode": "sampling.couple_mode",
    "canny_low": "hints.canny_low",
    "canny_high": "hints.canny_high",
    "canny_blur_ks": "hints.canny_blur",
    "use_broken_edges": "hints.broken_edges",
    "broken_drop_prob": "hints.broken_drop_prob",
    "soft_mask_blur": "mask.blur",
    "soft_mask_gamma": "mask.gamma",
    "soft_mask_invert": "mask.invert",
    "soft_mask_floor": "mask.floor",
    "soft_mask_ceiling": "mask.ceiling",
    "canny_scale": "control.canny_scale",
    "soft_scale": "control.softedge_scale",
    "start_frac": "control.start_frac",
    "end_frac": "control.end_frac",
    "gate_kind": "control.gate",
    "blend_profile": "blend.profile",
    "blend_start": "blend.start_frac",
    "blend_end": "blend.end_frac",
    "alpha_end": "blend.alpha_end",
}

LEGACY_LAYOUT_ALIASES = {
    "layout_scale": "blend_mask_scale",
    "layout_offset_x": "blend_mask_offset_x",
    "layout_offset_y": "blend_mask_offset_y",
    "layout_rotation_deg": "blend_mask_rotation_deg",
}


def legacy_layout_to_actual(
    scale: float, offset_x: float, offset_y: float, rotation_deg: float, width: int, height: int
) -> Tuple[float, float, float, float]:
    """Convert layout values from the old notebook into the geometry it really rendered.

    The old ``transform_frame`` passed an inverted matrix to ``cv2.warpAffine``,
    so the subject was drawn at ``1/scale``, rotated by ``-rotation`` and the
    offset was rotated/scaled with it.
    """
    s = max(float(scale), 1e-3)
    theta = math.radians(-float(rotation_deg))
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    bx, by = float(offset_x) * width, float(offset_y) * height
    tx = (cos_t * bx - sin_t * by) / s
    ty = (sin_t * bx + cos_t * by) / s
    return 1.0 / s, tx / width, ty / height, -float(rotation_deg)


def import_legacy_ui_json(path: str, reproduce_old_geometry: bool = True) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    params = payload.get("params", payload) if isinstance(payload, dict) else payload
    if isinstance(params, list):
        params = dict(zip(LEGACY_FIELD_ORDER, params))
    if not isinstance(params, dict):
        raise ValueError(f"Unsupported legacy config format in {path}")
    for alias, name in LEGACY_LAYOUT_ALIASES.items():
        params.setdefault(name, params.get(alias))

    cfg = Config()
    for legacy_key, key in LEGACY_TO_KEY.items():
        if params.get(legacy_key) is not None:
            set_value(cfg, key, params[legacy_key])
    if params.get("randomize_seed_each_generate"):
        cfg.sampling.seed_mode = "random"
    if str(params.get("mask_mode", "softedge")).lower() != "softedge":
        print(f"[config] Legacy mask_mode={params.get('mask_mode')!r} is no longer supported; using softedge.")

    layout = [params.get(k) for k in ("blend_mask_scale", "blend_mask_offset_x", "blend_mask_offset_y", "blend_mask_rotation_deg")]
    if any(v is not None for v in layout):
        scale, ox, oy, rot = (float(v) if v is not None else d for v, d in zip(layout, (1.0, 0.0, 0.0, 0.0)))
        if reproduce_old_geometry:
            scale, ox, oy, rot = legacy_layout_to_actual(scale, ox, oy, rot, cfg.sampling.width, cfg.sampling.height)
        cfg.layout = LayoutConfig(scale=scale, offset_x=ox, offset_y=oy, rotation_deg=rot)
    return cfg
