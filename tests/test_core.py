"""CPU tests for the geometry, mask, schedule, config and metric logic.

Run: python -m pytest tests   (or: python tests/test_core.py)
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camo.config import (  # noqa: E402
    Config, LayoutConfig, apply_overrides, import_legacy_ui_json, legacy_layout_to_actual, load_config,
    save_config, validate,
)
from camo.hints import apply_layout, break_edges, canny_edges  # noqa: E402
from camo.imageio import format_prompt, subject_name_from_path  # noqa: E402
from camo.masks import build_soft_mask  # noqa: E402
from camo.metrics import compute_metrics, ssim_map  # noqa: E402
from camo.schedules import blend_alpha, control_gate, layer_weights  # noqa: E402


def _blob_stats(img):
    ys, xs = np.nonzero(img > 0.5)
    return xs.mean(), ys.mean(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1


def _canvas_reference(img, layout):
    """Re-implementation of the canvas drawing (translate -> rotate -> drawImage scaled)."""
    h, w = img.shape
    out = np.zeros_like(img)
    s, th = layout.scale, math.radians(layout.rotation_deg)
    cx, cy = (w - 1) / 2 + layout.offset_x * w, (h - 1) / 2 + layout.offset_y * h
    ys, xs = np.mgrid[0:h, 0:w]
    lx, ly = xs - cx, ys - cy  # inverse map every destination pixel
    sx = (math.cos(th) * lx + math.sin(th) * ly) / s + (w - 1) / 2
    sy = (-math.sin(th) * lx + math.cos(th) * ly) / s + (h - 1) / 2
    return cv2.remap(img, sx.astype(np.float32), sy.astype(np.float32), cv2.INTER_LINEAR, borderValue=0)


def test_layout_matches_canvas_convention():
    img = np.zeros((200, 200), np.float32)
    img[90:110, 100:160] = 1.0  # bar to the right of the centre
    for layout in [
        LayoutConfig(scale=0.5),
        LayoutConfig(scale=2.0),
        LayoutConfig(offset_x=0.2, offset_y=-0.1),
        LayoutConfig(rotation_deg=90),
        LayoutConfig(scale=0.7, offset_x=-0.15, offset_y=0.2, rotation_deg=-35),
    ]:
        ours, ref = apply_layout(img, layout), _canvas_reference(img, layout)
        assert np.abs(ours - ref).mean() < 0.01, layout

    cx, cy, bw, bh = _blob_stats(apply_layout(img, LayoutConfig(scale=0.5)))
    assert abs(bw - 30) <= 2 and abs(bh - 10) <= 2  # smaller, not 2x bigger
    cx, cy, _, _ = _blob_stats(apply_layout(img, LayoutConfig(rotation_deg=90)))
    assert cy > 120  # +90 deg is clockwise on screen: the bar points down


def _old_transform(arr, scale, ox, oy, rot):
    """The notebook's transform_frame, kept verbatim-in-spirit to test the legacy conversion."""
    h, w = arr.shape[:2]
    tx, ty = -ox * w, -oy * h
    th = math.radians(rot)
    c, s = math.cos(th), math.sin(th)
    cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
    t_neg = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], np.float32)
    s_m = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], np.float32)
    r_m = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float32)
    t_pos = np.array([[1, 0, cx + tx], [0, 1, cy + ty], [0, 0, 1]], np.float32)
    m = np.linalg.inv(t_pos @ r_m @ s_m @ t_neg)[:2].astype(np.float32)
    return cv2.warpAffine(arr, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def test_legacy_layout_conversion_reproduces_old_render():
    img = np.zeros((256, 256), np.float32)
    img[100:140, 120:200] = 1.0
    for old in [(0.6, 0.1, -0.2, 30.0), (1.4, -0.25, 0.05, -70.0), (1.0, 0.3, 0.0, 0.0)]:
        scale, ox, oy, rot = legacy_layout_to_actual(*old, 256, 256)
        new = apply_layout(img, LayoutConfig(scale, ox, oy, rot))
        assert np.abs(new - _old_transform(img, *old)).mean() < 0.01, old


def test_legacy_json_import():
    params = {"animal_name": "butterfly", "alpha_end": 0.6, "soft_scale": 1.2, "blend_mask_scale": 0.5,
              "mask_mode": "softedge", "randomize_seed_each_generate": True}
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ui_config.json")
        with open(path, "w") as f:
            json.dump({"params": params}, f)
        cfg = validate(import_legacy_ui_json(path))
    assert cfg.prompt.subject_name == "butterfly"
    assert cfg.blend.alpha_end == 0.6 and cfg.control.softedge_scale == 1.2
    assert abs(cfg.layout.scale - 2.0) < 1e-9  # old 0.5 really rendered 2x
    assert cfg.sampling.seed_mode == "random"


def test_config_roundtrip_and_overrides():
    cfg = Config()
    apply_overrides(cfg, ["blend.alpha_end=0.7", "sampling.steps=30", "hints.broken_edges=false", "sweep={mask.gamma: [0.4, 0.6]}"])
    assert cfg.blend.alpha_end == 0.7 and cfg.sampling.steps == 30 and cfg.hints.broken_edges is False
    with tempfile.TemporaryDirectory() as tmp:
        path = save_config(cfg, os.path.join(tmp, "c.yaml"))
        again = load_config(path)
    assert again.to_dict() == cfg.to_dict()
    try:
        apply_overrides(Config(), ["blend.alpha_ned=0.3"])
        raise AssertionError("typo should fail")
    except KeyError:
        pass
    try:
        validate(apply_overrides(Config(), ["blend.profile=wobble"]))
        raise AssertionError("bad choice should fail")
    except ValueError:
        pass


def test_default_yaml_matches_dataclass_defaults():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_config(os.path.join(root, "configs", "default.yaml"))
    ref = Config()
    ref.paths = cfg.paths
    ref.sweep = cfg.sweep
    assert cfg.to_dict() == ref.to_dict()


def test_soft_mask_pipeline():
    soft = np.zeros((64, 64, 3), np.uint8)
    soft[20:44, 20:44] = 200
    stages = build_soft_mask(soft, Config().mask)
    for arr in (stages.gray, stages.smoothed, stages.remapped, stages.mask):
        assert arr.dtype == np.float32 and arr.min() >= 0 and arr.max() <= 1
    assert stages.mask[32, 32] > 0.9 and stages.mask[0, 0] == 0.0


def test_schedules():
    n = 29
    assert blend_alpha(0, n, 0.05, 0.56, 0.54, "decay") == 1.0
    assert abs(blend_alpha(n - 1, n, 0.05, 0.56, 0.54, "decay") - 0.54) < 1e-9
    values = [blend_alpha(i, n, 0.05, 0.56, 0.54, "decay") for i in range(n)]
    assert all(a >= b for a, b in zip(values, values[1:]))
    assert blend_alpha(0, n, 0.1, 0.6, 0.5, "ramp") == 0.0
    assert control_gate(0, n, 0.0, 0.84) == 1.0 and control_gate(n - 1, n, 0.0, 0.84) == 0.0
    w, mid = layer_weights("A", 12)
    assert len(w) == 12 and w[:4] == [0, 0, 0, 0] and mid == 0.9


def test_hints_and_names():
    rgb = np.zeros((64, 64, 3), np.uint8)
    cv2.circle(rgb, (32, 32), 15, (255, 255, 255), -1)
    edges = canny_edges(rgb, Config().hints)
    assert edges.shape == (64, 64, 3) and edges.max() == 255
    broken = break_edges(edges, 0.5, seed=1)
    assert 0 < (broken > 0).sum() < (edges > 0).sum()
    assert np.array_equal(broken, break_edges(edges, 0.5, seed=1))
    assert subject_name_from_path("/x/lizard_000d3a9260.jpg") == "lizard"
    assert subject_name_from_path("golden-retriever_01.png") == "golden retriever"
    assert format_prompt("a {animal} and {animals}", "cat") == "a cat and cat"


def test_metrics_are_region_weighted():
    rng = np.random.default_rng(0)
    bg = (rng.random((128, 128, 3)) * 255).astype(np.uint8)
    mask = np.zeros((128, 128), np.float32)
    mask[40:90, 40:90] = 1.0
    out = bg.copy()
    out[40:90, 40:90] = 0  # subject region fully changed, background untouched
    m = compute_metrics(out, bg, np.zeros_like(bg), mask)
    assert m["SSIM_bg"] > 0.9 and m["SSIM_subject"] < 0.2
    assert abs(float(ssim_map(bg[..., 0] / 255.0, bg[..., 0] / 255.0).mean()) - 1.0) < 1e-5


def test_variant_presets_roundtrip_through_text():
    from camo.presets import VARIANT_PRESETS, apply_variant, parse_variants, variants_to_text

    base = Config()
    for key, (_, variants) in VARIANT_PRESETS.items():
        parsed = parse_variants(variants_to_text(variants))
        assert [o for _, o in parsed] == [o for _, o in variants], key
        assert [l for l, _ in parsed] == [l for l, _ in variants], key
        for _, overrides in parsed:
            apply_variant(base, overrides)  # every preset is a valid config
    seeds = parse_variants("a: sampling.seed+=2\nblend.alpha_end=0.3, mask.gamma=0.5")
    assert apply_variant(base, seeds[0][1]).sampling.seed == base.sampling.seed + 2
    assert seeds[1][1] == {"blend.alpha_end": 0.3, "mask.gamma": 0.5}


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} tests passed")
