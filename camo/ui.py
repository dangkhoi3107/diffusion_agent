"""Gradio app: every pipeline feature is one click away.

Left:  config preset, uploads, all parameters (grouped by thesis stage), config file load/save.
Right: click-to-pick subject/background, Generate (N seeds), Stop, and tabs:
       Result | Layout canvas | Stages | Compare / ablation | Batch | History
"""
from __future__ import annotations

import glob
import html
import json
import os
import zipfile
from datetime import datetime
from typing import Any, Dict, List, Sequence, Tuple

import gradio as gr
import pandas as pd
import yaml

from .config import CHOICES, Config, LayoutConfig, get_value, load_config, save_config, set_value, validate
from .imageio import list_images, mask_to_rgba, thumbnail, to_pil
from .presets import CONFIG_PRESETS, TUNABLE_KEYS, VARIANT_PRESETS, apply_config_preset, grid_to_variants, parse_variants, variants_to_text
from .runner import (
    SUMMARY_METRICS, RunCancelled, Session, new_run_dir, pick_seed, run_batch, run_pair, run_variants, stage_images,
)
from .ui_canvas import JS_SYNC_LAYOUT, LAYOUT_ELEM_IDS, canvas_html

# (config key, widget, options). Layout must stay last: the canvas JS rewrites the last 4 inputs.
SECTIONS: List[Tuple[str, bool, List[Tuple[str, str, Dict[str, Any]]]]] = [
    ("Inputs & prompts", False, [
        ("paths.subjects", "text", {"label": "Subject image / folder (used when nothing is uploaded)"}),
        ("paths.backgrounds", "text", {"label": "Background image / folder (used when nothing is uploaded)"}),
        ("prompt.subject_name", "text", {"label": "Subject name (empty = from file name)"}),
        ("prompt.subject_template", "area", {"label": "Subject prompt template ({animal})"}),
        ("prompt.background", "area", {"label": "Background prompt"}),
        ("prompt.negative", "area", {"label": "Negative prompt"}),
    ]),
    ("Sampling - Stage 3", True, [
        ("sampling.steps", "slider", {"minimum": 1, "maximum": 100, "step": 1}),
        ("sampling.seed", "number", {}),
        ("sampling.seed_mode", "dropdown", {}),
        ("sampling.bg_strength", "slider", {"minimum": 0.0, "maximum": 1.0, "step": 0.01, "label": "bg strength (!)"}),
        ("sampling.guidance_sub", "slider", {"minimum": 0.0, "maximum": 10.0, "step": 0.1}),
        ("sampling.guidance_bg", "slider", {"minimum": 0.0, "maximum": 10.0, "step": 0.1}),
        ("sampling.sub_init_noise", "slider", {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
        ("sampling.couple_mode", "dropdown", {}),
        ("sampling.scheduler", "dropdown", {}),
        ("sampling.legacy_unipc_corrector", "checkbox", {"label": "Legacy UniPC corrector (reproduce old notebook)"}),
    ]),
    ("Structural hints - Stage 1", False, [
        ("hints.canny_low", "slider", {"minimum": 0, "maximum": 255, "step": 1}),
        ("hints.canny_high", "slider", {"minimum": 0, "maximum": 255, "step": 1}),
        ("hints.canny_blur", "slider", {"minimum": 1, "maximum": 15, "step": 2}),
        ("hints.broken_edges", "checkbox", {}),
        ("hints.broken_drop_prob", "slider", {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    ]),
    ("Soft mask - Stage 2", True, [
        ("mask.blur", "slider", {"minimum": 1, "maximum": 31, "step": 2}),
        ("mask.gamma", "slider", {"minimum": 0.1, "maximum": 5.0, "step": 0.05, "label": "gamma (!)"}),
        ("mask.floor", "slider", {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
        ("mask.ceiling", "slider", {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
        ("mask.invert", "checkbox", {}),
    ]),
    ("ControlNet - Stage 4", False, [
        ("control.softedge_scale", "slider", {"minimum": 0.0, "maximum": 2.0, "step": 0.01, "label": "softedge scale (!)"}),
        ("control.canny_scale", "slider", {"minimum": 0.0, "maximum": 2.0, "step": 0.01}),
        ("control.start_frac", "slider", {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
        ("control.end_frac", "slider", {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
        ("control.gate", "dropdown", {}),
        ("control.layer_weights", "dropdown", {}),
        ("control.guess_mode", "checkbox", {}),
    ]),
    ("Blending - Stage 5", True, [
        ("blend.alpha_end", "slider", {"minimum": 0.0, "maximum": 1.0, "step": 0.01, "label": "alpha end (!)"}),
        ("blend.profile", "dropdown", {}),
        ("blend.start_frac", "slider", {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
        ("blend.end_frac", "slider", {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    ]),
    ("Metrics", False, [
        ("metrics.enabled", "checkbox", {"label": "compute metrics"}),
        ("metrics.clip", "checkbox", {"label": "CLIP recognizability (downloads ~600MB once)"}),
    ]),
    ("Layout (synced with the canvas)", True, [
        ("layout.scale", "slider", {"minimum": 0.1, "maximum": 4.0, "step": 0.01}),
        ("layout.offset_x", "slider", {"minimum": -1.0, "maximum": 1.0, "step": 0.01, "label": "offset x (+ = right)"}),
        ("layout.offset_y", "slider", {"minimum": -1.0, "maximum": 1.0, "step": 0.01, "label": "offset y (+ = down)"}),
        ("layout.rotation_deg", "slider", {"minimum": -180.0, "maximum": 180.0, "step": 1.0, "label": "rotation deg (+ = clockwise)"}),
    ]),
]
FIELD_KEYS = [key for _, _, items in SECTIONS for key, _, _ in items]
assert FIELD_KEYS[-4:] == list(LAYOUT_ELEM_IDS)

SCOPES = [
    "Selected pair",
    "Selected subject x all backgrounds",
    "All subjects x selected background",
    "All subjects x all backgrounds",
]
MAX_THUMBS = 200
PRESET_CHOICES = [(title, key) for key, (title, _) in VARIANT_PRESETS.items()] + [("Custom (edit the list)", "custom")]


def _widget(cfg: Config, key: str, kind: str, opts: Dict[str, Any]):
    opts = dict(opts)
    label = opts.pop("label", key.split(".")[-1].replace("_", " "))
    value = get_value(cfg, key)
    if kind == "slider":
        return gr.Slider(value=value, label=label, elem_id=LAYOUT_ELEM_IDS.get(key), **opts)
    if kind == "dropdown":
        return gr.Dropdown(choices=list(CHOICES[key]), value=value, label=label)
    if kind == "checkbox":
        return gr.Checkbox(value=value, label=label)
    if kind == "number":
        return gr.Number(value=value, label=label, precision=0)
    return gr.Textbox(value=value, label=label, lines=2 if kind == "area" else 1)


def _upload_paths(value: Any) -> List[str]:
    items = value if isinstance(value, (list, tuple)) else ([value] if value else [])
    paths = []
    for item in items:
        if isinstance(item, dict):
            item = item.get("path") or item.get("name")
        elif not isinstance(item, (str, os.PathLike)):
            item = getattr(item, "path", None) or getattr(item, "name", None)
        if item and os.path.isfile(item):
            paths.append(os.fspath(item))
    return paths


def _choices(paths: Sequence[str], root: str) -> List[Tuple[str, str]]:
    def label(p: str) -> str:
        rel = os.path.relpath(p, root) if root and os.path.isdir(root) and p.startswith(os.path.abspath(root)) else ""
        return rel or os.path.basename(p)

    return [(label(p), p) for p in paths]


def _table(rows: Sequence[Dict[str, Any]], first: Sequence[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    cols = [c for c in first if any(c in r for r in rows)] + [m for m in SUMMARY_METRICS if any(m in r for r in rows)]
    df = pd.DataFrame(rows)[list(dict.fromkeys(cols))]
    return df.round(4)


def _zip_outputs(out_dir: str) -> str:
    export_dir = os.path.join(out_dir, "_exports")
    os.makedirs(export_dir, exist_ok=True)
    path = os.path.join(export_dir, f"outputs_{datetime.now():%Y%m%d_%H%M%S}.zip")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(out_dir):
            dirs[:] = [d for d in dirs if d != "_exports"]
            for name in files:
                if not name.endswith(".zip"):
                    full = os.path.join(root, name)
                    zf.write(full, os.path.relpath(full, out_dir))
    return path


def build_app(session: Session, base_cfg: Config) -> gr.Blocks:
    out_dir = os.path.abspath(base_cfg.paths.out_dir)

    # ---------------- helpers ----------------
    def config_from(values: Sequence[Any]) -> Config:
        cfg = base_cfg.copy()
        try:
            for key, value in zip(FIELD_KEYS, values):
                if value is not None:
                    set_value(cfg, key, value)
            return validate(cfg)
        except (ValueError, TypeError, KeyError) as exc:
            raise gr.Error(str(exc))

    def image_lists(s_up, b_up, s_path: str, b_path: str) -> Tuple[List[str], List[str]]:
        return _upload_paths(s_up) or list_images(s_path), _upload_paths(b_up) or list_images(b_path)

    def selection(cfg: Config, s_up, b_up, s_sel, b_sel):
        subjects, backgrounds = image_lists(s_up, b_up, cfg.paths.subjects, cfg.paths.backgrounds)
        if not subjects:
            raise gr.Error(f"No subject images: upload files or check the path {cfg.paths.subjects!r}")
        if not backgrounds:
            raise gr.Error(f"No background images: upload files or check the path {cfg.paths.backgrounds!r}")
        subject = s_sel if s_sel in subjects else subjects[0]
        background = b_sel if b_sel in backgrounds else backgrounds[0]
        return subjects, backgrounds, subject, background

    def scope_pairs(scope: str, subjects, backgrounds, subject, background) -> List[Tuple[str, str]]:
        s_list = subjects if scope in (SCOPES[2], SCOPES[3]) else [subject]
        b_list = backgrounds if scope in (SCOPES[1], SCOPES[3]) else [background]
        return [(s, b) for s in s_list for b in b_list]

    def render_canvas(cfg: Config, subject: str, background: str) -> str:
        identity = cfg.copy()
        identity.layout = LayoutConfig()
        prepared = session.preprocessor.prepare(identity, subject, background, pick_seed(cfg, 0))
        return canvas_html(to_pil(prepared.background), mask_to_rgba(prepared.mask.mask),
                           cfg.sampling.width, cfg.sampling.height, cfg.layout)

    def progress_status(progress):
        return lambda i, n, msg: progress(i / max(n, 1), desc=msg)

    # ---------------- handlers ----------------
    def on_reload(s_up, b_up, s_path, b_path, s_sel, b_sel):
        subjects, backgrounds = image_lists(s_up, b_up, s_path, b_path)
        s_val = s_sel if s_sel in subjects else (subjects[0] if subjects else None)
        b_val = b_sel if b_sel in backgrounds else (backgrounds[0] if backgrounds else None)
        note = f"{len(subjects)} subjects, {len(backgrounds)} backgrounds"
        if max(len(subjects), len(backgrounds)) > MAX_THUMBS:
            note += f" (thumbnails show the first {MAX_THUMBS})"
        return (
            [(thumbnail(p), os.path.basename(p)) for p in subjects[:MAX_THUMBS]],
            [(thumbnail(p), os.path.basename(p)) for p in backgrounds[:MAX_THUMBS]],
            gr.Dropdown(choices=_choices(subjects, s_path), value=s_val),
            gr.Dropdown(choices=_choices(backgrounds, b_path), value=b_val),
            note,
        )

    def pick(kind: str):
        def handler(upload, path, evt: gr.SelectData):
            items = _upload_paths(upload) or list_images(path)
            index = evt.index if isinstance(evt.index, int) else evt.index[0]
            return gr.Dropdown(value=items[index]) if 0 <= index < len(items) else gr.Dropdown()
        handler.__name__ = f"pick_{kind}"
        return handler

    def on_preset(name, *values):
        cfg = apply_config_preset(base_cfg, config_from(values), name)
        return [get_value(cfg, key) for key in FIELD_KEYS] + [f"Preset applied: {name}"]

    def on_canvas(s_up, b_up, s_sel, b_sel, *values):
        cfg = config_from(values)
        _, _, subject, background = selection(cfg, s_up, b_up, s_sel, b_sel)
        return render_canvas(cfg, subject, background), f"Canvas: {os.path.basename(subject)} on {os.path.basename(background)}"

    def on_preview(s_up, b_up, s_sel, b_sel, *values):
        cfg = config_from(values)
        _, _, subject, background = selection(cfg, s_up, b_up, s_sel, b_sel)
        prepared = session.preprocessor.prepare(cfg, subject, background, pick_seed(cfg, 0))
        m = prepared.mask.mask
        status = (f"Stages only (no diffusion) | softedge={prepared.softedge_source} | mask mean={m.mean():.3f} "
                  f"max={m.max():.3f} active(>0.25)={(m > 0.25).mean():.3f}")
        return [(img, name) for name, img in stage_images(prepared)], status

    def on_generate(n_seeds, s_up, b_up, s_sel, b_sel, *values, progress=gr.Progress()):
        session.reset_stop()
        cfg = config_from(values)
        _, _, subject, background = selection(cfg, s_up, b_up, s_sel, b_sel)
        n_seeds = max(1, int(n_seeds))
        run_dir = new_run_dir(cfg, "ui")
        base_seed = pick_seed(cfg, 0)
        results = []
        for k in range(n_seeds):
            seed = pick_seed(cfg, k) if cfg.sampling.seed_mode == "random" else base_seed + k
            try:
                results.append(run_pair(
                    session, cfg, subject, background, seed,
                    out_dir=os.path.join(run_dir, f"seed_{seed}") if n_seeds > 1 else run_dir,
                    progress=lambda i, n, k=k, seed=seed: progress((k + i / n) / n_seeds, desc=f"seed {seed}: step {i}/{n}"),
                ))
            except RunCancelled:
                break
        if not results:
            raise gr.Error("Stopped before the first image finished")
        first = results[0]
        rows = [{"seed": r.seed, "seconds": round(r.seconds, 1), **r.metrics} for r in results]
        stopped = " | STOPPED" if len(results) < n_seeds else ""
        status = f"{len(results)} image(s), {sum(r.seconds for r in results):.1f}s | prompt: {first.prompt} | saved: {run_dir}{stopped}"
        return (
            base_seed,
            first.output,
            [(r.output, f"seed {r.seed}") for r in results],
            [(img, name) for name, img in stage_images(first.prepared, first.output)],
            _table(rows, ["seed", "seconds"]),
            json.dumps([r.record() for r in results], indent=2, default=str),
            status,
        )

    def on_variant_preset(key):
        if key == "custom":
            return gr.Textbox()
        return variants_to_text(VARIANT_PRESETS[key][1])

    def on_build_grid(k1, v1, k2, v2, k3, v3):
        grid = {}
        for key, raw in ((k1, v1), (k2, v2), (k3, v3)):
            if key and str(raw or "").strip():
                grid[key] = [yaml.safe_load(x) for x in str(raw).split(",") if x.strip()]
        if not grid:
            raise gr.Error("Pick at least one parameter and type comma-separated values")
        return variants_to_text(grid_to_variants(grid)), "custom"

    def on_compare(variants_text, scope, s_up, b_up, s_sel, b_sel, *values, progress=gr.Progress()):
        session.reset_stop()
        cfg = config_from(values)
        subjects, backgrounds, subject, background = selection(cfg, s_up, b_up, s_sel, b_sel)
        try:
            variants = parse_variants(variants_text)
        except ValueError as exc:
            raise gr.Error(str(exc))
        pairs = scope_pairs(scope, subjects, backgrounds, subject, background)
        try:
            out = run_variants(session, cfg, variants, pairs, status=progress_status(progress))
        except (ValueError, KeyError, TypeError) as exc:
            raise gr.Error(f"Bad variant: {exc}")
        status = (f"Compare: {out['succeeded']}/{len(variants) * len(pairs)} images, {out['failed']} failed"
                  + (" | STOPPED" if out["stopped"] else "") + f" -> {out['run_dir']}")
        return out["comparison"], _table(out["summary"], ["variant", "n"]), out["gallery"], out.get("zip"), status

    def on_batch(scope, s_up, b_up, s_sel, b_sel, *values, progress=gr.Progress()):
        session.reset_stop()
        cfg = config_from(values)
        subjects, backgrounds, subject, background = selection(cfg, s_up, b_up, s_sel, b_sel)
        pairs = scope_pairs(scope, subjects, backgrounds, subject, background)
        out = run_batch(session, cfg, pairs, run_dir=new_run_dir(cfg, "ui_batch"), status=progress_status(progress))
        table = _table(out["rows"], ["subject_name", "background", "seed"])
        if not table.empty and "background" in table:
            table["background"] = table["background"].map(os.path.basename)
        status = (f"Batch: {out['succeeded']}/{out['pairs']} ok, {out['failed']} failed"
                  + (" | STOPPED" if out["stopped"] else "") + f" -> {out['run_dir']}")
        return out["gallery"], table, out.get("zip"), status

    def on_stop():
        session.request_stop()
        return "Stop requested: the current image stops at its next denoising step."

    def on_history():
        files = sorted(glob.glob(os.path.join(out_dir, "**", "output.png"), recursive=True), key=os.path.getmtime, reverse=True)[:60]
        return [(p, os.path.relpath(os.path.dirname(p), out_dir)) for p in files], f"{len(files)} most recent outputs in {out_dir}"

    def on_zip_all():
        path = _zip_outputs(out_dir)
        return path, f"Zipped all outputs: {path}"

    def on_save(*values):
        cfg = config_from(values)
        path = os.path.join(out_dir, "configs", f"ui_{datetime.now():%Y%m%d_%H%M%S}.yaml")
        save_config(cfg, path)
        return path, f"Saved config: {path}"

    def on_load(path):
        path = str(path or "").strip()
        if not os.path.isfile(path):
            raise gr.Error(f"Config not found: {path}")
        try:
            cfg = load_config(path)
        except Exception as exc:
            raise gr.Error(f"Could not load {path}: {exc}")
        return [get_value(cfg, key) for key in FIELD_KEYS] + [f"Loaded {path}. Click 'Refresh canvas' to redraw the layout."]

    # ---------------- layout ----------------
    with gr.Blocks() as demo:
        gr.Markdown("## Camouflage latent blending · SD1.5 + Canny/SoftEdge ControlNet\n"
                    "Click a thumbnail to choose the subject and background, set parameters on the left, then Generate, "
                    "Compare or Batch. `(!)` marks the most sensitive parameters. Every run is saved under "
                    f"`{out_dir}` together with its config.")
        with gr.Row():
            with gr.Column(scale=4, min_width=380):
                with gr.Row():
                    preset = gr.Dropdown(choices=list(CONFIG_PRESETS), value=list(CONFIG_PRESETS)[0], label="Config preset", scale=3)
                    btn_preset = gr.Button("Apply preset", scale=1)
                with gr.Accordion("Upload images (optional, overrides folders)", open=False):
                    subject_upload = gr.File(label="Subject images", file_count="multiple", file_types=["image"], type="filepath")
                    bg_upload = gr.File(label="Background images", file_count="multiple", file_types=["image"], type="filepath")
                widgets = {}
                for title, is_open, items in SECTIONS:
                    with gr.Accordion(title, open=is_open):
                        for key, kind, opts in items:
                            widgets[key] = _widget(base_cfg, key, kind, opts)
                with gr.Accordion("Config file (YAML or legacy ui_config JSON)", open=False):
                    config_path = gr.Textbox(label="Path")
                    with gr.Row():
                        btn_load = gr.Button("Load")
                        btn_save = gr.Button("Save current")

            with gr.Column(scale=6):
                with gr.Group():
                    with gr.Row():
                        subject_dd = gr.Dropdown(label="Subject", choices=[], allow_custom_value=True, scale=3)
                        bg_dd = gr.Dropdown(label="Background", choices=[], allow_custom_value=True, scale=3)
                        btn_reload = gr.Button("Reload images", scale=1)
                    with gr.Row():
                        subject_gallery = gr.Gallery(label="Subjects (click to select)", columns=6, height=190, allow_preview=False)
                        bg_gallery = gr.Gallery(label="Backgrounds (click to select)", columns=6, height=190, allow_preview=False)
                with gr.Row():
                    btn_generate = gr.Button("Generate", variant="primary", scale=3)
                    n_seeds = gr.Slider(1, 8, value=1, step=1, label="images (seed, seed+1, ...)", scale=3)
                    btn_stop = gr.Button("Stop", variant="stop", scale=1)
                status = gr.Textbox(label="Status", interactive=False, lines=2)

                with gr.Tabs():
                    with gr.Tab("Result"):
                        output_image = gr.Image(label="Output", type="pil", height=512)
                        seeds_gallery = gr.Gallery(label="All generated seeds", columns=4, height="auto")
                        metrics_table = gr.Dataframe(label="Metrics", interactive=False)
                        with gr.Accordion("Result JSON", open=False):
                            result_json = gr.Code(language="json")
                    with gr.Tab("Layout canvas"):
                        btn_canvas = gr.Button("Refresh canvas (after changing images or mask settings)")
                        try:
                            first_s, first_b = list_images(base_cfg.paths.subjects), list_images(base_cfg.paths.backgrounds)
                            initial = render_canvas(base_cfg, first_s[0], first_b[0]) if first_s and first_b else ""
                        except Exception as exc:
                            initial = f"<p>Canvas unavailable: {html.escape(str(exc))}</p>"
                        canvas = gr.HTML(initial or "<p>Choose images, then click Refresh canvas.</p>")
                    with gr.Tab("Stages"):
                        btn_preview = gr.Button("Preview stages (no diffusion, a few seconds)")
                        stages = gr.Gallery(label="Stage 1-2 intermediates", columns=4, height="auto")
                    with gr.Tab("Compare / ablation"):
                        gr.Markdown("Render the same images under several settings, one per line: "
                                    "`label: key=value, key=value`. Keys not listed keep the values from the left panel. "
                                    "`sampling.seed+=1` changes a value relative to the panel.")
                        with gr.Row():
                            variant_preset = gr.Dropdown(choices=PRESET_CHOICES, value="structure", label="Variant preset", scale=3)
                            compare_scope = gr.Dropdown(choices=SCOPES, value=SCOPES[0], label="Images", scale=3)
                        variants_box = gr.Textbox(value=variants_to_text(VARIANT_PRESETS["structure"][1]), lines=7,
                                                  label="Variants (editable)")
                        with gr.Accordion("Grid builder (cartesian product of up to 3 parameters)", open=False):
                            grid_inputs = []
                            for default_key, default_values in (("blend.alpha_end", "0.4, 0.55, 0.7"), ("mask.gamma", "0.4, 0.6"), (None, "")):
                                with gr.Row():
                                    grid_inputs.append(gr.Dropdown(choices=TUNABLE_KEYS, value=default_key, label="parameter", scale=2))
                                    grid_inputs.append(gr.Textbox(value=default_values, label="values (comma separated)", scale=3))
                            btn_grid = gr.Button("Build variants from grid")
                        btn_compare = gr.Button("Run comparison", variant="primary")
                        comparison_image = gr.Image(label="Comparison sheet (rows = image pairs, columns = variants)", type="filepath")
                        compare_table = gr.Dataframe(label="Mean metrics per variant", interactive=False)
                        with gr.Accordion("All images", open=False):
                            compare_gallery = gr.Gallery(columns=4, height="auto")
                        compare_zip = gr.File(label="ZIP (images, per-image metrics, variants.csv, summary.csv)")
                    with gr.Tab("Batch"):
                        with gr.Row():
                            batch_scope = gr.Dropdown(choices=SCOPES, value=SCOPES[3], label="Images", scale=3)
                            btn_batch = gr.Button("Run batch", variant="primary", scale=1)
                        batch_gallery = gr.Gallery(label="Outputs", columns=4, height="auto")
                        batch_table = gr.Dataframe(label="Metrics per image", interactive=False)
                        batch_zip = gr.File(label="ZIP")
                    with gr.Tab("History"):
                        with gr.Row():
                            btn_history = gr.Button("Show recent outputs")
                            btn_zip_all = gr.Button("Zip all outputs for download")
                        history_gallery = gr.Gallery(columns=5, height="auto")
                        history_zip = gr.File(label="All outputs")

        # ---------------- events ----------------
        fields = [widgets[key] for key in FIELD_KEYS]
        picked = [subject_upload, bg_upload, subject_dd, bg_dd]
        reload_io = ([subject_upload, bg_upload, widgets["paths.subjects"], widgets["paths.backgrounds"], subject_dd, bg_dd],
                     [subject_gallery, bg_gallery, subject_dd, bg_dd, status])
        for trigger in (btn_reload.click, subject_upload.change, bg_upload.change, widgets["paths.subjects"].submit,
                        widgets["paths.backgrounds"].submit, demo.load):
            trigger(on_reload, *reload_io, concurrency_limit=None)
        subject_gallery.select(pick("subject"), [subject_upload, widgets["paths.subjects"]], subject_dd, concurrency_limit=None)
        bg_gallery.select(pick("background"), [bg_upload, widgets["paths.backgrounds"]], bg_dd, concurrency_limit=None)

        # The canvas JS rewrites the LAST four inputs, so `fields` must always come last.
        btn_preset.click(on_preset, [preset] + fields, fields + [status], js=JS_SYNC_LAYOUT, concurrency_limit=None)
        btn_generate.click(on_generate, [n_seeds] + picked + fields,
                           [widgets["sampling.seed"], output_image, seeds_gallery, stages, metrics_table, result_json, status],
                           js=JS_SYNC_LAYOUT)
        btn_canvas.click(on_canvas, picked + fields, [canvas, status], js=JS_SYNC_LAYOUT)
        btn_preview.click(on_preview, picked + fields, [stages, status], js=JS_SYNC_LAYOUT)
        variant_preset.change(on_variant_preset, variant_preset, variants_box, concurrency_limit=None)
        btn_grid.click(on_build_grid, grid_inputs, [variants_box, variant_preset], concurrency_limit=None)
        btn_compare.click(on_compare, [variants_box, compare_scope] + picked + fields,
                          [comparison_image, compare_table, compare_gallery, compare_zip, status], js=JS_SYNC_LAYOUT)
        btn_batch.click(on_batch, [batch_scope] + picked + fields, [batch_gallery, batch_table, batch_zip, status], js=JS_SYNC_LAYOUT)
        btn_stop.click(on_stop, None, status, concurrency_limit=None)
        btn_history.click(on_history, None, [history_gallery, status], concurrency_limit=None)
        btn_zip_all.click(on_zip_all, None, [history_zip, status], concurrency_limit=None)
        btn_save.click(on_save, fields, [config_path, status], js=JS_SYNC_LAYOUT, concurrency_limit=None)
        btn_load.click(on_load, [config_path], fields + [status], concurrency_limit=None)
    return demo


def launch(cfg: Config) -> None:
    session = Session(cfg)
    session.generator  # load models before the first click
    demo = build_app(session, cfg)
    demo.queue(default_concurrency_limit=1).launch(
        share=cfg.ui.share,
        server_name="0.0.0.0",
        server_port=cfg.ui.port,
        allowed_paths=[os.path.abspath(cfg.paths.out_dir)],
    )
