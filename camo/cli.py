"""Command line entry point: ``python -m camo <command> [options]``."""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional

import yaml

from .config import IS_KAGGLE, apply_overrides, import_legacy_ui_json, load_config, save_config, set_value, validate
from .presets import CONFIG_PRESETS, VARIANT_PRESETS, grid_to_variants, parse_variants, variants_to_text

COMMANDS = {
    "ui": "Launch the Gradio app.",
    "run": "Generate the first pair.",
    "batch": "Generate every subject x every background.",
    "sweep": "Every pair under several settings: --preset NAME | --grid key=v1,v2 | --variants FILE | config `sweep:`.",
    "preview": "Hints + mask + layout only (no diffusion, fast).",
    "list-presets": "Show config presets and variant presets.",
    "import-legacy": "Convert an old ui_config_*.json into a YAML config.",
    "show-config": "Print the resolved config.",
}


def _parse_grid(items: Optional[List[str]]) -> Dict[str, list]:
    grid: Dict[str, list] = {}
    for item in items or []:
        key, _, raw = item.partition("=")
        grid[key.strip()] = [yaml.safe_load(v) for v in raw.split(",") if v.strip()]
    return grid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m camo", description="Camouflage latent blending (SD1.5 + dual ControlNet)")
    parser.add_argument("command", choices=list(COMMANDS), help=" | ".join(f"{k}: {v}" for k, v in COMMANDS.items()))
    parser.add_argument("--config", "-c", default=None, help="YAML config (or legacy .json)")
    parser.add_argument("--set", "-s", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="Override a config value, e.g. --set blend.alpha_end=0.6 (repeatable)")
    parser.add_argument("--config-preset", choices=list(CONFIG_PRESETS), help="Apply a named config preset before --set")
    parser.add_argument("--subjects", help="Subject image or folder (overrides paths.subjects)")
    parser.add_argument("--backgrounds", help="Background image or folder (overrides paths.backgrounds)")
    parser.add_argument("--out", help="Output root (overrides paths.out_dir)")
    parser.add_argument("--run-name", help="Fixed run folder name (reuse it with output.skip_existing=true to resume)")
    parser.add_argument("--preset", choices=list(VARIANT_PRESETS), help="sweep: named variant set (ablation)")
    parser.add_argument("--grid", action="append", metavar="KEY=V1,V2", help="sweep: grid axis, repeatable")
    parser.add_argument("--variants", metavar="FILE", help="sweep: text file, one 'label: key=value, ...' per line")
    parser.add_argument("--legacy-json", help="import-legacy: input ui_config_*.json")
    parser.add_argument("--save", help="import-legacy/show-config: write the YAML here")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "list-presets":
        print("Config presets (--config-preset):")
        for name, overrides in CONFIG_PRESETS.items():
            print(f"  {name!r}: {' '.join(overrides) or '(launch config)'}")
        print("\nVariant presets (sweep --preset):")
        for key, (title, variants) in VARIANT_PRESETS.items():
            print(f"  {key:<12} {title}\n" + "\n".join("      " + line for line in variants_to_text(variants).splitlines()))
        return 0

    if args.command == "import-legacy":
        if not args.legacy_json:
            raise SystemExit("--legacy-json is required")
        cfg = validate(import_legacy_ui_json(args.legacy_json))
        target = args.save or os.path.splitext(args.legacy_json)[0] + ".yaml"
        print(f"Saved {save_config(cfg, target)} (layout converted to the geometry the old notebook rendered)")
        return 0

    cfg = load_config(args.config)
    if args.config_preset:
        apply_overrides(cfg, CONFIG_PRESETS[args.config_preset])
    apply_overrides(cfg, args.overrides)
    for key, value in (("paths.subjects", args.subjects), ("paths.backgrounds", args.backgrounds), ("paths.out_dir", args.out)):
        if value:
            set_value(cfg, key, value)
    if cfg.paths.out_dir.startswith("/kaggle/") and not IS_KAGGLE:
        print("[config] Not running on Kaggle: paths.out_dir -> ./outputs")
        cfg.paths.out_dir = "./outputs"
    validate(cfg)

    if args.command == "show-config":
        print(yaml.safe_dump(cfg.to_dict(), sort_keys=False, allow_unicode=True))
        if args.save:
            save_config(cfg, args.save)
        return 0
    os.makedirs(cfg.paths.out_dir, exist_ok=True)

    if args.command == "ui":
        from .ui import launch

        launch(cfg)
        return 0

    from .runner import Session, all_pairs, new_run_dir, pick_seed, run_batch, run_pair, run_variants

    try:
        pairs = all_pairs(cfg)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))

    if args.command == "preview":
        from .imageio import labeled_grid
        from .preprocess import Preprocessor
        from .runner import stage_images

        pre = Preprocessor(cfg)
        run_dir = new_run_dir(cfg, "preview", args.run_name)
        subjects = list(dict.fromkeys(s for s, _ in pairs))
        for index, subject in enumerate(subjects):
            prepared = pre.prepare(cfg, subject, pairs[0][1], pick_seed(cfg, index))
            stages = stage_images(prepared)
            path = os.path.join(run_dir, f"{index:04d}_{os.path.splitext(os.path.basename(subject))[0]}.png")
            labeled_grid([(img, name) for name, img in stages], columns=len(stages), tile=192).save(path)
            print(f"[preview] {path} (softedge={prepared.softedge_source}, mask mean={prepared.mask.mask.mean():.3f})")
        return 0

    session = Session(cfg)
    if args.command == "run":
        run_dir = new_run_dir(cfg, "run", args.run_name)
        result = run_pair(session, cfg, pairs[0][0], pairs[0][1], pick_seed(cfg, 0), out_dir=run_dir)
        print(f"[run] {os.path.join(run_dir, 'output.png')} ({result.seconds:.1f}s)")
        for key, value in result.metrics.items():
            print(f"    {key:>28}: {value:.4f}" if isinstance(value, float) else f"    {key:>28}: {value}")
    elif args.command == "batch":
        summary = run_batch(session, cfg, pairs, run_dir=new_run_dir(cfg, "batch", args.run_name))
        print({k: summary.get(k) for k in ("run_dir", "succeeded", "failed", "csv", "grid", "zip")})
    elif args.command == "sweep":
        if args.preset:
            label, variants = args.preset, VARIANT_PRESETS[args.preset][1]
        elif args.variants:
            with open(args.variants, encoding="utf-8") as f:
                label, variants = "variants", parse_variants(f.read())
        else:
            grid = _parse_grid(args.grid) or cfg.sweep
            if not grid:
                raise SystemExit("sweep needs --preset, --grid, --variants or a `sweep:` section in the config")
            label, variants = "sweep", grid_to_variants(grid)
        summary = run_variants(session, cfg, variants, pairs, run_dir=new_run_dir(cfg, f"sweep_{label}", args.run_name))
        for row in summary["summary"]:
            print(row)
        print({k: summary.get(k) for k in ("run_dir", "succeeded", "failed", "summary_csv", "comparison", "zip")})
    return 0


if __name__ == "__main__":
    sys.exit(main())
