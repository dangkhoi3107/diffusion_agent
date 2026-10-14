"""Contact sheets for checking a rendered label-set batch by eye: one block per pair, a row per seed, the grid's
configs as columns (C00 first, then weak -> strong strength at each bg_strength), plus the background and subject.

    python3 scripts/batch_sheets.py outputs/label_b1v3            # -> outputs/label_b1v3/sheets/sheet_01.png ...
    python3 scripts/batch_sheets.py outputs/label_b1v3 --per 6 --thumb 112
"""
from __future__ import annotations

import argparse
import os
import sys

from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from camo.agent import ACTION_SPACE, SEEDS  # noqa: E402


def columns() -> list:
    """C00, then for each bg_strength the strength levels weak -> strong (ids without the seed suffix)."""
    cols = ["C00"]
    for bg in ACTION_SPACE["sampling.bg_strength"]:
        cols += [f"{s}_b{round(bg * 100):02d}" for s in ACTION_SPACE["subject.strength"]]
    return cols


def sheets(run: str, per: int = 7, thumb: int = 96) -> list:
    runs_dir = os.path.join(run, "runs")
    pairs = sorted(p for p in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, p)))
    cols, seeds = columns(), list(SEEDS)
    label_w, gap = 120, 6
    width = label_w + (len(cols) + 2) * thumb + gap
    out_dir = os.path.join(run, "sheets")
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for n in range(0, len(pairs), per):
        chunk = pairs[n:n + per]
        block = len(seeds) * thumb + gap * 2
        sheet = Image.new("RGB", (width, 18 + len(chunk) * block), "white")
        d = ImageDraw.Draw(sheet)
        for j, c in enumerate(["bg", "subject"] + cols):
            d.text((label_w + j * thumb + 2, 3), c, fill="black")
        for i, pair in enumerate(chunk):
            y0 = 18 + i * block
            d.text((4, y0 + 4), pair, fill="black")
            bg_id, subj_id = pair[:3], pair[3:]
            for j, name in enumerate((bg_id, subj_id)):
                f = os.path.join(run, "inputs_normalized", f"{name}.png")
                if os.path.exists(f):
                    sheet.paste(Image.open(f).convert("RGB").resize((thumb, thumb)), (label_w + j * thumb, y0))
            for r, seed in enumerate(seeds):
                suffix = "" if seed == 1 else f"_s{seed}"
                d.text((4, y0 + r * thumb + thumb // 2), f"seed {seed}", fill="gray")
                for j, cid in enumerate(cols):
                    f = os.path.join(runs_dir, pair, cid + suffix, "output.png")
                    x, y = label_w + (j + 2) * thumb, y0 + r * thumb
                    if os.path.exists(f):
                        sheet.paste(Image.open(f).convert("RGB").resize((thumb, thumb)), (x, y))
                    else:
                        d.rectangle([x, y, x + thumb - 1, y + thumb - 1], outline="red")
        path = os.path.join(out_dir, f"sheet_{n // per + 1:02d}.png")
        sheet.save(path)
        paths.append(path)
    return paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run")
    ap.add_argument("--per", type=int, default=7, help="pairs per sheet")
    ap.add_argument("--thumb", type=int, default=96)
    a = ap.parse_args()
    for p in sheets(a.run, a.per, a.thumb):
        print(p)
