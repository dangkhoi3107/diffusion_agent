"""Does a rendered config grid actually change the image? Per render: mean |config - C00| over the whole image and
inside the subject mask (0-255). Run before mass rendering a new grid (the v1 grid moved pixels < 1/255).

    python3 scripts/grid_check.py outputs/label_b1
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
from PIL import Image


def diffs(run: str) -> list:
    rows = []
    for pdir in sorted(glob.glob(os.path.join(run, "runs", "*"))):
        c00 = os.path.join(pdir, "C00", "output.png")
        if not os.path.exists(c00):
            continue
        base = np.asarray(Image.open(c00).convert("RGB"), np.float32)
        mpath = next(iter(sorted(glob.glob(os.path.join(pdir, "C00", "mask*final*.png")))), None)
        mask = (np.asarray(Image.open(mpath).convert("L").resize(base.shape[1::-1])) > 127) if mpath else None
        for cdir in sorted(glob.glob(os.path.join(pdir, "*"))):
            cid, out = os.path.basename(cdir), os.path.join(cdir, "output.png")
            if cid == "C00" or not os.path.exists(out):
                continue
            d = np.abs(np.asarray(Image.open(out).convert("RGB"), np.float32) - base).mean(-1)
            inside = float(d[mask].mean()) if mask is not None and mask.any() else None
            rows.append((os.path.basename(pdir), cid, float(d.mean()), inside))
    return rows


def main(run: str) -> int:
    rows = diffs(run)
    if not rows:
        sys.exit(f"no renders under {run}/runs")
    whole = np.array([r[2] for r in rows])
    inside = np.array([r[3] for r in rows if r[3] is not None])
    print(f"pairs {len({r[0] for r in rows})}, renders {len(rows)}")
    print(f"whole-image diff vs C00: median {np.median(whole):.1f}, under 2/255: {(whole < 2).mean():.0%}")
    if len(inside):
        print(f"in-mask diff vs C00: median {np.median(inside):.1f}, under 2/255: {(inside < 2).mean():.0%}")
    by = {}
    for _p, cid, w, m in rows:
        by.setdefault(cid, []).append(m if m is not None else w)
    for cid in sorted(by):
        print(f"  {cid}: in-mask median {np.median(by[cid]):.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "outputs/label_b1"))
