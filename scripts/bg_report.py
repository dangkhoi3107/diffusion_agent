"""Background map report: which background types work? -> outputs/compare/backgrounds.html (+ backgrounds.json)

    python3 scripts/bg_report.py                     # run bgmap_v1, labels of rater tac_gia
    open http://localhost:8765/backgrounds           # served by scripts/label_server.py

Groups the 47 DTD categories (camo/sources.py DTD_GROUPS), shows the 4 animals per background with your labels,
and the share of "good" per background and per group. Backgrounds flagged contains_animal_risk are listed apart.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
from collections import Counter, defaultdict

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from replay_agents import LABEL_VI, load_labels  # noqa: E402

OUT = os.path.join(ROOT, "outputs")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default="bgmap_v1")
    ap.add_argument("--rater", default="tac_gia")
    args = ap.parse_args()
    run_dir = os.path.join(OUT, args.run)
    with open(os.path.join(run_dir, "inputs.yaml"), encoding="utf-8") as f:
        inputs = yaml.safe_load(f)
    with open(os.path.join(run_dir, "trials_latest.json"), encoding="utf-8") as f:
        trials = [t for t in json.load(f) if t.get("status") == "ok"]
    labels = load_labels(args.rater)
    by_bg = defaultdict(list)
    for t in trials:
        lab = labels.get(f"{args.run}/{t['trial_id']}")
        by_bg[t["background_id"]].append((t, lab["label"] if lab else None))

    rows, groups = [], defaultdict(Counter)
    for bid, meta in inputs["backgrounds"].items():
        cells = sorted(by_bg.get(bid, []), key=lambda x: x[0]["subject_id"])
        c = Counter(l for _, l in cells if l)
        group = meta.get("group", "?")
        if not meta.get("contains_animal_risk"):
            groups[group].update(c)
        rows.append({"id": bid, "material": meta.get("material"), "group": group, "risk": meta.get("contains_animal_risk"),
                     "good": c.get("good", 0), "labelled": sum(c.values()), "counts": dict(c),
                     "cells": [(t["subject_label"], t["pair_id"], l) for t, l in cells]})
    summary = {g: {"good": c.get("good", 0), "labelled": sum(c.values()), "counts": dict(c)} for g, c in groups.items()}
    os.makedirs(os.path.join(OUT, "compare"), exist_ok=True)
    with open(os.path.join(OUT, "compare", "backgrounds.json"), "w", encoding="utf-8") as f:
        json.dump({"run": args.run, "rater": args.rater, "groups": summary, "backgrounds": rows}, f, ensure_ascii=False, indent=1)

    def cell(subject, pair, label):
        return (f"<figure><img loading='lazy' src='/files/{args.run}/runs/{pair}/C00/output.png'>"
                f"<figcaption>{html.escape(str(subject))} · <b class='l-{label}'>{LABEL_VI.get(label)}</b></figcaption></figure>")

    order = sorted(rows, key=lambda r: (r["group"], -r["good"], r["material"] or ""))
    body = []
    for group in sorted({r["group"] for r in rows}):
        s = summary.get(group, {"good": 0, "labelled": 0})
        body.append(f"<h2>{html.escape(group)} <small>{s['good']}/{s['labelled']} đạt</small></h2>")
        for r in [x for x in order if x["group"] == group]:
            flag = " ⚠️ nền có sẵn động vật/người" if r["risk"] else ""
            body.append(f"<div class='bg'><div class='name'><b>{r['id']}</b> {html.escape(str(r['material']))}{flag}<br>"
                        f"<small>{r['good']}/{r['labelled']} đạt</small></div>"
                        f"<img class='src' loading='lazy' src='/files/{args.run}/inputs_normalized/{r['id']}.png'>"
                        + "".join(cell(*c) for c in r["cells"]) + "</div>")
    total_good = sum(s["good"] for s in summary.values())
    total_lab = sum(s["labelled"] for s in summary.values())
    page = f"""<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bản đồ nền</title><style>
:root{{--bg:#f6f6f3;--panel:#fff;--ink:#1c1c1a;--muted:#6b6b65;--line:#dcdcd5;--good:#2f6f5e;--bad:#b91c1c;--mid:#a16207}}
@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#151514;--panel:#1f1f1d;--ink:#ecece6;--muted:#a3a39b;--line:#34342f;--good:#6fbfa6;--bad:#f87171;--mid:#facc15}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.4 system-ui,sans-serif}} main{{max-width:1200px;margin:0 auto;padding:16px}}
.bg{{display:grid;grid-template-columns:150px repeat(5,1fr);gap:8px;align-items:start;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:8px;margin:8px 0}}
@media (max-width:800px){{.bg{{grid-template-columns:1fr 1fr}}}}
img{{width:100%;border-radius:5px;display:block}} .src{{outline:2px dashed var(--line)}} figure{{margin:0}} figcaption,small{{color:var(--muted);font-size:12px}}
.l-good{{color:var(--good)}} .l-missing,.l-broken{{color:var(--bad)}} .l-too_visible,.l-subtle{{color:var(--mid)}} h2 small{{font-weight:400}}
</style></head><body><main><h1>Bản đồ nền · {args.run}</h1>
<p><small>47 loại texture DTD × 4 con vật, cấu hình mặc định. Nhãn của {html.escape(args.rater)}: {total_good}/{total_lab} đạt (không tính nền có sẵn động vật).
Ô viền đứt = ảnh nền gốc. Chấm thêm ở <a href="/">trang gán nhãn</a> rồi chạy lại <code>python3 scripts/bg_report.py</code>.</small></p>
{''.join(body)}</main></body></html>"""
    with open(os.path.join(OUT, "compare", "backgrounds.html"), "w", encoding="utf-8") as f:
        f.write(page)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print("Open http://localhost:8765/backgrounds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
