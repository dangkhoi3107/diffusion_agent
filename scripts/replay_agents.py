"""Before/after comparison: replay search agents on the labelled grid and build outputs/compare/index.html.

    python3 scripts/replay_agents.py                       # default grid_v5, rater tac_gia, budget 5
    python3 scripts/replay_agents.py --llm                 # also the OpenAI Proposer (needs OPENAI_API_KEY)

"Before" = the thesis default render (C00) of each pair. "After" = the image an agent selects within the budget.
Every config of the action space was rendered once (pilot/grid_v5.yaml) and labelled once in the quick-label UI;
agents are replayed on those labels, so all policies are compared on the same images and the same judgments.
A config that looks identical to C00 (mean |delta| < 2/255, hidden in the label UI) inherits C00's label.
A search that reaches an unlabelled config stops and is reported as "incomplete", never counted as success.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from camo.agent import Controller, LLMPolicy, RandomPolicy, RulePolicy, grid_id  # noqa: E402

OUT = os.path.join(ROOT, "outputs")
LABEL_VI = {"too_visible": "Quá rõ", "good": "Đạt", "subtle": "Hơi chìm", "missing": "Mất hẳn", "broken": "Lỗi",
            "unsure": "Không chắc", None: "chưa chấm"}


class Incomplete(Exception):
    pass


def load_labels(rater: str) -> Dict[str, dict]:
    latest: Dict[str, dict] = {}
    path = os.path.join(OUT, "labels.jsonl")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r["rater"] == rater:
                        latest[r["uid"]] = r
    return latest


def grid_labels(run: str, rater: str, inherit_below: float = 2.0) -> Tuple[Dict[Tuple[str, str], Tuple[Optional[str], str]], dict]:
    """(pair, config_id) -> (label, source) with source own | inherited_from_C00 | none."""
    from label_server import collect

    labels = load_labels(rater)
    items = collect(0.0, True, runs={run})  # explicit run: works for retired runs too
    out: Dict[Tuple[str, str], Tuple[Optional[str], str]] = {}
    for i in items:
        own = labels.get(i["uid"])
        out[(i["pair_id"], i["config_id"])] = (own["label"], "own") if own else (None, "none")
    for i in items:
        key = (i["pair_id"], i["config_id"])
        if out[key][0] is None and i["diff_vs_c00"] is not None and i["diff_vs_c00"] < inherit_below:
            base = out.get((i["pair_id"], i.get("base_id", "C00")), (None, "none"))  # C00 of the same seed
            if base[0]:
                out[key] = (base[0], "inherited_from_C00")
    meta = {i["pair_id"]: {"subject": i["subject_label"], "background": i["background_id"]} for i in items}
    return out, meta


def replay(pair: str, policy, budget: int, table) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        ctl = Controller(os.path.join(tmp, "events.jsonl"))
        search = ctl.start(pair, budget, policy)

        def evaluate(trial):
            label, _src = table.get((pair, grid_id(trial.config)), (None, "none"))
            if label is None:
                raise Incomplete(grid_id(trial.config))
            return label

        try:
            ctl.run(search, policy, lambda cfg: grid_id(cfg), evaluate)
            status = "done"
        except Incomplete as missing:
            status = f"incomplete (cần nhãn {missing})"
        sel = search.selected()
        return {"status": status, "selected": grid_id(sel.config) if sel else None,
                "renders": len([t for t in search.trials if t.label]),
                "path": [(grid_id(t.config), t.label) for t in search.trials if t.label]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default="grid_v5")
    ap.add_argument("--rater", default="tac_gia")
    ap.add_argument("--budget", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=5, help="random policy repeats")
    ap.add_argument("--llm", action="store_true")
    args = ap.parse_args()
    if not os.path.isdir(os.path.join(OUT, args.run)):
        sys.exit(f"outputs/{args.run} not found (render the grid first)")
    table, meta = grid_labels(args.run, args.rater)
    pairs = sorted(meta)
    policies = [("rule", lambda: RulePolicy())] + [(f"random#{s}", lambda s=s: RandomPolicy(seed=s)) for s in range(args.seeds)]
    if args.llm:
        from camo.llm import openai_text
        policies.append(("llm", lambda: LLMPolicy(openai_text())))

    results: Dict[str, Dict[str, dict]] = defaultdict(dict)
    for name, make in policies:
        for pair in pairs:
            results[name][pair] = replay(pair, make(), args.budget, table)

    def summary(name_prefix: str) -> dict:
        rows = [r for n, per in results.items() if n.split("#")[0] == name_prefix for r in per.values()]
        done = [r for r in rows if r["status"] == "done"]
        return {"runs": len(rows), "complete": len(done), "success": sum(r["selected"] is not None for r in done),
                "mean_renders": round(sum(r["renders"] for r in done) / max(len(done), 1), 2)}

    before = Counter(table.get((p, "C00"), (None,))[0] for p in pairs)
    report = {"run": args.run, "rater": args.rater, "budget": args.budget, "pairs": len(pairs),
              "before_default": {"good": before.get("good", 0), "labelled": sum(v for k, v in before.items() if k)},
              "labels": {"own": sum(1 for v in table.values() if v[1] == "own"),
                         "inherited": sum(1 for v in table.values() if v[1] == "inherited_from_C00"),
                         "missing": sum(1 for v in table.values() if v[0] is None), "total": len(table)},
              "policies": {n: summary(n) for n in dict.fromkeys(n.split("#")[0] for n in results)},
              "per_pair": results}
    os.makedirs(os.path.join(OUT, "compare"), exist_ok=True)
    with open(os.path.join(OUT, "compare", "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    write_html(report, meta, table, args.run)
    print(json.dumps({k: report[k] for k in ("before_default", "labels", "policies")}, ensure_ascii=False, indent=1))
    print("Open http://localhost:8765/compare (label server) or outputs/compare/index.html")
    return 0


def write_html(report: dict, meta: dict, table, run: str) -> None:
    def img(pair: str, cid: Optional[str]) -> str:
        if not cid:
            return "<div class='none'>không có ảnh đạt</div>"
        label = table.get((pair, cid), (None,))[0]
        return (f"<figure><img src='/files/{run}/runs/{pair}/{cid}/output.png' loading='lazy'>"
                f"<figcaption>{cid} · <b class='l-{label}'>{LABEL_VI.get(label)}</b></figcaption></figure>")

    pol = report["policies"]
    lab = report["labels"]
    rows = []
    for pair in sorted(meta):
        rule = report["per_pair"]["rule"][pair]
        rows.append(
            f"<tr><td><b>{pair}</b><br><small>{html.escape(str(meta[pair]['subject']))} + {meta[pair]['background']}</small></td>"
            f"<td>{img(pair, 'C00')}</td><td>{img(pair, rule['selected'])}"
            f"<small>{html.escape(rule['status'])} · {rule['renders']} lượt: "
            + " → ".join(f"{c} ({LABEL_VI.get(l)})" for c, l in rule["path"]) + "</small></td></tr>")
    policy_rows = "".join(f"<tr><td>{n}</td><td>{s['success']}/{s['complete']}</td><td>{s['runs'] - s['complete']}</td>"
                          f"<td>{s['mean_renders']}</td></tr>" for n, s in pol.items())
    page = f"""<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Trước và sau agent</title><style>
:root{{--bg:#f6f6f3;--panel:#fff;--ink:#1c1c1a;--muted:#6b6b65;--line:#dcdcd5;--good:#2f6f5e;--bad:#b91c1c;--mid:#a16207}}
@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#151514;--panel:#1f1f1d;--ink:#ecece6;--muted:#a3a39b;--line:#34342f;--good:#6fbfa6;--bad:#f87171;--mid:#facc15}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif}} main{{max-width:1100px;margin:0 auto;padding:16px}}
table{{border-collapse:collapse;width:100%;background:var(--panel)}} th,td{{border-bottom:1px solid var(--line);padding:8px;vertical-align:top;text-align:left}}
figure{{margin:0}} img{{width:100%;max-width:300px;border-radius:6px;display:block}} small,.none{{color:var(--muted)}}
.l-good{{color:var(--good)}} .l-missing,.l-broken{{color:var(--bad)}} .l-too_visible,.l-subtle{{color:var(--mid)}}
.cards{{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0}} .card{{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px 14px}}
.card b{{font-size:22px;display:block}}</style></head><body><main>
<h1>Trước và sau khi thêm agent · {run}</h1>
<p class="none">Nhãn của <b>{report['rater']}</b>; ngân sách {report['budget']} lượt render/cặp. Ảnh trùng mặc định thừa hưởng nhãn mặc định.
Nhãn: {lab['own']} tự chấm, {lab['inherited']} thừa hưởng, <b>{lab['missing']} còn thiếu</b> / {lab['total']}.</p>
<div class="cards"><div class="card">Trước (mặc định luận văn)<b>{report['before_default']['good']}/{report['before_default']['labelled']}</b>cặp đạt</div>
{''.join(f"<div class='card'>Sau · {n}<b>{s['success']}/{s['complete']}</b>cặp đạt · {s['mean_renders']} lượt TB</div>" for n, s in pol.items())}</div>
<table><tr><th>Agent</th><th>Đạt / chạy trọn</th><th>Thiếu nhãn</th><th>Lượt render TB</th></tr>{policy_rows}</table>
<h2>Từng cặp: trước (mặc định) và sau (agent rule chọn)</h2>
<table><tr><th>Cặp</th><th>Trước</th><th>Sau (rule)</th></tr>{''.join(rows)}</table></main></body></html>"""
    with open(os.path.join(OUT, "compare", "index.html"), "w", encoding="utf-8") as f:
        f.write(page)


if __name__ == "__main__":
    sys.exit(main())
