"""How well does the VLM Observer agree with your labels? Runs it on images you labelled and reports agreement.

    python3 scripts/eval_observer.py --limit 30                 # needs OPENAI_API_KEY; ~2 calls per image
    python3 scripts/eval_observer.py --limit 30 --few-shot 1    # + 1 author-labelled example per label (other pairs)
    python3 scripts/eval_observer.py --fake                     # offline dry run (no API), checks the plumbing

Observer labels are appended to outputs/labels.jsonl under rater "observer_llm" (or observer_fake), so the replay
script can also run fully automatic agents on them. The report goes to outputs/observer_eval.json: exact agreement,
"good vs not good" agreement, confusion matrix, false accepts (Observer says good, you did not).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from camo.agent import LABELS  # noqa: E402
from camo.llm import CALLS, Observer, fake_vision, openai_vision  # noqa: E402
import label_server  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rater", default="tac_gia")
    ap.add_argument("--runs", nargs="*", help="limit to these runs (default: all labelled)")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--few-shot", type=int, default=0, help="author examples per label, from other pairs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--zoom", action="store_true", help="add an enlarged crop at the subject location (rater knows where to look)")
    args = ap.parse_args()
    out = label_server.OUT
    items = {i["uid"]: i for i in label_server.collect(0.0, True)}
    mine = {}
    with open(os.path.join(out, "labels.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r["rater"] == args.rater and r["uid"] in items and r["label"] != "unsure":
                    mine[r["uid"]] = r["label"]
    pool = [u for u in mine if not args.runs or items[u]["run"] in args.runs]
    random.Random(args.seed).shuffle(pool)
    targets = pool[: args.limit]
    path = lambda rel: os.path.join(out, rel[len("files/"):])  # noqa: E731
    by_label = defaultdict(list)
    for u in pool:
        by_label[mine[u]].append(u)
    rater = "observer_fake" if args.fake else "observer_llm"
    see = fake_vision(lambda p: random.Random(p).choice(LABELS[:5])) if args.fake else openai_vision()

    rows = []
    for n, uid in enumerate(targets, 1):
        it = items[uid]
        examples = []
        for label in LABELS[:5]:  # few-shot from other pairs only: no answer leakage
            cands = [u for u in by_label[label] if items[u]["pair_id"] != it["pair_id"] and u != uid]
            examples += [(path(items[u]["out"]), label) for u in cands[: args.few_shot]]
        zoom = None
        if args.zoom:
            import tempfile

            from camo.llm import subject_zoom
            zoom = subject_zoom(path(it["out"]), path(it["subj"]), os.path.join(tempfile.gettempdir(), f"zoom_{n}.png"))
        label, reason, extra = Observer(see, examples).judge(path(it["out"]), path(it["subj"]), path(it["bg"]),
                                                              it["subject_label"] or "object", zoom)
        rows.append({"uid": uid, "human": mine[uid], "observer": label, "reason": reason, **extra})
        with open(os.path.join(out, "labels.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({"id": f"obs_{datetime.now(timezone.utc):%Y%m%d%H%M%S}_{n}", "created_utc": datetime.now(timezone.utc).isoformat(),
                                "rater": rater, "uid": uid, "run": it["run"], "trial_id": it["trial_id"], "pair_id": it["pair_id"],
                                "config_id": it["config_id"], "label": label, "note": reason, "few_shot": args.few_shot,
                                "schema": "quick_label_v1"}, ensure_ascii=False) + "\n")
        print(f"[{n}/{len(targets)}] {uid}: you={mine[uid]:<11} observer={label:<11} {reason[:70]}", flush=True)

    exact = sum(r["human"] == r["observer"] for r in rows)
    binary = sum((r["human"] == "good") == (r["observer"] == "good") for r in rows)
    false_accept = [r["uid"] for r in rows if r["observer"] == "good" and r["human"] != "good"]
    confusion = Counter(f"{r['human']}->{r['observer']}" for r in rows)
    report = {"rater": args.rater, "observer": rater, "n": len(rows), "few_shot": args.few_shot,
              "exact_agreement": round(exact / max(len(rows), 1), 3), "good_vs_not_agreement": round(binary / max(len(rows), 1), 3),
              "false_accepts": false_accept, "abstained": sum(r["observer"] == "unsure" for r in rows),
              "confusion": dict(confusion), "api_calls": len(CALLS),
              "tokens": {"input": sum(c.get("input_tokens") or 0 for c in CALLS), "output": sum(c.get("output_tokens") or 0 for c in CALLS)},
              "rows": rows}
    tag = f"_fs{args.few_shot}" + ("_zoom" if args.zoom else "") + ("_fake" if args.fake else "")
    report["detail"] = os.environ.get("CAMO_IMAGE_DETAIL", "high")
    with open(os.path.join(out, f"observer_eval{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    with open(os.path.join(out, "observer_eval.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: report[k] for k in ("n", "exact_agreement", "good_vs_not_agreement", "abstained", "api_calls", "tokens")}, indent=1))
    print(f"false accepts (observer good, you not): {len(false_accept)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
