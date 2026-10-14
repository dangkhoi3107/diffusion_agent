"""Local labeling server: open the page, press a key per image, labels are written to outputs/labels.jsonl.

    python3 scripts/label_server.py            # http://localhost:8765
    python3 scripts/label_server.py --all      # also show configs that look identical to C00

Reads every outputs/<run>/trials_latest.json. Audit configs whose output differs from the same run's C00 by less
than --min-diff (mean |delta| in 0-255 levels) are skipped: they are visually the same image as C00. A seed-2 config
(id ending in _s2) is compared with C00_s2, the default render with the same seed.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.parse
from functools import lru_cache
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from camo.agent import default_id  # noqa: E402
OUT = os.path.join(ROOT, "outputs")
LABELS = os.path.join(OUT, "labels.jsonl")


@lru_cache(maxsize=None)
def _pixels(path: str) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB").resize((128, 128)), dtype=np.float32)


SET_ASIDE = ("stale_", "oldgrid_")  # renders kept for the record (old v1 grid), never to be labelled


def collect(min_diff: float, show_all: bool, runs=None) -> list:
    items = []
    for run in sorted(os.listdir(OUT)):
        index = os.path.join(OUT, run, "trials_latest.json")
        if not os.path.isfile(index) or (runs is not None and run not in runs) or run.startswith(SET_ASIDE):
            continue
        if runs is None and os.path.exists(os.path.join(OUT, run, "RETIRED")):  # an old grid: off the label page,
            continue                                                            # its labels still feed training
        with open(index, encoding="utf-8") as f:
            trials = [t for t in json.load(f) if t.get("status") == "ok"]
        bg_meta = {}
        inputs_yaml = os.path.join(OUT, run, "inputs.yaml")
        if os.path.isfile(inputs_yaml):
            import yaml

            with open(inputs_yaml, encoding="utf-8") as f:
                bg_meta = (yaml.safe_load(f) or {}).get("backgrounds") or {}
        for t in trials:
            rel = f"{run}/runs/{t['pair_id']}/{t['config_id']}/output.png"
            if not os.path.isfile(os.path.join(OUT, rel)):
                continue
            base_id = default_id(t["config_id"])
            base = os.path.join(OUT, run, "runs", t["pair_id"], base_id, "output.png")
            diff = None
            if t["config_id"] != base_id and os.path.isfile(base):
                diff = float(np.abs(_pixels(os.path.join(OUT, rel)) - _pixels(base)).mean())
                if diff < min_diff and not show_all:
                    continue
            items.append({**{k: t.get(k) for k in ("trial_id", "pair_id", "config_id", "subject_label", "category",
                                                     "subject_id", "background_id", "overrides")},
                          "run": run, "uid": f"{run}/{t['trial_id']}", "diff_vs_c00": diff, "base_id": base_id,
                          "bg_material": (bg_meta.get(t.get("background_id")) or {}).get("material"),
                          "bg_group": (bg_meta.get(t.get("background_id")) or {}).get("group"),
                          "out": "files/" + rel, "subj": f"files/{run}/inputs_normalized/{t['subject_id']}.png",
                          "bg": f"files/{run}/inputs_normalized/{t['background_id']}.png"})
    return items


class Handler(SimpleHTTPRequestHandler):
    min_diff, show_all = 2.0, False  # items are rescanned on every request: new runs appear after a page reload

    def log_message(self, *args):  # keep the terminal quiet
        pass

    def end_headers(self):  # the page changes while labelling goes on: never serve a cached copy
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path in ("/", "/index.html"):
            self.path = "/tools/label.html"
            return SimpleHTTPRequestHandler.do_GET(self)
        if path == "/api/items":
            return self._json(collect(self.min_diff, self.show_all))
        if path in ("/compare", "/compare/", "/backgrounds", "/backgrounds/"):
            # rebuilt on every visit so the numbers follow the latest labels
            script, page = ("replay_agents.py", "index.html") if path.startswith("/compare") else ("bg_report.py", "backgrounds.html")
            subprocess.run([sys.executable, os.path.join(ROOT, "scripts", script)], cwd=ROOT, capture_output=True, timeout=300)
            if not os.path.exists(os.path.join(OUT, "compare", page)):
                return self.send_error(404, "report not available yet (run not rendered?)")
            self.path = "/outputs/compare/" + page
            return SimpleHTTPRequestHandler.do_GET(self)
        if path == "/api/status":  # written by scripts/labelset.py run (Kaggle batch progress)
            status = {}
            if os.path.exists(os.path.join(OUT, ".labelset_status.json")):
                with open(os.path.join(OUT, ".labelset_status.json"), encoding="utf-8") as f:
                    status = json.load(f)
            return self._json(status)
        if path == "/api/labels":
            recs = []
            if os.path.exists(LABELS):
                with open(LABELS, encoding="utf-8") as f:
                    recs = [json.loads(line) for line in f if line.strip()]
            return self._json(recs)
        if path.startswith("/files/"):
            rel = os.path.normpath(urllib.parse.unquote(path[len("/files/"):]))
            if rel.startswith("..") or not rel.endswith(".png"):
                return self.send_error(403)
            self.path = "/outputs/" + rel
        elif not path.startswith(("/tools/", "/outputs/compare/")):
            return self.send_error(404)
        return SimpleHTTPRequestHandler.do_GET(self)

    def do_POST(self):
        if urllib.parse.urlparse(self.path).path != "/api/label":
            return self.send_error(404)
        rec = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if not rec.get("uid") or not rec.get("label"):
            return self._json({"ok": False}, 400)
        with open(LABELS, "a", encoding="utf-8") as f:  # append-only; the newest record per (rater, uid) wins
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return self._json({"ok": True})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--min-diff", type=float, default=2.0)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    Handler.min_diff, Handler.show_all = args.min_diff, args.all
    items = collect(args.min_diff, args.all)
    runs = sorted({i["run"] for i in items})
    print(f"{len(items)} images to label from {', '.join(runs)} -> labels saved to {LABELS}", flush=True)
    print(f"Open http://localhost:{args.port}", flush=True)
    os.chdir(ROOT)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
