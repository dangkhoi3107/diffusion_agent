"""GPU render endpoint for the local agent demo (runs on Kaggle).

    bash run.sh render_server --token SECRET            # prints RENDER_URL=https://xxxx.gradio.live

API (gradio_client): /render(token, subject_png, background_png, label, overrides_json) streams progress strings
"step i/n" and finally a JSON status plus the output image. The caller sends the (already normalized) input images,
so the server needs no dataset. Exits after --idle minutes without requests (saves GPU quota).
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import tempfile
import threading
import time
from typing import Iterator, Optional, Tuple

from .config import IS_KAGGLE, apply_overrides, load_config

LAST_REQUEST = time.time()


def build(cfg_path: Optional[str], token: str):
    import gradio as gr

    from .runner import Session, run_pair

    base = load_config(cfg_path)
    session = Session(base)
    print("[render_server] loading models before publishing the URL ...", flush=True)
    session.generator  # noqa: B018 - warm up so the first demo render takes ~10 s, not ~90 s
    session.clip(base)
    lock = threading.Lock()  # one render at a time on the GPU

    def render(tok: str, subject: str, background: str, label: str, overrides: str) -> Iterator[Tuple[str, Optional[str]]]:
        global LAST_REQUEST
        LAST_REQUEST = time.time()
        if tok != token:
            yield json.dumps({"ok": False, "error": "bad token"}), None
            return
        cfg = base.copy()
        cfg.prompt.subject_name = label
        apply_overrides(cfg, json.loads(overrides or "[]"))
        out_dir = tempfile.mkdtemp(prefix="render_")
        events: "queue.Queue" = queue.Queue()

        def work():
            try:
                with lock:
                    events.put(("progress", "loaded"))
                    result = run_pair(session, cfg, subject, background, cfg.sampling.seed, out_dir=out_dir,
                                      progress=lambda i, n: events.put(("progress", f"step {i}/{n}")))
                events.put(("done", result))
            except Exception as exc:  # report, keep the server alive
                events.put(("error", f"{type(exc).__name__}: {exc}"))

        threading.Thread(target=work, daemon=True).start()
        while True:
            kind, value = events.get()
            if kind == "progress":
                yield value, None
            elif kind == "error":
                yield json.dumps({"ok": False, "error": value}), None
                return
            else:
                status = {"ok": True, "seconds": round(value.seconds, 2), "steps_run": value.steps_run,
                          "prompt": value.prompt, "metrics": value.metrics}
                yield json.dumps(status), os.path.join(out_dir, "output.png")
                return

    with gr.Blocks() as app:
        inputs = [gr.Textbox(), gr.Image(type="filepath"), gr.Image(type="filepath"), gr.Textbox(), gr.Textbox()]
        outputs = [gr.Textbox(), gr.Image(type="filepath")]
        gr.Button("render").click(render, inputs, outputs, api_name="render")
    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--token", required=True)
    ap.add_argument("--config", default=os.path.join(root, "configs", "default.yaml"))
    ap.add_argument("--idle", type=float, default=40.0, help="minutes without requests before exiting")
    ap.add_argument("--max-hours", type=float, default=4.0)
    ap.add_argument("--port", type=int, default=7862)
    args = ap.parse_args()
    app = build(args.config, args.token)
    app.queue()
    _, local_url, share_url = app.launch(server_name="0.0.0.0", server_port=args.port, share=True,
                                         prevent_thread_lock=True)
    print(f"RENDER_URL={share_url or local_url}", flush=True)
    started = time.time()
    while time.time() - LAST_REQUEST < args.idle * 60 and time.time() - started < args.max_hours * 3600:
        time.sleep(20)
    print("[render_server] idle/max time reached, exiting", flush=True)
    app.close()


if __name__ == "__main__":
    main()
