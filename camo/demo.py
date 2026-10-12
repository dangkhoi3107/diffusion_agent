"""Live demo of the two-agent loop: Proposer suggests -> GPU renders -> Observer (or you) labels -> repeat.

Every step streams into the "Agent trace" panel: LLM tokens as they arrive, render progress (step i/n), the two
Observer passes, the label and the next decision.

    python -m camo.demo --simulate                                    # no GPU: cheap composite stand-in
    python -m camo.demo --render-url https://xxxx.gradio.live         # renders on the Kaggle GPU (render_server)
    python -m camo.demo --local-gpu                                   # renders on this machine's GPU

LLM roles need OPENAI_API_KEY in this machine's environment; rule/random + human work without it.
The render URL/token can also come from outputs/.render_url and outputs/.render_token (written by kaggle_run).
"""
from __future__ import annotations

import argparse
import html
import json
import os
import queue
import shutil
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
from PIL import Image

from . import agent
from .agent import Controller, LLMPolicy, RandomPolicy, RulePolicy
from .config import IS_KAGGLE, load_config
from .pilot import load_inputs, normalize

LABEL_TEXT = {"too_visible": "1 Quá rõ", "good": "2 Đạt", "subtle": "3 Hơi chìm", "missing": "4 Mất hẳn",
              "broken": "5 Lỗi", "unsure": "0 Không chắc"}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Emit = Callable[[Tuple[str, str]], None]


# ------------------------------------------------------------------
# Renderers: (subject, background, label, cfg, out_dir, emit) -> output path
# ------------------------------------------------------------------
class SimRenderer:
    name = "mô phỏng (không phải diffusion)"

    def __call__(self, subject, background, label, cfg, out_dir, emit):
        bg = np.asarray(Image.open(background).convert("RGB"), dtype=np.float32)
        sub = np.asarray(Image.open(subject).convert("L").resize(bg.shape[1::-1]), dtype=np.float32)
        gy, gx = np.gradient(sub)
        edges = np.clip(np.hypot(gx, gy) / 40.0, 0, 1)[..., None]
        for i in range(1, 6):
            time.sleep(0.15)
            emit(("progress", f"step {i}/5"))
        cfg = cfg or {"blend.alpha_end": 1.0, "sampling.bg_strength": 0.52}
        a = cfg["blend.alpha_end"] * (0.6 + cfg["sampling.bg_strength"])
        path = os.path.join(out_dir, "output.png")
        Image.fromarray(np.clip(bg * (1 - 0.55 * a * edges), 0, 255).astype(np.uint8)).save(path)
        return path, {"renderer": "simulate"}


class ReplayRenderer:
    """Real diffusion outputs rendered beforehand on the GPU (pilot/grid_v5.yaml covers the whole action space):
    a config is looked up instead of rendered, so a demo runs instantly and offline. cfg=None -> thesis default C00."""
    name = "ảnh diffusion thật, render sẵn trên GPU"

    def __init__(self, run_dir: str):
        self.run_dir = os.path.abspath(run_dir)

    def __call__(self, subject, background, label, cfg, out_dir, emit, pair_id: str = ""):
        cid = agent.grid_id(cfg) if cfg else "C00"
        src = os.path.join(self.run_dir, "runs", pair_id, cid, "output.png")
        if not os.path.exists(src):
            raise FileNotFoundError(f"{pair_id}/{cid} not in {self.run_dir}")
        for i in range(1, 30, 4):  # replay pacing so the trace shows progress like a live render
            time.sleep(0.05)
            emit(("progress", f"step {i}/29 (phát lại)"))
        path = os.path.join(out_dir, "output.png")
        shutil.copy(src, path)
        return path, {"renderer": "replay", "source": src}


class RemoteRenderer:
    name = "GPU Kaggle"

    def __init__(self, url: str, token: str):
        from gradio_client import Client

        self.client, self.token = Client(url, verbose=False), token

    def __call__(self, subject, background, label, cfg, out_dir, emit):
        from gradio_client import handle_file

        job = self.client.submit(self.token, handle_file(subject), handle_file(background), label,
                                 json.dumps(agent.overrides(cfg) if cfg else []), api_name="/render")
        status, image = None, None
        for status, image in job:
            if image is None and not status.startswith("{"):
                emit(("progress", status))
        status, image = job.result()
        info = json.loads(status)
        if not info.get("ok"):
            raise RuntimeError(info.get("error"))
        path = os.path.join(out_dir, "output.png")
        shutil.copy(image, path)
        return path, info


class LocalGPURenderer:
    name = "GPU máy này"

    def __init__(self, cfg_path: str):
        from .runner import Session

        self.base = load_config(cfg_path)
        self.session = Session(self.base)

    def __call__(self, subject, background, label, cfg, out_dir, emit):
        from .config import apply_overrides
        from .runner import run_pair

        run_cfg = self.base.copy()
        run_cfg.prompt.subject_name = label
        apply_overrides(run_cfg, agent.overrides(cfg) if cfg else [])
        r = run_pair(self.session, run_cfg, subject, background, run_cfg.sampling.seed, out_dir=out_dir,
                     progress=lambda i, n: emit(("progress", f"step {i}/{n}")))
        return os.path.join(out_dir, "output.png"), {"seconds": r.seconds, "metrics": r.metrics}


def pump(fn: Callable[[Emit], Any]) -> Iterator[Tuple[str, str]]:
    """Run a blocking call in a thread and yield its emitted messages live; the call's result is the return value."""
    q: "queue.Queue" = queue.Queue()
    box: Dict[str, Any] = {}

    def target():
        try:
            box["value"] = fn(q.put)
        except Exception as exc:  # surfaced in the trace by the caller
            box["error"] = exc
        finally:
            q.put(None)

    threading.Thread(target=target, daemon=True).start()
    while (msg := q.get()) is not None:
        yield msg
    if "error" in box:
        raise box["error"]
    return box.get("value")


# ------------------------------------------------------------------
class Demo:
    def __init__(self, inputs: List[str], out: str, renderer, default_pair: str = "frozen_inputs:P12"):
        self.default_pair = default_pair
        self.out = os.path.abspath(out)
        self.renderer = renderer
        self.pairs: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]] = {}
        for path in inputs:
            if not os.path.exists(path):
                continue
            data = load_inputs(path)
            tag = os.path.basename(os.path.dirname(path))
            inv = normalize(data, os.path.join(self.out, "_inputs", tag), 512)
            for pid, (s, b) in data["pairs"].items():
                if s in inv["items"] and b in inv["items"]:
                    self.pairs[f"{tag}:{pid}"] = (inv["items"][s], inv["items"][b])

    def render(self, pair: str, cfg, out_dir: str, emit):
        s, b = self.pairs[pair]
        args = (s["normalized_path"], b["normalized_path"], s["label"], cfg, out_dir, emit)
        if isinstance(self.renderer, ReplayRenderer):
            return self.renderer(*args, pair_id=pair.split(":", 1)[1])
        return self.renderer(*args)

    def choices(self) -> List[str]:
        return [f"{k} · {s['label']} + {b.get('material', b.get('label', ''))}" for k, (s, b) in self.pairs.items()]

    def policy(self, name: str, emit: Emit):
        if name == "llm":
            from .llm import openai_text

            return LLMPolicy(openai_text(on_delta=lambda d: emit(("delta", d))))
        return RandomPolicy() if name == "random" else RulePolicy()

    def observe(self, pair: str, output: str, emit: Emit) -> Tuple[str, str]:
        from .llm import Observer, openai_vision

        base = openai_vision(on_delta=lambda d: emit(("delta", d)))

        def see(prompt, images):
            emit(("phase", "👁️ Observer lượt A: chỉ xem ảnh, không biết tên subject" if "once" in prompt
                  else "👁️ Observer lượt B: có ảnh subject + nền, chọn nhãn"))
            return base(prompt, images)

        s, b = self.pairs[pair]
        label, reason, extra = Observer(see).judge(output, s["normalized_path"], b["normalized_path"], s["label"])
        return label, f"observer: {reason} (lượt A thấy: {extra.get('first_view')})"


def build(demo: Demo):
    import gradio as gr

    def board(state, trace: List[str]):
        if not state:
            return None, None, None, [], "Chọn cặp ảnh, agent rồi bấm **Bắt đầu**.", "", "\n\n".join(trace)
        search = Controller(state["log"]).load()
        gallery, rows = [], []
        for t in search.trials:
            cfg = ", ".join(f"{k.split('.')[-1]}={v}" for k, v in t.config.items())
            if t.output and os.path.exists(t.output):
                gallery.append((t.output, f"#{t.index} {cfg} → {LABEL_TEXT.get(t.label, 'chờ nhãn')}"))
            rows.append(f"<tr><td>{t.index}</td><td>{html.escape(cfg)}</td><td>{html.escape(t.proposed_by)}</td>"
                        f"<td>{html.escape(t.rationale)}</td><td><b>{LABEL_TEXT.get(t.label, '…')}</b></td>"
                        f"<td>{html.escape(t.note or '')}</td></tr>")
        last = next((t.output for t in reversed(search.trials) if t.output), None)
        sel = search.selected()
        status = (f"**{state['pair']}** · proposer `{state['proposer']}` · observer `{state['observer']}` · render "
                  f"{demo.renderer.name} · {len([t for t in search.trials if t.label])}/{search.budget} lượt · ")
        if search.done:
            status += f"✅ xong: chọn #{sel.index}" if sel else "⛔ hết ngân sách, chưa có ảnh đạt"
        elif search.pending and search.pending.output:
            status += "⏳ chờ bạn gán nhãn ảnh mới nhất"
        table = ("<table><tr><th>#</th><th>config</th><th>agent</th><th>lý do</th><th>nhãn</th><th>ghi chú</th></tr>"
                 + "".join(rows) + "</table>")
        before = state.get("before")
        after = sel.output if sel else None
        return last, before, after, gallery, status, table, "\n\n".join(trace)

    def run(state, trace):
        """Generator: streams until the search is done or waits for a human label."""
        if not state:
            yield (state, trace, *board(state, trace))
            return
        ctl = Controller(state["log"])
        while True:
            search = ctl.load()
            if search.done:
                sel = search.selected()
                trace.append(f"🏁 **Kết thúc**: {'chọn #%d' % sel.index if sel else 'không có ảnh đạt trong ngân sách'}")
                yield (state, trace, *board(state, trace))
                return
            if search.pending and search.pending.output and search.pending.label is None:
                trace.append("⏳ **Chờ bạn gán nhãn** ảnh mới nhất (bấm nút bên phải)")
                yield (state, trace, *board(state, trace))
                return
            # 1. Proposer
            trace.append(f"🧠 **Proposer `{state['proposer']}`** đang chọn config tiếp theo…")
            yield (state, trace, *board(state, trace))
            stream = ""
            try:
                gen = pump(lambda emit: ctl.propose(search, demo.policy(state["proposer"], emit)))
                while True:
                    kind, text = next(gen)
                    stream += text
                    trace[-1] = f"🧠 **Proposer `{state['proposer']}`** đang suy nghĩ…\n```\n{stream[-600:]}\n```"
                    yield (state, trace, *board(state, trace))
            except StopIteration as stop:
                trial = stop.value
            except Exception as exc:
                trace.append(f"❌ Proposer lỗi: `{exc}`")
                yield (state, trace, *board(state, trace))
                return
            cfg = ", ".join(f"{k.split('.')[-1]}={v}" for k, v in trial.config.items())
            trace[-1] = f"🧠 **Proposer** → #{trial.index}: `{cfg}`  \n_{html.escape(trial.rationale)}_"
            yield (state, trace, *board(state, trace))
            # 2. Render
            out_dir = os.path.join(state["dir"], f"t{trial.index:02d}")
            os.makedirs(out_dir, exist_ok=True)
            trace.append(f"🎨 **Render** ({demo.renderer.name})…")
            yield (state, trace, *board(state, trace))
            started = time.time()
            try:
                gen = pump(lambda emit: demo.render(state["pair"], trial.config, out_dir, emit))
                while True:
                    _, text = next(gen)
                    trace[-1] = f"🎨 **Render** ({demo.renderer.name}): {text}"
                    yield (state, trace, *board(state, trace))
            except StopIteration as stop:
                path, _info = stop.value
            except Exception as exc:
                trace.append(f"❌ Render lỗi: `{exc}`")
                yield (state, trace, *board(state, trace))
                return
            ctl.rendered(search, trial, path, round(time.time() - started, 2))
            trace[-1] = f"🎨 **Render xong** #{trial.index} trong {time.time() - started:.1f}s"
            yield (state, trace, *board(state, trace))
            # 3. Observer
            if state["observer"] != "llm":
                continue  # loop back: waits for the human label
            stream = ""
            trace.append("👁️ **Observer** …")
            try:
                gen = pump(lambda emit: demo.observe(state["pair"], path, emit))
                while True:
                    kind, text = next(gen)
                    if kind == "phase":
                        stream = ""
                        trace.append(text)
                    else:
                        stream += text
                        trace[-1] = trace[-1].split("\n```")[0] + f"\n```\n{stream[-600:]}\n```"
                    yield (state, trace, *board(state, trace))
            except StopIteration as stop:
                label, reason = stop.value
            except Exception as exc:
                trace.append(f"❌ Observer lỗi: `{exc}` → bạn gán nhãn tay")
                yield (state, trace, *board(state, trace))
                return
            ctl.labelled(search, trial, label, reason)
            trace.append(f"🏷️ **Nhãn #{trial.index}: {LABEL_TEXT[label]}** ({html.escape(reason)})")
            yield (state, trace, *board(state, trace))

    def start(pair, proposer, observer, budget):
        key = pair.split(" · ")[0]
        run_dir = os.path.join(demo.out, "searches", f"{key.replace(':', '_')}_{proposer}_{observer}_{datetime.now():%H%M%S}")
        state = {"log": os.path.join(run_dir, "events.jsonl"), "dir": run_dir, "pair": key, "proposer": proposer, "observer": observer}
        Controller(state["log"]).start(key, int(budget), RulePolicy() if proposer == "rule" else RandomPolicy() if proposer == "random" else LLMPolicy(lambda p: ""))
        trace = [f"▶️ **Bắt đầu** {key}: ngân sách {int(budget)} lượt, dừng khi Đạt",
                 f"📷 **Ảnh trước (mặc định luận văn, không agent)**: đang render ({demo.renderer.name})…"]
        yield (state, trace, *board(state, trace))
        s, b = demo.pairs[key]
        before_dir = os.path.join(run_dir, "before_default")
        os.makedirs(before_dir, exist_ok=True)
        try:
            gen = pump(lambda emit: demo.render(key, None, before_dir, emit))
            while True:
                _, text = next(gen)
                trace[-1] = f"📷 **Ảnh trước (mặc định luận văn)**: {text}"
                yield (state, trace, *board(state, trace))
        except StopIteration as stop:
            state["before"] = stop.value[0]
            trace[-1] = "📷 **Ảnh trước (mặc định luận văn)** xong. Giờ agent bắt đầu tìm."
        except Exception as exc:
            trace[-1] = f"📷 Ảnh trước lỗi: `{exc}` (bỏ qua, agent vẫn chạy)"
        yield from run(state, trace)

    def label(state, trace, value, note):
        if state:
            ctl = Controller(state["log"])
            search = ctl.load()
            target = search.pending if search.pending and search.pending.output else (search.trials[-1] if search.trials else None)
            if target is not None:  # also overrules the observer's last label
                ctl.labelled(search, target, value, ("human: " + note) if note else "human")
                trace.append(f"🙋 **Bạn gán #{target.index}: {LABEL_TEXT[value]}**" + (f" ({html.escape(note)})" if note else ""))
        yield from run(state, trace)

    with gr.Blocks(title="CamouBlend agents") as app:
        gr.Markdown("## CamouBlend · 2 agent tìm config (streaming)\nProposer đề xuất → GPU render → Observer/bạn gán nhãn → lặp đến khi **Đạt** hoặc hết lượt.")
        state, trace = gr.State({}), gr.State([])
        with gr.Row():
            choices = demo.choices()
            default = next((c for c in choices if c.startswith(demo.default_pair)), choices[0] if choices else None)
            pair = gr.Dropdown(choices, value=default, label="Cặp ảnh", scale=3)
            proposer = gr.Radio(["rule", "random", "llm"], value="rule", label="Proposer")
            observer = gr.Radio(["human", "llm"], value="human", label="Observer")
            budget = gr.Slider(2, 10, value=5, step=1, label="Ngân sách")
        with gr.Row():
            b_start = gr.Button("▶ Bắt đầu", variant="primary")
            b_run = gr.Button("⏩ Chạy tiếp")
        status = gr.Markdown()
        with gr.Row():
            with gr.Column(scale=3):
                current = gr.Image(label="Ảnh mới nhất", height=460)
                with gr.Row():
                    before = gr.Image(label="TRƯỚC: mặc định luận văn", height=220)
                    after = gr.Image(label="SAU: ảnh agent chọn", height=220)
            with gr.Column(scale=3):
                live = gr.Markdown(label="Agent trace", height=460)
            with gr.Column(scale=2):
                note = gr.Textbox(label="Ghi chú (tuỳ chọn)")
                buttons = [(gr.Button(LABEL_TEXT[v]), v) for v in ("too_visible", "good", "subtle", "missing", "broken", "unsure")]
        gallery = gr.Gallery(label="Các lượt", columns=6, height=230)
        table = gr.HTML()
        outs = [state, trace, current, before, after, gallery, status, table, live]
        b_start.click(start, [pair, proposer, observer, budget], outs)
        b_run.click(run, [state, trace], outs)
        def on_label(value):
            def handler(st, tr, n):
                yield from label(st, tr, value, n)
            return handler

        for btn, value in buttons:
            btn.click(on_label(value), [state, trace, note], outs)
    return app


def _read(path: str) -> Optional[str]:
    return open(path).read().strip() if os.path.exists(path) else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--inputs", nargs="*", default=[os.path.join(ROOT, "pilot", "frozen_inputs", "inputs.yaml"),
                                                    os.path.join(ROOT, "pilot", "bgmap_inputs", "inputs.yaml")])
    ap.add_argument("--out", default="/kaggle/working/demo" if IS_KAGGLE else os.path.join(ROOT, "outputs", "demo"))
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--local-gpu", action="store_true")
    ap.add_argument("--replay", help="run dir with pre-rendered grid (e.g. outputs/grid_v5): real images, instant, no GPU")
    ap.add_argument("--render-url", default=os.environ.get("CAMO_RENDER_URL") or _read(os.path.join(ROOT, "outputs", ".render_url")))
    ap.add_argument("--render-token", default=os.environ.get("CAMO_RENDER_TOKEN") or _read(os.path.join(ROOT, "outputs", ".render_token")))
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--pair", default="frozen_inputs:P12", help="pair selected when the page opens")
    ap.add_argument("--port", type=int, default=7861)
    args = ap.parse_args()
    if args.replay:
        renderer = ReplayRenderer(args.replay)
        args.inputs = [os.path.join(ROOT, "pilot", "frozen_inputs", "inputs.yaml")]  # the grid's pairs
    elif args.local_gpu:
        renderer = LocalGPURenderer(args.config)
    elif args.simulate or not args.render_url:
        if not args.simulate:
            print("[demo] no --render-url: using the simulated renderer")
        renderer = SimRenderer()
    else:
        renderer = RemoteRenderer(args.render_url, args.render_token or "")
    demo = Demo(args.inputs, args.out, renderer, args.pair)
    build(demo).queue().launch(server_name="0.0.0.0" if IS_KAGGLE else "127.0.0.1", server_port=args.port,
                               share=args.share or IS_KAGGLE)


if __name__ == "__main__":
    main()
