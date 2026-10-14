"""Live demo of the two-agent loop: Proposer suggests -> GPU renders -> Observer (or you) labels -> repeat.

Every step streams into the "Agent trace" panel: LLM tokens as they arrive, render progress (step i/n), the two
Observer passes, the label and the next decision.

    python -m camo.demo --simulate                                    # no GPU: cheap composite stand-in
    python -m camo.demo --render-url https://xxxx.gradio.live         # renders on the Kaggle GPU (render_server)
    python -m camo.demo --local-gpu                                   # renders on this machine's GPU
    ~/.venvs/camo-ml/bin/python -m camo.demo --replay outputs/label_b1v3   # self-trained Model 1 + Model 2 on a
                                                                      # rendered, labelled label-set batch (no GPU)

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
        cfg = cfg or dict(agent.START)
        levels = agent.ACTION_SPACE["subject.strength"]  # faint -> strong
        a = (levels.index(cfg["subject.strength"]) + 1) / len(levels) * (0.6 + cfg["sampling.bg_strength"])
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

    def __call__(self, subject, background, label, cfg, out_dir, emit, extra: Optional[List[str]] = None):
        from gradio_client import handle_file

        ov = [o for o in (agent.overrides(cfg) if cfg else []) if not (extra and o.startswith("sampling.seed="))]
        job = self.client.submit(self.token, handle_file(subject), handle_file(background), label,
                                 json.dumps(ov + list(extra or [])), api_name="/render")
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


# ------------------------------------------------------------------
# A rendered label-set batch and the self-trained models (Model 1: Observer + Head C, Model 2: Proposer). torch is
# needed for the models: run the demo with ~/.venvs/camo-ml/bin/python.
# ------------------------------------------------------------------
class LabelSet:
    """outputs/label_b*: its pairs, normalized inputs and your labels (a render within 2/255 of the default render
    with the same seed inherits that label, as on the label page)."""

    def __init__(self, run_dir: str, rater: str = "tac_gia"):
        import yaml

        self.run_dir, self.run = os.path.abspath(run_dir), os.path.basename(os.path.abspath(run_dir))
        with open(os.path.join(self.run_dir, "trials_latest.json"), encoding="utf-8") as f:
            trials = [t for t in json.load(f) if t.get("status") == "ok"]
        with open(os.path.join(self.run_dir, "inputs.yaml"), encoding="utf-8") as f:
            inputs = yaml.safe_load(f) or {}
        split = {}
        manifest = os.path.join(ROOT, "pilot", "labelset", "manifest.json")
        if os.path.exists(manifest):
            with open(manifest, encoding="utf-8") as f:
                m = json.load(f)
            split = {k: v.get("split") for part in ("subjects", "backgrounds") for k, v in m[part].items()}
        norm = os.path.join(self.run_dir, "inputs_normalized")
        self.pairs: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]] = {}
        for t in trials:
            if t["pair_id"] in self.pairs:
                continue
            s_meta = (inputs.get("subjects") or {}).get(t["subject_id"], {})
            b_meta = (inputs.get("backgrounds") or {}).get(t["background_id"], {})
            test = "test" in (split.get(t["subject_id"]), split.get(t["background_id"]))
            self.pairs[t["pair_id"]] = (
                {"normalized_path": os.path.join(norm, f"{t['subject_id']}.png"), "label": t["subject_label"],
                 "species": s_meta.get("species") or t["subject_label"], "shape": s_meta.get("shape")},
                {"normalized_path": os.path.join(norm, f"{t['background_id']}.png"), "id": t["background_id"],
                 "material": b_meta.get("material", ""), "group": b_meta.get("group"), "test": test})
        self.labels: Dict[str, str] = {}
        path = os.path.join(ROOT, "outputs", "labels.jsonl")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        if r.get("rater") == rater:
                            self.labels[r["uid"]] = r["label"]

    def output(self, pair: str, cid: str) -> str:
        return os.path.join(self.run_dir, "runs", pair, cid, "output.png")

    def truth(self, pair: str, cid: str) -> Optional[str]:
        own = self.labels.get(f"{self.run}/{pair}_{cid}")
        if own or cid.startswith("C00"):
            return own
        base = agent.default_id(cid)
        if not (os.path.exists(self.output(pair, cid)) and os.path.exists(self.output(pair, base))):
            return None
        small = lambda p: np.asarray(Image.open(p).convert("RGB").resize((128, 128)), np.float32)  # noqa: E731
        if np.abs(small(self.output(pair, cid)) - small(self.output(pair, base))).mean() < 2.0:
            return self.labels.get(f"{self.run}/{pair}_{base}")
        return None


class Models:
    """Lazy-loaded self-trained models; one DINOv2 backbone shared by Head C and the Observer."""

    def __init__(self):
        self._observer = self._proposer = self._head = None
        self._bg: Dict[str, np.ndarray] = {}

    @property
    def observer(self):
        if self._observer is None:
            from .observer import NetObserver

            self._observer = NetObserver()
        return self._observer

    def recognize(self, subject_path: str) -> Dict[str, Any]:
        """Head C: species (or human) of the subject cutout; "unknown" below the trained threshold."""
        if self._head is None:
            meta = json.load(open(os.path.join(ROOT, "outputs", "recognizer", "meta.json"), encoding="utf-8"))
            head = dict(np.load(os.path.join(ROOT, "outputs", "recognizer", "head.npz")))
            self._head = (meta, head)
        meta, head = self._head
        x = self.observer.backbone.embed([subject_path])
        z = ((x - head["mean"]) / head["scale"]) @ head["W"].T + head["b"]
        z = np.exp(z - z.max(1, keepdims=True))
        p = (z / z.sum(1, keepdims=True))[0]
        top = np.argsort(-p)[:3]
        cls = meta["classes"][int(top[0])]
        return {"class": cls, "prompt": meta["prompt_name"].get(cls, cls), "prob": float(p[top[0]]),
                "unknown": bool(p[top[0]] < meta["tau"]), "top3": [(meta["classes"][int(i)], float(p[i])) for i in top]}

    def background_feature(self, path: str) -> np.ndarray:
        if path not in self._bg:
            from .backbone import FeatureCache

            key = os.path.relpath(path, os.path.join(ROOT, "outputs"))
            cache = FeatureCache(os.path.join(ROOT, "outputs", "observer", "feats_dinov2.npz"))
            self._bg[path] = cache.vec[key] if key in cache.vec else self.observer.backbone.embed([path])[0]
        return self._bg[path]

    def proposer(self, background: Callable[[str], np.ndarray]):
        from . import proposer as P

        if self._proposer is None:
            self._proposer = P.load(os.path.join(ROOT, "outputs", "proposer", "proposer.pt"))
        return P.NetPolicy(self._proposer, background)


class Named:
    """Stands in for a policy where only its name is logged (Controller.start)."""

    def __init__(self, name: str):
        self.name = name


class Catalog:
    """Every subject and background of the label set (pilot/labelset/manifest.json), normalized on demand exactly as
    camo.pilot does, so any subject x background pair can be rendered live (most of the 252 x 90 were never rendered)."""

    def __init__(self, out: str):
        with open(os.path.join(ROOT, "pilot", "labelset", "manifest.json"), encoding="utf-8") as f:
            m = json.load(f)
        self.subjects, self.backgrounds = m["subjects"], m["backgrounds"]
        self.dir = os.path.join(out, "_inputs", "labelset")
        os.makedirs(self.dir, exist_ok=True)

    def source(self, item_id: str) -> str:
        import glob

        hits = [p for p in sorted(glob.glob(os.path.join(ROOT, "pilot", "labelset", "*", "candidates", f"{item_id}.*")))
                if not p.endswith(".json")]
        if not hits:
            raise FileNotFoundError(f"{item_id}: no candidate image under pilot/labelset/*/candidates")
        return hits[0]

    def normalized(self, item_id: str) -> str:
        from .pilot import normalize_image

        path = os.path.join(self.dir, f"{item_id}.png")
        if not os.path.exists(path):
            img, _alpha, _info = normalize_image(self.source(item_id), 512, "pad" if item_id in self.subjects else "crop")
            img.save(path)
        return path

    def subject_choices(self) -> List[str]:
        return [f"{k} · {v.get('species') or v['label']} ({v.get('shape') or v.get('category')})" + (" · TEST" if v.get("split") == "test" else "")
                for k, v in sorted(self.subjects.items(), key=lambda kv: (kv[1].get("shape") or "", kv[0]))]

    def background_choices(self) -> List[str]:
        return [f"{k} · {v['material']} ({v['group']})" + (" · TEST" if v.get("split") == "test" else "")
                for k, v in sorted(self.backgrounds.items())]

    def pair(self, subject_id: str, background_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        s, b = self.subjects[subject_id], self.backgrounds[background_id]
        return ({"normalized_path": self.normalized(subject_id), "label": s["label"], "species": s.get("species") or s["label"],
                 "shape": s.get("shape")},
                {"normalized_path": self.normalized(background_id), "id": background_id, "material": b["material"],
                 "group": b["group"], "test": "test" in (s.get("split"), b.get("split"))})


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
    def __init__(self, inputs: List[str], out: str, renderer, default_pair: str = "frozen_inputs:P12",
                 labelset: Optional[LabelSet] = None, live_only: bool = False,
                 live_subject_id: Optional[str] = None, live_background_id: Optional[str] = None):
        self.default_pair = default_pair
        self.live_only = live_only
        self.live_subject_id = live_subject_id
        self.live_background_id = live_background_id
        self.out = os.path.abspath(out)
        self.renderer = renderer
        self.labelset = labelset
        self.models = Models()
        self.catalog = Catalog(self.out) if labelset is not None else None
        self.live_renderer = None  # RemoteRenderer, created at the first live render (the GPU server may start later)
        self.base_seed = 1234  # configs/default.yaml sampling.seed; main() sets it from the config
        self.pairs: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]] = {}
        if labelset is not None:
            self.pairs = {f"{labelset.run}:{pid}": v for pid, v in labelset.pairs.items()}
            inputs = []
        for path in inputs:
            if not os.path.exists(path):
                continue
            data = load_inputs(path)
            tag = os.path.basename(os.path.dirname(path))
            inv = normalize(data, os.path.join(self.out, "_inputs", tag), 512)
            for pid, (s, b) in data["pairs"].items():
                if s in inv["items"] and b in inv["items"]:
                    self.pairs[f"{tag}:{pid}"] = (inv["items"][s], inv["items"][b])

    def add_live_pair(self, subject_choice: str, background_choice: str) -> str:
        sid, bid = subject_choice.split(" · ")[0], background_choice.split(" · ")[0]
        key = f"live:{bid}{sid}"
        self.pairs[key] = self.catalog.pair(sid, bid)
        return key

    def renderer_for(self, pair: str):
        if not pair.startswith("live:"):
            return self.renderer
        if self.live_renderer is None:
            url, token = _read(os.path.join(ROOT, "outputs", ".render_url")), _read(os.path.join(ROOT, "outputs", ".render_token"))
            if not url:
                raise RuntimeError("chưa có server render GPU: lưu RENDER_URL bằng scripts/kaggle_run.py --serve-url …")
            self.live_renderer = RemoteRenderer(url, token or "")
        return self.live_renderer

    def render(self, pair: str, cfg, out_dir: str, emit):
        s, b = self.pairs[pair]
        if pair.startswith("live:"):  # the label set's per-pair seed: an already rendered pair gives the same image
            from .pilot import pair_seed

            pid = pair.split(":", 1)[1]
            seed = pair_seed(pid, (agent.SEEDS[(cfg or {}).get("seed", 1)] or self.base_seed) if cfg else self.base_seed)
            return self.renderer_for(pair)(s["normalized_path"], b["normalized_path"], s.get("prompt_label") or s["label"],
                                           cfg, out_dir, emit, extra=[f"sampling.seed={seed}", "sampling.seed_mode=fixed"])
        args = (s["normalized_path"], b["normalized_path"], s["label"], cfg, out_dir, emit)
        if isinstance(self.renderer, ReplayRenderer):
            return self.renderer(*args, pair_id=pair.split(":", 1)[1])
        return self.renderer(*args)

    def choices(self) -> List[str]:
        return [f"{k} · {s['label']} + {b.get('material', b.get('label', ''))}" + (" · TEST (chưa từng thấy)" if b.get("test") else "")
                for k, (s, b) in self.pairs.items()]

    def policy(self, name: str, emit: Emit):
        if name == "model2":
            return self.models.proposer(lambda pair_id: self.models.background_feature(self.pairs[pair_id][1]["normalized_path"]))
        if name == "llm":
            from .llm import openai_text

            return LLMPolicy(openai_text(on_delta=lambda d: emit(("delta", d))))
        return RandomPolicy() if name == "random" else RulePolicy()

    def truth(self, pair: str, cid: str) -> Optional[str]:
        if self.labelset is None or pair.startswith("live:"):  # a live pair has no labels of yours
            return None
        return self.labelset.truth(pair.split(":", 1)[1], cid)

    def observe_self_trained(self, pair: str, output: str, cid: str, who: str) -> Tuple[str, str]:
        """who = model1: the self-trained Observer judges, your label is shown beside it; labels: your label."""
        mine = self.truth(pair, cid)
        if who == "labels" and mine is None:
            raise RuntimeError(f"no label of yours for {pair} {cid}")
        s, b = self.pairs[pair]
        label, reason, extra = self.models.observer.judge(output, s["normalized_path"], b["normalized_path"], s["label"])
        probs = ", ".join(f"{LABEL_TEXT[c][2:]} {v:.0%}" for c, v in sorted(extra["probs"].items(), key=lambda kv: -kv[1])[:3])
        if who == "labels":  # your label decides; Model 1's opinion is shown beside it
            return mine, f"nhãn anh đã gán · Model 1 nghĩ: {LABEL_TEXT[label][2:]} ({probs})" + (" ✅" if label == mine else " ✗")
        agree = "" if mine is None else (" · ✅ khớp nhãn của anh" if mine == label else f" · nhãn của anh: {LABEL_TEXT[mine][2:]}")
        return label, f"Model 1: {probs}{agree}"

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

    def render_metrics_line(info: Dict[str, Any]) -> str:
        metrics = info.get("metrics") or {}
        names = (("SSIM_bg", "SSIM nền"), ("PSNR_bg", "PSNR nền"), ("S_edge", "Biên subject"),
                 ("hidden_score", "Điểm ẩn"), ("clip_text_gain", "CLIP text gain"),
                 ("clip_image_gain", "CLIP image gain"))
        parts = []
        for key, title in names:
            value = metrics.get(key)
            if isinstance(value, (int, float)) and np.isfinite(value):
                parts.append(f"{title} {value:.2f}" + (" dB" if key == "PSNR_bg" else ""))
        return " · ".join(parts) if parts else "chưa có metrics ảnh"

    def training_metrics() -> str:
        if demo.labelset is None:
            return ""
        try:
            with open(os.path.join(ROOT, "outputs", "observer", "report.json"), encoding="utf-8") as f:
                observer = json.load(f)["variants"]["dinov2_pair_mlp"]["test"]
            with open(os.path.join(ROOT, "outputs", "proposer", "report.json"), encoding="utf-8") as f:
                proposer = json.load(f)
            test, cv = proposer["test"], proposer["cv"]
        except (OSError, KeyError, ValueError):
            return "<p>Chưa đọc được báo cáo train local.</p>"
        pct = lambda value: f"{value * 100:.1f}%"
        def bar(label: str, value: float, color: str) -> str:
            return (f"<div style='display:flex;align-items:center;gap:8px;margin:7px 0;font-size:13px'>"
                    f"<span style='width:110px'>{html.escape(label)}</span>"
                    f"<span style='flex:1;height:13px;background:#e6eae1;border-radius:3px;overflow:hidden'>"
                    f"<span style='display:block;width:{100 * value:.1f}%;height:100%;background:{color}'></span></span>"
                    f"<b style='width:52px;text-align:right'>{pct(value)}</b></div>")
        return ("<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:12px 0'>"
                f"<div style='padding:16px;border:1px solid #d9ddd3;border-radius:8px'><b>Observer · test {observer['n']} ảnh</b><br>"
                f"{bar('Đúng 5 lớp', observer['exact'], '#4c6a37')}"
                f"{bar('Precision Đạt', observer['good_precision'], '#a35f1a')}"
                f"{bar('Recall Đạt', observer['good_recall'], '#a35f1a')}"
                f"{bar('F1 Lỗi', observer['broken_f1'], '#56636b')}</div>"
                f"<div style='padding:16px;border:1px solid #d9ddd3;border-radius:8px'><b>Proposer · test {test['net']['n']} cặp</b><br>"
                f"{bar('GRU success@5', test['net']['success'], '#4c6a37')}"
                f"{bar('Rule success@5', test['rule']['success'], '#a35f1a')}"
                f"Lượt render TB {test['net']['mean_renders']:.2f} (Rule {test['rule']['mean_renders']:.2f})</div>"
                f"<div style='padding:16px;border:1px solid #d9ddd3;border-radius:8px'><b>Hai agent tự chạy · CV {cv['net_model1_judge']['n']} cặp</b><br>"
                f"{bar('Success@5', cv['net_model1_judge']['success'], '#a3402f')}"
                f"Số test end-to-end chưa đo.<br>{pct(test['net']['success'])} bên trái dùng nhãn người để dừng.</div></div>")

    def board(state, trace: List[str]):
        if not state:
            return None, None, None, [], "Chọn con vật và nền rồi bấm **Bắt đầu**.", "", "\n\n".join(trace)
        search = Controller(state["log"]).load()
        gallery, rows = [], []
        for t in search.trials:
            cfg = ", ".join(f"{k.split('.')[-1]}={v}" for k, v in t.config.items())
            if t.output and os.path.exists(t.output):
                gallery.append((t.output, f"#{t.index} {cfg} → {LABEL_TEXT.get(t.label, 'chờ nhãn')}"))
            info = state.get("render_info", {}).get(str(t.index), {})
            rows.append(f"<tr><td>{t.index}</td><td>{html.escape(cfg)}</td><td>{html.escape(t.proposed_by)}</td>"
                        f"<td>{html.escape(t.rationale)}</td><td><b>{LABEL_TEXT.get(t.label, '…')}</b></td>"
                        f"<td>{html.escape(render_metrics_line(info))}</td><td>{html.escape(t.note or '')}</td></tr>")
        last = next((t.output for t in reversed(search.trials) if t.output), None)
        sel = search.selected()
        rname = RemoteRenderer.name if state["pair"].startswith("live:") else demo.renderer.name
        status = (f"**{state['pair']}** · proposer `{state['proposer']}` · observer `{state['observer']}` · render "
                  f"{rname} · {len([t for t in search.trials if t.label])}/{search.budget} lượt · ")
        rendered = [t for t in search.trials if t.seconds is not None]
        if rendered:
            status += f"{len(rendered)} ảnh agent / {sum(t.seconds for t in rendered):.1f}s GPU · "
            latest_info = state.get("render_info", {}).get(str(rendered[-1].index), {})
            status += f"metrics ảnh mới nhất: {render_metrics_line(latest_info)} · "
        if search.done:
            status += f"✅ Observer báo Đạt: chọn #{sel.index}" if sel else "⛔ hết ngân sách, Observer chưa báo Đạt"
        elif search.pending and search.pending.output:
            status += "⏳ chờ bạn gán nhãn ảnh mới nhất"
        table = ("<table><tr><th>#</th><th>config</th><th>agent</th><th>lý do</th><th>nhãn</th><th>metrics ảnh</th><th>ghi chú</th></tr>"
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
            rname = RemoteRenderer.name if state["pair"].startswith("live:") else demo.renderer.name
            trace.append(f"🎨 **Render** ({rname})…")
            yield (state, trace, *board(state, trace))
            started = time.time()
            try:
                gen = pump(lambda emit: demo.render(state["pair"], trial.config, out_dir, emit))
                while True:
                    _, text = next(gen)
                    trace[-1] = f"🎨 **Render** ({rname}): {text}"
                    yield (state, trace, *board(state, trace))
            except StopIteration as stop:
                path, _info = stop.value
            except Exception as exc:
                trace.append(f"❌ Render lỗi: `{exc}`")
                yield (state, trace, *board(state, trace))
                return
            ctl.rendered(search, trial, path, round(time.time() - started, 2))
            state.setdefault("render_info", {})[str(trial.index)] = _info
            with open(os.path.join(out_dir, "render_info.json"), "w", encoding="utf-8") as f:
                json.dump(_info, f, ensure_ascii=False, indent=2)
            trace[-1] = (f"🎨 **Render xong** #{trial.index} trong {time.time() - started:.1f}s  \n"
                         f"📏 {render_metrics_line(_info)}")
            yield (state, trace, *board(state, trace))
            # 3. Observer
            if state["observer"] in ("model1", "labels"):
                try:
                    label, reason = demo.observe_self_trained(state["pair"], path, agent.grid_id(trial.config), state["observer"])
                except Exception as exc:
                    trace.append(f"❌ Observer lỗi: `{exc}` → bạn gán nhãn tay")
                    yield (state, trace, *board(state, trace))
                    return
                ctl.labelled(search, trial, label, reason)
                who = "Model 1 (tự train)" if state["observer"] == "model1" else "Nhãn của anh"
                trace.append(f"👁️ **{who} chấm #{trial.index}: {LABEL_TEXT[label]}**  \n_{html.escape(reason)}_")
                yield (state, trace, *board(state, trace))
                continue
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

    def start(pair, proposer, observer, budget, live_subject=None, live_background=None):
        if demo.live_only and not (live_subject and live_background):
            raise gr.Error("Chọn cả con vật và nền để render thật trên Kaggle GPU.")
        live = bool(live_subject and live_background and demo.catalog is not None)
        key = demo.add_live_pair(live_subject, live_background) if live else pair.split(" · ")[0]
        if live and observer == "labels":
            observer = "model1"  # a new pair has no labels of yours: Model 1 judges, you can overrule on the right
        run_dir = os.path.join(demo.out, "searches", f"{key.replace(':', '_')}_{proposer}_{observer}_{datetime.now():%H%M%S}")
        state = {"log": os.path.join(run_dir, "events.jsonl"), "dir": run_dir, "pair": key, "proposer": proposer, "observer": observer}
        renderer_name = RemoteRenderer.name if live else demo.renderer.name
        Controller(state["log"]).start(key, int(budget), Named("net" if proposer == "model2" else proposer))
        trace = [f"▶️ **Bắt đầu** {key}: ngân sách {int(budget)} lượt, dừng khi Đạt"]
        s, b = demo.pairs[key]
        if demo.labelset is not None:
            trace.append("🔎 **Model 1 · Head C** đang nhận diện con vật…")
            yield (state, trace, *board(state, trace))
            try:
                r = demo.models.recognize(s["normalized_path"])
                top = ", ".join(f"{c} {p:.0%}" for c, p in r["top3"])
                trace[-1] = (f"🔎 **Model 1 · Head C nhận diện: {'unknown (hỏi lại người dùng)' if r['unknown'] else r['class']}** "
                             f"({top}) → prompt `a hidden {r['prompt']} silhouette…` · đáp án: {s.get('species')}")
                if live:  # the live render uses the recognised name; unknown -> the name the user gives (the true label here)
                    s["prompt_label"] = s["label"] if r["unknown"] else r["prompt"]
                    trace.append(f"✍️ Prompt dùng cho render: `{s['prompt_label']}`"
                                 + (" (Head C không chắc → tên do người dùng cung cấp)" if r["unknown"] else " (từ Head C)"))
            except Exception as exc:
                trace[-1] = f"🔎 Head C lỗi: `{exc}`"
        if live and observer == "model1":
            trace.append("ℹ️ Cặp mới chưa có nhãn của anh: **Model 1 chấm**, anh có thể ghi đè bằng các nút bên phải.")
        trace.append(f"📷 **Ảnh trước (mặc định luận văn, không agent)**: đang render ({renderer_name})…")
        yield (state, trace, *board(state, trace))
        before_dir = os.path.join(run_dir, "before_default")
        os.makedirs(before_dir, exist_ok=True)
        try:
            gen = pump(lambda emit: demo.render(key, None, before_dir, emit))
            while True:
                _, text = next(gen)
                trace[-1] = f"📷 **Ảnh trước (mặc định luận văn)**: {text}"
                yield (state, trace, *board(state, trace))
        except StopIteration as stop:
            state["before"], before_info = stop.value
            with open(os.path.join(before_dir, "render_info.json"), "w", encoding="utf-8") as f:
                json.dump(before_info, f, ensure_ascii=False, indent=2)
            trace[-1] = ("📷 **Ảnh trước (mặc định luận văn)** xong. Giờ agent bắt đầu tìm.  \n"
                         f"📏 {render_metrics_line(before_info)}")
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
        if demo.labelset is not None:
            gr.HTML(training_metrics())
            gr.Markdown("[Mở biểu đồ train đầy đủ ↗](http://localhost:8770/metrics_dashboard.html) · đường loss/return chỉ có cho Proposer; Observer chỉ lưu chỉ số CV/test.")
        state, trace = gr.State({}), gr.State([])
        with gr.Row():
            choices = demo.choices()
            default = next((c for c in choices if c.startswith(demo.default_pair)), choices[0] if choices else None)
            pair = gr.State(default) if demo.live_only else gr.Dropdown(choices, value=default, label="Cặp ảnh", scale=3)
            self_trained = demo.labelset is not None
            proposer = gr.Radio((["model2"] if self_trained else []) + ["rule", "random", "llm"],
                                value="model2" if self_trained else "rule", label="Proposer (model2 = Model 2 tự train)")
            observer = gr.Radio((["model1"] if demo.live_only else ["model1", "labels"] if self_trained else []) + ["human", "llm"],
                                value="model1" if self_trained else "human",
                                label="Observer (model1 = Model 1 tự train · labels = nhãn anh đã gán)")
            budget = gr.Slider(2, 10, value=5, step=1, label="Ngân sách")
        live_subject = live_background = None
        if demo.catalog is not None:
            selected_key = default.split(" · ")[0] if default else demo.default_pair
            selected_s, selected_b = demo.pairs[selected_key]
            sid = os.path.splitext(os.path.basename(selected_s["normalized_path"]))[0]
            bid = os.path.splitext(os.path.basename(selected_b["normalized_path"]))[0]
            sid = demo.live_subject_id or sid
            bid = demo.live_background_id or bid
            subject_choices, background_choices = demo.catalog.subject_choices(), demo.catalog.background_choices()
            default_subject = next((c for c in subject_choices if c.startswith(sid + " · ")), None)
            default_background = next((c for c in background_choices if c.startswith(bid + " · ")), None)
            with gr.Row():
                live_subject = gr.Dropdown(subject_choices, value=default_subject, label="GPU THẬT · con vật", scale=3)
                live_background = gr.Dropdown(background_choices, value=default_background, label="GPU THẬT · nền", scale=3)
            gr.Markdown("_Chọn con vật và nền. **Bắt đầu** sẽ tạo mọi ảnh mới trên Kaggle GPU; cặp này không phát lại ảnh cũ._" if demo.live_only
                        else "_Hai ô GPU THẬT đang chọn sẵn cặp dưới đây. Bấm **Bắt đầu** sẽ render ảnh mới trên Kaggle; xoá một trong hai ô nếu muốn phát lại ảnh có sẵn._")
        def input_paths(pair_choice, subject_choice=None, background_choice=None):
            if subject_choice and background_choice and demo.catalog is not None:
                sid, bid = subject_choice.split(" · ")[0], background_choice.split(" · ")[0]
                return demo.catalog.normalized(sid), demo.catalog.normalized(bid)
            if demo.live_only:
                return None, None
            key = pair_choice.split(" · ")[0] if pair_choice else demo.default_pair
            if key not in demo.pairs:
                return None, None
            s, b = demo.pairs[key]
            return s["normalized_path"], b["normalized_path"]

        initial_subject, initial_background = input_paths(default, default_subject, default_background) if live_subject else input_paths(default)
        mode_text = gr.Markdown("**🎨 GPU THẬT · Kaggle sẽ tạo ảnh mới**" if live_subject and default_subject and default_background
                                else "**Chọn đủ hai input để render thật**" if demo.live_only
                                else "**⚡ PHÁT LẠI · dùng ảnh đã render sẵn**")
        with gr.Row():
            input_subject = gr.Image(value=initial_subject, label="INPUT · con vật", height=250, interactive=False)
            input_background = gr.Image(value=initial_background, label="INPUT · nền", height=250, interactive=False)
        def preview(pair_choice, subject_choice=None, background_choice=None):
            s, b = input_paths(pair_choice, subject_choice, background_choice)
            mode = "**🎨 GPU THẬT · Kaggle sẽ tạo ảnh mới**" if subject_choice and background_choice else (
                "**Chọn đủ hai input để render thật**" if demo.live_only else "**⚡ PHÁT LẠI · dùng ảnh đã render sẵn**")
            return s, b, mode
        preview_inputs = [pair] + ([live_subject, live_background] if live_subject else [])
        for choice in (preview_inputs[1:] if demo.live_only else preview_inputs):
            choice.change(preview, preview_inputs, [input_subject, input_background, mode_text])
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
        gr.Markdown("_Metrics ảnh là chỉ số kỹ thuật (SSIM/PSNR nền, biên, độ ẩn, CLIP); không thay thế nhãn chất lượng của Observer/người._")
        table = gr.HTML()
        outs = [state, trace, current, before, after, gallery, status, table, live]
        b_start.click(start, [pair, proposer, observer, budget] + ([live_subject, live_background] if live_subject else []), outs)
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
    ap.add_argument("--labelset", help="label-set catalog for the trained models and live subject/background choices")
    ap.add_argument("--live-only", action="store_true", help="show only the live Kaggle GPU path; require both input selections")
    ap.add_argument("--live-subject-id", help="subject ID preselected in the live input dropdown")
    ap.add_argument("--live-background-id", help="background ID preselected in the live input dropdown")
    ap.add_argument("--render-url", default=os.environ.get("CAMO_RENDER_URL") or _read(os.path.join(ROOT, "outputs", ".render_url")))
    ap.add_argument("--render-token", default=os.environ.get("CAMO_RENDER_TOKEN") or _read(os.path.join(ROOT, "outputs", ".render_token")))
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--pair", default="frozen_inputs:P12", help="pair selected when the page opens")
    ap.add_argument("--port", type=int, default=7861)
    args = ap.parse_args()
    if args.labelset and args.replay:
        ap.error("choose --labelset or --replay, not both")
    if args.labelset:
        args.replay = args.labelset
    labelset = None
    if args.replay:
        renderer = ReplayRenderer(args.replay)
        if os.path.exists(os.path.join(args.replay, "trials_latest.json")) and os.path.basename(os.path.abspath(args.replay)).startswith("label_b"):
            labelset = LabelSet(args.replay)  # a label-set batch: its own pairs, your labels, self-trained models
        else:
            args.inputs = [os.path.join(ROOT, "pilot", "frozen_inputs", "inputs.yaml")]  # the grid's pairs
    elif args.local_gpu:
        renderer = LocalGPURenderer(args.config)
    elif args.simulate or not args.render_url:
        if not args.simulate:
            print("[demo] no --render-url: using the simulated renderer")
        renderer = SimRenderer()
    else:
        renderer = RemoteRenderer(args.render_url, args.render_token or "")
    if args.live_only and labelset is None:
        ap.error("--live-only needs a label-set --replay directory for the subject/background catalog and trained models")
    demo = Demo(args.inputs, args.out, renderer, args.pair, labelset, args.live_only,
                args.live_subject_id, args.live_background_id)
    demo.base_seed = int(load_config(args.config).sampling.seed)
    build(demo).queue().launch(server_name="0.0.0.0" if IS_KAGGLE else "127.0.0.1", server_port=args.port,
                               share=args.share or IS_KAGGLE)


if __name__ == "__main__":
    main()
