"""LLM adapters and the VLM Observer. Provider calls are plain functions so tests and demos can inject fakes.

    text  = openai_text()           # prompt -> reply (Proposer)
    image = openai_vision()         # (prompt, [image paths]) -> reply (Observer)

Key from the OPENAI_API_KEY environment variable, else from the file ~/.config/camo/openai_api_key (chmod 600)
so processes started outside your shell (the app, scripts) find it too. Never stored in configs or logs. Model per PRD §17 (overridable
with CAMO_LLM_MODEL). Every call is recorded (latency, token usage, request id) in `CALLS` for cost accounting.
"""
from __future__ import annotations

import base64
import json
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .agent import LABELS

DEFAULT_MODEL = os.environ.get("CAMO_LLM_MODEL", "gpt-5.4-mini-2026-03-17")
IMAGE_DETAIL = os.environ.get("CAMO_IMAGE_DETAIL", "high")
CALLS: List[Dict[str, Any]] = []

TextFn = Callable[[str], str]
VisionFn = Callable[[str, Sequence[str]], str]


def _client():
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError("pip install openai") from exc
    key = os.environ.get("OPENAI_API_KEY") or _key_from_file()
    if not key:
        raise RuntimeError("OpenAI key missing: set OPENAI_API_KEY or create ~/.config/camo/openai_api_key")
    return OpenAI(api_key=key)


KEY_FILE = os.path.expanduser("~/.config/camo/openai_api_key")


def _key_from_file() -> Optional[str]:
    try:
        with open(KEY_FILE, encoding="utf-8") as f:
            return f.read().strip() or None
    except OSError:
        return None


def has_key() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY") or _key_from_file())


def _data_url(path: str) -> str:
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def _call(model: str, content: List[Dict[str, Any]], role: str, on_delta: Optional[Callable[[str], None]] = None) -> str:
    """One Responses API call. With on_delta the reply is streamed token by token (shown live in the demo)."""
    client = _client()
    started = time.time()
    request = {"model": model, "input": [{"role": "user", "content": content}]}
    if os.environ.get("CAMO_REASONING"):  # PRD §17: "none" for the main study; left unset unless configured
        request["reasoning"] = {"effort": os.environ["CAMO_REASONING"]}
    usage, rid, text = None, None, ""
    if on_delta is None:
        resp = client.responses.create(**request)
        usage, rid, text = getattr(resp, "usage", None), getattr(resp, "id", None), resp.output_text
    else:
        parts: List[str] = []
        for event in client.responses.create(stream=True, **request):
            kind = getattr(event, "type", "")
            if kind == "response.output_text.delta":
                parts.append(event.delta)
                on_delta(event.delta)
            elif kind == "response.completed":
                usage, rid = getattr(event.response, "usage", None), getattr(event.response, "id", None)
        text = "".join(parts)
    CALLS.append({"role": role, "model": model, "seconds": round(time.time() - started, 3), "request_id": rid,
                  "input_tokens": getattr(usage, "input_tokens", None), "output_tokens": getattr(usage, "output_tokens", None)})
    return text


def openai_text(model: str = DEFAULT_MODEL, on_delta: Optional[Callable[[str], None]] = None) -> TextFn:
    return lambda prompt: _call(model, [{"type": "input_text", "text": prompt}], "proposer", on_delta)


def openai_vision(model: str = DEFAULT_MODEL, on_delta: Optional[Callable[[str], None]] = None) -> VisionFn:
    def run(prompt: str, images: Sequence[str]) -> str:
        # detail=high: camouflaged outlines are faint; a downscaled image loses exactly what the judge must see
        content = [{"type": "input_text", "text": prompt}] + [{"type": "input_image", "image_url": _data_url(p), "detail": IMAGE_DETAIL}
                                                               for p in images]
        return _call(model, content, "observer", on_delta)
    return run


def parse_json(text: str) -> Dict[str, Any]:
    return json.loads(re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M))


class Observer:
    """Two-pass image judge (PRD §16.4): pass A sees only the output (no target name) and says what it recognises;
    pass B sees output + subject + background + the expected label and returns one quick label. Unparseable replies
    abstain as 'unsure' instead of guessing."""

    def __init__(self, see: VisionFn, examples: Sequence[Tuple[str, str]] = ()):
        """examples: (image path, author label) pairs from OTHER pairs, shown in pass B only (few-shot calibration)."""
        self.see = see
        self.examples = list(examples)

    def first_view(self, output: str) -> str:
        reply = self.see(
            'Look at this image once. Is there a hidden object or face integrated into it? '
            'Reply JSON: {"sees_object": true/false, "what": "short noun or null"}', [output])
        try:
            data = parse_json(reply)
            return str(data.get("what")) if data.get("sees_object") else "nothing"
        except (ValueError, TypeError):
            return "unparsed"

    def judge(self, output: str, subject: str, background: str, target: str,
              zoom: Optional[str] = None) -> Tuple[str, str, Dict[str, Any]]:
        """zoom: optional enlarged crop of image 1 where the subject was placed (the human rater knows where to look)."""
        blind = self.first_view(output)
        reply = self.see(json.dumps({
            "task": f"Image 1 is a generated camouflage image that should contain a '{target}' blended into the "
                    "background (image 3), shaped like the subject reference (image 2). Judge image 1.",
            "first_view_answer": blind,
            "labels": {"too_visible": "recognised instantly but looks pasted on / sticker-like",
                       "good": "blends into the background and is still recognised at first glance",
                       "subtle": "only found after looking carefully",
                       "missing": "subject not visible",
                       "broken": "deformed subject or damaged background",
                       "unsure": "cannot decide"},
            **({"zoom": f"image {4 + len(self.examples)} is an enlarged crop of image 1 at the subject's location"} if zoom else {}),
            **({"author_examples": {f"image {i + 4}": label for i, (_, label) in enumerate(self.examples)},
                "examples_note": "Images 4+ are other pairs labelled by the author; match the author's standard."}
               if self.examples else {}),
            "reply_format": {"label": "one of the labels", "confidence": "0-1", "reason": "one sentence"},
        }), [output, subject, background, *[path for path, _ in self.examples], *([zoom] if zoom else [])])
        try:
            data = parse_json(reply)
            label = data.get("label")
            if label not in LABELS:
                raise ValueError(label)
            return label, str(data.get("reason", ""))[:300], {"first_view": blind, "confidence": data.get("confidence")}
        except (ValueError, TypeError, AttributeError):
            return "unsure", "observer reply unparseable -> abstain", {"first_view": blind, "raw": reply[:500]}


def subject_zoom(output: str, subject: str, out_path: str, pad: float = 0.12, size: int = 512) -> Optional[str]:
    """Crop `output` to the subject's bounding box (non-gray pixels of the normalized subject, which shares the
    output's frame at layout scale 1) and enlarge it. None if the subject box cannot be found."""
    import numpy as np
    from PIL import Image

    sub = np.asarray(Image.open(subject).convert("RGB"), dtype=np.int16)
    out = Image.open(output).convert("RGB")
    if sub.shape[:2] != (out.height, out.width):
        return None
    ys, xs = np.nonzero(np.abs(sub - 128).sum(axis=2) > 24)
    if len(xs) < 50:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    px, py = int((x1 - x0) * pad), int((y1 - y0) * pad)
    box = (max(0, x0 - px), max(0, y0 - py), min(out.width, x1 + px), min(out.height, y1 + py))
    crop = out.crop(box)
    crop.thumbnail((size, size)) if max(crop.size) > size else None
    scale = size / max(crop.size)
    crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.LANCZOS).save(out_path)
    return out_path


def fake_vision(label_for: Callable[[str], str]) -> VisionFn:
    """Offline stand-in used by tests and the demo without an API key."""
    def run(prompt: str, images: Sequence[str]) -> str:
        if "Look at this image once" in prompt:
            return '{"sees_object": true, "what": "animal"}'
        return json.dumps({"label": label_for(images[0]), "confidence": 0.5, "reason": "fake observer"})
    return run
