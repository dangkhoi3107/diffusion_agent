"""Frozen image backbones of Model 1 (docs/AGENT_PLAN.html §2): image -> one feature vector, cached on disk.

    dinov2  facebook/dinov2-small (ViT-S/14, 224 px): concat(CLS, mean of patch tokens) = 768 numbers, the same
            feature as scripts/train_recognizer.py (Head C)
    clip    openai/clip-vit-base-patch32: projected image embedding = 512 numbers (baseline)

Needs torch + transformers. Locally they live in their own venv: ~/.venvs/camo-ml (CPU torch).
"""
from __future__ import annotations

import os
import time
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

BACKBONES = {"dinov2": "facebook/dinov2-small", "clip": "openai/clip-vit-base-patch32"}


def load_rgb(path: str):
    """RGB image; transparent cutouts are put over gray 128 as camo.pilot normalizes them."""
    from PIL import Image

    img = Image.open(path)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (128, 128, 128, 255))
        bg.alpha_composite(img)
        img = bg
    return img.convert("RGB")


class Backbone:
    def __init__(self, name: str = "dinov2", device: Optional[str] = None):
        import torch
        from transformers import AutoImageProcessor, AutoModel, CLIPVisionModelWithProjection

        if name not in BACKBONES:
            raise ValueError(f"unknown backbone {name!r}, choose from {sorted(BACKBONES)}")
        self.name, self.torch = name, torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cpu":
            torch.set_num_threads(os.cpu_count() or 4)
        self.proc = AutoImageProcessor.from_pretrained(BACKBONES[name])
        cls = CLIPVisionModelWithProjection if name == "clip" else AutoModel
        self.model = cls.from_pretrained(BACKBONES[name]).eval().to(self.device)
        self.dim = 512 if name == "clip" else 768

    def embed(self, paths: Sequence[str], batch: int = 32, log: Optional[Callable[[str], None]] = None) -> np.ndarray:
        out: List[np.ndarray] = []
        t0 = time.time()
        with self.torch.inference_mode():
            for i in range(0, len(paths), batch):
                inputs = self.proc(images=[load_rgb(p) for p in paths[i:i + batch]], return_tensors="pt").to(self.device)
                res = self.model(**inputs)
                if self.name == "clip":
                    f = res.image_embeds
                else:
                    h = res.last_hidden_state
                    f = self.torch.cat([h[:, 0], h[:, 1:].mean(1)], 1)
                out.append(f.float().cpu().numpy())
                if log and (i // batch) % 10 == 0:
                    log(f"{self.name}: {i + len(out[-1])}/{len(paths)} images ({time.time() - t0:.0f}s)")
        return np.concatenate(out).astype("float32") if out else np.zeros((0, self.dim), "float32")


class FeatureCache:
    """{key: vector} of one backbone in one .npz. A key is a path relative to outputs/; its vector is recomputed when
    the file's size or mtime changed (a batch unzipped again)."""

    def __init__(self, path: str):
        self.path = path
        self.vec: Dict[str, np.ndarray] = {}
        self.stamp: Dict[str, str] = {}
        if os.path.exists(path):
            with np.load(path, allow_pickle=False) as z:
                for k, s, v in zip(z["keys"], z["stamps"], z["X"]):
                    self.vec[str(k)], self.stamp[str(k)] = v, str(s)

    @staticmethod
    def _stamp(path: str) -> str:
        st = os.stat(path)
        return f"{st.st_size}:{int(st.st_mtime)}"

    def missing(self, files: Dict[str, str]) -> List[str]:
        """files: {key: absolute path}; keys whose vector is absent or stale."""
        return [k for k, p in files.items() if self.stamp.get(k) != self._stamp(p)]

    def update(self, backbone: Backbone, files: Dict[str, str], log: Optional[Callable[[str], None]] = None,
               save_every: int = 2000) -> int:
        todo = self.missing(files)
        for i in range(0, len(todo), save_every):  # save in chunks: an interrupted run keeps what it computed
            keys = todo[i:i + save_every]
            X = backbone.embed([files[k] for k in keys], log=log)
            for k, v in zip(keys, X):
                self.vec[k], self.stamp[k] = v, self._stamp(files[k])
            self.save()
        return len(todo)

    def get(self, keys: Sequence[str]) -> np.ndarray:
        return np.stack([self.vec[k] for k in keys]).astype("float32")

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        keys = sorted(self.vec)
        tmp = self.path + ".tmp.npz"
        np.savez(tmp, keys=np.array(keys), stamps=np.array([self.stamp[k] for k in keys]),
                 X=np.stack([self.vec[k] for k in keys]) if keys else np.zeros((0, 1), "float32"))
        os.replace(tmp, self.path)
