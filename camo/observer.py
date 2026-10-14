"""Model 1, heads A + B (docs/AGENT_PLAN.html §2): how visible is the subject in a render, and is the render broken?

Input per render: frozen backbone features of the render and of its background, x = [f_r, f_b, f_r - f_b]
(ablation "render": f_r only). Head A is ordinal (CORAL): 3 yes/no logits "beyond missing / subtle / good" that share
one weight vector and differ only in their bias, so P(level > k) can never increase with k. Head B is one logit for
"broken"; a broken render has no visibility level, so it is left out of the CORAL loss. "unsure" labels are dropped.

Stage A (this file): backbone frozen, features cached (camo/backbone.py), only this small MLP is trained, seconds on
a CPU. Training and evaluation: scripts/train_observer.py. Stage B adds LoRA to the backbone (GPU).
numpy is enough to import this module; fit / predict / NetObserver need torch.
"""
from __future__ import annotations

import os
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

VIS = ("missing", "subtle", "good", "too_visible")  # ordinal, lowest first
CLASSES = VIS + ("broken",)
MODES = ("pair", "render")
DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "observer",
                             "observer_dinov2.pt")


def inputs(f_render: np.ndarray, f_bg: np.ndarray, mode: str = "pair") -> np.ndarray:
    if mode == "render":
        return f_render
    if mode == "pair":
        return np.concatenate([f_render, f_bg, f_render - f_bg], 1)
    raise ValueError(f"mode {mode!r} not in {MODES}")


def encode(labels: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """label -> (visibility level 0..3, or -1 for broken; broken 0/1)."""
    if any(l not in CLASSES for l in labels):
        raise ValueError(f"labels outside {CLASSES}: {sorted(set(labels) - set(CLASSES))}")
    level = np.array([VIS.index(l) if l in VIS else -1 for l in labels])
    broken = np.array([l == "broken" for l in labels], dtype=np.float32)
    return level, broken


def coral_targets(level: np.ndarray) -> np.ndarray:
    """level k -> K-1 binary targets [k > 0, k > 1, k > 2]."""
    return (level[:, None] > np.arange(len(VIS) - 1)[None, :]).astype(np.float32)


def class_probs(p_beyond: np.ndarray, p_broken: np.ndarray) -> np.ndarray:
    """CORAL P(level > k) (N x 3) + P(broken) (N) -> probabilities over CLASSES (N x 5), rows sum to 1."""
    edges = np.concatenate([np.ones((len(p_beyond), 1)), p_beyond, np.zeros((len(p_beyond), 1))], 1)
    vis = np.clip(edges[:, :-1] - edges[:, 1:], 0, 1)  # rank-consistent by construction, clip guards rounding
    vis = vis / vis.sum(1, keepdims=True)
    return np.concatenate([vis * (1 - p_broken[:, None]), p_broken[:, None]], 1)


def decide(p_beyond: np.ndarray, p_broken: np.ndarray, broken_threshold: float = 0.5) -> List[str]:
    """CORAL decision: level = number of thresholds passed with P > 0.5; broken wins above its threshold."""
    level = (p_beyond > 0.5).sum(1)
    return ["broken" if b >= broken_threshold else VIS[k] for k, b in zip(level, p_broken)]


@dataclass
class FitConfig:
    hidden: int = 256  # 0 = linear heads on the standardized features (linear probe)
    dropout: float = 0.3
    steps: int = 1000  # optimizer steps on random minibatches: the same cost for 400 or 7,000 labels
    batch: int = 256
    lr: float = 3e-3
    weight_decay: float = 1e-2
    broken_weight: float = 1.0  # lambda in L = L_CORAL + lambda * BCE_broken
    balance: bool = True  # weight visibility classes by 1/sqrt(frequency): subtle is rare
    seed: int = 0


def _net(in_dim: int, cfg: FitConfig):
    import torch
    from torch import nn

    class Heads(nn.Module):
        def __init__(self):
            super().__init__()
            if cfg.hidden:
                self.body = nn.Sequential(nn.Dropout(cfg.dropout / 2), nn.Linear(in_dim, cfg.hidden), nn.GELU(),
                                          nn.Dropout(cfg.dropout))
                d = cfg.hidden
            else:
                self.body, d = nn.Dropout(cfg.dropout / 2), in_dim
            self.vis = nn.Linear(d, 1, bias=False)  # CORAL: one weight vector ...
            self.vis_bias = nn.Parameter(torch.tensor([1.0, 0.0, -1.0]))  # ... and one bias per threshold
            self.broken = nn.Linear(d, 1)

        def forward(self, x):
            h = self.body(x)
            return self.vis(h) + self.vis_bias, self.broken(h).squeeze(1)

    return Heads()


def fit(X: np.ndarray, labels: Sequence[str], cfg: Optional[FitConfig] = None) -> Dict[str, Any]:
    """Train heads A + B on features X (N x D). Returns a bundle for predict() / save()."""
    import torch
    import torch.nn.functional as F

    cfg = cfg or FitConfig()
    torch.manual_seed(cfg.seed)
    level, broken = encode(labels)
    mean, std = X.mean(0), X.std(0) + 1e-6
    x = torch.tensor((X - mean) / std, dtype=torch.float32)
    is_vis = torch.tensor(level >= 0)
    t_vis = torch.tensor(coral_targets(np.maximum(level, 0)))
    t_broken = torch.tensor(broken)
    counts = Counter(level[level >= 0].tolist())
    w = np.array([1 / np.sqrt(counts[k]) if cfg.balance else 1.0 for k in np.maximum(level, 0)], np.float32)
    w_vis = torch.tensor(w / w[level >= 0].mean() if (level >= 0).any() else w)
    n_pos = float(broken.sum())
    pos_weight = torch.tensor(min((len(broken) - n_pos) / n_pos, 20.0) if n_pos else 1.0)
    net = _net(X.shape[1], cfg)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    gen = torch.Generator().manual_seed(cfg.seed)
    net.train()
    for _ in range(cfg.steps):
        b = torch.randint(0, len(x), (min(cfg.batch, len(x)),), generator=gen)
        logit_vis, logit_broken = net(x[b])
        v = is_vis[b]
        loss_vis = F.binary_cross_entropy_with_logits(logit_vis[v], t_vis[b][v], reduction="none").mean(1)
        loss = (loss_vis * w_vis[b][v]).mean() if v.any() else torch.tensor(0.0)
        loss = loss + cfg.broken_weight * F.binary_cross_entropy_with_logits(logit_broken, t_broken[b], pos_weight=pos_weight)
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    return {"state": net.state_dict(), "mean": mean, "std": std, "in_dim": X.shape[1], "cfg": asdict(cfg),
            "classes": CLASSES, "n_train": len(labels), "label_counts": dict(Counter(labels))}


def _forward(bundle: Dict[str, Any], X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    import torch

    net = _net(bundle["in_dim"], FitConfig(**bundle["cfg"]))
    net.load_state_dict(bundle["state"])
    net.eval()
    with torch.inference_mode():
        lv, lb = net(torch.tensor((X - bundle["mean"]) / bundle["std"], dtype=torch.float32))
    return torch.sigmoid(lv).numpy(), torch.sigmoid(lb).numpy()


def predict(bundle: Dict[str, Any], X: np.ndarray) -> Tuple[List[str], np.ndarray]:
    """-> (label per row, probabilities over CLASSES N x 5)."""
    p_beyond, p_broken = _forward(bundle, X)
    return decide(p_beyond, p_broken), class_probs(p_beyond, p_broken)


def metrics(truth: Sequence[str], pred: Sequence[str]) -> Dict[str, Any]:
    """Agreement with the user's labels. exact: 5 classes; good_vs_not: what the agent's stop rule uses;
    ordinal: on renders neither side calls broken, mean |level difference| and share within one level."""
    t, p = list(truth), list(pred)
    n = len(t)
    if not n:
        return {"n": 0}
    both = [(VIS.index(a), VIS.index(b)) for a, b in zip(t, p) if a in VIS and b in VIS]
    tp = sum(a == b == "broken" for a, b in zip(t, p))
    fp = sum(b == "broken" != a for a, b in zip(t, p))
    fn = sum(a == "broken" != b for a, b in zip(t, p))
    recall = {c: round(sum(a == b == c for a, b in zip(t, p)) / t.count(c), 3) for c in CLASSES if t.count(c)}
    # "good" is rare (about 1 render in 10), so good_vs_not is high even for "never good": the agent's stop rule
    # needs good precision (a false good ends the search on a bad image) and good recall
    g_tp = sum(a == b == "good" for a, b in zip(t, p))
    g_pred, g_true = p.count("good"), t.count("good")
    return {"n": n, "exact": round(sum(a == b for a, b in zip(t, p)) / n, 3),
            "good_precision": round(g_tp / g_pred, 3) if g_pred else None,
            "good_recall": round(g_tp / g_true, 3) if g_true else None,
            "good_vs_not": round(sum((a == "good") == (b == "good") for a, b in zip(t, p)) / n, 3),
            "ordinal_mae": round(float(np.mean([abs(a - b) for a, b in both])), 3) if both else None,
            "within_one": round(float(np.mean([abs(a - b) <= 1 for a, b in both])), 3) if both else None,
            "macro_recall": round(float(np.mean(list(recall.values()))), 3),
            "broken_f1": round(2 * tp / (2 * tp + fp + fn), 3) if tp + fp + fn else None,
            "recall": recall, "truth_counts": dict(Counter(t)),
            "confusion": dict(Counter(f"{a}->{b}" for a, b in zip(t, p) if a != b))}


def save(bundle: Dict[str, Any], path: str = DEFAULT_MODEL, **meta) -> None:
    import torch

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({**bundle, **meta}, path)


def load(path: str = DEFAULT_MODEL) -> Dict[str, Any]:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


class NetObserver:
    """Model 1 in the agent loop: same judge() as camo.llm.Observer, so the Controller and the demo can swap it in."""

    def __init__(self, path: str = DEFAULT_MODEL, backbone=None):
        from camo.backbone import Backbone

        self.bundle = load(path)
        self.mode = self.bundle.get("mode", "pair")
        self.backbone = backbone or Backbone(self.bundle.get("backbone", "dinov2"))

    def probs(self, output: str, background: str) -> Dict[str, float]:
        f = self.backbone.embed([output, background])
        _labels, P = predict(self.bundle, inputs(f[:1], f[1:], self.mode))
        return {c: round(float(v), 4) for c, v in zip(CLASSES, P[0])}

    def judge(self, output: str, subject: str, background: str, target: str,
              zoom: Optional[str] = None) -> Tuple[str, str, Dict[str, Any]]:
        f = self.backbone.embed([output, background])
        labels, P = predict(self.bundle, inputs(f[:1], f[1:], self.mode))
        probs = {c: round(float(v), 4) for c, v in zip(CLASSES, P[0])}
        top = ", ".join(f"{c} {v:.2f}" for c, v in sorted(probs.items(), key=lambda kv: -kv[1])[:2])
        return labels[0], f"Model 1 ({self.bundle.get('backbone', 'dinov2')}): {top}", {"probs": probs}
