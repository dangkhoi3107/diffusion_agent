"""Model 2 (Proposer, docs/AGENT_PLAN.html §3): a small GRU that picks the next config of the 15-config grid.

Input per step: the config just tried (one-hot over CONFIG_IDS) + its judgement (probabilities over OBS; a human
label is one-hot, Model 1 gives soft probabilities). Initial state h0 = tanh(W . background feature) with a 32-unit
bottleneck so the policy learns "this kind of background wants these configs" rather than single images. Output:
one score per config; tried configs are masked; greedy at test time, sampled while training.

Training runs on the replay simulator: every pair of the label set is rendered and labelled on the full grid, so
"try config X" is a table lookup (a Board).
  1. behaviour cloning: at every state the target is the uniform distribution over the untried good configs;
  2. REINFORCE: +1 on a good image, -0.1 per render, -1 when the budget ends without one (stopping on a "good" that
     the human label says is not good also counts as -1). With `noisy`, part of the episodes see Model 1's
     predicted probabilities (out-of-fold) instead of the human label, so the policy gets used to a noisy judge.

The simulator, boards and baselines are numpy only; torch is imported lazily (the net, fit, NetPolicy).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .agent import START, Policy, Search, Trial, all_configs, grid_id, validate

OBS = ("too_visible", "good", "subtle", "missing", "broken")  # judgement channels fed back to the policy
CONFIGS = all_configs()
CONFIG_IDS = [grid_id(c) for c in CONFIGS]
CONFIG_OF = dict(zip(CONFIG_IDS, CONFIGS))
STEP_DIM = len(CONFIG_IDS) + len(OBS)
BUDGET = 5
REWARD_GOOD, REWARD_FAIL, COST_RENDER = 1.0, -1.0, 0.1


@dataclass
class Board:
    """One labelled pair: true label of every grid config, background feature, optional Model 1 probabilities."""
    pair: str
    labels: Dict[str, str]
    bg: np.ndarray
    probs: Dict[str, np.ndarray] = field(default_factory=dict)  # config id -> probabilities over OBS
    split: str = "train"
    bg_group: Optional[str] = None

    def goods(self) -> List[str]:
        return [c for c in CONFIG_IDS if self.labels.get(c) == "good"]


def one_hot(label: Optional[str]) -> np.ndarray:
    v = np.zeros(len(OBS), np.float32)
    if label in OBS:
        v[OBS.index(label)] = 1.0
    return v


def observe(board: Board, cid: str, noisy: bool) -> Tuple[np.ndarray, str]:
    """(probabilities fed to the policy, label the search acts on) for rendering config `cid`."""
    if noisy and cid in board.probs:
        p = np.asarray(board.probs[cid], np.float32)
        return p, OBS[int(np.argmax(p))]
    return one_hot(board.labels[cid]), board.labels[cid]


def step_input(cid: str, p: np.ndarray) -> np.ndarray:
    x = np.zeros(STEP_DIM, np.float32)
    x[CONFIG_IDS.index(cid)] = 1.0
    x[len(CONFIG_IDS):] = p
    return x


History = List[Tuple[str, np.ndarray, str]]  # (config id, probabilities seen, judged label)


@dataclass
class Episode:
    pair: str
    history: History
    success: bool  # stopped on a config whose human label is good
    reward: float

    @property
    def renders(self) -> int:
        return len(self.history)

    @property
    def path(self) -> List[str]:
        return [h[0] for h in self.history]


def rollout(board: Board, choose: Callable[[History], str], budget: int = BUDGET, noisy: bool = False) -> Episode:
    """Run one search on the board. It stops when the judge says good, when the budget is spent, or when the
    policy repeats a config (no new information: a fixed policy renders once)."""
    history: History = []
    for _ in range(budget):
        cid = choose(history)
        if cid in {h[0] for h in history}:
            break
        p, judged = observe(board, cid, noisy)
        history.append((cid, p, judged))
        if judged == "good":
            break
    success = bool(history) and history[-1][2] == "good" and board.labels[history[-1][0]] == "good"
    reward = (REWARD_GOOD if success else REWARD_FAIL) - COST_RENDER * len(history)
    return Episode(board.pair, history, success, reward)


# ------------------------------------------------------------------
# Baselines on the same simulator
# ------------------------------------------------------------------
def agent_chooser(policy: Policy, pair: str) -> Callable[[History], str]:
    """Adapter: a camo.agent Policy (Rule, Random, Fixed) as a simulator chooser."""
    def choose(history: History) -> str:
        search = Search(pair, BUDGET, policy.name,
                        [Trial(i, dict(CONFIG_OF[c]), policy.name, label=lab) for i, (c, _p, lab) in enumerate(history)])
        cfg, _why = policy.propose(search)
        return grid_id(validate(cfg))
    return choose


def oracle_success(board: Board) -> bool:
    return bool(board.goods())


def summarize(episodes: Sequence[Episode], boards: Sequence[Board]) -> Dict[str, Any]:
    n = max(len(episodes), 1)
    hits = [e for e in episodes if e.success]
    return {"n": len(episodes), "success": round(len(hits) / n, 3),
            "mean_renders": round(sum(e.renders for e in episodes) / n, 2),
            "renders_to_good": round(sum(e.renders for e in hits) / max(len(hits), 1), 2),
            "mean_reward": round(sum(e.reward for e in episodes) / n, 3),
            "first_try_good": round(sum(e.success and e.renders == 1 for e in episodes) / n, 3),
            "oracle": round(sum(oracle_success(b) for b in boards) / max(len(boards), 1), 3)}


# ------------------------------------------------------------------
# The net (torch, lazy)
# ------------------------------------------------------------------
@dataclass
class FitConfig:
    hidden: int = 32
    bc_epochs: int = 200
    rl_epochs: int = 120
    lr: float = 1e-2
    rl_lr: float = 2e-3  # self-play fine-tunes the cloned policy: a smaller step keeps what BC learned
    entropy: float = 0.01
    noisy_share: float = 0.5  # share of self-play episodes judged by Model 1 (boards that have its probabilities)
    weight_decay: float = 1e-2
    pca: int = 16  # background feature -> its top principal components (fit on the training backgrounds, no labels)
    bg_dropout: float = 0.2
    seed: int = 0


def _net(bg_dim: int, hidden: int, bg_dropout: float = 0.0):
    import torch
    from torch import nn

    class ProposerNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.h0 = nn.Sequential(nn.Dropout(bg_dropout), nn.Linear(bg_dim, hidden), nn.Tanh())
            self.gru = nn.GRUCell(STEP_DIM, hidden)
            self.head = nn.Linear(hidden, len(CONFIG_IDS))

        def forward(self, bg, steps):
            """bg (B, bg_dim); steps (B, T, STEP_DIM) -> logits (B, T + 1, n): the choice after 0..T steps."""
            h = self.h0(bg)
            out = [self.head(h)]
            for t in range(steps.shape[1]):
                h = self.gru(steps[:, t], h)
                out.append(self.head(h))
            return torch.stack(out, 1)

    return ProposerNet()


def _projection(boards: Sequence[Board], k: int) -> Dict[str, np.ndarray]:
    """Standardize, then keep the top-k principal components (not whitened: the strong directions stay strong):
    a few hundred boards cannot support a 768-number input (on a synthetic task the net memorised the noise
    dimensions and fell below the rule)."""
    X = np.stack([b.bg for b in boards]).astype(np.float64)
    mean, std = X.mean(0), X.std(0) + 1e-6
    Z = (X - mean) / std
    k = max(1, min(k, len(boards) - 1, Z.shape[1])) if k else 0
    if not k:
        return {"mean": mean.astype(np.float32), "std": std.astype(np.float32), "W": None}
    _u, sv, vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    W = (vt[:k].T / np.sqrt(k)).astype(np.float32)  # unit-ish scale overall, relative variances kept
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32), "W": W}


def project(bundle: Dict[str, Any], bg: np.ndarray) -> np.ndarray:
    z = (np.asarray(bg, np.float32) - bundle["mean"]) / bundle["std"]
    return z @ bundle["W"] if bundle.get("W") is not None else z


def _masked(logits, tried):
    return logits.masked_fill(tried, float("-inf"))


def _bc_targets(board: Board, path: List[str]) -> List[Optional[np.ndarray]]:
    """Per prefix 0..len(path): uniform over the untried good configs (None once a good was found or none is left)."""
    goods = set(board.goods())
    out: List[Optional[np.ndarray]] = []
    for t in range(len(path) + 1):
        tried = set(path[:t])
        left = [c for c in CONFIG_IDS if c in goods and c not in tried]
        if goods & tried or not left:
            out.append(None)
            continue
        y = np.zeros(len(CONFIG_IDS), np.float32)
        y[[CONFIG_IDS.index(c) for c in left]] = 1.0 / len(left)
        out.append(y)
    return out


def _encode(boards: Sequence[Board], histories: Sequence[History], bundle: Dict[str, Any], T: int):
    import torch

    B = len(boards)
    bg = torch.tensor(np.stack([project(bundle, b.bg) for b in boards]), dtype=torch.float32)
    steps = torch.zeros(B, T, STEP_DIM)
    tried = torch.zeros(B, T + 1, len(CONFIG_IDS), dtype=torch.bool)
    for i, h in enumerate(histories):
        for t, (cid, p, _lab) in enumerate(h[:T]):
            steps[i, t] = torch.from_numpy(step_input(cid, p))
            tried[i, t + 1:, CONFIG_IDS.index(cid)] = True
    return bg, steps, tried


def fit(boards: Sequence[Board], cfg: Optional[FitConfig] = None, log: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """Behaviour cloning, then REINFORCE, on `boards` (train split only). Returns a bundle for predict / NetPolicy."""
    import torch

    cfg = cfg or FitConfig()
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    proj = _projection(boards, cfg.pca)
    bg_dim = proj["W"].shape[1] if proj["W"] is not None else len(proj["mean"])
    net = _net(bg_dim, cfg.hidden, cfg.bg_dropout)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    bundle = {"net": net, **proj, "config": dict(cfg.__dict__), "config_ids": list(CONFIG_IDS), "obs": OBS}
    T = BUDGET

    def sample_path(board: Board, explore: float) -> History:
        if rng.random() < explore:  # random untried order: covers states the current policy would not visit
            order = rng.sample(CONFIG_IDS, T)
            return [(c, *observe(board, c, False)) for c in order]
        ep = rollout(board, chooser(bundle, board.bg, greedy=False, rng=rng), T)
        net.train()
        return ep.history

    for epoch in range(cfg.bc_epochs):
        explore = max(0.2, 1.0 - epoch / max(cfg.bc_epochs * 0.6, 1))  # DAgger-style: more on-policy states later
        hists = [sample_path(b, explore) for b in boards]
        net.train()
        bg, steps, tried = _encode(boards, hists, bundle, T)
        logits = _masked(net(bg, steps), tried)
        loss, n = 0.0, 0
        for i, (b, h) in enumerate(zip(boards, hists)):
            for t, y in enumerate(_bc_targets(b, [c for c, _p, _l in h])[:T + 1]):
                if y is None:
                    continue
                logp = torch.log_softmax(logits[i, t], -1)
                y_t = torch.from_numpy(y)
                loss = loss - (y_t * torch.where(y_t > 0, logp, torch.zeros_like(logp))).sum()
                n += 1
        if n:
            opt.zero_grad()
            (loss / n).backward()
            opt.step()
        if log and (epoch + 1) % 20 == 0:
            log(f"bc {epoch + 1}/{cfg.bc_epochs} loss {float(loss) / max(n, 1):.3f}")

    for group in opt.param_groups:
        group["lr"] = cfg.rl_lr
    for epoch in range(cfg.rl_epochs):
        logps, ents, rets = [], [], []
        for b in boards:
            noisy = bool(b.probs) and rng.random() < cfg.noisy_share
            history: History = []
            net.train()
            h = net.h0(torch.tensor(project(bundle, b.bg), dtype=torch.float32)[None])
            lp, en, rw = [], [], []
            for _ in range(T):
                logits = net.head(h)[0]
                mask = torch.zeros(len(CONFIG_IDS), dtype=torch.bool)
                mask[[CONFIG_IDS.index(c) for c, _p, _l in history]] = True
                dist = torch.distributions.Categorical(logits=_masked(logits, mask))
                a = dist.sample()
                cid = CONFIG_IDS[int(a)]
                p, judged = observe(b, cid, noisy)
                history.append((cid, p, judged))
                lp.append(dist.log_prob(a))
                en.append(dist.entropy())
                rw.append(-COST_RENDER)
                if judged == "good":
                    break
                h = net.gru(torch.from_numpy(step_input(cid, p))[None], h)
            ok = history[-1][2] == "good" and b.labels[history[-1][0]] == "good"
            rw[-1] += REWARD_GOOD if ok else REWARD_FAIL
            g = np.cumsum(rw[::-1])[::-1]  # return to go
            logps.append(lp)
            ents.append(en)
            rets.append(g)
        width = max(len(g) for g in rets)
        base = [np.mean([g[t] for g in rets if len(g) > t]) for t in range(width)]  # per-step baseline
        loss = 0.0
        for lp, en, g in zip(logps, ents, rets):
            for t in range(len(lp)):
                loss = loss - (float(g[t] - base[t]) * lp[t] + cfg.entropy * en[t])
        opt.zero_grad()
        (loss / len(boards)).backward()
        opt.step()
        if log and (epoch + 1) % 20 == 0:
            log(f"rl {epoch + 1}/{cfg.rl_epochs} mean return {np.mean([g[0] for g in rets]):.3f}")
    net.eval()
    return bundle


def scores(bundle: Dict[str, Any], bg: np.ndarray, history: History) -> np.ndarray:
    """Masked softmax over CONFIG_IDS for the next step (tried configs get 0)."""
    import torch

    net = bundle["net"]
    was_training = net.training
    net.eval()  # no dropout when choosing
    with torch.no_grad():
        h = net.h0(torch.tensor(project(bundle, bg), dtype=torch.float32)[None])
        for cid, p, _lab in history:
            h = net.gru(torch.from_numpy(step_input(cid, p))[None], h)
        logits = net.head(h)[0]
        mask = torch.zeros(len(CONFIG_IDS), dtype=torch.bool)
        mask[[CONFIG_IDS.index(c) for c, _p, _l in history]] = True
        out = torch.softmax(_masked(logits, mask), -1).numpy()
    net.train(was_training)
    return out


def chooser(bundle: Dict[str, Any], bg: np.ndarray, greedy: bool = True, rng: Optional[random.Random] = None):
    def choose(history: History) -> str:
        p = scores(bundle, bg, history)
        if greedy:
            return CONFIG_IDS[int(np.argmax(p))]
        r = (rng or random).random() * p.sum()
        return CONFIG_IDS[int(np.searchsorted(np.cumsum(p), r, side="right").clip(0, len(p) - 1))]
    return choose


def save(bundle: Dict[str, Any], path: str, **meta) -> None:
    import torch

    torch.save({"state": bundle["net"].state_dict(), "mean": bundle["mean"], "std": bundle["std"], "W": bundle["W"],
                "config": bundle["config"], "config_ids": bundle["config_ids"], "obs": list(bundle["obs"]), **meta}, path)


def load(path: str) -> Dict[str, Any]:
    import torch

    d = torch.load(path, map_location="cpu", weights_only=False)
    if d["config_ids"] != CONFIG_IDS or tuple(d["obs"]) != OBS:
        raise ValueError(f"{path} was trained on another grid or label set")
    net = _net(d["W"].shape[1] if d.get("W") is not None else len(d["mean"]), d["config"]["hidden"])
    net.load_state_dict(d["state"])
    net.eval()
    return {**d, "net": net}


class NetPolicy(Policy):
    """Model 2 behind the same interface as RulePolicy, so Controller, event log and demo use it unchanged.
    `background` gives the pair's background feature; `judgement(trial)` may return Model 1's probabilities over
    OBS (default: the one-hot of the trial's label)."""
    name = "net"

    def __init__(self, bundle: Dict[str, Any], background: Callable[[str], np.ndarray],
                 judgement: Optional[Callable[[Trial], np.ndarray]] = None):
        self.bundle, self.background = bundle, background
        self.judgement = judgement or (lambda t: one_hot(t.label))

    def propose(self, search: Search) -> Tuple[Dict[str, float], str]:
        history = [(grid_id(t.config), self.judgement(t), t.label) for t in search.trials if t.label]
        p = scores(self.bundle, self.background(search.pair_id), history)
        best = int(np.argmax(p))
        if not np.isfinite(p).all() or p[best] <= 0:  # every config tried
            return dict(START), "net: grid exhausted"
        return dict(CONFIG_OF[CONFIG_IDS[best]]), f"net: p={p[best]:.2f}"
