"""Config search around the generator (M3 baselines + M4 proposer hook). Plain Python, no agent framework.

One search = one subject/background pair, a budget of renders, and a feedback label per render:

    propose (policy) -> validate (action space) -> render (renderer) -> label (human/evaluator) -> log -> repeat

Labels are the quick-label vocabulary of tools/label.html. Everything the policy sees is in the event log, so a
search can stop at any point (e.g. waiting for a human label) and resume later from the same log.
"""
from __future__ import annotations

import json
import os
import random
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

LABELS = ("too_visible", "good", "subtle", "missing", "broken", "unsure")

# Action space v2 (2026-10-12, knob sweep outputs/label_sweep): the constant blend alpha only matters between 0 and
# ~0.2 (the v1 grid 0.2-1.0 of grid_v5/label_b1 rendered near-identical images); inside the subject the change from
# alpha 0 -> 0.04 -> 0.08 -> 0.12 -> 0.2 is roughly even. bg_strength 0.40 -> 0.55 raises visibility about as much;
# 0.57+ broke textured backgrounds (pilot labels). The two knobs add up. Other knobs had little effect (end_frac,
# softedge +-0.2) and are not searched. v1 (grid_v5) was alpha (0.2, 0.35, 0.5, 0.7, 1.0) x bg (0.45, 0.52).
ACTION_SPACE: Dict[str, Tuple[float, ...]] = {
    "blend.alpha_end": (0.0, 0.04, 0.08, 0.12, 0.2),
    "sampling.bg_strength": (0.40, 0.48, 0.55),
}
FIXED = {"blend.profile": "constant"}  # alpha_end is the constant blend weight
START = {"blend.alpha_end": 0.2, "sampling.bg_strength": 0.48}  # closest to the thesis default C00 (sweep: ~a020_b50)

Config = Dict[str, float]


def grid_id(cfg: Config) -> str:
    """Stable config id used by the replay grid (pilot/grid_v5.yaml), e.g. a050_b45."""
    return f"a{round(cfg['blend.alpha_end'] * 100):03d}_b{round(cfg['sampling.bg_strength'] * 100):02d}"


def all_configs() -> List[Config]:
    return [dict(zip(ACTION_SPACE, vals)) for vals in _product(ACTION_SPACE.values())]


def key_of(cfg: Config) -> Tuple:
    return tuple(cfg[k] for k in ACTION_SPACE)


def validate(cfg: Dict[str, Any]) -> Config:
    """Only whitelisted keys, only allowed values: a proposal outside the space is rejected, never clipped silently."""
    unknown = set(cfg) - set(ACTION_SPACE)
    if unknown:
        raise ValueError(f"keys outside the action space: {sorted(unknown)}")
    out: Config = {}
    for k, allowed in ACTION_SPACE.items():
        v = float(cfg.get(k, START[k]))
        match = [a for a in allowed if abs(a - v) < 1e-9]
        if not match:
            raise ValueError(f"{k}={v} not in {allowed}")
        out[k] = match[0]
    return out


def step(cfg: Config, k: str, direction: int) -> Optional[Config]:
    """Neighbouring value on one axis, or None at the edge."""
    values = ACTION_SPACE[k]
    i = values.index(cfg[k]) + direction
    if not 0 <= i < len(values):
        return None
    return {**cfg, k: values[i]}


@dataclass
class Trial:
    index: int
    config: Config
    proposed_by: str
    rationale: str = ""
    output: Optional[str] = None  # image path, set after rendering
    label: Optional[str] = None  # set after evaluation
    note: Optional[str] = None
    seconds: Optional[float] = None


@dataclass
class Search:
    pair_id: str
    budget: int
    policy: str
    trials: List[Trial] = field(default_factory=list)

    @property
    def done(self) -> bool:
        labelled = [t for t in self.trials if t.label]
        return len(labelled) >= self.budget or any(t.label == "good" for t in labelled)

    @property
    def pending(self) -> Optional[Trial]:
        """A rendered trial still waiting for its label (resume point), or a proposed one not yet rendered."""
        return next((t for t in self.trials if t.label is None), None)

    def tried(self) -> set:
        return {key_of(t.config) for t in self.trials}

    def selected(self) -> Optional[Trial]:
        """Final pick: the last 'good' trial, else None (no fake accept)."""
        good = [t for t in self.trials if t.label == "good"]
        return good[-1] if good else None


# ------------------------------------------------------------------
# Policies: history -> next config (+ short rationale)
# ------------------------------------------------------------------
class Policy:
    name = "policy"

    def propose(self, search: Search) -> Tuple[Config, str]:
        raise NotImplementedError


class FixedDefault(Policy):
    name = "default"

    def propose(self, search):
        return dict(START), "thesis default"


class RandomPolicy(Policy):
    """Uniform over untried configs (seeded per pair, so reruns are reproducible)."""
    name = "random"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def propose(self, search):
        rng = random.Random(f"{self.seed}-{search.pair_id}-{len(search.trials)}")
        grid = all_configs()
        untried = [c for c in grid if key_of(c) not in search.tried()]
        return rng.choice(untried or grid), "random untried config"


class RulePolicy(Policy):
    """Coordinate rule from the pilot labels (heuristic, checked on dev only):
    too_visible -> lower alpha; subtle/missing -> raise alpha; broken -> lower bg_strength; unsure -> lower alpha."""
    name = "rule"
    MOVES = {
        "too_visible": [("blend.alpha_end", -1), ("sampling.bg_strength", -1)],
        "subtle": [("blend.alpha_end", +1), ("sampling.bg_strength", +1)],
        "missing": [("blend.alpha_end", +1), ("sampling.bg_strength", +1)],
        "broken": [("sampling.bg_strength", -1), ("blend.alpha_end", -1)],
        "unsure": [("blend.alpha_end", -1), ("blend.alpha_end", +1)],
    }

    def propose(self, search):
        labelled = [t for t in search.trials if t.label]
        if not labelled:
            return dict(START), "start from the thesis default"
        last = labelled[-1]
        for k, d in self.MOVES.get(last.label, []):
            cand = step(last.config, k, d)
            if cand and key_of(cand) not in search.tried():
                return cand, f"last={last.label}: {k} {'+' if d > 0 else '-'}1 step"
        return RandomPolicy().propose(search)[0], f"last={last.label}: rule exhausted, random untried"


class LLMPolicy(Policy):
    """Proposer backed by a text LLM. `complete(prompt) -> str` is injected (OpenAI, Claude, ... or a fake in tests).
    The reply must be strict JSON {"config": {...}, "rationale": "..."}; invalid or repeated proposals get a bounded
    repair, then fall back to the rule policy and the fallback is logged."""
    name = "llm"

    def __init__(self, complete: Callable[[str], str], max_repairs: int = 1):
        self.complete = complete
        self.max_repairs = max_repairs
        self.calls: List[Dict[str, Any]] = []

    def prompt(self, search: Search, error: str = "") -> str:
        history = [{"config": t.config, "label": t.label, "note": t.note} for t in search.trials if t.label]
        return json.dumps({
            "task": "Choose the next generator config so the subject blends into the background yet is recognised "
                    "at first glance. Labels: too_visible, good, subtle, missing, broken, unsure.",
            "action_space": {k: list(v) for k, v in ACTION_SPACE.items()},
            "effects": {"blend.alpha_end": "higher = subject more visible",
                        "sampling.bg_strength": "higher = background repainted more, subject stronger; too high breaks texture"},
            "history": history,
            "renders_left": search.budget - len(history),
            "already_tried": [dict(zip(ACTION_SPACE, k)) for k in search.tried()],
            "reply_format": {"config": {k: "one allowed value" for k in ACTION_SPACE}, "rationale": "one sentence"},
            **({"previous_reply_error": error} if error else {}),
        }, ensure_ascii=False)

    def propose(self, search):
        error = ""
        for attempt in range(self.max_repairs + 1):
            text = self.complete(self.prompt(search, error))
            self.calls.append({"attempt": attempt, "reply": text[:2000]})
            try:
                reply = json.loads(re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M))
                cfg = validate(reply["config"])
                if key_of(cfg) in search.tried():
                    raise ValueError("config already tried")
                return cfg, "llm: " + str(reply.get("rationale", ""))[:300]
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                error = f"{type(exc).__name__}: {exc}"
        cfg, why = RulePolicy().propose(search)
        return cfg, f"llm invalid after {self.max_repairs + 1} tries ({error}); fallback rule: {why}"


def _product(lists):
    out = [[]]
    for values in lists:
        out = [o + [v] for o in out for v in values]
    return out


# ------------------------------------------------------------------
# Controller with an append-only event log (resume = replay the log)
# ------------------------------------------------------------------
class Controller:
    def __init__(self, log_path: str):
        self.log_path = log_path

    def _event(self, kind: str, **data) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": datetime.now(timezone.utc).isoformat(timespec="seconds"), "event": kind, **data},
                               ensure_ascii=False) + "\n")

    def load(self) -> Optional[Search]:
        if not os.path.exists(self.log_path):
            return None
        search = None
        with open(self.log_path, encoding="utf-8") as f:
            for line in f:
                e = json.loads(line)
                if e["event"] == "start":
                    search = Search(e["pair_id"], e["budget"], e["policy"])
                elif e["event"] == "propose":
                    search.trials.append(Trial(e["index"], e["config"], e["proposed_by"], e.get("rationale", "")))
                elif e["event"] == "render":
                    t = search.trials[e["index"]]
                    t.output, t.seconds = e["output"], e.get("seconds")
                elif e["event"] == "label":
                    t = search.trials[e["index"]]
                    t.label, t.note = e["label"], e.get("note")
        return search

    def start(self, pair_id: str, budget: int, policy: Policy) -> Search:
        existing = self.load()
        if existing:
            return existing
        self._event("start", pair_id=pair_id, budget=budget, policy=policy.name, action_space={k: list(v) for k, v in ACTION_SPACE.items()},
                    fixed=FIXED)
        return Search(pair_id, budget, policy.name)

    def propose(self, search: Search, policy: Policy) -> Trial:
        if search.pending:
            return search.pending
        if search.done:
            raise RuntimeError("search finished")
        cfg, why = policy.propose(search)
        cfg = validate(cfg)
        trial = Trial(len(search.trials), cfg, policy.name, why)
        search.trials.append(trial)
        self._event("propose", index=trial.index, config=cfg, proposed_by=policy.name, rationale=why)
        if isinstance(policy, LLMPolicy) and policy.calls:
            self._event("llm_calls", index=trial.index, calls=policy.calls[-(policy.max_repairs + 1):])
        return trial

    def rendered(self, search: Search, trial: Trial, output: str, seconds: Optional[float] = None) -> None:
        trial.output, trial.seconds = output, seconds
        self._event("render", index=trial.index, output=output, seconds=seconds)

    def labelled(self, search: Search, trial: Trial, label: str, note: Optional[str] = None) -> None:
        if label not in LABELS:
            raise ValueError(f"label must be one of {LABELS}")
        trial.label, trial.note = label, note
        self._event("label", index=trial.index, label=label, note=note)
        if search.done:
            sel = search.selected()
            self._event("finish", selected=sel.index if sel else None, renders=len([t for t in search.trials if t.output]))

    def run(self, search: Search, policy: Policy, render: Callable[[Config], str],
            evaluate: Callable[[Trial], Optional[str]]) -> Search:
        """Non-interactive loop (offline replay, simulated evaluator). evaluate() returning None pauses the search."""
        while not search.done:
            trial = self.propose(search, policy)
            if trial.output is None:
                self.rendered(search, trial, render({**FIXED, **trial.config}))
            label = evaluate(trial)
            if label is None:
                break
            self.labelled(search, trial, label)
        return search


def overrides(cfg: Config) -> List[str]:
    """--set style overrides for the generator."""
    return [f"{k}={v}" for k, v in {**FIXED, **cfg}.items()]
