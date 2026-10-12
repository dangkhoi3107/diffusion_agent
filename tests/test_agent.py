"""CPU tests for the config search: action space, policies, controller log/resume, LLM repair/fallback.

Run: python -m pytest tests/test_agent.py
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camo import agent  # noqa: E402
from camo.agent import ACTION_SPACE, START, Controller, LLMPolicy, RandomPolicy, RulePolicy, validate  # noqa: E402


def visibility_world(good_alpha=0.5):
    """Simulated evaluator: alpha above the sweet spot is too visible, below it too hidden."""
    def evaluate(trial):
        a = trial.config["blend.alpha_end"]
        if trial.config["sampling.bg_strength"] >= 0.57:
            return "broken"
        if abs(a - good_alpha) < 1e-9:
            return "good"
        return "too_visible" if a > good_alpha else "missing"
    return evaluate


def fake_render(tmp_path):
    def render(cfg):
        path = tmp_path / f"r_{cfg['blend.alpha_end']}_{cfg['sampling.bg_strength']}.png"
        path.write_text("img")
        return str(path)
    return render


def test_validate_rejects_outside_values_and_keys():
    assert validate({"blend.alpha_end": 0.08}) == {"blend.alpha_end": 0.08, "sampling.bg_strength": 0.48}
    with pytest.raises(ValueError):
        validate({"blend.alpha_end": 0.55})
    with pytest.raises(ValueError):
        validate({"mask.gamma": 0.5})


def test_rule_policy_walks_alpha_down_to_good(tmp_path):
    ctl = Controller(str(tmp_path / "log.jsonl"))
    s = ctl.start("P02", budget=5, policy=RulePolicy())
    ctl.run(s, RulePolicy(), fake_render(tmp_path), visibility_world(0.08))
    assert [t.config["blend.alpha_end"] for t in s.trials] == [0.2, 0.12, 0.08]
    assert s.selected().config["blend.alpha_end"] == 0.08 and s.done


def test_budget_and_no_fake_accept(tmp_path):
    ctl = Controller(str(tmp_path / "log.jsonl"))
    s = ctl.start("P09", budget=3, policy=RandomPolicy())
    ctl.run(s, RandomPolicy(seed=1), fake_render(tmp_path), lambda t: "missing")
    assert len(s.trials) == 3 and s.selected() is None
    assert len({agent.key_of(t.config) for t in s.trials}) == 3  # never repeats


def test_resume_from_log_after_pause(tmp_path):
    log = str(tmp_path / "log.jsonl")
    ctl = Controller(log)
    s = ctl.start("P13", budget=5, policy=RulePolicy())
    answers = iter(["too_visible", None])  # second render waits for a human label
    ctl.run(s, RulePolicy(), fake_render(tmp_path), lambda t: next(answers))
    resumed = Controller(log).load()
    assert len(resumed.trials) == 2 and resumed.pending.output and resumed.pending.label is None
    ctl2 = Controller(log)
    ctl2.labelled(resumed, resumed.pending, "good")
    assert Controller(log).load().selected().index == 1


def test_llm_policy_parses_repairs_and_falls_back(tmp_path):
    ctl = Controller(str(tmp_path / "log.jsonl"))
    s = ctl.start("P05", budget=3, policy=RulePolicy())
    replies = iter([
        '```json\n{"config": {"blend.alpha_end": 0.12, "sampling.bg_strength": 0.48}, "rationale": "lower alpha"}\n```',
        '{"config": {"blend.alpha_end": 0.55}}',            # invalid value -> repair
        '{"config": {"blend.alpha_end": 0.08, "sampling.bg_strength": 0.40}, "rationale": "go lower"}',
        "not json", "still not json",                        # -> fallback to rule
    ])
    llm = LLMPolicy(lambda prompt: next(replies), max_repairs=1)
    labels = iter(["too_visible", "too_visible", "missing"])
    ctl.run(s, llm, fake_render(tmp_path), lambda t: next(labels))
    why = [t.rationale for t in s.trials]
    assert any("llm: lower alpha" in w for w in why)
    assert any("fallback rule" in w for w in why)
    prompt = json.loads(llm.prompt(s))
    assert prompt["action_space"]["blend.alpha_end"] == list(ACTION_SPACE["blend.alpha_end"])
    events = [json.loads(line)["event"] for line in open(tmp_path / "log.jsonl")]
    assert "llm_calls" in events and events[-1] == "finish"


def test_observer_two_pass_and_abstain(tmp_path):
    from camo.llm import Observer, fake_vision

    seen = []

    def vision(prompt, images):
        seen.append((prompt[:30], len(images)))
        return fake_vision(lambda p: "good")(prompt, images)

    label, reason, extra = Observer(vision).judge("out.png", "subj.png", "bg.png", "cat")
    assert label == "good" and extra["first_view"] == "animal"
    assert seen[0][1] == 1 and seen[1][1] == 3  # pass A: output only, no target name; pass B: with references
    assert "cat" not in json.dumps(seen[0])
    bad = Observer(lambda p, imgs: '{"sees_object": false}' if "once" in p else "garbage")
    assert bad.judge("o", "s", "b", "cat")[0] == "unsure"


def test_two_agents_close_the_loop(tmp_path):
    """Proposer (LLM, faked) + Observer (VLM, faked) run a search with no human."""
    from camo.llm import Observer, fake_vision

    world = visibility_world(0.08)
    configs = {}

    def render(cfg):
        path = fake_render(tmp_path)(cfg)
        configs[path] = cfg
        return path

    def observer_label(path):
        return world(agent.Trial(0, {k: configs[path][k] for k in ACTION_SPACE}, "x"))

    obs = Observer(fake_vision(observer_label))
    step_down = iter([0.12, 0.08])
    llm = LLMPolicy(lambda prompt: json.dumps({"config": {"blend.alpha_end": next(step_down), "sampling.bg_strength": 0.48},
                                               "rationale": "too visible, lower alpha"}))
    ctl = Controller(str(tmp_path / "log.jsonl"))
    s = ctl.start("P11", budget=5, policy=llm)
    ctl.run(s, llm, render, lambda t: obs.judge(t.output, "s.png", "b.png", "fish")[0])
    assert s.selected() and s.selected().config["blend.alpha_end"] == 0.08


def test_openai_streaming_adapter_collects_deltas(monkeypatch):
    from types import SimpleNamespace

    from camo import llm

    events = [SimpleNamespace(type="response.output_text.delta", delta='{"config": '),
              SimpleNamespace(type="response.output_text.delta", delta='{}}'),
              SimpleNamespace(type="response.completed",
                              response=SimpleNamespace(id="resp_1", usage=SimpleNamespace(input_tokens=10, output_tokens=4)))]

    class FakeResponses:
        def create(self, stream=False, **request):
            assert request["model"] and request["input"][0]["content"][0]["type"] == "input_text"
            return iter(events) if stream else SimpleNamespace(output_text="{}", usage=None, id="r0")

    monkeypatch.setattr(llm, "_client", lambda: SimpleNamespace(responses=FakeResponses()))
    got = []
    text = llm.openai_text(on_delta=got.append)("hello")
    assert text == '{"config": {}}' and got == ['{"config": ', '{}}']
    assert llm.CALLS[-1]["request_id"] == "resp_1" and llm.CALLS[-1]["output_tokens"] == 4
    assert llm.openai_text()("hi") == "{}"


def test_openai_key_from_file_when_env_missing(tmp_path, monkeypatch):
    from camo import llm

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(llm, "KEY_FILE", str(tmp_path / "missing"))
    assert not llm.has_key()
    (tmp_path / "k").write_text("sk-test-123\n")
    monkeypatch.setattr(llm, "KEY_FILE", str(tmp_path / "k"))
    assert llm.has_key() and llm._key_from_file() == "sk-test-123"
