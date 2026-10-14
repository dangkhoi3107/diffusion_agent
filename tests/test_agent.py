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


LEVELS = ACTION_SPACE["subject.strength"]  # faint -> strong


def visibility_world(good="a008"):
    """Simulated evaluator: a stronger subject than the sweet spot is too visible, a weaker one too hidden."""
    def evaluate(trial):
        level = LEVELS.index(trial.config["subject.strength"])
        if trial.config["sampling.bg_strength"] >= 0.57:
            return "broken"
        if level == LEVELS.index(good):
            return "good"
        return "too_visible" if level > LEVELS.index(good) else "missing"
    return evaluate


def fake_render(tmp_path):
    def render(cfg):
        path = tmp_path / f"r_{cfg['subject.strength']}_{cfg['sampling.bg_strength']}.png"
        path.write_text("img")
        return str(path)
    return render


def test_validate_rejects_outside_values_and_keys():
    assert validate({"subject.strength": "g50"}) == {"subject.strength": "g50", "sampling.bg_strength": 0.52, "seed": 1}
    assert validate({"sampling.bg_strength": "0.60", "seed": 2}) == {"subject.strength": "a020", "sampling.bg_strength": 0.60,
                                                                     "seed": 2}
    for bad in ({"subject.strength": "a050"}, {"sampling.bg_strength": 0.48}, {"seed": 3}, {"mask.gamma": 0.5}):
        with pytest.raises(ValueError):
            validate(bad)


def test_render_overrides_are_the_generator_settings():
    assert agent.render_overrides({"subject.strength": "a012", "sampling.bg_strength": 0.44}) == {
        "blend.profile": "constant", "blend.alpha_end": 0.12, "sampling.bg_strength": 0.44}
    assert "sampling.guidance_sub=5.0" in agent.overrides({"subject.strength": "g50", "sampling.bg_strength": 0.52})
    assert len({agent.grid_id(c) for c in agent.all_configs()}) == len(agent.all_configs()) == \
        len(LEVELS) * len(ACTION_SPACE["sampling.bg_strength"]) * len(ACTION_SPACE["seed"])
    # seed 2: same settings plus the second per-pair seed; its id and its default render carry the _s2 suffix
    s2 = {"subject.strength": "a012", "sampling.bg_strength": 0.44, "seed": 2}
    assert agent.render_overrides(s2)["sampling.seed"] == agent.SEEDS[2] and agent.grid_id(s2) == "a012_b44_s2"
    assert agent.default_id("a012_b44_s2") == "C00_s2" and agent.default_id("a012_b44") == "C00"


@pytest.mark.parametrize("good, path", [("a008", ["a020", "a012", "a008"]), ("g50", ["a020", "g50"])])
def test_rule_policy_walks_the_strength_axis_to_good(tmp_path, good, path):
    ctl = Controller(str(tmp_path / "log.jsonl"))
    s = ctl.start("P02", budget=5, policy=RulePolicy())
    ctl.run(s, RulePolicy(), fake_render(tmp_path), visibility_world(good))
    assert [t.config["subject.strength"] for t in s.trials] == path
    assert s.selected().config["subject.strength"] == good and s.done


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
        '```json\n{"config": {"subject.strength": "a008", "sampling.bg_strength": 0.52}, "rationale": "weaker subject"}\n```',
        '{"config": {"subject.strength": "a050"}}',          # invalid value -> repair
        '{"config": {"subject.strength": "a008", "sampling.bg_strength": 0.44}, "rationale": "go lower"}',
        "not json", "still not json",                        # -> fallback to rule
    ])
    llm = LLMPolicy(lambda prompt: next(replies), max_repairs=1)
    labels = iter(["too_visible", "too_visible", "missing"])
    ctl.run(s, llm, fake_render(tmp_path), lambda t: next(labels))
    why = [t.rationale for t in s.trials]
    assert any("llm: weaker subject" in w for w in why)
    assert any("fallback rule" in w for w in why)
    prompt = json.loads(llm.prompt(s))
    assert prompt["action_space"]["subject.strength"] == list(LEVELS) and "g50" in prompt["effects"]["subject.strength"]
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

    world = visibility_world("a008")
    configs = {}

    def render(cfg):
        path = fake_render(tmp_path)(cfg)
        configs[path] = cfg
        return path

    def observer_label(path):
        return world(agent.Trial(0, {k: configs[path][k] for k in ACTION_SPACE}, "x"))

    obs = Observer(fake_vision(observer_label))
    step_down = iter(["g50", "a008"])
    llm = LLMPolicy(lambda prompt: json.dumps({"config": {"subject.strength": next(step_down), "sampling.bg_strength": 0.52},
                                               "rationale": "too visible, weaker subject"}))
    ctl = Controller(str(tmp_path / "log.jsonl"))
    s = ctl.start("P11", budget=5, policy=llm)
    ctl.run(s, llm, render, lambda t: obs.judge(t.output, "s.png", "b.png", "fish")[0])
    assert s.selected() and s.selected().config["subject.strength"] == "a008"
    assert [t.config["subject.strength"] for t in s.trials] == ["g50", "a008"]


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
