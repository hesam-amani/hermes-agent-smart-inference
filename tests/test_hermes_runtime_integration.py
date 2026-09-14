from types import SimpleNamespace

from agent.hermes_runtime_integration import route_turn, routed_turn
from agent.smart_router import (
    CostPolicy,
    ModelCandidate,
    ModelRef,
    Requirements,
    Task,
)


def candidate(name, **kwargs):
    return ModelCandidate(
        ref=ModelRef(
            provider="test",
            model=name,
        ),
        **kwargs,
    )


def test_route_turn_returns_smart_inference_decision():
    decision = route_turn(
        prompt="fix this Python bug",
        candidates=[
            candidate(
                "general",
                free=True,
                quality=0.9,
                coding=0.4,
            ),
            candidate(
                "coder",
                free=True,
                quality=0.8,
                coding=1.0,
            ),
        ],
    )

    assert decision.task is Task.CODING
    assert decision.primary.model == "coder"


def test_route_turn_respects_explicit_requirements():
    decision = route_turn(
        prompt="do the task",
        candidates=[
            candidate(
                "no-tools",
                free=True,
                tools=False,
            ),
            candidate(
                "tools",
                free=True,
                tools=True,
            ),
        ],
        requirements=Requirements(tools=True),
    )

    assert decision.primary.model == "tools"


def test_routed_turn_restores_runtime_state():
    agent = SimpleNamespace(
        provider="original-provider",
        model="original-model",
        base_url="https://original.example",
        api_key="original-key",
        api_mode="original",
        _fallback_chain=[],
        _fallback_model=None,
        _fallback_index=0,
    )

    calls = []

    def switch_model(**kwargs):
        calls.append(kwargs)
        agent.provider = kwargs["new_provider"]
        agent.model = kwargs["new_model"]

    agent.switch_model = switch_model

    decision = route_turn(
        prompt="fix this Python bug",
        candidates=[
            candidate(
                "coder",
                free=True,
                coding=1.0,
                reasoning=True,
            ),
        ],
    )

    with routed_turn(agent, decision):
        assert agent.provider == "test"
        assert agent.model == "coder"

    assert calls == [
        {
            "new_model": "coder",
            "new_provider": "test",
        }
    ]

    assert agent.provider == "original-provider"
    assert agent.model == "original-model"
    assert agent.base_url == "https://original.example"
    assert agent.api_key == "original-key"
    assert agent.api_mode == "original"
    assert agent._fallback_chain == []
    assert agent._fallback_model is None
    assert agent._fallback_index == 0


def test_routed_turn_does_not_switch_when_already_on_selected_model():
    agent = SimpleNamespace(
        provider="test",
        model="coder",
        base_url="https://example",
        api_key="key",
        api_mode="api",
        _fallback_chain=[],
        _fallback_model=None,
        _fallback_index=0,
    )

    calls = []

    def switch_model(**kwargs):
        calls.append(kwargs)

    agent.switch_model = switch_model

    decision = route_turn(
        prompt="fix this Python bug",
        candidates=[
            candidate(
                "coder",
                free=True,
                coding=1.0,
            ),
        ],
        cost_policy=CostPolicy.FREE_ONLY,
    )

    with routed_turn(agent, decision):
        assert agent.provider == "test"
        assert agent.model == "coder"

    assert calls == []
