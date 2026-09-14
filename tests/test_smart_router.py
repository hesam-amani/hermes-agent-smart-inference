from agent.smart_router import (
    CostPolicy,
    InferenceRequest,
    ModelCandidate,
    ModelRef,
    Requirements,
    SmartInference,
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


def test_free_only_excludes_paid_models():
    inference = SmartInference()

    models = [
        candidate("paid", free=False),
        candidate("free", free=True),
    ]

    decision = inference.choose(
        InferenceRequest(
            "hello",
            cost_policy=CostPolicy.FREE_ONLY,
        ),
        models,
    )

    assert decision.primary.model == "free"
    assert all(
        model.model != "paid"
        for model in decision.ranked
    )


def test_tools_are_hard_capability_requirement():
    inference = SmartInference()

    models = [
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
    ]

    decision = inference.choose(
        InferenceRequest(
            "use a tool",
            requirements=Requirements(tools=True),
        ),
        models,
    )

    assert decision.primary.model == "tools"


def test_coding_task_prefers_coding_model():
    inference = SmartInference()

    models = [
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
    ]

    decision = inference.choose(
        InferenceRequest(
            "fix this Python function bug"
        ),
        models,
    )

    assert decision.task is Task.CODING
    assert decision.primary.model == "coder"


def test_reasoning_task_prefers_reasoning_model():
    inference = SmartInference()

    models = [
        candidate(
            "general",
            free=True,
            quality=0.9,
        ),
        candidate(
            "reasoner",
            free=True,
            quality=0.8,
            reasoning=True,
        ),
    ]

    decision = inference.choose(
        InferenceRequest(
            "prove why this solution is correct"
        ),
        models,
    )

    assert decision.task is Task.REASONING
    assert decision.primary.model == "reasoner"


def test_multiple_requirements_are_hard_filters():
    inference = SmartInference()

    models = [
        candidate(
            "partial",
            free=True,
            reasoning=True,
            coding=True,
            context_window=50_000,
        ),
        candidate(
            "complete",
            free=True,
            reasoning=True,
            coding=True,
            context_window=128_000,
        ),
    ]

    decision = inference.choose(
        InferenceRequest(
            "work on the whole codebase",
            requirements=Requirements(
                reasoning=True,
                coding=True,
                long_context=True,
            ),
        ),
        models,
    )

    assert decision.primary.model == "complete"


def test_structured_output_is_hard_requirement():
    inference = SmartInference()

    models = [
        candidate(
            "plain",
            free=True,
            structured_output=False,
        ),
        candidate(
            "structured",
            free=True,
            structured_output=True,
        ),
    ]

    decision = inference.choose(
        InferenceRequest(
            "return JSON",
            requirements=Requirements(
                structured_output=True,
            ),
        ),
        models,
    )

    assert decision.primary.model == "structured"


def test_ranking_is_deterministic_on_ties():
    inference = SmartInference()

    models = [
        candidate(
            "z-model",
            free=True,
            quality=0.8,
        ),
        candidate(
            "a-model",
            free=True,
            quality=0.8,
        ),
    ]

    decision = inference.choose(
        InferenceRequest("hello"),
        models,
    )

    assert [
        model.model
        for model in decision.ranked
    ] == [
        "a-model",
        "z-model",
    ]


def test_no_candidate_is_explicit_failure():
    inference = SmartInference()

    try:
        inference.choose(
            InferenceRequest(
                "hello",
                cost_policy=CostPolicy.FREE_ONLY,
            ),
            [
                candidate(
                    "paid",
                    free=False,
                )
            ],
        )
    except LookupError as exc:
        assert "No model satisfies" in str(exc)
    else:
        raise AssertionError(
            "expected LookupError"
        )
