"""Thin translation layer between Hermes metadata and Smart Inference."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from .smart_router import (
    CostPolicy,
    InferenceDecision,
    InferenceRequest,
    ModelCandidate,
    ModelRef,
    Requirements,
    choose,
)


def _value(
    metadata: Any,
    name: str,
    default: Any = None,
) -> Any:
    if isinstance(metadata, Mapping):
        return metadata.get(name, default)

    return getattr(metadata, name, default)


def candidate_from_hermes_metadata(
    provider: str,
    model: str,
    metadata: Any,
    *,
    free_models: set[tuple[str, str]] | None = None,
) -> ModelCandidate:
    """Translate Hermes model metadata into a Smart Inference candidate."""

    free_keys = {
        (
            str(provider_name).strip().lower(),
            str(model_name).strip().lower(),
        )
        for provider_name, model_name in (free_models or set())
    }

    key = (
        provider.strip().lower(),
        model.strip().lower(),
    )

    context = _value(metadata, "context_window", None)

    try:
        context = int(context) if context is not None else None
    except (TypeError, ValueError):
        context = None

    modalities = tuple(
        _value(metadata, "input_modalities", ()) or ()
    )

    return ModelCandidate(
        ref=ModelRef(
            provider=provider,
            model=model,
        ),
        free=key in free_keys,
        context_window=context,
        reasoning=bool(
            _value(metadata, "reasoning", False)
        ),
        tools=bool(
            _value(metadata, "tool_call", False)
        ),
        vision=bool(
            _value(metadata, "attachment", False)
            or "image" in modalities
        ),
        structured_output=bool(
            _value(metadata, "structured_output", False)
        ),
        quality=float(
            _value(metadata, "quality", 0.5) or 0.5
        ),
        coding=float(
            _value(metadata, "coding", 0.0) or 0.0
        ),
        research=float(
            _value(metadata, "research", 0.0) or 0.0
        ),
        financial=float(
            _value(metadata, "financial", 0.0) or 0.0
        ),
    )


def build_candidates(
    provider_models: Mapping[str, Iterable[str]],
    metadata_lookup: Callable[[str, str], Any],
    *,
    free_models: set[tuple[str, str]] | None = None,
) -> list[ModelCandidate]:
    """Build candidates from Hermes' existing provider/model inventory."""

    return [
        candidate_from_hermes_metadata(
            provider,
            model,
            metadata_lookup(provider, model),
            free_models=free_models,
        )
        for provider, models in provider_models.items()
        for model in models
    ]


def infer_hermes_request(
    prompt: str,
    *,
    cost_policy: CostPolicy = CostPolicy.FREE_ONLY,
    requirements: Requirements | None = None,
) -> InferenceRequest:
    """Translate a Hermes turn into a Smart Inference request."""

    return InferenceRequest(
        prompt=prompt,
        requirements=requirements,
        cost_policy=cost_policy,
    )


def choose_hermes_model(
    prompt: str,
    candidates: Iterable[ModelCandidate],
    *,
    cost_policy: CostPolicy = CostPolicy.FREE_ONLY,
    requirements: Requirements | None = None,
) -> InferenceDecision:
    """Choose a model without performing any provider/runtime operation."""

    request = infer_hermes_request(
        prompt,
        cost_policy=cost_policy,
        requirements=requirements,
    )

    return choose(request, candidates)


__all__ = [
    "build_candidates",
    "candidate_from_hermes_metadata",
    "choose_hermes_model",
    "infer_hermes_request",
]
