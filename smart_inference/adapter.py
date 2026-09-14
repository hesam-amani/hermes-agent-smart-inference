"""Adapters from external model metadata into Smart Inference candidates.

This module deliberately knows nothing about Hermes runtime state, clients,
credentials, transports, retries, or fallback behavior.
"""

from __future__ import annotations

from typing import Any

from .router import ModelCandidate, ModelRef


def _number(metadata: Any, name: str, default: float = 0.0) -> float:
    value = getattr(metadata, name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _text(metadata: Any, *names: str) -> str:
    values: list[str] = []

    for name in names:
        value = getattr(metadata, name, None)
        if value:
            values.append(str(value))

    return " ".join(values).lower()


def _contains(text: str, *terms: str) -> bool:
    return any(term in text for term in terms)


def candidate_from_metadata(
    provider: str,
    model: str,
    metadata: Any,
    *,
    free: bool = False,
) -> ModelCandidate:
    """Convert provider/model metadata into a Smart Inference candidate.

    Hermes currently exposes strong capability metadata for things such as
    reasoning, tools, vision, structured output and context size.

    Coding/research/financial/quality are intentionally conservative heuristic
    priors because they are not first-class Hermes ModelInfo capabilities.
    """

    model_text = _text(
        metadata,
        "id",
        "name",
        "family",
        "provider_id",
    )

    context_window = int(
        _number(
            metadata,
            "context_window",
            _number(metadata, "max_input", 0.0),
        )
    )

    reasoning = bool(getattr(metadata, "reasoning", False))
    tools = bool(getattr(metadata, "tool_call", False))
    vision = bool(getattr(metadata, "attachment", False))

    modalities = getattr(metadata, "input_modalities", None)
    if modalities:
        try:
            modality_text = " ".join(str(item).lower() for item in modalities)
            vision = vision or "image" in modality_text
        except TypeError:
            pass

    structured_output = bool(
        getattr(metadata, "structured_output", False)
    )

    # Conservative model-family priors.
    coding = 0.75 if _contains(
        model_text,
        "coder",
        "coding",
        "code",
        "dev",
        "developer",
    ) else 0.25

    research = 0.70 if _contains(
        model_text,
        "research",
        "reason",
        "thinking",
        "think",
    ) else 0.25

    financial = 0.55 if _contains(
        model_text,
        "finance",
        "financial",
        "trader",
        "trading",
    ) else 0.20

    quality = 0.65 if reasoning else 0.50

    return ModelCandidate(
        ref=ModelRef(
            provider=str(provider),
            model=str(model),
        ),
        free=free,
        context_window=context_window,
        reasoning=reasoning,
        tools=tools,
        vision=vision,
        structured_output=structured_output,
        quality=quality,
        coding=coding,
        research=research,
        financial=financial,
    )


def metadata_is_free(metadata: Any) -> bool:
    """Return True only when Hermes explicitly reports zero input/output cost."""

    input_cost = _number(metadata, "cost_input")
    output_cost = _number(metadata, "cost_output")

    return input_cost == 0.0 and output_cost == 0.0
