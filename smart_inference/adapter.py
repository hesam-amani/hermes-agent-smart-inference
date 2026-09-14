"""Hermes-neutral adapter helpers for Smart Inference."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .router import ModelCandidate, ModelRef


def _value(metadata: Any, name: str, default: Any = None) -> Any:
    if isinstance(metadata, Mapping):
        return metadata.get(name, default)
    return getattr(metadata, name, default)


def candidate_from_metadata(
    provider: str,
    model: str,
    metadata: Any,
    *,
    free: bool = False,
) -> ModelCandidate:
    context = _value(metadata, "context_window")
    try:
        context = int(context) if context is not None else None
    except (TypeError, ValueError):
        context = None

    modalities = tuple(_value(metadata, "input_modalities", ()) or ())
    name = f"{provider}/{model}".lower()
    family = str(_value(metadata, "family", "") or "").lower()

    # Hermes ModelInfo does not claim task-specific quality scores. These are
    # deliberately conservative priors, not fabricated provider metadata.
    coding = 0.75 if any(x in name or x in family for x in ("coder", "code", "dev")) else 0.25
    research = 0.70 if any(x in name or x in family for x in ("research", "reason", "thinking")) else 0.25
    financial = 0.55 if any(x in name for x in ("finance", "financial", "trader")) else 0.20
    quality = 0.65 if bool(_value(metadata, "reasoning", False)) else 0.50

    return ModelCandidate(
        ref=ModelRef(provider=provider, model=model),
        free=free,
        context_window=context,
        reasoning=bool(_value(metadata, "reasoning", False)),
        tools=bool(_value(metadata, "tool_call", False)),
        vision=bool(_value(metadata, "attachment", False) or "image" in modalities),
        structured_output=bool(_value(metadata, "structured_output", False)),
        quality=quality,
        coding=coding,
        research=research,
        financial=financial,
    )


__all__ = ["candidate_from_metadata"]
