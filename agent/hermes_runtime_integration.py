"""Hermes runtime integration helpers.

Smart Inference decides which model should handle a turn. Hermes remains
responsible for clients, credentials, transports, retries, cooldowns, and
fallback execution.

Automatic routing is turn-scoped: Hermes' native model-switch machinery is
used for activation, then the original runtime state is restored when the
turn ends.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import Any

from .hermes_adapter import build_candidates
from .smart_router import (
    CostPolicy,
    InferenceDecision,
    InferenceRequest,
    ModelCandidate,
    ModelRef,
    Requirements,
    SmartInference,
)

_SNAPSHOT_NAMES = (
    "model",
    "provider",
    "base_url",
    "api_mode",
    "api_key",
    "client",
    "_anthropic_client",
    "_anthropic_api_key",
    "_anthropic_base_url",
    "_is_anthropic_oauth",
    "_config_context_length",
    "_bedrock_region",
    "_use_prompt_caching",
    "_use_native_cache_layout",
    "_cached_system_prompt",
    "_fallback_chain",
    "_fallback_model",
    "_fallback_index",
    "_fallback_activated",
    "_primary_runtime",
    "_client_kwargs",
)

_LOGGER = logging.getLogger(__name__)


def _runtime_snapshot(agent: Any) -> dict[str, Any]:
    """Capture runtime state without copying live client objects."""

    snapshot: dict[str, Any] = {}
    missing = object()

    for name in _SNAPSHOT_NAMES:
        value = getattr(agent, name, missing)

        if value is missing:
            continue

        if name in {
            "_fallback_chain",
            "_client_kwargs",
            "_primary_runtime",
        }:
            if isinstance(value, dict):
                snapshot[name] = dict(value)
            elif isinstance(value, list):
                snapshot[name] = [
                    dict(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                snapshot[name] = value
        else:
            snapshot[name] = value

    return snapshot


def _restore_snapshot(
    agent: Any,
    snapshot: dict[str, Any],
) -> None:
    """Restore a previously captured runtime snapshot."""

    for name, value in snapshot.items():
        try:
            setattr(agent, name, value)
        except (AttributeError, TypeError) as exc:
            _LOGGER.warning(
                "Unable to restore runtime attribute %s: %s",
                name,
                exc,
            )
            continue


def _candidate_metadata_from_hermes(
    provider: str,
    model: str,
) -> Any:
    """Resolve rich model metadata through Hermes' canonical metadata layer."""

    from agent.models_dev import get_model_info  # pyright: ignore[reportMissingImports]

    return get_model_info(provider, model)


def discover_candidates(
    providers: Iterable[str],
    *,
    provider_models: dict[str, Iterable[str]],
    free_models: set[tuple[str, str]] | None = None,
) -> list[ModelCandidate]:
    """Translate Hermes' already-discovered models into candidates.

    Discovery itself remains Hermes' responsibility. This function only
    translates an existing provider/model inventory.
    """

    selected_models = {
        provider: provider_models[provider]
        for provider in providers
        if provider in provider_models
    }

    return build_candidates(
        selected_models,
        _candidate_metadata_from_hermes,
        free_models=free_models,
    )


def route_turn(
    *,
    prompt: str,
    candidates: Iterable[ModelCandidate],
    cost_policy: CostPolicy = CostPolicy.FREE_ONLY,
    requirements: Requirements | None = None,
) -> InferenceDecision:
    """Perform the per-turn Smart Inference decision."""

    request = InferenceRequest(
        prompt=prompt,
        requirements=requirements,
        cost_policy=cost_policy,
    )

    return SmartInference().choose(
        request,
        candidates,
    )


def _as_switch_kwargs(
    ref: ModelRef,
) -> dict[str, Any]:
    """Build the minimal target passed to Hermes' native model switch."""

    return {
        "new_model": ref.model,
        "new_provider": ref.provider,
    }


@contextmanager
def routed_turn(
    agent: Any,
    decision: InferenceDecision,
) -> Iterator[InferenceDecision]:
    """Activate one inference decision for exactly one user turn."""

    snapshot = _runtime_snapshot(agent)

    try:
        current_provider = str(
            snapshot.get("provider") or ""
        ).strip()

        current_model = str(
            snapshot.get("model") or ""
        ).strip()

        if (
            decision.primary.provider == current_provider
            and decision.primary.model == current_model
        ):
            yield decision
            return

        agent.switch_model(
            **_as_switch_kwargs(decision.primary)
        )

        if current_provider and current_model:
            original_fallback = {
                "provider": current_provider,
                "model": current_model,
                "base_url": snapshot.get("base_url") or "",
                "api_key": snapshot.get("api_key") or "",
                "api_mode": snapshot.get("api_mode") or "",
            }

            current_chain = list(
                getattr(agent, "_fallback_chain", []) or []
            )

            duplicate = any(
                isinstance(item, dict)
                and str(
                    item.get("provider") or ""
                ).strip().lower()
                == current_provider.lower()
                and str(
                    item.get("model") or ""
                ).strip()
                == current_model
                for item in current_chain
            )

            if not duplicate:
                agent._fallback_chain = [
                    original_fallback,
                    *current_chain,
                ]
                agent._fallback_model = original_fallback
                agent._fallback_index = 0

        yield decision

    finally:
        _restore_snapshot(agent, snapshot)


__all__ = [
    "discover_candidates",
    "route_turn",
    "routed_turn",
]
