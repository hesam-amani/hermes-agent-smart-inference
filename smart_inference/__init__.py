"""Deterministic Smart Inference model-selection engine."""

from .smart_router import (
    CostPolicy,
    InferenceDecision,
    InferenceRequest,
    ModelCandidate,
    ModelRef,
    Requirements,
    SmartInference,
    Task,
    choose,
)

__all__ = [
    "CostPolicy",
    "InferenceDecision",
    "InferenceRequest",
    "ModelCandidate",
    "ModelRef",
    "Requirements",
    "SmartInference",
    "Task",
    "choose",
]
