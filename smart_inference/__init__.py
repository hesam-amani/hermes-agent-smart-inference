"""Small, provider-neutral Smart Inference engine."""

from .adapter import candidate_from_metadata
from .router import (
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
    "candidate_from_metadata",
    "choose",
]
