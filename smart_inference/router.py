"""Small deterministic decision engine for Hermes model selection."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum


class CostPolicy(str, Enum):
    FREE_ONLY = "free_only"
    FREE_PREFERRED = "free_preferred"
    ANY = "any"


class Task(str, Enum):
    CODING = "coding"
    REASONING = "reasoning"
    RESEARCH = "research"
    FINANCIAL = "financial"
    VISION = "vision"
    GENERAL = "general"


@dataclass(frozen=True)
class Requirements:
    reasoning: bool = False
    coding: bool = False
    research: bool = False
    financial: bool = False
    vision: bool = False
    tools: bool = False
    structured_output: bool = False
    long_context: bool = False
    min_context: int | None = None


@dataclass(frozen=True)
class InferenceRequest:
    prompt: str
    requirements: Requirements | None = None
    cost_policy: CostPolicy = CostPolicy.FREE_ONLY


@dataclass(frozen=True)
class ModelRef:
    provider: str
    model: str

    @property
    def key(self) -> str:
        return f"{self.provider}/{self.model}"


@dataclass(frozen=True)
class ModelCandidate:
    ref: ModelRef
    free: bool = False
    context_window: int | None = None
    reasoning: bool = False
    tools: bool = False
    vision: bool = False
    structured_output: bool = False
    quality: float = 0.5
    coding: float = 0.0
    research: float = 0.0
    financial: float = 0.0


@dataclass(frozen=True)
class InferenceDecision:
    task: Task
    requirements: Requirements
    primary: ModelRef
    ranked: tuple[ModelRef, ...]
    scores: dict[str, float]
    rationale: str

    @property
    def fallbacks(self) -> tuple[ModelRef, ...]:
        return self.ranked[1:]


def _has_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


class SmartInference:
    """Pure deterministic decision engine with no provider/runtime ownership."""

    def classify(self, prompt: str) -> Task:
        text = prompt.lower()
        if _has_any(text, ("screenshot", "image", "photo", "picture", "vision")):
            return Task.VISION
        if _has_any(text, ("xauusd", "forex", "trading", "trade", "market", "ohlc", "position")):
            return Task.FINANCIAL
        if _has_any(text, ("research", "sources", "cite", "citations", "investigate", "latest", "compare")):
            return Task.RESEARCH
        if _has_any(text, ("code", "python", "typescript", "javascript", "bug", "compile", "function", "repo", "repository", "codebase", "api")):
            return Task.CODING
        if _has_any(text, ("prove", "derive", "analyze", "analyse", "reason", "why", "solve", "architecture")):
            return Task.REASONING
        return Task.GENERAL

    def infer_requirements(self, prompt: str, task: Task | None = None) -> Requirements:
        text = prompt.lower()
        return Requirements(
            tools=_has_any(text, ("use a tool", "use tools", "call an api", "browse", "search the web", "run a command")),
            structured_output=_has_any(text, ("json", "structured output", "schema", "return a table")),
            long_context=_has_any(text, ("whole repository", "entire repository", "whole codebase", "large document", "long context", "entire codebase")),
        )

    def filter_candidates(self, candidates: Iterable[ModelCandidate], request: InferenceRequest, requirements: Requirements) -> list[ModelCandidate]:
        result: list[ModelCandidate] = []
        for candidate in candidates:
            if request.cost_policy is CostPolicy.FREE_ONLY and not candidate.free:
                continue
            if requirements.reasoning and not candidate.reasoning:
                continue
            if requirements.coding and candidate.coding <= 0:
                continue
            if requirements.research and candidate.research <= 0:
                continue
            if requirements.financial and candidate.financial <= 0:
                continue
            if requirements.vision and not candidate.vision:
                continue
            if requirements.tools and not candidate.tools:
                continue
            if requirements.structured_output and not candidate.structured_output:
                continue
            if requirements.min_context is not None and (candidate.context_window or 0) < requirements.min_context:
                continue
            if requirements.long_context and (candidate.context_window or 0) < 100_000:
                continue
            result.append(candidate)
        return result

    def score(self, candidate: ModelCandidate, task: Task, policy: CostPolicy) -> float:
        task_fit = {
            Task.CODING: candidate.coding,
            Task.RESEARCH: candidate.research,
            Task.FINANCIAL: candidate.financial,
            Task.REASONING: 1.0 if candidate.reasoning else 0.0,
            Task.VISION: 1.0 if candidate.vision else 0.0,
            Task.GENERAL: candidate.quality,
        }[task]
        context_bonus = min((candidate.context_window or 0) / 200_000, 1.0)
        free_bonus = 0.15 if policy is CostPolicy.FREE_PREFERRED and candidate.free else 0.0
        return 0.55 * task_fit + 0.30 * candidate.quality + 0.15 * context_bonus + free_bonus

    def choose(self, request: InferenceRequest, candidates: Iterable[ModelCandidate]) -> InferenceDecision:
        task = self.classify(request.prompt)
        requirements = request.requirements if request.requirements is not None else self.infer_requirements(request.prompt, task)
        filtered = self.filter_candidates(candidates, request, requirements)
        if not filtered:
            raise LookupError("No model satisfies the inferred requirements and cost policy")
        ranked_candidates = sorted(filtered, key=lambda c: (-self.score(c, task, request.cost_policy), c.ref.key))
        scores = {c.ref.key: self.score(c, task, request.cost_policy) for c in ranked_candidates}
        ranked = tuple(c.ref for c in ranked_candidates)
        primary = ranked[0]
        return InferenceDecision(
            task=task,
            requirements=requirements,
            primary=primary,
            ranked=ranked,
            scores=scores,
            rationale=f"{primary.key} selected for {task.value}: meets all hard requirements and has the highest deterministic score ({scores[primary.key]:.3f}) among eligible candidates.",
        )


def choose(request: InferenceRequest, candidates: Iterable[ModelCandidate]) -> InferenceDecision:
    return SmartInference().choose(request, candidates)
