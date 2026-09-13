# Smart Inference

A small decision engine for automatic model selection in Hermes Agent.

## Purpose

Smart Inference answers one question:

> **Given this request and the models Hermes already knows about, which model should handle it?**

```text
request
  ↓
Smart Inference
  ├─ classify task
  ├─ infer requirements
  ├─ filter capabilities
  └─ rank candidates
  ↓
provider + model
  ↓
Hermes executes
```

The public API is intentionally small:

```python
decision = inference.choose(request, candidates)
```

The result contains a primary `provider/model` and a ranked list of alternatives.

## Boundary

Smart Inference does **not** own:

- provider clients or API calls
- credentials
- transport
- retries or cooldowns
- streaming
- provider fallback execution
- a second model catalog
- a daemon, gateway, database, embeddings, or RAG system

Hermes remains the runtime and source of truth for provider/model discovery and
capabilities. The Hermes adapter only translates that existing metadata into the
small candidate shape consumed by the decision engine.

## Requirements, not one-label routing

A request can require several capabilities at once:

```text
coding + reasoning + long_context
```

rather than being forced into a single `CODING` or `REASONING` bucket.

Hard requirements filter candidates first. Ranking then considers task fit,
model quality, and context capacity.

## Integration seam

Hermes already has a shared model-switch/resolution pipeline. Smart Inference
should sit immediately before that pipeline and return a plain provider/model
selection. Hermes then performs its normal provider resolution, credential
handling, API-mode selection, and execution.

Explicit `/model` selections remain authoritative; automatic inference must
never silently replace an explicit user choice.
