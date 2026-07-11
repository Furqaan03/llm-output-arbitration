# LLM Output Arbitration System

A multi-agent pipeline that takes any LLM-generated output and routes it to three
competing critic models that independently evaluate it for accuracy, logical
consistency, and completeness — then synthesizes their critiques (and their
disagreements) into a single confidence-scored verdict with actionable callouts.

## Why this exists

Instead of building yet another system that *generates* answers, this one
*catches bad answers*. It demonstrates the evaluation mindset AI teams hire for
but rarely see — and the multi-model design catches issues single-model
self-evaluation misses.

## Architecture

```
src/critics/schema.py         structured Critique format + the 3 critic roles, each
                               bound to a DIFFERENT provider (GPT-4o / Claude / Llama)
src/critics/llm_critic.py     routes each dimension through its assigned provider
src/orchestration/graph.py    parallel critic dispatch (fan-out/fan-in) with graceful
                               degradation + disagreement detection + all-pass short-circuit
src/adjudicator/verdict.py    weighs critiques + disagreements into a final verdict
src/api/main.py               FastAPI: /v1/arbitrate, /v1/arbitrate/batch
```

## Design decisions

- **Each critic runs on a different model, on purpose.** The accuracy critic is
  GPT-4o, logic is Claude, completeness is Llama. The disagreements *between*
  models are the most valuable signal — three critics sharing one model would share
  one set of blind spots and rubber-stamp each other.
- **Critics fan out in parallel, not sequentially.** All three dispatch
  simultaneously and fan in when complete, keeping latency to the slowest single
  critic rather than the sum — this is how real multi-agent systems are built.
- **Graceful degradation, not abort.** If one critic's provider call fails, it's
  recorded as `failed` and the system still produces a verdict from the remaining
  critics — with confidence downgraded to reflect the missing dimension. One dead
  provider doesn't take down the whole arbitration.
- **All-agree short-circuits the adjudicator.** If every critic returns a clean
  high score with no issues, there's nothing to adjudicate — it returns a
  high-confidence pass without spending an adjudicator LLM call.
- **The critic function is injected.** Orchestration, disagreement detection, the
  short-circuit, and confidence labeling are all tested against a fake critic
  function — so the actual multi-agent logic is verified offline, no three API keys
  required.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
cp .env.example .env       # OPENAI_API_KEY + ANTHROPIC_API_KEY (Llama via local Ollama)
uvicorn src.api.main:app --reload
```

## Example

```bash
curl -X POST localhost:8000/v1/arbitrate -H "Content-Type: application/json" \
  -d '{"output": "The Eiffel Tower, built in 1802, is in Berlin.", "question": "Where is the Eiffel Tower?"}'
# -> {"verdict": {"quality_score": 2, "confirmed_issues": [...], "confidence": "high"}, "disagreements": [...]}
```

## Tests

```bash
pytest tests/ -v
```

10 tests covering parallel dispatch, graceful degradation on critic failure,
disagreement detection (issue-presence and severity-gap), the all-pass
short-circuit, and adjudicator confidence labeling (high / low-on-failure) — all
offline via an injected fake critic, no API keys required.

## Docker

```bash
docker compose up --build   # API + local Ollama for the Llama-backed critic
```

## Status

Phases 1-3 complete (critic architecture with per-provider routing, parallel
orchestration + disagreement detection, adjudicator) plus the arbitration API.
The LangGraph state machine is implemented as an explicit fan-out/fan-in with an
injectable critic function (testable offline); Phase 4's verdict-explorer UI and
Phase 5's critic-behavior analytics are not built.
