"""LLM-backed critic: routes each dimension through its assigned provider.

Different models per critic is deliberate — disagreement between models is the
most valuable signal; shared models would share blind spots."""
from __future__ import annotations

import json

from src.critics.schema import CRITIC_ROLES, Critique, Issue


def _call_openai(model: str, system: str, user: str) -> dict:
    from openai import OpenAI

    resp = OpenAI().chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(resp.choices[0].message.content or "{}")


def _call_anthropic(model: str, system: str, user: str) -> dict:
    from anthropic import Anthropic

    resp = Anthropic().messages.create(
        model=model, max_tokens=1024, system=system,
        messages=[{"role": "user", "content": user + "\n\nRespond with only valid JSON."}],
    )
    text = "".join(b.text for b in resp.content if hasattr(b, "text"))
    return json.loads(text)


def _call_ollama(model: str, system: str, user: str) -> dict:
    import httpx

    with httpx.Client(timeout=120) as client:
        resp = client.post("http://localhost:11434/api/chat", json={
            "model": model, "format": "json", "stream": False,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        })
        resp.raise_for_status()
        return json.loads(resp.json()["message"]["content"])


def llm_critic(dimension: str, output: str, question: str) -> Critique:
    role = CRITIC_ROLES[dimension]
    model = role["model"]
    system = (
        f"You are the {dimension} critic. {role['instruction']} "
        'Respond as JSON: {"score": 1-5, "issues": [{"quote": "...", "problem": "...", "severity": 1-5}], '
        '"self_confidence": 0.0-1.0}.'
    )
    user = f"Question: {question}\n\nOutput to evaluate:\n{output}"

    if model.startswith("gpt"):
        parsed = _call_openai(model, system, user)
    elif model.startswith("claude"):
        parsed = _call_anthropic(model, system, user)
    else:
        parsed = _call_ollama(model, system, user)

    return Critique(
        dimension=dimension,
        score=int(parsed.get("score", 3)),
        issues=[Issue(**i) for i in parsed.get("issues", [])],
        self_confidence=float(parsed.get("self_confidence", 0.5)),
        critic_model=model,
    )
