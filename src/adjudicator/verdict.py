"""Adjudicator: weigh critiques + disagreements into a single verdict."""
from __future__ import annotations

import json

from openai import OpenAI
from pydantic import BaseModel, Field

from src.critics.schema import Critique
from src.orchestration.graph import Disagreement


class ConfirmedIssue(BaseModel):
    problem: str
    severity: int
    evidence: str


class Verdict(BaseModel):
    quality_score: int          # 1..10
    confidence: str             # "low" | "medium" | "high"
    confirmed_issues: list[ConfirmedIssue] = Field(default_factory=list)
    dismissed_flags: list[str] = Field(default_factory=list)
    summary: str = ""


def _confidence_label(critiques: list[Critique], had_failure: bool) -> str:
    active = [c for c in critiques if not c.failed]
    if not active:
        return "low"
    avg_self = sum(c.self_confidence for c in active) / len(active)
    if had_failure or avg_self < 0.5:
        return "low"
    return "high" if avg_self >= 0.8 else "medium"


def build_verdict(output: str, question: str, critiques: list[Critique],
                  disagreements: list[Disagreement], all_passed: bool,
                  client: OpenAI | None = None) -> Verdict:
    """Short-circuits to a clean pass if all critics agreed; otherwise the adjudicator
    LLM reasons through each disagreement."""
    had_failure = any(c.failed for c in critiques)

    if all_passed:
        return Verdict(quality_score=9, confidence=_confidence_label(critiques, had_failure),
                       confirmed_issues=[], dismissed_flags=[], summary="All critics agreed the output is high quality.")

    client = client or OpenAI()
    critiques_json = json.dumps([c.model_dump() for c in critiques], indent=2)
    disagreements_json = json.dumps([d.__dict__ for d in disagreements], indent=2)
    prompt = (
        "You are an adjudicator. Given the original output, three critic reports, and detected "
        "disagreements, resolve conflicts and produce a final verdict. Reason through each disagreement.\n\n"
        f"ORIGINAL OUTPUT:\n{output}\n\nQUESTION:\n{question}\n\n"
        f"CRITIQUES:\n{critiques_json}\n\nDISAGREEMENTS:\n{disagreements_json}\n\n"
        'Respond as JSON: {"quality_score": 1-10, "confirmed_issues": [{"problem": "...", "severity": 1-5, '
        '"evidence": "..."}], "dismissed_flags": ["issues you overruled, with reasoning"], "summary": "one paragraph"}.'
    )
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    parsed = json.loads(resp.choices[0].message.content or "{}")
    return Verdict(
        quality_score=int(parsed.get("quality_score", 5)),
        confidence=_confidence_label(critiques, had_failure),
        confirmed_issues=[ConfirmedIssue(**i) for i in parsed.get("confirmed_issues", [])],
        dismissed_flags=parsed.get("dismissed_flags", []),
        summary=parsed.get("summary", ""),
    )
