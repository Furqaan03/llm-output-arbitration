"""Structured critique format shared by all critics."""
from __future__ import annotations

from pydantic import BaseModel, Field


class Issue(BaseModel):
    quote: str            # the span of the original output the issue refers to
    problem: str
    severity: int         # 1 (minor) .. 5 (critical)


class Critique(BaseModel):
    dimension: str        # "accuracy" | "logic" | "completeness"
    score: int            # 1 (bad) .. 5 (good)
    issues: list[Issue] = Field(default_factory=list)
    self_confidence: float  # 0..1 — how sure the critic is of its own assessment
    critic_model: str = ""
    failed: bool = False    # set if the critic's API call failed


# Each critic role is assigned a different provider/model so they don't share blind spots.
CRITIC_ROLES = {
    "accuracy": {"model": "gpt-4o", "instruction": "Check whether claims are verifiable and internally consistent."},
    "logic": {"model": "claude-sonnet-4-5", "instruction": "Check whether the reasoning follows and conclusions are supported."},
    "completeness": {"model": "llama3", "instruction": "Check whether the response addresses all parts of the question and flag gaps."},
}
