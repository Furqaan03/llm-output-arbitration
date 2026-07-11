"""Orchestration: parallel critic dispatch (with graceful degradation) + disagreement
detection. A LangGraph-style state machine, but with an injectable critic function
so the fan-out/fan-in and disagreement logic are testable without live models."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

from src.critics.schema import CRITIC_ROLES, Critique

CriticFn = Callable[[str, str, str], Critique]  # (dimension, output, question) -> Critique


@dataclass
class Disagreement:
    kind: str            # "issue_presence" | "severity_gap" | "unique_finding"
    description: str
    dimensions: list[str] = field(default_factory=list)


@dataclass
class ArbitrationState:
    critiques: list[Critique]
    disagreements: list[Disagreement]
    all_passed: bool     # short-circuit signal: every critic gave a clean high score


def dispatch_critics(output: str, question: str, critic_fn: CriticFn) -> list[Critique]:
    """Fan out to all critics in parallel; fan in when all complete. A critic whose
    call fails is recorded as failed (graceful degradation) rather than aborting."""
    def run(dimension: str) -> Critique:
        try:
            return critic_fn(dimension, output, question)
        except Exception as exc:  # noqa: BLE001 — degrade, don't abort the whole arbitration
            return Critique(dimension=dimension, score=3, issues=[], self_confidence=0.0,
                            critic_model=CRITIC_ROLES[dimension]["model"], failed=True)

    with ThreadPoolExecutor(max_workers=len(CRITIC_ROLES)) as pool:
        return list(pool.map(run, CRITIC_ROLES.keys()))


def detect_disagreements(critiques: list[Critique]) -> list[Disagreement]:
    """Flags where critics disagree: on whether an issue exists, on severity by >2,
    or where one critic found issues the others missed entirely."""
    disagreements: list[Disagreement] = []
    active = [c for c in critiques if not c.failed]

    issue_counts = {c.dimension: len(c.issues) for c in active}
    with_issues = [d for d, n in issue_counts.items() if n > 0]
    without_issues = [d for d, n in issue_counts.items() if n == 0]
    if with_issues and without_issues:
        disagreements.append(Disagreement(
            kind="issue_presence",
            description=f"{with_issues} found issues; {without_issues} found none.",
            dimensions=with_issues + without_issues,
        ))

    scores = [(c.dimension, c.score) for c in active]
    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            if abs(scores[i][1] - scores[j][1]) > 2:
                disagreements.append(Disagreement(
                    kind="severity_gap",
                    description=f"{scores[i][0]} scored {scores[i][1]}, {scores[j][0]} scored {scores[j][1]}.",
                    dimensions=[scores[i][0], scores[j][0]],
                ))
    return disagreements


def arbitrate(output: str, question: str, critic_fn: CriticFn) -> ArbitrationState:
    critiques = dispatch_critics(output, question, critic_fn)
    active = [c for c in critiques if not c.failed]
    all_passed = bool(active) and all(c.score >= 5 and not c.issues for c in active)
    disagreements = [] if all_passed else detect_disagreements(critiques)
    return ArbitrationState(critiques=critiques, disagreements=disagreements, all_passed=all_passed)
