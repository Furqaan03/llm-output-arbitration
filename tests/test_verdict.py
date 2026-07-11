from src.adjudicator.verdict import build_verdict
from src.critics.schema import Critique


def _critique(dim, score, conf=0.9, failed=False):
    return Critique(dimension=dim, score=score, issues=[], self_confidence=conf, critic_model="test", failed=failed)


def test_all_passed_returns_clean_verdict_without_llm():
    critiques = [_critique(d, 5) for d in ("accuracy", "logic", "completeness")]
    # all_passed=True short-circuits — no client needed
    verdict = build_verdict("out", "q", critiques, [], all_passed=True)
    assert verdict.quality_score == 9
    assert verdict.confirmed_issues == []


def test_confidence_high_with_confident_critics():
    critiques = [_critique(d, 5, conf=0.9) for d in ("accuracy", "logic", "completeness")]
    verdict = build_verdict("out", "q", critiques, [], all_passed=True)
    assert verdict.confidence == "high"


def test_confidence_low_when_critic_failed():
    critiques = [_critique("accuracy", 5, conf=0.9), _critique("logic", 3, failed=True),
                 _critique("completeness", 5, conf=0.9)]
    verdict = build_verdict("out", "q", critiques, [], all_passed=True)
    assert verdict.confidence == "low"
