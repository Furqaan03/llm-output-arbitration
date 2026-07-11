from src.critics.schema import Critique, Issue
from src.orchestration.graph import arbitrate, detect_disagreements, dispatch_critics


def _critique(dim, score, issues=None, conf=0.9):
    return Critique(dimension=dim, score=score, issues=issues or [], self_confidence=conf, critic_model="test")


def _make_critic_fn(mapping):
    def fn(dimension, output, question):
        return mapping[dimension]
    return fn


def test_dispatch_runs_all_critics():
    mapping = {
        "accuracy": _critique("accuracy", 4),
        "logic": _critique("logic", 5),
        "completeness": _critique("completeness", 3),
    }
    critiques = dispatch_critics("out", "q", _make_critic_fn(mapping))
    assert {c.dimension for c in critiques} == {"accuracy", "logic", "completeness"}


def test_graceful_degradation_on_critic_failure():
    def failing(dimension, output, question):
        if dimension == "logic":
            raise RuntimeError("provider down")
        return _critique(dimension, 4)

    critiques = dispatch_critics("out", "q", failing)
    logic = next(c for c in critiques if c.dimension == "logic")
    assert logic.failed is True
    # other critics still succeeded
    assert all(not c.failed for c in critiques if c.dimension != "logic")


def test_disagreement_on_issue_presence():
    critiques = [
        _critique("accuracy", 4, issues=[Issue(quote="x", problem="p", severity=3)]),
        _critique("logic", 5),
        _critique("completeness", 5),
    ]
    dis = detect_disagreements(critiques)
    assert any(d.kind == "issue_presence" for d in dis)


def test_disagreement_on_severity_gap():
    critiques = [_critique("accuracy", 1), _critique("logic", 5), _critique("completeness", 4)]
    dis = detect_disagreements(critiques)
    assert any(d.kind == "severity_gap" for d in dis)


def test_all_passed_short_circuits():
    mapping = {d: _critique(d, 5) for d in ("accuracy", "logic", "completeness")}
    state = arbitrate("out", "q", _make_critic_fn(mapping))
    assert state.all_passed is True
    assert state.disagreements == []


def test_not_all_passed_when_issues_exist():
    mapping = {
        "accuracy": _critique("accuracy", 5, issues=[Issue(quote="x", problem="p", severity=2)]),
        "logic": _critique("logic", 5),
        "completeness": _critique("completeness", 5),
    }
    state = arbitrate("out", "q", _make_critic_fn(mapping))
    assert state.all_passed is False
