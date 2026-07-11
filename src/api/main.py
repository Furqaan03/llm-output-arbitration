"""FastAPI: arbitrate an LLM output through 3 critics + an adjudicator."""
from __future__ import annotations

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from src.adjudicator.verdict import build_verdict
from src.critics.llm_critic import llm_critic
from src.orchestration.graph import arbitrate

load_dotenv()

app = FastAPI(title="LLM Output Arbitration System")


class ArbitrateRequest(BaseModel):
    output: str
    question: str = ""


@app.post("/v1/arbitrate")
def arbitrate_endpoint(req: ArbitrateRequest) -> dict:
    state = arbitrate(req.output, req.question, llm_critic)
    verdict = build_verdict(req.output, req.question, state.critiques, state.disagreements, state.all_passed)
    return {
        "verdict": verdict.model_dump(),
        "critiques": [c.model_dump() for c in state.critiques],
        "disagreements": [d.__dict__ for d in state.disagreements],
    }


@app.post("/v1/arbitrate/batch")
def arbitrate_batch(reqs: list[ArbitrateRequest]) -> dict:
    results = []
    for req in reqs:
        state = arbitrate(req.output, req.question, llm_critic)
        verdict = build_verdict(req.output, req.question, state.critiques, state.disagreements, state.all_passed)
        results.append({"quality_score": verdict.quality_score, "confidence": verdict.confidence,
                        "issue_count": len(verdict.confirmed_issues)})
    return {"results": results}
