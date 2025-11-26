# api_server.py
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from governance_drift_engine import analyze_governance_window

app = FastAPI(title="Governance Drift Engine API")


# ----- Request / Response Models -----

class DriftRequest(BaseModel):
    window_sys_id: Optional[str] = None   # sys_id from ServiceNow (optional)
    window_text: str
    skipped_approvals_delta: float
    emergency_without_CAB_count: int
    after_hours_critical_delta: float


class DriftResponse(BaseModel):
    window_sys_id: Optional[str] = None
    drift_score: float
    root_causes: List[str]
    recommended_actions: List[str]
    confidence: float
    ml_drift_score: float
    embedding_drift_score: float
    combined_drift_score: float


# ----- Endpoint -----

@app.post("/governance/drift", response_model=DriftResponse)
def governance_drift(req: DriftRequest):
    result = analyze_governance_window(
        window_text=req.window_text,
        skipped=req.skipped_approvals_delta,
        emergency=req.emergency_without_CAB_count,
        after_hours=req.after_hours_critical_delta,
    )

    # result already has drift_score, root_causes, recommended_actions, confidence,
    # and drift components. Just attach the window_sys_id for ServiceNow mapping.
    return DriftResponse(
        window_sys_id=req.window_sys_id,
        drift_score=result.get("drift_score", 0.0),
        root_causes=result.get("root_causes", []),
        recommended_actions=result.get("recommended_actions", []),
        confidence=result.get("confidence", 0.0),
        ml_drift_score=result.get("ml_drift_score", 0.0),
        embedding_drift_score=result.get("embedding_drift_score", 0.0),
        combined_drift_score=result.get("combined_drift_score", 0.0),
    )
