"""
main.py — FastAPI Application
Post-Operative Brain Tumor Recovery Analysis Agent
UCSD-PTGBM-BraTS-2024 | Level 3 Medical AI
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from agent import RecoveryAgentState, recovery_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NeuroRecovery AI Agent",
    description="Level 3 Medical AI Agent — BraTS-2024 · LangGraph · Ollama.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# FIX: allow all origins so test_stack.py, curl, and any frontend can reach the API.
# Tighten to ["http://localhost:3000"] when deploying to production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,   # must be False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ClinicalMetadata(BaseModel):
    patient_id:          str            = Field(..., description="De-identified patient ID.")
    tumor_grade:         str            = Field("GBM")
    resection_extent:    str            = Field("gross_total")
    treatment_protocol:  str            = Field("Stupp")
    weeks_post_surgery:  int            = Field(..., ge=0, le=520)
    kps_score:           int            = Field(..., ge=0, le=100)
    idh_mutation_status: Optional[str]  = Field(None)
    mgmt_methylation:    Optional[bool] = Field(None)
    additional_notes:    Optional[str]  = Field(None, max_length=2000)

    @field_validator("kps_score")
    @classmethod
    def kps_must_be_multiple_of_10(cls, v: int) -> int:
        if v % 10 != 0:
            raise ValueError("KPS score must be a multiple of 10 (0–100).")
        return v


class AnalyzeRecoveryRequest(BaseModel):
    clinical_metadata: ClinicalMetadata
    image_features:    Optional[list[float]] = Field(None)
    session_id:        Optional[str]         = Field(None)


class AnalyzeRecoveryResponse(BaseModel):
    session_id:              str
    patient_id:              str
    final_report:            str
    literature_sources:      list[dict[str, Any]]
    warnings:                list[str]
    retrieval_attempts:      int
    processing_time_seconds: float
    status:                  str = "success"

# ---------------------------------------------------------------------------
# Exception handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s %s", request.method, request.url)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"Internal server error: {exc}"},
    )

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["ops"])
async def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "neurorecovery-agent"}


@app.get("/graph-schema", tags=["ops"])
async def graph_schema() -> dict[str, Any]:
    return {
        "nodes": [
            "extract_features_tool",
            "retrieve_literature_node",
            "synthesize_report_node",
        ],
        "edges": [
            {"from": "__start__",              "to": "extract_features_tool",    "type": "fixed"},
            {"from": "extract_features_tool",  "to": "retrieve_literature_node", "type": "fixed"},
            {"from": "retrieve_literature_node","to": "synthesize_report_node | retrieve_literature_node",
             "type": "conditional", "condition": "verify_literature_edge"},
            {"from": "synthesize_report_node", "to": "__end__",                  "type": "fixed"},
        ],
        "state_keys": list(RecoveryAgentState.__annotations__.keys()),
    }


@app.post(
    "/analyze-recovery",
    response_model=AnalyzeRecoveryResponse,
    status_code=status.HTTP_200_OK,
    tags=["agent"],
)
async def analyze_recovery(payload: AnalyzeRecoveryRequest) -> AnalyzeRecoveryResponse:
    session_id = payload.session_id or str(uuid.uuid4())
    patient_id = payload.clinical_metadata.patient_id

    logger.info("▶ /analyze-recovery | session=%s | patient=%s", session_id, patient_id)

    initial_state: RecoveryAgentState = {
        "image_features":     payload.image_features,
        "clinical_metadata":  payload.clinical_metadata.model_dump(),
        "literature_results": [],
        "retrieval_retry":    False,
        "retrieval_attempts": 0,
        "final_report":       None,
        "warnings":           [],
    }

    t0 = time.perf_counter()
    try:
        final_state: RecoveryAgentState = await recovery_graph.ainvoke(
            initial_state,
            config={"configurable": {"thread_id": session_id}},
        )
    except Exception as exc:
        logger.exception("Graph execution failed for session=%s", session_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent graph execution failed: {exc}",
        ) from exc

    elapsed = round(time.perf_counter() - t0, 3)
    logger.info(
        "✔ /analyze-recovery | session=%s | elapsed=%.3fs | warnings=%d",
        session_id, elapsed, len(final_state.get("warnings", [])),
    )

    if not final_state.get("final_report"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Graph completed but no report was generated.",
        )

    return AnalyzeRecoveryResponse(
        session_id=session_id,
        patient_id=patient_id,
        final_report=final_state["final_report"],
        literature_sources=final_state.get("literature_results", []),
        warnings=final_state.get("warnings", []),
        retrieval_attempts=final_state.get("retrieval_attempts", 0),
        processing_time_seconds=elapsed,
        status="success",
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
