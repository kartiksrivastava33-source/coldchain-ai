"""ColdChain AI FastAPI backend.

Run from the project root:
    uv run uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload

Local development only: this API has no authentication. Keep it on localhost.
All operational actions remain SIMULATED_ONLY.
"""
import logging
from contextlib import asynccontextmanager
from typing import Any
from src.evaluation import evaluate_report
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from src.orchestrator import open_agent, resume_approval, start_investigation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("coldchain-api")

agent = None
checkpoint_connection = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent, checkpoint_connection
    agent, checkpoint_connection = open_agent()
    logger.info("ColdChain AI agent initialized.")
    try:
        yield
    finally:
        if checkpoint_connection is not None:
            checkpoint_connection.close()
        agent = None
        checkpoint_connection = None


app = FastAPI(
    title="ColdChain AI API",
    description=(
        "Shipment investigations using MySQL telemetry, Qdrant SOP retrieval, "
        "live weather, LangGraph reasoning, human approval and simulated actions."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class InvestigationRequest(BaseModel):
    question: str = Field(..., min_length=10, max_length=5000)


class ApprovalRequest(BaseModel):
    approved: bool
    approver: str = Field(default="api_operator", min_length=1, max_length=100)


def require_agent():
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent is not initialized.")
    return agent


def thread_config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 40}


def serialize_interrupts(result: dict) -> list[dict]:
    items = result.get("__interrupt__", ())
    serialized = []
    for item in items:
        value = getattr(item, "value", item)
        serialized.append(value if isinstance(value, dict) else {"message": str(value)})
    return serialized


@app.get("/")
async def root():
    return {"name": "ColdChain AI API", "status": "running", "docs": "/docs"}


@app.get("/health")
async def health():
    return {"status": "healthy" if agent is not None else "starting",
            "agent_initialized": agent is not None}


@app.post("/investigations")
async def create_investigation(payload: InvestigationRequest):
    """Run an investigation and pause at the human-approval interrupt."""
    current_agent = require_agent()
    try:
        result, _, thread_id = await run_in_threadpool(
            start_investigation, current_agent, payload.question.strip()
        )
        interrupts = serialize_interrupts(result)
        tool_count = sum(
            getattr(message, "type", None) == "tool"
            for message in result.get("messages", [])
        )
        return {
            "thread_id": thread_id,
            "status": "pending_approval" if interrupts else "completed",
            "investigation_report": result.get("investigation_report", ""),
            "proposed_action": result.get("proposed_action", {}),
            "approval_request": interrupts,
            "tool_results": tool_count,
            "replanning_rounds": result.get("replan_round", 0),
            "evaluation": evaluate_report(
            result.get("investigation_report", ""),
            result.get("proposed_action", {})),
        }
    except Exception as error:
        logger.exception("Investigation failed.")
        raise HTTPException(
            status_code=502,
            detail=f"Investigation failed: {type(error).__name__}: {error}",
        ) from error


@app.post("/investigations/{thread_id}/approval")
async def approve_investigation(thread_id: str, payload: ApprovalRequest):
    """Resume a paused investigation; only simulated actions can execute."""
    current_agent = require_agent()
    try:
        result = await run_in_threadpool(
            resume_approval,
            current_agent,
            thread_config(thread_id),
            payload.approved,
            payload.approver.strip(),
        )
        return {
            "thread_id": thread_id,
            "status": "simulated_executed" if payload.approved else "rejected",
            "investigation_report": result.get("investigation_report", ""),
            "action_result": result.get("action_result", {}),
            "approval_decision": result.get("approval_decision", {}),
        }
    except Exception as error:
        logger.exception("Approval/resume failed for %s.", thread_id)
        raise HTTPException(
            status_code=502,
            detail=f"Could not resume investigation: {type(error).__name__}: {error}",
        ) from error
