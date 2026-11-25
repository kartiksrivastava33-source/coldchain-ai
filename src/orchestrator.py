import json
import os
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt

from src.agent_tools import (
    fetch_corridor_conditions,
    query_telemetry_db,
    search_compliance_sop,
)


# ---------------------------------------------------------------------
# Project configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError("GOOGLE_API_KEY is missing from the project .env file.")

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
MAX_REPLAN_ROUNDS = int(os.getenv("MAX_REPLAN_ROUNDS", "2"))
MAX_TOOL_RESULTS = int(os.getenv("MAX_TOOL_RESULTS", "8"))

TOOLS = [
    query_telemetry_db,
    fetch_corridor_conditions,
    search_compliance_sop,
]

LLM = ChatGoogleGenerativeAI(
    model=MODEL_NAME
)
LLM_WITH_TOOLS = LLM.bind_tools(TOOLS)


def safe_invoke(model, messages, max_retries=3):
    """Retry transient Gemini/API failures with exponential backoff."""
    retryable_statuses = {"429", "500", "502", "503", "504"}

    for attempt in range(max_retries):
        try:
            return model.invoke(messages)
        except Exception as error:
            status = str(
                getattr(error, "code", None)
                or getattr(error, "status_code", None)
                or ""
            )
            error_text = str(error).lower()
            is_retryable = (
                status in retryable_statuses
                or any(
                    marker in error_text
                    for marker in (
                        "429", "500", "502", "503", "504",
                        "resource_exhausted", "service_unavailable",
                        "deadline_exceeded", "temporarily unavailable",
                        "high demand",
                    )
                )
            )

            if not is_retryable or attempt == max_retries - 1:
                raise

            wait_seconds = 2 ** (attempt + 1)
            print(
                f"\\n[Gemini Retry] Temporary API failure "
                f"(attempt {attempt + 1}/{max_retries}). "
                f"Retrying in {wait_seconds}s..."
            )
            time.sleep(wait_seconds)



SYSTEM_PROMPT = """
You are ColdChain AI, a supply-chain shipment investigation agent.

AVAILABLE TOOLS
- query_telemetry_db: retrieve shipment telemetry from MySQL.
- fetch_corridor_conditions: retrieve current weather for coordinates.
- search_compliance_sop: retrieve company SOP passages from Qdrant.

MANDATORY INVESTIGATION SEQUENCE
1. Identify the requested record_id and query MySQL first for that exact record.
   Request all available fields: Record_ID, Timestamp, Latitude, Longitude,
   Current_Temperature_C, Cargo_Condition_Code, Risk_Classification,
   Delay_Probability, Port_Congestion_Level, Route_Risk_Index.
2. Use the returned shipment coordinates for current weather. Never use
   arbitrary or fixed coordinates when shipment coordinates are available.
3. Search SOP clauses relevant to the observed temperature, risk, delay and
   congestion values. Base the search query on the actual shipment values.
4. If a tool returns an error, disclose it. Never claim its evidence was
   retrieved successfully.
5. Use another tool only when it can materially add evidence. Do not repeat
   successful calls without a clear reason.

INTERPRETATION RULES
- Inspect Timestamp. Never describe historical records as currently in transit.
- Live weather is current supplementary context, never historical weather.
- Report risk classification, delay probability and route risk separately.
- Do not infer cargo type from Cargo_Condition_Code.
- The 0–4°C rule applies only to fresh perishables. If cargo type is unknown,
  state that the reading is within/outside that numerical range but cargo
  compliance cannot be confirmed.
- Above 4°C is a critical breach only when fresh perishables are established.
- Below 0°C is outside the stated range and should be flagged for human review.
- Tier 2 escalation requires BOTH High Risk and delay probability > 0.65.
- Port diversion applies only to Los Angeles / Long Beach when congestion > 7.
  Do not infer port identity from GPS coordinates.
- Never invent cargo type, destination port, historical weather, traffic,
  shipment status, or missing facts.
- Separate database evidence, SOP evidence, live external context, inference,
  and unknown information.
- Never claim a real operational action has been executed.

The final report MUST include every available telemetry field, explicitly
including Route Risk Index, and use all seven sections:
1. Investigation Summary
2. Shipment Telemetry
3. Risk Assessment
4. Environmental Conditions
5. Applicable SOP Rules
6. Recommended Actions
7. Evidence Quality and Missing Information

Be concise, evidence-based, and transparent about limitations.
"""


class AgentState(MessagesState):
    replan_round: int
    evidence_audit: str
    investigation_report: str
    proposed_action: dict
    approval_decision: dict
    action_result: dict


def content_to_text(content: Any) -> str:
    """Convert normal text and Gemini content blocks into printable text."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)

    return str(content)


def count_tool_results(messages) -> int:
    return sum(isinstance(message, ToolMessage) for message in messages)


# ---------------------------------------------------------------------
# Investigation nodes
# ---------------------------------------------------------------------

def reasoning_node(state: AgentState):
    print("\n[Reasoner] Investigating shipment...")
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = safe_invoke(LLM_WITH_TOOLS, messages)
    return {"messages": [response]}


def route_initial_reasoner(state: AgentState):
    """Bound the initial tool loop so tool use cannot run indefinitely."""
    if count_tool_results(state["messages"]) >= MAX_TOOL_RESULTS:
        return "evaluator"
    return tools_condition(state)


def evidence_evaluator(state: AgentState):
    print("\n[Evidence Evaluator] Reviewing investigation...")

    transcript = "\n\n".join(
        f"{message.type.upper()}:\n{content_to_text(message.content)}"
        for message in state["messages"]
    )

    prompt = f"""
You are the evidence auditor for ColdChain AI.

Review the investigation. Check:
1. Are telemetry values supported by database tool results?
2. Is every requested telemetry metric reported, especially Route_Risk_Index?
3. Are SOP recommendations supported by retrieved SOP text?
3. Is cargo type known before asserting fresh-perishable compliance?
4. Is port identity established before applying port-specific rules?
5. Was weather queried using the shipment's actual returned coordinates?
6. Is live weather clearly separated from historical shipment conditions?
7. Are risk classification, delay probability and route risk kept separate?
8. Is a historical record incorrectly described as currently active?
9. Are uncertainties, tool failures and missing information disclosed?
10. Can an available tool materially resolve any missing information?

Do not invent missing information. Do not claim an action was executed.

Return:
- Confirmed facts
- Potential risks
- Applicable SOP rules and whether their conditions are met
- Unsupported or uncertain conclusions
- Missing information
- Whether another available tool call could materially improve the investigation

INVESTIGATION:
{transcript}
"""

    response = safe_invoke(LLM, [HumanMessage(content=prompt)])
    audit = content_to_text(response.content)

    return {
        "evidence_audit": audit,
        "messages": [AIMessage(content="Evidence audit:\n" + audit)],
    }


def replanning_node(state: AgentState):
    round_number = state.get("replan_round", 0) + 1
    tool_results = count_tool_results(state["messages"])

    print(f"\n[Replanner] Round {round_number}")

    transcript = "\n\n".join(
        f"{message.type.upper()}:\n{content_to_text(message.content)}"
        for message in state["messages"]
    )

    prompt = f"""
You are the bounded investigation planner for ColdChain AI.

Decide whether available tools can retrieve additional information that
materially improves this investigation.

Available tools:
- query_telemetry_db
- fetch_corridor_conditions
- search_compliance_sop

Rules:
- Do not repeat completed queries without a clear reason.
- Use shipment GPS coordinates from the telemetry result for current weather.
- Current weather cannot reconstruct historical weather.
- The available tools cannot retrieve external cargo manifests or an
  unrepresented destination-port field.
- If no available tool can resolve a gap, disclose the limitation.
- Call a tool only when genuinely useful; otherwise produce the final report.
- Never claim an operational action was executed.

Final report MUST contain all seven sections:
1. Investigation Summary
2. Shipment Telemetry: record ID, timestamp, coordinates, temperature,
   cargo-condition code, risk classification, delay probability,
   port congestion and route risk index.
3. Risk Assessment
4. Environmental Conditions: weather coordinates and observation time if
   available; label weather as live/current.
5. Applicable SOP Rules, including each trigger's status
6. Recommended Actions
7. Evidence Quality and Missing Information

EVIDENCE AUDIT:
{state.get("evidence_audit", "")}

FULL TRANSCRIPT:
{transcript}
"""

    # Force a plain final answer when either limit is reached.
    if round_number > MAX_REPLAN_ROUNDS or tool_results >= MAX_TOOL_RESULTS:
        print("[Replanner] Limit reached; generating final report.")
        response = safe_invoke(
            LLM,
            [
                HumanMessage(
                    content=prompt + "\nDo not call tools. Produce the final report now."
                )
            ],
        )
        response = AIMessage(content=content_to_text(response.content))
    else:
        print("[Replanner] Checking whether more evidence is needed...")
        response = safe_invoke(LLM_WITH_TOOLS, [HumanMessage(content=prompt)])

    return {
        "messages": [response],
        "replan_round": round_number,
    }


def route_replanner(state: AgentState):
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", [])

    if tool_calls and count_tool_results(state["messages"]) < MAX_TOOL_RESULTS:
        return "tools"

    return "finish"


def capture_final_report(state: AgentState):
    report = content_to_text(state["messages"][-1].content)
    print("\n[Report] Final investigation report prepared.")
    return {"investigation_report": report}


# ---------------------------------------------------------------------
# Human approval and simulated action execution
# ---------------------------------------------------------------------

def find_shipment_id(state: AgentState) -> str:
    """Use the ID in telemetry output, falling back to the user's request."""
    for message in state.get("messages", []):
        if isinstance(message, ToolMessage):
            body = content_to_text(message.content)
            match = re.search(r"\bRecord_ID\b\s*[:=]?\s*['\"]?(\d+)", body, re.I)
            if match:
                return match.group(1)
            tuple_match = re.search(r"\(\s*(\d+)\s*,", body)
            if tuple_match and "COLUMNS:" in body:
                return tuple_match.group(1)
    for message in state.get("messages", []):
        if message.type == "human":
            match = re.search(r"\brecord(?:_|\s*)id\s*[:=#]?\s*(\d+)", content_to_text(message.content), re.I)
            if match:
                return match.group(1)
    return "unknown"


def propose_action(state: AgentState):
    """Propose a simulated review task; never a real dispatch instruction."""
    proposal = {
        "action_id": str(uuid4()),
        "shipment_id": find_shipment_id(state),
        "action": "create_dispatch_review",
        "description": "Create a human review task for this shipment record.",
        "reason": (
            "Review the investigation findings and delay-risk evidence. "
            "This proposal does not assume the historical shipment is active."
        ),
        "execution_mode": "SIMULATED_ONLY",
        "report_excerpt": state.get("investigation_report", "")[:1200],
    }

    print("\n[Action Planner] Prepared simulated review action.")
    return {"proposed_action": proposal}


def human_approval(state: AgentState):
    decision = interrupt({
        "message": "Human approval is required before simulated execution.",
        "action": state["proposed_action"],
        "choices": ["approve", "reject"],
    })
    return {"approval_decision": decision}


def execute_and_audit(state: AgentState):
    proposal = state["proposed_action"]
    decision = state.get("approval_decision", {})

    approved = (
        isinstance(decision, dict)
        and decision.get("approved") is True
    )

    approver = (
        decision.get("approver", "unspecified")
        if isinstance(decision, dict)
        else "unspecified"
    )

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action_id": proposal["action_id"],
        "shipment_id": proposal["shipment_id"],
        "action": proposal["action"],
        "approver": approver,
        "approval_status": "APPROVED" if approved else "REJECTED",
        "execution_status": "SIMULATED_EXECUTED" if approved else "REJECTED",
        "execution_mode": "SIMULATED_ONLY",
    }

    log_directory = PROJECT_ROOT / "logs"
    log_directory.mkdir(parents=True, exist_ok=True)
    audit_file = log_directory / "action_audit.jsonl"

    with audit_file.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n[Action Executor] {record['execution_status']}")
    print(f"[Audit Logger] Saved {audit_file}")

    return {
        "action_result": record,
        "messages": [AIMessage(content=json.dumps(record, indent=2))],
    }


# ---------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------

def build_graph(checkpointer):
    builder = StateGraph(AgentState)

    builder.add_node("reasoner", reasoning_node)
    builder.add_node("initial_tools", ToolNode(TOOLS))
    builder.add_node("evaluator", evidence_evaluator)
    builder.add_node("replanner", replanning_node)
    builder.add_node("replan_tools", ToolNode(TOOLS))
    builder.add_node("capture_report", capture_final_report)
    builder.add_node("propose_action", propose_action)
    builder.add_node("human_approval", human_approval)
    builder.add_node("execute_and_audit", execute_and_audit)

    builder.add_edge(START, "reasoner")

    builder.add_conditional_edges(
        "reasoner",
        route_initial_reasoner,
        {
            "tools": "initial_tools",
            "__end__": "evaluator",
            "evaluator": "evaluator",
        },
    )
    builder.add_edge("initial_tools", "reasoner")

    builder.add_edge("evaluator", "replanner")
    builder.add_conditional_edges(
        "replanner",
        route_replanner,
        {
            "tools": "replan_tools",
            "finish": "capture_report",
        },
    )
    builder.add_edge("replan_tools", "replanner")

    builder.add_edge("capture_report", "propose_action")
    builder.add_edge("propose_action", "human_approval")
    builder.add_edge("human_approval", "execute_and_audit")
    builder.add_edge("execute_and_audit", END)

    return builder.compile(checkpointer=checkpointer)


def open_agent():
    """Open the graph with a persistent SQLite checkpoint database."""
    checkpoint_directory = PROJECT_ROOT / "data"
    checkpoint_directory.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_directory / "coldchain_checkpoints.sqlite"
    connection = sqlite3.connect(
        str(checkpoint_path),
        check_same_thread=False,
    )
    checkpointer = SqliteSaver(connection)
    checkpointer.setup()

    return build_graph(checkpointer), connection


def start_investigation(app, question: str, thread_id: str | None = None):
    """Start an investigation and return state, config, and thread ID."""
    thread_id = thread_id or str(uuid4())
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 40,
    }

    initial_state = {
        "messages": [HumanMessage(content=question)],
        "replan_round": 0,
        "evidence_audit": "",
        "investigation_report": "",
        "proposed_action": {},
        "approval_decision": {},
        "action_result": {},
    }

    result = app.invoke(initial_state, config=config)
    return result, config, thread_id


def resume_approval(app, config: dict, approved: bool, approver: str):
    """Resume a paused approval request."""
    return app.invoke(
        Command(resume={
            "approved": approved,
            "approver": approver,
        }),
        config=config,
    )
