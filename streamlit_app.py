"""
ColdChain AI — Streamlit dashboard

Run the FastAPI backend in one terminal:
    uv run python -m uvicorn src.api:app --host 127.0.0.1 --port 8000

Run this dashboard in another terminal:
    uv run streamlit run streamlit_app.py
"""
import json
import re
from datetime import datetime
from typing import Any

import requests
import streamlit as st

st.set_page_config(
    page_title="ColdChain AI | Control Tower",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_API_URL = "http://127.0.0.1:8000"
DEFAULT_QUESTION = (
    "Investigate shipment record ID 1. Retrieve its telemetry from MySQL, "
    "search relevant cold-chain SOP procedures using Qdrant, and fetch current "
    "weather using the shipment's actual GPS coordinates. Evaluate temperature, "
    "delay probability, risk classification, port congestion and route risk index. "
    "Distinguish historical shipment data from live weather. Identify missing "
    "information, evaluate evidence quality, and recommend appropriate actions. "
    "Do not execute any action without explicit human approval."
)


def api_request(method: str, endpoint: str, **kwargs) -> Any:
    """Call the FastAPI backend and return decoded JSON."""
    base_url = st.session_state.api_url.rstrip("/")
    url = f"{base_url}{endpoint}"
    try:
        response = requests.request(method, url, timeout=300, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"Cannot connect to the API at {base_url}. "
            "Start the FastAPI backend first."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            "The API request timed out. The agent may still be processing; "
            "check the FastAPI terminal before submitting again."
        ) from exc
    except requests.exceptions.HTTPError as exc:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        raise RuntimeError(f"API returned HTTP {response.status_code}: {detail}") from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"API request failed: {exc}") from exc


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, indent=2, ensure_ascii=False)
    return str(value)


def render_report(report: str):
    if report:
        st.markdown(report)
    else:
        st.info("The API did not return an investigation report.")



def render_evaluation(evaluation: Any):
    """Display deterministic report-quality checks returned by the API."""
    st.subheader("Report quality evaluation")
    if not isinstance(evaluation, dict):
        st.info(
            "No evaluation was returned by the API. Confirm that src.api imports "
            "evaluate_report and includes the 'evaluation' field in its response."
        )
        return

    score = evaluation.get("score")
    passed = evaluation.get("passed")
    total = evaluation.get("total")
    status = str(evaluation.get("status", "REVIEW")).upper()

    cols = st.columns(3)
    cols[0].metric(
        "Quality checks",
        f"{passed}/{total}" if passed is not None and total is not None else "N/A",
    )
    if score is not None:
        score_value = float(score)
    if score_value <= 1:
        score_value *= 100
        score_display = f"{score_value:.0f}%"
    else:
        score_display = "N/A"

    cols[1].metric("Completeness score", score_display)
    if status == "PASS":
        cols[2].success("PASS")
    else:
        cols[2].warning(status)

    checks = evaluation.get("checks", {})
    if isinstance(checks, dict) and checks:
        rows = []
        for name, value in checks.items():
            if isinstance(value, dict):
                passed_check = value.get("passed", False)
                detail = value.get("detail", value.get("message", ""))
            else:
                passed_check = bool(value)
                detail = ""
            rows.append({
                "Check": str(name).replace("_", " ").title(),
                "Result": "PASS" if passed_check else "REVIEW",
                "Details": safe_text(detail),
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)

    st.caption(
        "This deterministic check measures report completeness and safety wording; "
        "it does not prove factual correctness of the shipment analysis."
    )

def show_metrics(report: str, result: dict):
    """Extract known report values for a quick visual summary."""
    patterns = {
        "Shipment ID": r"Record ID:\**\s*(\d+)",
        "Temperature": r"(?:Current Temperature(?: \(°C\))?|Temperature):\**\s*([-+]?\d+(?:\.\d+)?)\s*°?C",
        "Delay probability": r"Delay Probability:\**\s*([0-9.]+)",
        "Risk classification": r"Risk Classification:\**\s*([A-Za-z ]+)",
        "Port congestion": r"Port Congestion Level:\**\s*([0-9.]+)",
        "Route risk index": r"Route Risk Index:\**\s*([0-9.]+)",
    }
    found = {}
    for label, pattern in patterns.items():
        match = re.search(pattern, report, re.IGNORECASE)
        if match:
            found[label] = match.group(1).strip()

    if found:
        st.subheader("Investigation snapshot")
        keys = list(found.keys())
        columns = st.columns(min(3, len(keys)))
        for index, key in enumerate(keys):
            with columns[index % len(columns)]:
                value = found[key]
                if key == "Delay probability":
                    try:
                        number = float(value)
                        if number <= 1:
                            value = f"{number * 100:.1f}%"
                    except ValueError:
                        pass
                st.metric(key, value)

    st.caption(
        f"Tool results: {result.get('tool_results', 'N/A')} · "
        f"Replanning rounds: {result.get('replanning_rounds', 'N/A')}"
    )


# ----------------------------- Session state -----------------------------

if "api_url" not in st.session_state:
    st.session_state.api_url = DEFAULT_API_URL
if "last_investigation" not in st.session_state:
    st.session_state.last_investigation = None
if "last_approval" not in st.session_state:
    st.session_state.last_approval = None
if "question" not in st.session_state:
    st.session_state.question = DEFAULT_QUESTION


# -------------------------------- Sidebar --------------------------------

with st.sidebar:
    st.title("❄️ ColdChain AI")
    st.caption("Agentic Supply Chain Intelligence Platform")
    st.divider()

    st.subheader("Backend connection")
    st.session_state.api_url = st.text_input(
        "FastAPI base URL",
        value=st.session_state.api_url,
        help="Local default: http://127.0.0.1:8000",
    ).strip().rstrip("/")

    if st.button("Check backend health", use_container_width=True):
        try:
            health = api_request("GET", "/health")
            if health.get("agent_initialized"):
                st.success("Backend and agent are ready.")
            else:
                st.warning(f"Backend responded: {health}")
        except RuntimeError as error:
            st.error(str(error))

    st.divider()
    st.subheader("System components")
    st.markdown(
        "- **LLM:** Gemini\n"
        "- **Orchestration:** LangGraph\n"
        "- **Operational data:** MySQL\n"
        "- **SOP retrieval:** BGE-M3 + Qdrant\n"
        "- **Weather:** Open-Meteo\n"
        "- **API:** FastAPI"
    )
    st.divider()
    st.caption("Actions are simulated only. This dashboard does not dispatch shipments.")


# -------------------------------- Main UI --------------------------------

st.title("ColdChain AI Control Tower")
st.write(
    "Investigate shipment telemetry, retrieve cold-chain procedures, assess "
    "environmental context, and review proposed actions with human approval."
)

tab_investigate, tab_approval, tab_architecture = st.tabs(
    ["🔎 Investigation", "🛡️ Approval & audit", "🏗️ Platform"]
)

with tab_investigate:
    st.subheader("New shipment investigation")
    st.text_area(
        "Investigation request",
        key="question",
        height=150,
        help="Mention a shipment record ID and the evidence you want the agent to inspect.",
    )

    col_run, col_clear = st.columns([1, 1])
    with col_run:
        run_clicked = st.button(
            "Run investigation",
            type="primary",
            use_container_width=True,
        )
    with col_clear:
        clear_clicked = st.button("Clear previous result", use_container_width=True)

    if clear_clicked:
        st.session_state.last_investigation = None
        st.session_state.last_approval = None
        st.rerun()

    if run_clicked:
        if len(st.session_state.question.strip()) < 10:
            st.warning("Enter a more detailed investigation request.")
        else:
            try:
                with st.spinner(
                    "Agent is retrieving telemetry, searching SOPs, checking weather "
                    "and evaluating evidence. This can take a few minutes..."
                ):
                    payload = {"question": st.session_state.question.strip()}
                    result = api_request("POST", "/investigations", json=payload)
                    st.session_state.last_investigation = result
                    st.session_state.last_approval = None
            except RuntimeError as error:
                st.error(str(error))

    result = st.session_state.last_investigation
    if result:
        st.divider()
        status = result.get("status", "unknown")
        if status == "pending_approval":
            st.warning("Investigation complete — human approval is pending.")
        elif status == "completed":
            st.success("Investigation completed.")

        st_metrics_report = result.get("investigation_report", "")
        show_metrics(st_metrics_report, result)

        render_evaluation(result.get("evaluation"))

        with st.expander("Full investigation report", expanded=True):
            render_report(st_metrics_report)

        proposed_action = result.get("proposed_action") or {}
        if proposed_action:
            st.subheader("Proposed action")
            st.json(proposed_action)
            st.info(
                "The proposed action is SIMULATED_ONLY. It will not contact a driver, "
                "change a shipment, or perform a real-world dispatch."
            )
            if result.get("thread_id"):
                st.caption(f"Approval thread: `{result['thread_id']}`")


with tab_approval:
    st.subheader("Human approval")
    st.write(
        "Approve or reject the pending simulated review action. "
        "The decision is sent to the FastAPI backend and recorded in the audit log."
    )

    pending = st.session_state.last_investigation
    if not pending or not pending.get("thread_id"):
        st.info("Run an investigation first. Its pending approval will appear here.")
    elif pending.get("status") != "pending_approval":
        st.info("There is no pending approval for the latest investigation.")
    else:
        action = pending.get("proposed_action") or {}
        st.markdown(f"**Action:** `{action.get('action', 'Not supplied')}`")
        st.markdown(f"**Shipment:** `{action.get('shipment_id', 'Unknown')}`")
        st.markdown(f"**Mode:** `{action.get('execution_mode', 'Unknown')}`")
        st.markdown(f"**Reason:** {action.get('reason', 'Not supplied')}")

        approver = st.text_input("Approver", value="dashboard_operator")
        approve_col, reject_col = st.columns(2)

        with approve_col:
            approve_clicked = st.button(
                "Approve simulated action",
                type="primary",
                use_container_width=True,
            )
        with reject_col:
            reject_clicked = st.button(
                "Reject action",
                use_container_width=True,
            )

        if approve_clicked or reject_clicked:
            if not approver.strip():
                st.warning("Enter the approver identifier.")
            else:
                approved = bool(approve_clicked)
                try:
                    with st.spinner("Recording approval decision..."):
                        approval_result = api_request(
                            "POST",
                            f"/investigations/{pending['thread_id']}/approval",
                            json={
                                "approved": approved,
                                "approver": approver.strip(),
                            },
                        )
                    st.session_state.last_approval = approval_result
                    st.session_state.last_investigation["status"] = (
                        "approved" if approved else "rejected"
                    )
                except RuntimeError as error:
                    st.error(str(error))

    approval_result = st.session_state.last_approval
    if approval_result:
        st.divider()
        if approval_result.get("status") == "simulated_executed":
            st.success("Simulated action approved and audit record written.")
        else:
            st.warning("Action rejected and rejection recorded.")
        st.json(approval_result.get("action_result") or approval_result)


with tab_architecture:
    st.subheader("System architecture")
    st.graphviz_chart(
        """
        digraph ColdChain {
            rankdir=TB;
            node [shape=box, style="rounded,filled", fillcolor="#EAF2F8", color="#426782"];
            User [label="User / Operator"];
            UI [label="Streamlit Control Tower"];
            API [label="FastAPI"];
            Agent [label="LangGraph Agent"];
            Gemini [label="Gemini Reasoning"];
            MySQL [label="MySQL Telemetry"];
            Qdrant [label="BGE-M3 + Qdrant SOP RAG"];
            Weather [label="Open-Meteo"];
            Approval [label="Human Approval"];
            Audit [label="JSONL Audit Log"];
            User -> UI -> API -> Agent;
            Agent -> Gemini;
            Agent -> MySQL;
            Agent -> Qdrant;
            Agent -> Weather;
            Agent -> Approval -> Audit;
        }
        """
    )

    st.subheader("Technology stack")
    st.markdown(
        "| Layer | Technology | Responsibility |\n"
        "|---|---|---|\n"
        "| Interface | Streamlit | Investigation and approval UI |\n"
        "| API | FastAPI | HTTP endpoints |\n"
        "| Orchestration | LangGraph | Tool calling, evaluation and replanning |\n"
        "| Reasoning | Gemini | Evidence-based analysis |\n"
        "| Retrieval | BGE-M3 + Qdrant | SOP semantic search |\n"
        "| Database | MySQL | Shipment telemetry |\n"
        "| External context | Open-Meteo | Current weather |\n"
        "| Audit | JSONL + SQLite | Action records and graph checkpoints |"
    )

st.divider()
st.caption(
    f"ColdChain AI · Local development dashboard · "
    f"API: {st.session_state.api_url} · "
    f"Rendered {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)
