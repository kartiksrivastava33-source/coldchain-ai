# ColdChain AI

**Agentic Supply Chain Intelligence Platform**

ColdChain AI is an evidence-driven shipment investigation platform. It combines shipment telemetry, cold-chain standard operating procedures (SOPs), live weather context, and an orchestrated AI workflow to produce investigation reports and propose operational actions for human review.

**Maintainer:** Kartik Srivastava

## What it does

- Retrieves shipment telemetry from a MySQL operational view.
- Searches cold-chain SOP guidance with BGE-M3 embeddings and Qdrant.
- Retrieves current weather context from Open-Meteo using shipment coordinates.
- Orchestrates tool use, evidence evaluation, and bounded replanning with LangGraph.
- Produces an investigation report that distinguishes historical shipment records from live environmental observations.
- Pauses proposed actions for human approval and records decisions in an audit log.
- Exposes the workflow through a FastAPI backend and Streamlit control-tower dashboard.

> **Safety boundary:** Operational actions are simulated only. The application does not dispatch vehicles, contact drivers, or modify real shipments.

## Architecture

```text
Operator
   |
Streamlit Control Tower
   |
FastAPI
   |
LangGraph Investigation Workflow
   |-- Gemini reasoning
   |-- MySQL shipment telemetry
   |-- BGE-M3 + Qdrant SOP retrieval
   |-- Open-Meteo weather context
   |-- Evidence evaluation and bounded replanning
   |-- Human approval interrupt
   |
SQLite checkpoints + JSONL audit log
```

## Technology stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph |
| LLM reasoning | Google Gemini |
| API | FastAPI |
| Dashboard | Streamlit |
| Operational database | MySQL |
| Vector database | Qdrant |
| Embeddings | BGE-M3 |
| Weather context | Open-Meteo |
| Checkpointing | SQLite |
| Testing | pytest |
| Environment and dependencies | Python 3.12+, uv, Docker Compose |

## Repository layout

- `src/` — agent tools, orchestration, API, evaluation, and SOP ingestion logic
- `notebooks/` — exploratory analysis, visualizations, incident analysis, ML feasibility, and retrieval evaluation
- `scripts/` — database and service setup/utility scripts
- `data/` — source dataset and SOP policy material
- `tests/` — automated tests
- `streamlit_app.py` — Streamlit dashboard
- `run_agent.py` — command-line investigation entry point
- `docker-compose.yml` — Qdrant service

## Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- Docker Desktop (for the local Qdrant service)
- MySQL 8
- A Google Gemini API key

## Local setup

1. Clone the repository and enter the project directory:

   ```powershell
   git clone https://github.com/kartiksrivastava33-source/coldchain-ai.git
   cd coldchain-ai
   ```

2. Create and activate the environment, then install dependencies:

   ```powershell
   uv sync --frozen
   ```

3. Create a `.env` file in the project root. Configure the following variables with your own local credentials:

   ```env
   MYSQL_HOST=localhost
   MYSQL_PORT=3306
   MYSQL_DATABASE=coldchain_db
   MYSQL_AGENT_USER=
   MYSQL_AGENT_PASSWORD=

   QDRANT_URL=http://localhost:6333
   QDRANT_API_KEY=

   GOOGLE_API_KEY=
   GEMINI_MODEL=gemini-3.5-flash-lite
   ```

   Never commit `.env` or share API keys and database passwords.

4. Start Qdrant:

   ```powershell
   docker compose up -d qdrant
   ```

5. Configure MySQL and ingest the SOP documents into the Qdrant collection before running an investigation. The agent expects the configured MySQL database/view and the `coldchain-sop` Qdrant collection to exist.

## Run the application

Start the FastAPI backend in one terminal:

```powershell
uv run python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Start the Streamlit dashboard in a second terminal:

```powershell
uv run streamlit run streamlit_app.py
```

Open the local dashboard at http://localhost:8501. The API documentation is available at http://127.0.0.1:8000/docs.

For a command-line investigation:

```powershell
uv run python run_agent.py
```

## Tests

Run the automated test suite:

```powershell
uv run python -m pytest tests/ -v
```

## Data and interpretation notes

- The supplied logistics dataset is historical; it should not be presented as a live fleet feed.
- Current weather is fetched separately and must not be described as historical shipment weather.
- The dataset does not identify cargo type. Temperature SOP rules for fresh perishables must therefore be applied only when cargo applicability is established.
- The dataset does not identify the port. The Long Beach / Los Angeles congestion diversion procedure must not be generalized to shipments whose port is unknown.
- Exploratory ML results are documented in the notebooks. Do not interpret anomaly/SOP overlap as model accuracy or claim predictive performance unsupported by evaluation.

## Maintainer

Kartik Srivastava

