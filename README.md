# ColdChain AI — Notebooks

These notebooks are designed to run from the project root.

## Setup

Activate the project's virtual environment and install dependencies from `requirements.txt`.
For plotting and notebook execution, ensure `matplotlib` and `ipykernel` are installed:

```powershell
uv pip install matplotlib ipykernel
python -m ipykernel install --user --name coldchain --display-name "Python (ColdChain)"
```

Open the project folder in VS Code, select the `Python (ColdChain)` kernel, and run notebooks in order.

## Notebooks

1. `01_data_exploration.ipynb` — schema, data quality, summaries and time coverage.
2. `02_data_analysis_visualization.ipynb` — distributions, temperature bands, correlations and label checks.
3. `03_incident_analysis.ipynb` — executes `src/incident_engine.py` and reports incident events and affected records.
4. `04_ml_feasibility_and_experiments.ipynb` — target and leakage checks plus a commented baseline template.
5. `05_sop_retrieval_evaluation.ipynb` — local keyword retrieval baseline and expected-topic coverage.

## Important

- Keep raw data unchanged.
- Run notebooks and save their outputs before committing if you want GitHub visitors to see the results.
- The temperature SOP rules apply to fresh perishables, but the dataset does not identify cargo type.
- The congestion diversion rule in the SOP is specific to the Port of Long Beach / LA, while the dataset does not identify the port.
- The ML notebook intentionally does not invent a target or metrics.
- Notebook 05 is a lexical baseline, not a completed LLM/RAG evaluation.
