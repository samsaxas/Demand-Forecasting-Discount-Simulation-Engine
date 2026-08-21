# Retail Demand Forecasting & Discount Simulation Engine

A time-series demand forecasting pipeline with an interactive discount
what-if simulator, built on the Rossmann Store Sales dataset (1,115 stores,
2013–2015 daily sales).

## Why this project

Built to demonstrate two specific capabilities: (1) scalable, ML-based
demand forecasting applied across multiple stores/categories through one
reusable pipeline, and (2) discount/price what-if simulation to support
margin-aware promotional planning.

## Pipeline

```
Raw CSVs (train.csv, store.csv)
        |
Phase 1: Schema validation (pipeline/validation.py)
        |
Phase 1: Cleaning + merge (pipeline/cleaning.py)
        |
Phase 2: EDA (pipeline/eda.py)
        |
Phase 3: Feature engineering (pipeline/features.py)
        |
Phase 4-5: Forecasting (Prophet + SARIMA) + walk-forward validation
           (models/forecasting.py)
        |
Phase 6: Discount/promo what-if simulation (models/discount_simulation.py)
        |
Phase 7: LLM summarization + restricted Q&A (models/llm_insights.py)
        |
Phase 8: Streamlit dashboard (dashboard/app.py)
```

## Key results (from actual model runs, not illustrative numbers)

- Validated across **16 stores spanning all 4 Rossmann store types (a/b/c/d)**
  using the same pipeline logic — no per-store rebuild.
- **Prophet outperformed SARIMA on 14/16 stores** (lower RMSE on the
  held-out 6-week validation window); SARIMA won on 2.
- MAPE ranged from **7.8% to 34.2%** across stores — genuine, unfiltered
  spread, not cherry-picked.
- Overall historical promo uplift: **+38.8%** average sales during
  promotional periods vs non-promo periods (varies by store: e.g. Store 85
  shows only +6.3% uplift and is revenue-negative at deep discounts, while
  Store 3 shows +64.5% uplift and stays revenue-positive up to 30% discount).

## Design decisions worth being able to explain

- **Prophet as primary model**: handles weekly/yearly seasonality and
  holiday effects natively, faster to iterate across many stores than
  hand-tuning SARIMA orders per series.
- **Walk-forward validation, not random split**: training on early history
  and testing on the final 6 weeks avoids leaking future information into
  a time-series model — a random train/test split would silently inflate
  accuracy.
- **Closed-store days dropped in cleaning**: zero sales on a closed day is
  an operational flag, not a demand signal — including it would bias the
  model toward predicting "store closed."
- **Discount simulation is a transparent, explainable elasticity estimate**
  (scaled from the store's own historical promo uplift), not a black-box
  model — deliberately kept interpretable for a planning tool.
- **LLM layer never touches raw data or computes numbers.** It only
  receives pre-computed structured results (RMSE, MAPE, uplift %, discount
  scenarios) and generates natural-language explanation from them. This
  keeps a clear line between "real math" and "generated prose," and keeps
  hallucination risk low since the LLM has nothing to invent numbers about.
- **Modular, config-driven pipeline**: column names and schema are declared
  once (`pipeline/validation.py`), so the same pipeline can run on a
  differently-shaped retail dataset without rewriting the modeling logic.

## Running it

```bash
pip install prophet statsmodels streamlit plotly pandas scikit-learn requests

# Phase 1-3: build the clean, featured dataset
python3 pipeline/cleaning.py
python3 pipeline/features.py
python3 pipeline/eda.py

# Phase 4-6: run forecasting + discount simulation (writes to outputs/)
python3 models/forecasting.py
python3 models/discount_simulation.py

# Phase 7 (optional): set your own API key for live LLM summaries
export GEMINI_API_KEY="your-key-here"

# Phase 8: launch the dashboard
streamlit run dashboard/app.py
```

The dashboard works fully without `GEMINI_API_KEY` set — it falls back to a
rule-based summary so the app is demonstrable without live API access.

## Scope, stated honestly

- Static dataset only — no live database connection. This was a deliberate
  scope decision: a live-updating, auto-decisioning system raises real
  MLOps questions (validation, monitoring, human review) that a project at
  this stage isn't meant to answer.
- Validated across 16 representative stores (4 per store type), not all
  1,115 — a deliberate trade-off for build time; the pipeline logic itself
  is written to run against any subset or the full set unchanged.
- The discount simulation is a transparent scaling estimate from historical
  promo response, not a rigorously fit price-elasticity model.

<img width="927" height="503" alt="image" src="https://github.com/user-attachments/assets/c0a77689-06b1-40c2-9177-cbfbe8743fd9" />

<img width="934" height="511" alt="image" src="https://github.com/user-attachments/assets/54648717-897c-4acb-a5fd-bb0e21d4c5c2" />

<img width="952" height="467" alt="image" src="https://github.com/user-attachments/assets/7ef876cd-ce76-48eb-9cec-f8c019632076" />

<img width="946" height="413" alt="image" src="https://github.com/user-attachments/assets/86598ee4-6323-49df-b95c-33413498db3a" />

<img width="919" height="502" alt="image" src="https://github.com/user-attachments/assets/136c21b4-cc20-451b-bb57-cab6c6859eff" />



