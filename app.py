"""
Phase 8 — Streamlit Dashboard
Retail Demand Forecasting & Discount Simulation Engine

Run with: streamlit run dashboard/app.py

Only cleaned, validated, pre-computed results reach this UI — raw data is
never displayed directly (see pipeline/validation.py and pipeline/cleaning.py).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from models.llm_insights import generate_summary, answer_question, rule_based_summary

st.set_page_config(page_title="Demand Forecasting & Discount Simulation", layout="wide")

# --- Load pre-computed results (see models/forecasting.py, discount_simulation.py) ---
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@st.cache_data
def load_results():
    with open(os.path.join(BASE, "outputs/forecast_results.json")) as f:
        forecasts = json.load(f)
    with open(os.path.join(BASE, "outputs/discount_results.json")) as f:
        discounts = json.load(f)
    store_meta = pd.read_csv(os.path.join(BASE, "data/store.csv"))
    return forecasts, discounts, store_meta


forecasts, discounts, store_meta = load_results()
store_ids = sorted(forecasts.keys(), key=int)

# --- Header ---
st.title("Retail Demand Forecasting & Discount Simulation Engine")
st.caption(
    "Time-series forecasting (Prophet / SARIMA) with walk-forward validation, "
    "and a discount what-if simulator, applied across multiple stores through "
    "one reusable pipeline."
)

# --- Sidebar: store/category selector ---
st.sidebar.header("Select Store")
store_id = st.sidebar.selectbox("Store ID", store_ids)

store_row = store_meta[store_meta["Store"] == int(store_id)]
store_type = store_row["StoreType"].values[0] if len(store_row) else "?"
assortment = store_row["Assortment"].values[0] if len(store_row) else "?"

st.sidebar.markdown(f"**Store Type:** {store_type}")
st.sidebar.markdown(f"**Assortment:** {assortment}")
st.sidebar.markdown("---")
st.sidebar.caption(
    f"Pipeline validated across {len(store_ids)} stores spanning all 4 "
    f"Rossmann store types (a/b/c/d) — demonstrating the same modeling "
    f"logic scales across categories without a per-store rebuild."
)

forecast_data = forecasts[store_id]
discount_data = discounts.get(store_id, {})

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "Demand Forecast", "Model Validation", "Discount Simulation", "AI Insights & Q&A"
])

# === TAB 1: Demand Forecast ===
with tab1:
    st.subheader(f"Store {store_id} — Forecast vs Actual (held-out validation window)")

    best_model = forecast_data.get("best_model", "prophet")
    best_data = forecast_data.get(best_model, {})

    if "dates" in best_data:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=best_data["dates"], y=best_data["actuals"],
            name="Actual Sales", mode="lines+markers", line=dict(color="#2563eb")
        ))
        fig.add_trace(go.Scatter(
            x=best_data["dates"], y=best_data["predictions"],
            name=f"{best_model.upper()} Forecast", mode="lines+markers",
            line=dict(color="#f97316", dash="dash")
        ))
        fig.update_layout(
            xaxis_title="Date", yaxis_title="Sales",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info(f"Best-performing model for this store: **{best_model.upper()}** "
                f"(selected by lowest RMSE on the held-out window)")
    else:
        st.warning("No forecast data available for this store.")

# === TAB 2: Model Validation ===
with tab2:
    st.subheader("Walk-Forward Validation Metrics")
    st.caption(
        "Models trained on early history, tested on the final 6 weeks — "
        "never on randomly shuffled data, which would leak future "
        "information into a time-series model."
    )

    col1, col2 = st.columns(2)
    prophet_m = forecast_data.get("prophet", {})
    sarima_m = forecast_data.get("sarima", {})

    with col1:
        st.markdown("**Prophet**")
        st.metric("RMSE", f"{prophet_m.get('rmse', 0):.1f}")
        st.metric("MAPE", f"{prophet_m.get('mape', 0):.1f}%")

    with col2:
        st.markdown("**SARIMA**")
        st.metric("RMSE", f"{sarima_m.get('rmse', 0):.1f}")
        st.metric("MAPE", f"{sarima_m.get('mape', 0):.1f}%")

    st.markdown("---")
    st.subheader("Accuracy Across All Validated Stores")
    rows = []
    for sid in store_ids:
        f = forecasts[sid]
        rows.append({
            "Store": sid,
            "Prophet MAPE": f.get("prophet", {}).get("mape"),
            "SARIMA MAPE": f.get("sarima", {}).get("mape"),
            "Best Model": f.get("best_model"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# === TAB 3: Discount Simulation ===
with tab3:
    st.subheader(f"Store {store_id} — Discount What-If Simulator")

    if not discount_data or discount_data.get("uplift_pct") is None:
        st.warning("Insufficient promo history for this store to simulate discounts.")
    else:
        st.metric(
            "Historical Promo Uplift",
            f"{discount_data['uplift_pct']:.1f}%",
            help="Average sales increase during past promotional periods vs non-promo periods"
        )

        discount_pct = st.slider("Hypothetical discount depth (%)", 5, 40, 20, step=5)

        scenarios = discount_data.get("scenarios", {})
        scenario = scenarios.get(str(discount_pct))

        if scenario:
            c1, c2, c3 = st.columns(3)
            c1.metric("Baseline Sales", f"{scenario['baseline_sales']:,.0f}")
            c2.metric("Projected Sales", f"{scenario['projected_sales']:,.0f}",
                       f"{scenario['projected_uplift_pct']:+.1f}%")
            c3.metric("Projected Revenue Impact", f"{scenario['projected_revenue_change_pct']:+.1f}%")

            if scenario["projected_revenue_change_pct"] < 0:
                st.warning(
                    "At this discount depth, projected volume gains don't offset the "
                    "margin given up — revenue is projected to decline for this store."
                )
            else:
                st.success(
                    "At this discount depth, projected volume gains outweigh the "
                    "margin given up — revenue is projected to increase for this store."
                )

        st.caption(
            "Simulation scales this store's own historical promo uplift by discount "
            "depth relative to a 20% reference promo — a transparent planning "
            "estimate, not a causal elasticity model."
        )

# === TAB 4: AI Insights & Q&A ===
with tab4:
    st.subheader("AI-Generated Summary")
    st.caption(
        "The LLM only receives the pre-computed numbers shown in the other "
        "tabs — it does not see raw data and does not compute any figures itself."
    )

    if st.button("Generate Summary"):
        with st.spinner("Generating..."):
            summary = generate_summary(store_id, forecast_data, discount_data)
            if "[LLM not configured" in summary:
                summary = rule_based_summary(store_id, forecast_data, discount_data)
                st.caption("(Showing rule-based summary — set GEMINI_API_KEY for live LLM summaries)")
        st.write(summary)

    st.markdown("---")
    st.subheader("Ask a Question About This Store's Results")
    st.caption("Restricted to the forecasting and discount data shown above.")

    question = st.text_input("Your question")
    if question:
        with st.spinner("Thinking..."):
            answer = answer_question(store_id, forecast_data, discount_data, question)
        st.write(answer)

st.markdown("---")
st.caption(
    "Built on the Rossmann Store Sales dataset. Pipeline: schema validation → "
    "cleaning → feature engineering → Prophet/SARIMA forecasting → walk-forward "
    "validation → discount simulation → LLM summarization."
)
