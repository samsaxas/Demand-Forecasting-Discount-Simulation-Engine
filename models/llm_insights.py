"""
Phase 7 — LLM Summarization + Restricted Q&A

IMPORTANT DESIGN PRINCIPLE:
The LLM never sees raw data and never computes numbers. It only receives
ALREADY-COMPUTED, pre-validated structured results (forecast metrics,
promo uplift %, discount scenarios) and turns them into natural-language
explanation. This keeps the LLM in a summarization/explanation role, not
an analysis role — the numbers are real math, the prose is generated text.

API note: this module is written for the Gemini API (matching the original
project plan). Swap in the Anthropic API by changing `call_llm()` if the
Gemini API isn't reachable from your environment.
"""

import os
import json
import requests

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-3.5-flash:generateContent"
)

SYSTEM_INSTRUCTIONS = """You are a data-insight assistant for a retail demand
forecasting dashboard. You will be given ONLY pre-computed structured results
(forecast accuracy metrics, promo uplift percentages, discount scenario
projections). Rules:
1. Only discuss numbers present in the provided data. Never invent figures.
2. If asked about anything outside the provided data, say plainly that you
   can only answer questions about the forecasting results shown on this
   dashboard.
3. Keep answers concise and business-focused (2-4 sentences unless asked
   for detail).
4. Do not make investment, hiring, or legal recommendations — only describe
   what the data shows.
"""


def _build_context(store_id: str, forecast_data: dict, discount_data: dict) -> str:
    """Packages only the pre-computed, structured results for this store."""
    context = {
        "store_id": store_id,
        "forecast_accuracy": {
            "prophet_rmse": forecast_data.get("prophet", {}).get("rmse"),
            "prophet_mape": forecast_data.get("prophet", {}).get("mape"),
            "sarima_rmse": forecast_data.get("sarima", {}).get("rmse"),
            "sarima_mape": forecast_data.get("sarima", {}).get("mape"),
            "best_model": forecast_data.get("best_model"),
        },
        "promo_uplift_pct": discount_data.get("uplift_pct"),
        "avg_sales_with_promo": discount_data.get("avg_sales_promo"),
        "avg_sales_without_promo": discount_data.get("avg_sales_no_promo"),
        "discount_scenarios": discount_data.get("scenarios"),
    }
    return json.dumps(context, indent=2)


def call_llm(prompt: str, context: str) -> str:
    """Calls the Gemini API with the system instructions + context + user prompt."""
    if not GEMINI_API_KEY:
        return ("[LLM not configured: set GEMINI_API_KEY to enable live "
                "summaries and Q&A. Showing computed data only.]")

    payload = {
        "contents": [{
            "parts": [{
                "text": f"{SYSTEM_INSTRUCTIONS}\n\nDATA (this is the ONLY data "
                        f"you may reference):\n{context}\n\nQUESTION:\n{prompt}"
            }]
        }]
    }

    try:
        resp = requests.post(
            f"{GEMINI_URL}?key={GEMINI_API_KEY}",
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"[LLM call failed: {e}]"


def generate_summary(store_id: str, forecast_data: dict, discount_data: dict) -> str:
    """Auto-generates a natural-language summary of this store's results."""
    context = _build_context(store_id, forecast_data, discount_data)
    prompt = (
        "Summarize this store's forecast accuracy and discount sensitivity "
        "in 2-3 sentences, in plain business language for a category manager."
    )
    return call_llm(prompt, context)


def answer_question(store_id: str, forecast_data: dict, discount_data: dict,
                     question: str) -> str:
    """Answers a user question, restricted to this store's computed results."""
    context = _build_context(store_id, forecast_data, discount_data)
    return call_llm(question, context)


def rule_based_summary(store_id: str, forecast_data: dict, discount_data: dict) -> str:
    """
    Fallback summary generated WITHOUT an LLM call — used when no API key is
    configured, so the dashboard is still fully functional and demonstrable
    without live API access.
    """
    best = forecast_data.get("best_model", "prophet")
    metrics = forecast_data.get(best, {})
    rmse, mape_val = metrics.get("rmse"), metrics.get("mape")
    uplift = discount_data.get("uplift_pct")

    lines = [f"Store {store_id}: the {best.upper()} model forecasts demand with "]
    if rmse is not None and mape_val is not None:
        lines.append(f"an average error of {mape_val:.1f}% (MAPE) on held-out data. ")
    if uplift is not None:
        sensitivity = "high" if uplift > 30 else "moderate" if uplift > 10 else "low"
        lines.append(
            f"This store shows {sensitivity} promo sensitivity, with sales "
            f"rising {uplift:.1f}% during promotional periods historically."
        )
    return "".join(lines)


if __name__ == "__main__":
    with open("outputs/forecast_results.json") as f:
        forecasts = json.load(f)
    with open("outputs/discount_results.json") as f:
        discounts = json.load(f)

    store_id = list(forecasts.keys())[0]
    print("Rule-based fallback summary (no API key needed):")
    print(rule_based_summary(store_id, forecasts[store_id], discounts.get(store_id, {})))

    print("\nLLM summary (requires GEMINI_API_KEY):")
    print(generate_summary(store_id, forecasts[store_id], discounts.get(store_id, {})))
