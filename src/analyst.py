"""Guardrailed analyst tools with optional LangChain tool-calling agent."""
from pathlib import Path
import os
import pandas as pd
import duckdb
from .modeling import load_model, FEATURES, explain_with_shap
from .decision_engine import rank_retention_opportunities

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "artifacts" / "retainiQ.duckdb"


def query_business_metrics(question: str) -> str:
    q = question.lower()
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        if "month-to-month" in q or "month to month" in q:
            row = con.execute("SELECT COUNT(*) n, AVG(churn) rate, AVG(monthly_charges) charges FROM customers WHERE contract_type='Month-to-month'").fetchone()
            return f"Month-to-month customers: {row[0]:,}; observed churn rate: {row[1]:.1%}; average monthly charge: ${row[2]:.2f}."
        if "churn rate" in q or "overall churn" in q or "overall" in q:
            row = con.execute("SELECT COUNT(*) n, AVG(churn) rate FROM customers").fetchone()
            return f"Across {row[0]:,} customers, the observed churn rate is {row[1]:.1%}."
        if "contract" in q or "segment" in q:
            rows = con.execute("SELECT contract_type, COUNT(*) n, AVG(churn) rate FROM customers GROUP BY 1 ORDER BY rate DESC").fetchall()
            return "Churn by contract: " + "; ".join(f"{r[0]}: {r[1]:,} customers, {r[2]:.1%} churn" for r in rows)
        if "support" in q:
            rows = con.execute("SELECT tech_support, COUNT(*) n, AVG(churn) rate FROM customers GROUP BY 1 ORDER BY rate DESC").fetchall()
            return "Churn by tech support: " + "; ".join(f"support={r[0]}: {r[1]:,} customers, {r[2]:.1%} churn" for r in rows)
        return "I can answer questions about overall churn, contract segments, month-to-month customers, and technical support."
    finally:
        con.close()


def predict_customer(customer: dict) -> dict:
    bundle = load_model(); model = bundle["model"]
    frame = pd.DataFrame([{k: customer.get(k) for k in FEATURES}])
    probability = float(model.predict_proba(frame)[0, 1])
    return {"churn_probability": round(probability, 4), "risk_band": "High" if probability >= .60 else "Medium" if probability >= .30 else "Low", "model": bundle.get("model_name", "Unknown")}


def explain_customer(customer: dict) -> str:
    shap_result = explain_with_shap(customer)
    if shap_result["available"]:
        top = shap_result["values"][:5]
        details = ", ".join(f"{x['feature']} ({x['shap_value']:+.3f})" for x in top)
        return f"Top model-specific SHAP contributions: {details}. Positive/negative values indicate direction in the model output space, not causal effects."
    reasons = []
    if customer.get("contract_type") == "Month-to-month": reasons.append("month-to-month contract")
    if float(customer.get("tenure_months", 99)) < 12: reasons.append("short tenure")
    if customer.get("tech_support") == "No": reasons.append("no technical support")
    if customer.get("autopay") == "No": reasons.append("autopay disabled")
    if float(customer.get("usage_score", 100)) < 25: reasons.append("low usage score")
    if int(customer.get("complaints_90d", 0)) >= 2: reasons.append("multiple recent complaints")
    if float(customer.get("monthly_charges", 0)) > 95: reasons.append("high monthly charges")
    return "SHAP explanation is unavailable for the current model/runtime. Rule-based review signals: " + (", ".join(reasons) if reasons else "no obvious rule-based flags") + ". These are descriptive signals, not proof of causation."


def _try_langchain_agent(question: str):
    """Optional real tool-calling agent. Returns None unless an API key and packages exist."""
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.tools import tool
        from langchain.agents import create_agent
        @tool
        def churn_analytics(query: str) -> str:
            """Read-only, allowlisted SQL analytics for overall churn, contract segments, month-to-month customers, and support."""
            return query_business_metrics(query)
        @tool
        def customer_prediction(customer_id: str) -> str:
            """Predict churn for a customer ID present in the local customer dataset."""
            data = pd.read_csv(ROOT / "data" / "customers.csv")
            row = data.loc[data.customer_id == customer_id]
            if row.empty: return "Customer ID not found. Ask the user to select a valid customer ID."
            return str(predict_customer(row.iloc[0].to_dict()))
        @tool
        def customer_explanation(customer_id: str) -> str:
            """Explain the model risk signals for a customer ID present in the local dataset."""
            data = pd.read_csv(ROOT / "data" / "customers.csv")
            row = data.loc[data.customer_id == customer_id]
            if row.empty: return "Customer ID not found. Ask the user to select a valid customer ID."
            return explain_customer(row.iloc[0].to_dict())
        agent = create_agent(ChatOpenAI(model=os.getenv("RETAINIQ_LLM_MODEL", "gpt-4o-mini"), temperature=0), tools=[churn_analytics, customer_prediction, customer_explanation], system_prompt="You are RetainIQ, a careful business analyst. Use tools for numerical claims. Never invent data. Treat predictions as estimates and SHAP as association, not causation. If information is unavailable, say so. Never execute arbitrary SQL or recommend discriminatory treatment.")
        result = agent.invoke({"messages": [{"role": "user", "content": question}]})
        messages = result.get("messages", [])
        return str(getattr(messages[-1], "content", messages[-1])) if messages else None
    except Exception:
        return None


def answer_question(question: str) -> str:
    agent_answer = _try_langchain_agent(question)
    if agent_answer:
        return agent_answer
    return query_business_metrics(question)
