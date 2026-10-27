from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
from src.analyst import predict_customer, explain_customer, answer_question
from src.modeling import FEATURES

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "customers.csv"
app = FastAPI(title="RetainIQ API", version="1.0.0", description="Customer churn decision intelligence demo API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class Customer(BaseModel):
    tenure_months: int = Field(ge=0, le=120)
    contract_type: str
    internet_service: str = "Fiber optic"
    tech_support: str = "No"
    autopay: str = "No"
    monthly_charges: float = Field(ge=0)
    usage_score: float = Field(ge=0, le=100)
    complaints_90d: int = Field(ge=0)

class Question(BaseModel):
    question: str = Field(min_length=3, max_length=500)

@app.get("/")
def root(): return {"service": "RetainIQ API", "docs": "/docs", "health": "/health"}

@app.get("/health")
def health(): return {"status": "ok", "service": "RetainIQ"}

@app.get("/metrics")
def metrics():
    if not DATA_PATH.exists(): raise HTTPException(status_code=404, detail="Dataset not found. Run: python -m src.data_pipeline")
    df = pd.read_csv(DATA_PATH)
    total = len(df); churned = int(df["churn"].sum())
    segment = df.groupby("contract_type")["churn"].agg(["count", "mean"]).reset_index()
    return {"total_customers": total, "churned_customers": churned, "churn_rate": float(df["churn"].mean()), "avg_monthly_charges": float(df["monthly_charges"].mean()), "contract_breakdown": [{"contract": r["contract_type"], "customers": int(r["count"]), "churn_rate": float(r["mean"])} for _, r in segment.iterrows()]}

@app.get("/customers")
def customers(limit: int = Query(100, ge=1, le=1000), search: str = "", risk: str = "all"):
    if not DATA_PATH.exists(): raise HTTPException(status_code=404, detail="Dataset not found. Run: python -m src.data_pipeline")
    df = pd.read_csv(DATA_PATH)
    if search: df = df[df["customer_id"].astype(str).str.contains(search, case=False, na=False)]
    # Risk segments are indicative segments based on observable business rules, not model probabilities.
    def segment(row):
        score = (row["contract_type"] == "Month-to-month") * 2 + (row["tenure_months"] < 12) + (row["autopay"] == "No") + (row["tech_support"] == "No") + (row["usage_score"] < 25) + (row["complaints_90d"] >= 2)
        return "High" if score >= 4 else ("Medium" if score >= 2 else "Low")
    df["risk_segment"] = df.apply(segment, axis=1)
    if risk.lower() in {"high", "medium", "low"}: df = df[df["risk_segment"].str.lower() == risk.lower()]
    cols = ["customer_id", "tenure_months", "contract_type", "monthly_charges", "usage_score", "complaints_90d", "churn", "risk_segment"]
    return {"customers": df[cols].head(limit).to_dict(orient="records"), "count": int(len(df))}

@app.post("/predict")
def predict(customer: Customer):
    try: return predict_customer(customer.model_dump())
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain")
def explain(customer: Customer):
    try: return {"explanation": explain_customer(customer.model_dump())}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
def ask(payload: Question):
    try: return {"answer": answer_question(payload.question)}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))
