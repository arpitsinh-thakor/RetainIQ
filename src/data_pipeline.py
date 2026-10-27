"""Reproducible demo data pipeline for RetainIQ.

Uses a generated subscription dataset so the project runs immediately. Replace
this generator with a governed real-data ingestion connector for production.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DB_PATH = ROOT / "artifacts" / "retainiQ.duckdb"


def generate_demo_data(n: int = 3500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 73, n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[.56, .25, .19])
    internet = rng.choice(["Fiber optic", "DSL", "No"], n, p=[.48, .36, .16])
    support = rng.choice(["Yes", "No"], n, p=[.32, .68])
    autopay = rng.choice(["Yes", "No"], n, p=[.54, .46])
    monthly = np.clip(rng.normal(72, 22, n) + (internet == "Fiber optic") * 12, 18, 160).round(2)
    usage = np.clip(rng.normal(45, 18, n) + np.minimum(tenure, 24) * .25, 1, 100).round(1)
    complaints = rng.poisson(1.0 + (usage < 25) * .7, n)
    logit = (-2.05 + (contract == "Month-to-month") * 1.25 + (contract == "One year") * -.65
             + (tenure < 12) * .85 + (internet == "Fiber optic") * .35
             + (support == "No") * .55 + (autopay == "No") * .45
             + (usage < 25) * .75 + complaints * .22 + (monthly > 95) * .35)
    probability = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, probability)
    df = pd.DataFrame({
        "customer_id": [f"CUST-{i:05d}" for i in range(1, n + 1)],
        "tenure_months": tenure,
        "contract_type": contract,
        "internet_service": internet,
        "tech_support": support,
        "autopay": autopay,
        "monthly_charges": monthly,
        "usage_score": usage,
        "complaints_90d": complaints,
        "customer_value": (monthly * (tenure + 6) * rng.uniform(.75, 1.25, n)).round(2),
        "churn": churn,
    })
    return df


def validate_data(df: pd.DataFrame) -> list[str]:
    issues = []
    required = {"customer_id", "tenure_months", "contract_type", "monthly_charges", "churn"}
    missing = required - set(df.columns)
    if missing: issues.append(f"Missing required columns: {sorted(missing)}")
    if "customer_id" in df and df.customer_id.duplicated().any(): issues.append("Duplicate customer_id values found")
    if "churn" in df and not set(df.churn.dropna().unique()).issubset({0, 1}): issues.append("churn must contain only 0/1 values")
    if df.isnull().sum().sum(): issues.append(f"Null values found: {int(df.isnull().sum().sum())}")
    return issues


def run_pipeline(n: int = 3500, seed: int = 42) -> pd.DataFrame:
    DATA_DIR.mkdir(exist_ok=True); (ROOT / "artifacts").mkdir(exist_ok=True)
    csv_path = DATA_DIR / "customers.csv"
    df = generate_demo_data(n, seed)
    issues = validate_data(df)
    if issues: raise ValueError("Data validation failed: " + "; ".join(issues))
    df.to_csv(csv_path, index=False)
    con = duckdb.connect(str(DB_PATH))
    con.register("incoming_customers", df)
    con.execute("CREATE OR REPLACE TABLE customers AS SELECT * FROM incoming_customers")
    con.close()
    return df

if __name__ == "__main__":
    frame = run_pipeline()
    print(f"Created {len(frame):,} demo customers; churn rate={frame.churn.mean():.1%}")
    print(f"CSV: {DATA_DIR / 'customers.csv'}\nDuckDB: {DB_PATH}")
