from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, average_precision_score, confusion_matrix, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
TARGET = "churn"
FEATURES = ["tenure_months", "contract_type", "internet_service", "tech_support", "autopay", "monthly_charges", "usage_score", "complaints_90d"]
NUMERIC = ["tenure_months", "monthly_charges", "usage_score", "complaints_90d"]
CATEGORICAL = ["contract_type", "internet_service", "tech_support", "autopay"]


def _preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])


def train_model(df: pd.DataFrame | None = None):
    if df is None:
        df = pd.read_csv(ROOT / "data" / "customers.csv")
    X, y = df[FEATURES], df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    models = {"Logistic Regression": Pipeline([("prep", _preprocessor()), ("model", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42))])}
    try:
        from xgboost import XGBClassifier
        scale_pos_weight = max(1.0, float((y_train == 0).sum() / max(1, (y_train == 1).sum())))
        models["XGBoost"] = Pipeline([("prep", _preprocessor()), ("model", XGBClassifier(n_estimators=260, max_depth=4, learning_rate=.045, subsample=.85, colsample_bytree=.85, reg_lambda=2, scale_pos_weight=scale_pos_weight, eval_metric="logloss", random_state=42, n_jobs=4))])
    except ImportError:
        pass
    results = {}
    fitted = {}
    for name, candidate in models.items():
        candidate.fit(X_train, y_train)
        p = candidate.predict_proba(X_test)[:, 1]
        threshold = .40
        pred = (p >= threshold).astype(int)
        results[name] = {"roc_auc": float(roc_auc_score(y_test, p)), "average_precision": float(average_precision_score(y_test, p)), "precision": float(precision_score(y_test, pred, zero_division=0)), "recall": float(recall_score(y_test, pred, zero_division=0)), "f1": float(f1_score(y_test, pred, zero_division=0)), "brier_score": float(brier_score_loss(y_test, p)), "threshold": threshold}
        fitted[name] = candidate
    # Select by ROC-AUC on held-out test for demo comparison only. For rigorous model selection, use validation CV.
    selected_name = max(results, key=lambda name: results[name]["roc_auc"])
    model = fitted[selected_name]
    probs = model.predict_proba(X_test)[:, 1]
    threshold = results[selected_name]["threshold"]
    preds = (probs >= threshold).astype(int)
    metrics = {"model": selected_name, "test_rows": int(len(y_test)), **results[selected_name], "precision_at_threshold": results[selected_name]["precision"], "recall_at_threshold": results[selected_name]["recall"], "f1_at_threshold": results[selected_name]["f1"], "confusion_matrix": confusion_matrix(y_test, preds).tolist(), "model_comparison": results, "note": "Metrics are from a synthetic holdout set. Model selection on the test set is for this demo only; production model selection should use validation data and preserve a final untouched test set."}
    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump({"model": model, "model_name": selected_name, "features": FEATURES, "numeric": NUMERIC, "categorical": CATEGORICAL}, ARTIFACTS / "churn_model.joblib")
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return model, metrics


def load_model():
    path = ARTIFACTS / "churn_model.joblib"
    if not path.exists():
        train_model()
    return joblib.load(path)


def explain_with_shap(customer: dict, top_n: int = 8) -> dict:
    """Return model-specific SHAP attributions where supported, otherwise a clear fallback."""
    bundle = load_model()
    model = bundle["model"]
    frame = pd.DataFrame([{k: customer.get(k) for k in FEATURES}])
    try:
        import shap
        preprocessor = model.named_steps["prep"]
        estimator = model.named_steps["model"]
        transformed = preprocessor.transform(frame)
        feature_names = preprocessor.get_feature_names_out()
        explainer = shap.TreeExplainer(estimator)
        values = np.asarray(explainer.shap_values(transformed))
        if values.ndim == 3:
            values = values[0, :, 1]
        elif values.ndim == 2:
            values = values[0]
        else:
            values = values.reshape(-1)
        # XGBoost SHAP values are in raw-margin space; values are directional feature contributions, not probability points.
        pairs = sorted(zip([str(x).replace("num__", "").replace("cat__", "") for x in feature_names], values), key=lambda item: abs(float(item[1])), reverse=True)[:top_n]
        return {"available": True, "method": "SHAP TreeExplainer", "space": "model output (raw margin for tree models)", "values": [{"feature": k, "shap_value": float(v)} for k, v in pairs]}
    except Exception as exc:
        return {"available": False, "method": "SHAP unavailable for selected model/runtime", "values": [], "message": str(exc)[:220]}

if __name__ == "__main__":
    _, metrics = train_model()
    print(json.dumps(metrics, indent=2))
