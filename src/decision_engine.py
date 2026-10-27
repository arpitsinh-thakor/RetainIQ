import pandas as pd


def rank_retention_opportunities(df: pd.DataFrame, probabilities, offer_cost: float = 25.0, intervention_success: float = .30) -> pd.DataFrame:
    """Estimate incremental value under explicit assumptions.

    Expected incremental value = P(churn) * P(offer prevents churn) * customer value - offer cost.
    This is a prioritization heuristic, not a causal estimate until treatment-effect data exists.
    """
    result = df.copy()
    result["churn_probability"] = probabilities
    result["risk_band"] = pd.cut(result.churn_probability, [-.001, .30, .60, 1.001], labels=["Low", "Medium", "High"])
    result["expected_retained_value"] = result.churn_probability * intervention_success * result.customer_value
    result["expected_net_value"] = result.expected_retained_value - offer_cost
    result["recommended_action"] = result.apply(lambda r: "Review retention offer" if r.expected_net_value > 0 and r.churn_probability >= .30 else "Monitor" if r.churn_probability >= .30 else "No offer needed", axis=1)
    return result.sort_values(["expected_net_value", "churn_probability"], ascending=False)
