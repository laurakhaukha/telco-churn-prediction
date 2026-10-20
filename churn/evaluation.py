"""Evaluation helpers: calibration error, bootstrap intervals, importance, feature ablation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from churn import business
from churn.features import ADDON_SERVICES, ENGINEERED_FEATURES, build_preprocessor

# Columns that only make sense together are shuffled together, otherwise the model
# is scored on customers that cannot exist:
# * TotalCharges ~= tenure x MonthlyCharges (correlation 0.9996), so shuffling
#   tenure alone pairs a 2-month customer with a 5-year bill.
# * MonthlyCharges is almost fully determined by the services a customer has
#   (R^2 = 0.999 from service dummies), so price and services form one group.
FEATURE_GROUPS = {
    "Services & monthly price": ["InternetService", *ADDON_SERVICES, "PhoneService",
                                 "MultipleLines", "MonthlyCharges"],
    "Tenure & total charges": ["tenure", "TotalCharges"],
    "Contract type": ["Contract"],
    "Payment method & billing": ["PaymentMethod", "PaperlessBilling"],
    "Demographics": ["gender", "SeniorCitizen", "Partner", "Dependents"],
}


def expected_calibration_error(y_true, proba, n_bins: int = 10) -> float:
    """Weighted mean |observed rate - mean prediction| over equal-frequency bins."""
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    bins = np.array_split(np.argsort(proba), n_bins)
    return float(sum(len(b) * abs(y_true[b].mean() - proba[b].mean()) for b in bins) / len(proba))


def grouped_permutation_importance(model, X: pd.DataFrame, y, groups=FEATURE_GROUPS,
                                   n_repeats: int = 20, seed: int = 0) -> pd.DataFrame:
    """Drop in ROC-AUC when each group of columns is shuffled jointly (same row order)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    base = roc_auc_score(y, model.predict_proba(X)[:, 1])
    rows = []
    for name, cols in groups.items():
        drops = []
        for _ in range(n_repeats):
            idx = rng.permutation(len(X))
            shuffled = X.copy()
            for col in cols:
                shuffled[col] = X[col].to_numpy()[idx]
            drops.append(base - roc_auc_score(y, model.predict_proba(shuffled)[:, 1]))
        rows.append({"feature": name, "columns": cols,
                     "mean": float(np.mean(drops)), "std": float(np.std(drops))})
    return pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)


def feature_ablation(estimator, X, y, cv) -> list[dict]:
    """Cross-validated ROC-AUC of ``estimator`` with different engineered-feature sets."""
    variants = {"raw columns only": ()}
    variants.update({f"+ {name}": (name,) for name in ENGINEERED_FEATURES})
    variants["+ all four"] = tuple(ENGINEERED_FEATURES)
    rows = []
    for label, extras in variants.items():
        est = clone(estimator)
        scale = isinstance(est.named_steps["prep"].named_steps["encode"].transformers[0][1],
                           StandardScaler)
        est.set_params(prep=build_preprocessor(scale_numeric=scale, extras=extras))
        scores = cross_val_score(est, X, y, cv=cv, scoring="roc_auc")
        rows.append({"features": label, "roc_auc_mean": float(scores.mean()),
                     "roc_auc_std": float(scores.std())})
    return rows


def bootstrap_intervals(y, proba, monthly_charges, contract, a: business.CampaignAssumptions,
                        n_boot: int = 2000, seed: int = 0) -> dict:
    """95% bootstrap intervals for test ROC-AUC and campaign profit per 1,000 customers."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    proba = np.asarray(proba)
    gains = business.realized_profit(y, monthly_charges, a)
    model_mask = business.expected_profit(proba, monthly_charges, a) > 0
    rule_mask = np.asarray(contract) == "Month-to-month"
    n = len(y)

    aucs, model_profit, rule_profit, everyone_profit = [], [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        aucs.append(roc_auc_score(y[idx], proba[idx]))
        g = gains[idx]
        model_profit.append(g[model_mask[idx]].sum() / n * 1000)
        rule_profit.append(g[rule_mask[idx]].sum() / n * 1000)
        everyone_profit.append(g.sum() / n * 1000)

    def ci(values):
        lo, hi = np.percentile(values, [2.5, 97.5])
        return [float(lo), float(hi)]

    model_profit, rule_profit = np.array(model_profit), np.array(rule_profit)
    everyone_profit = np.array(everyone_profit)
    return {
        "n_boot": len(aucs),
        "roc_auc_95ci": ci(aucs),
        "model_profit_per_1000_95ci": ci(model_profit),
        "model_minus_rule_per_1000_95ci": ci(model_profit - rule_profit),
        "model_beats_rule_share": float(np.mean(model_profit > rule_profit)),
        "model_minus_everyone_per_1000_95ci": ci(model_profit - everyone_profit),
    }
