"""Retention-campaign economics: turning churn probabilities into decisions.

Each month the retention team can contact customers with an offer (e.g. a
discount). The model is only useful if acting on it makes money, so this module
puts a price on every decision:

* every contact costs ``contact_cost``;
* a contacted customer who would have stayed anyway still accepts the offer,
  so the company pays ``incentive_cost`` for nothing;
* a contacted would-be churner is retained with probability ``save_rate``; a
  saved customer is worth ``gross_margin * value_months * MonthlyCharges``
  minus the incentive.

Expected profit of contacting a customer with churn probability ``p``::

    E[profit] = p * save_rate * (value - incentive) - (1 - p) * incentive - contact_cost

The default numbers are illustrative assumptions, not facts from the dataset.
Swap in real figures from the business before acting on the results.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np


@dataclass(frozen=True)
class CampaignAssumptions:
    contact_cost: float = 5.0      # $ per customer contacted (call / email + staff time)
    incentive_cost: float = 50.0   # $ cost of the retention offer when accepted
    save_rate: float = 0.30        # share of would-be churners the offer retains
    value_months: int = 12         # months of revenue a retained customer is worth
    gross_margin: float = 0.60     # share of revenue that is profit

    def to_dict(self) -> dict:
        return asdict(self)

    def with_(self, **changes) -> "CampaignAssumptions":
        return replace(self, **changes)


def customer_value(monthly_charges, a: CampaignAssumptions) -> np.ndarray:
    """Gross profit from keeping a customer for ``value_months`` more months."""
    return np.asarray(monthly_charges, dtype=float) * a.gross_margin * a.value_months


def expected_profit(churn_prob, monthly_charges, a: CampaignAssumptions) -> np.ndarray:
    """Expected profit of contacting each customer, given a churn probability."""
    p = np.asarray(churn_prob, dtype=float)
    value = customer_value(monthly_charges, a)
    return p * a.save_rate * (value - a.incentive_cost) - (1 - p) * a.incentive_cost - a.contact_cost


def realized_profit(churned, monthly_charges, a: CampaignAssumptions) -> np.ndarray:
    """Profit of contacting each customer given whether they actually churned.

    Uses the expected value over the save-rate coin flip, so results are
    deterministic.
    """
    y = np.asarray(churned).astype(bool)
    value = customer_value(monthly_charges, a)
    gain_if_churner = a.save_rate * (value - a.incentive_cost)
    return np.where(y, gain_if_churner, -a.incentive_cost) - a.contact_cost


def profit_curve(scores, churned, monthly_charges, a: CampaignAssumptions):
    """Cumulative realized profit when contacting customers in descending score order.

    Returns ``(share_targeted, cumulative_profit)``, both starting at 0.
    """
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    gains = realized_profit(churned, monthly_charges, a)[order]
    n = len(gains)
    share = np.arange(n + 1) / n
    cumulative = np.concatenate([[0.0], np.cumsum(gains)])
    return share, cumulative


def _summarise(mask, churned, monthly_charges, a: CampaignAssumptions) -> dict:
    mask = np.asarray(mask, dtype=bool)
    y = np.asarray(churned).astype(bool)
    gains = realized_profit(y, monthly_charges, a)
    n = len(y)
    profit = float(gains[mask].sum())
    return {
        "customers_contacted": int(mask.sum()),
        "share_contacted": float(mask.mean()),
        "churners_reached": int((mask & y).sum()),
        "share_of_churners_reached": float((mask & y).sum() / max(y.sum(), 1)),
        "precision": float(y[mask].mean()) if mask.any() else 0.0,
        "profit": profit,
        "profit_per_1000_customers": profit / n * 1000,
    }


def compare_policies(churn_prob, churned, monthly_charges, contract, a: CampaignAssumptions) -> dict:
    """Evaluate targeting strategies on labelled data.

    * ``model``: contact a customer when the model's expected profit is positive
    * ``random``: contact the same number of customers chosen at random (expected value)
    * ``month_to_month_rule``: contact everyone on a month-to-month contract
    * ``everyone`` / ``nobody``: the two trivial strategies
    * ``perfect_foresight``: contact exactly the churners worth saving (upper bound)
    """
    y = np.asarray(churned).astype(bool)
    monthly = np.asarray(monthly_charges, dtype=float)
    gains = realized_profit(y, monthly, a)
    model_mask = expected_profit(churn_prob, monthly, a) > 0

    results = {
        "model": _summarise(model_mask, y, monthly, a),
        "month_to_month_rule": _summarise(np.asarray(contract) == "Month-to-month", y, monthly, a),
        "everyone": _summarise(np.ones_like(y), y, monthly, a),
        "nobody": _summarise(np.zeros_like(y), y, monthly, a),
        "perfect_foresight": _summarise(y & (gains > 0), y, monthly, a),
    }

    k = int(model_mask.sum())
    random_profit = float(gains.mean() * k)
    results["random"] = {
        "customers_contacted": k,
        "share_contacted": k / len(y),
        "churners_reached": float(y.mean() * k),
        "share_of_churners_reached": k / len(y),
        "precision": float(y.mean()),
        "profit": random_profit,
        "profit_per_1000_customers": random_profit / len(y) * 1000,
    }
    return results


def sensitivity(churn_prob, churned, monthly_charges, contract, a: CampaignAssumptions,
                save_rates=(0.1, 0.2, 0.3, 0.4, 0.5)) -> list[dict]:
    """Profit per 1,000 customers for each policy across different offer save rates."""
    rows = []
    for rate in save_rates:
        res = compare_policies(churn_prob, churned, monthly_charges, contract, a.with_(save_rate=rate))
        rows.append({
            "save_rate": rate,
            "model": res["model"]["profit_per_1000_customers"],
            "model_share_contacted": res["model"]["share_contacted"],
            "month_to_month_rule": res["month_to_month_rule"]["profit_per_1000_customers"],
            "everyone": res["everyone"]["profit_per_1000_customers"],
        })
    return rows
