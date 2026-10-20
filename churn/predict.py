"""Score new customers and produce a contact list for the retention team.

Usage:
    python -m churn.predict --input customers.csv --output contact_list.csv

The input needs the same columns as the training data (``Churn`` is optional).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from churn import config
from churn.business import CampaignAssumptions, expected_profit
from churn.data import clean


def score(customers: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    df = clean(customers)
    assumptions = CampaignAssumptions(**bundle["assumptions"])
    proba = bundle["model"].predict_proba(df[bundle["feature_columns"]])[:, 1]
    profit = expected_profit(proba, df["MonthlyCharges"], assumptions)
    out = pd.DataFrame({
        config.ID_COL: df[config.ID_COL],
        "churn_probability": proba.round(4),
        "expected_profit_if_contacted": profit.round(2),
        "contact": profit > 0,
    })
    return out.sort_values("expected_profit_if_contacted", ascending=False).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Score customers for churn risk.")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("contact_list.csv"))
    ap.add_argument("--model", type=Path, default=config.MODEL_PATH)
    args = ap.parse_args()

    if not args.model.exists():
        raise SystemExit(f"No model at {args.model}. Train one first: python -m churn.pipeline")
    bundle = joblib.load(args.model)
    scored = score(pd.read_csv(args.input), bundle)
    scored.to_csv(args.output, index=False)

    n = int(scored["contact"].sum())
    total = scored.loc[scored["contact"], "expected_profit_if_contacted"].sum()
    print(f"Scored {len(scored):,} customers -> {args.output}")
    print(f"Recommend contacting {n:,} ({n / len(scored):.0%}), expected profit ${total:,.0f}")


if __name__ == "__main__":
    main()
