"""Synthetic customers with the exact Telco schema, so tests and CI need no download."""

from __future__ import annotations

import numpy as np
import pandas as pd

from churn.features import ADDON_SERVICES


def make_fake_telco(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 73, n)
    tenure[:3] = 0  # brand-new customers have a blank TotalCharges, like the real data
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.2, 0.25])
    internet = rng.choice(["Fiber optic", "DSL", "No"], n, p=[0.45, 0.35, 0.2])
    phone = rng.choice(["Yes", "No"], n, p=[0.9, 0.1])

    df = pd.DataFrame({
        "customerID": [f"{i:04d}-FAKE" for i in range(n)],
        "gender": rng.choice(["Female", "Male"], n),
        "SeniorCitizen": rng.choice([0, 1], n, p=[0.84, 0.16]),
        "Partner": rng.choice(["Yes", "No"], n),
        "Dependents": rng.choice(["Yes", "No"], n, p=[0.3, 0.7]),
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": np.where(phone == "No", "No phone service", rng.choice(["Yes", "No"], n)),
        "InternetService": internet,
    })
    for col in ADDON_SERVICES:
        df[col] = np.where(internet == "No", "No internet service", rng.choice(["Yes", "No"], n))
    df["Contract"] = contract
    df["PaperlessBilling"] = rng.choice(["Yes", "No"], n)
    df["PaymentMethod"] = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], n)

    n_addons = (df[ADDON_SERVICES] == "Yes").sum(axis=1).to_numpy()
    monthly = (20 + 50 * (internet == "Fiber optic") + 25 * (internet == "DSL")
               + 5 * n_addons + rng.normal(0, 2, n)).round(2)
    df["MonthlyCharges"] = monthly
    total = (monthly * tenure).round(2)
    df["TotalCharges"] = [" " if t == 0 else f"{v}" for t, v in zip(tenure, total)]

    logit = (-1.2 + 1.6 * (contract == "Month-to-month") - 0.04 * tenure
             + 0.7 * (internet == "Fiber optic"))
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))
    df["Churn"] = np.where(churn, "Yes", "No")
    return df
