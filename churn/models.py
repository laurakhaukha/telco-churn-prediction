"""Candidate models and their hyperparameter grids."""

from __future__ import annotations

from dataclasses import dataclass, field

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from churn.config import RANDOM_STATE
from churn.features import build_preprocessor


@dataclass
class Candidate:
    name: str
    label: str
    pipeline: Pipeline
    param_grid: dict = field(default_factory=dict)


def get_candidates(fast: bool = False) -> list[Candidate]:
    """Model zoo. ``fast=True`` shrinks grids and ensembles for tests/CI."""
    n_trees = 50 if fast else 400

    def grid(full: dict, small: dict) -> dict:
        return small if fast else full

    return [
        Candidate(
            name="baseline",
            label="Baseline (predict churn rate)",
            pipeline=Pipeline([
                ("prep", build_preprocessor()),
                ("model", DummyClassifier(strategy="prior")),
            ]),
        ),
        Candidate(
            name="logreg",
            label="Logistic regression",
            pipeline=Pipeline([
                ("prep", build_preprocessor(scale_numeric=True)),
                ("model", LogisticRegression(max_iter=5000)),
            ]),
            param_grid=grid(
                {"model__C": [0.01, 0.1, 1.0, 10.0]},
                {"model__C": [0.1, 1.0]},
            ),
        ),
        Candidate(
            name="random_forest",
            label="Random forest",
            pipeline=Pipeline([
                ("prep", build_preprocessor(scale_numeric=False)),
                ("model", RandomForestClassifier(
                    n_estimators=n_trees, n_jobs=-1, random_state=RANDOM_STATE)),
            ]),
            param_grid=grid(
                {"model__min_samples_leaf": [3, 10, 25], "model__max_features": ["sqrt", 0.3]},
                {"model__min_samples_leaf": [10]},
            ),
        ),
        Candidate(
            name="gradient_boosting",
            label="Gradient boosting",
            pipeline=Pipeline([
                ("prep", build_preprocessor(scale_numeric=False)),
                ("model", HistGradientBoostingClassifier(
                    max_iter=50 if fast else 300, l2_regularization=1.0,
                    early_stopping=False, random_state=RANDOM_STATE)),
            ]),
            param_grid=grid(
                {
                    "model__learning_rate": [0.03, 0.1],
                    "model__max_leaf_nodes": [7, 15],
                    "model__min_samples_leaf": [20, 60],
                },
                {"model__learning_rate": [0.1]},
            ),
        ),
    ]
