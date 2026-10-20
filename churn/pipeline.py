"""End-to-end pipeline: data -> model selection -> evaluation -> business case -> report.

Usage:
    python -m churn.pipeline            # full run (about a minute)
    python -m churn.pipeline --fast     # small grids, for a quick check
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split

from churn import business, config, plots
from churn.data import load_clean
from churn.evaluation import (
    bootstrap_intervals,
    expected_calibration_error,
    feature_ablation,
    grouped_permutation_importance,
)
from churn.models import get_candidates

SCORING = {"roc_auc": "roc_auc", "average_precision": "average_precision", "brier": "neg_brier_score"}


def select_models(X_train, y_train, cv, fast: bool, log=print) -> tuple[dict, dict]:
    """Tune every candidate with stratified CV on the training set only."""
    fitted, cv_results = {}, {}
    for cand in get_candidates(fast=fast):
        start = time.perf_counter()
        if cand.param_grid:
            search = GridSearchCV(cand.pipeline, cand.param_grid, scoring=SCORING,
                                  refit="roc_auc", cv=cv, n_jobs=-1)
            search.fit(X_train, y_train)
            i, res = search.best_index_, search.cv_results_
            stats = {m: (res[f"mean_test_{m}"][i], res[f"std_test_{m}"][i]) for m in SCORING}
            fitted[cand.name] = search.best_estimator_
            best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
        else:
            res = cross_validate(cand.pipeline, X_train, y_train, scoring=SCORING, cv=cv)
            stats = {m: (res[f"test_{m}"].mean(), res[f"test_{m}"].std()) for m in SCORING}
            fitted[cand.name] = cand.pipeline.fit(X_train, y_train)
            best_params = {}
        cv_results[cand.name] = {
            "label": cand.label,
            "roc_auc_mean": float(stats["roc_auc"][0]),
            "roc_auc_std": float(stats["roc_auc"][1]),
            "average_precision_mean": float(stats["average_precision"][0]),
            "average_precision_std": float(stats["average_precision"][1]),
            "brier_mean": float(-stats["brier"][0]),
            "best_params": best_params,
        }
        log(f"  {cand.label:<30} CV ROC-AUC {stats['roc_auc'][0]:.4f} ± {stats['roc_auc'][1]:.4f}"
            f"  ({time.perf_counter() - start:.1f}s)")
    return fitted, cv_results


def evaluate_on_test(fitted: dict, X_test, y_test) -> tuple[dict, dict]:
    probas, metrics = {}, {}
    for name, model in fitted.items():
        p = model.predict_proba(X_test)[:, 1]
        probas[name] = p
        metrics[name] = {
            "roc_auc": float(roc_auc_score(y_test, p)),
            "average_precision": float(average_precision_score(y_test, p)),
            "brier": float(brier_score_loss(y_test, p)),
            "log_loss": float(log_loss(y_test, p)),
            "calibration_error": expected_calibration_error(y_test, p),
        }
    return probas, metrics


def run(
    data_path: Path = config.DATA_PATH,
    reports_dir: Path = config.REPORTS_DIR,
    model_path: Path = config.MODEL_PATH,
    assumptions: business.CampaignAssumptions = business.CampaignAssumptions(),
    fast: bool = False,
    log=print,
) -> dict:
    t0 = time.perf_counter()
    reports_dir, model_path = Path(reports_dir), Path(model_path)
    figures = reports_dir / "figures"

    # 1. Data: one stratified split; the test set is touched only at step 4 ------
    df = load_clean(data_path)
    X, y = df[config.FEATURE_COLS], df[config.TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE)
    cv = StratifiedKFold(n_splits=3 if fast else config.CV_FOLDS, shuffle=True,
                         random_state=config.RANDOM_STATE)
    log(f"Data: {len(df):,} customers, churn rate {y.mean():.1%} | "
        f"train {len(X_train):,} / test {len(X_test):,}")

    # 2. Exploratory figure (training data only) ---------------------------------
    plots.churn_drivers(X_train.assign(Churn=y_train), figures / "churn_drivers.png")

    # 3. Model selection with cross-validation on the training set ---------------
    log("Cross-validating candidates on the training set:")
    fitted, cv_results = select_models(X_train, y_train, cv, fast=fast, log=log)
    contenders = {k: v for k, v in cv_results.items() if k != "baseline"}
    selected = max(contenders, key=lambda k: contenders[k]["roc_auc_mean"])
    model = fitted[selected]
    log(f"Selected: {cv_results[selected]['label']} (highest mean CV ROC-AUC)")

    log("Feature ablation for the selected model (CV on training set):")
    ablation = feature_ablation(model, X_train, y_train, cv)
    for row in ablation:
        log(f"  {row['features']:<20} {row['roc_auc_mean']:.4f}")

    # 4. Held-out test evaluation -------------------------------------------------
    probas, test_metrics = evaluate_on_test(fitted, X_test, y_test)
    labels = {k: v["label"] for k, v in cv_results.items()}
    model_probas = {k: v for k, v in probas.items() if k != "baseline"}
    plots.roc_pr_curves(y_test, model_probas, labels, figures / "roc_pr_curves.png")
    plots.calibration(y_test, model_probas, labels, figures / "calibration.png")

    # 5. Business case: who should the retention team contact? -------------------
    p_test = probas[selected]
    monthly = X_test["MonthlyCharges"].to_numpy()
    contract = X_test["Contract"].to_numpy()
    policies = business.compare_policies(p_test, y_test, monthly, contract, assumptions)
    exp_profit = business.expected_profit(p_test, monthly, assumptions)
    share, cumulative = business.profit_curve(exp_profit, y_test, monthly, assumptions)
    plots.profit_curve(share, cumulative, policies["model"]["share_contacted"],
                       policies["model"]["profit"], len(y_test), figures / "profit_curve.png")
    sens = business.sensitivity(p_test, y_test, monthly, contract, assumptions)
    boot = bootstrap_intervals(y_test, p_test, monthly, contract, assumptions,
                               n_boot=200 if fast else 2000, seed=config.RANDOM_STATE)

    # 6. Explainability -----------------------------------------------------------
    importances = grouped_permutation_importance(
        model, X_test, y_test, n_repeats=3 if fast else 30, seed=config.RANDOM_STATE)
    plots.permutation_importance_plot(importances, figures / "permutation_importance.png")

    # 7. Persist model and metrics ------------------------------------------------
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "model_name": selected,
        "assumptions": assumptions.to_dict(),
        "feature_columns": config.FEATURE_COLS,
        "sklearn_version": sklearn.__version__,
    }, model_path)

    metrics = {
        "data": {
            "n_customers": int(len(df)),
            "churn_rate": float(y.mean()),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        },
        "cross_validation": cv_results,
        "selected_model": selected,
        "feature_ablation": ablation,
        "test": test_metrics,
        "campaign": {
            "assumptions": assumptions.to_dict(),
            "policies": policies,
            "bootstrap": boot,
            "sensitivity_to_save_rate": sens,
        },
        "grouped_permutation_importance": importances.round(5).to_dict(orient="records"),
        "versions": {"scikit-learn": sklearn.__version__, "pandas": pd.__version__,
                     "numpy": np.__version__},
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # 8. Console summary ----------------------------------------------------------
    tm = test_metrics[selected]
    lo, hi = boot["roc_auc_95ci"]
    log("\nTest set (never used for training or model selection):")
    log(f"  ROC-AUC {tm['roc_auc']:.3f} (95% CI {lo:.3f}-{hi:.3f}) | PR-AUC "
        f"{tm['average_precision']:.3f} | Brier {tm['brier']:.3f} | "
        f"calibration error {tm['calibration_error']:.3f}")
    log("Retention campaign, profit per 1,000 customers:")
    for key, name in [("model", "model policy"), ("random", "random (same no.)"),
                      ("month_to_month_rule", "month-to-month rule"), ("everyone", "contact everyone"),
                      ("perfect_foresight", "perfect foresight")]:
        pol = policies[key]
        log(f"  {name:<20} {pol['profit_per_1000_customers']:>10,.0f}  "
            f"(contacts {pol['share_contacted']:.0%})")
    lo, hi = boot["model_minus_rule_per_1000_95ci"]
    log(f"  model - rule: 95% CI {lo:,.0f} to {hi:,.0f}")
    log(f"\nSaved model to {model_path}, report to {reports_dir} "
        f"({time.perf_counter() - t0:.0f}s)")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=config.DATA_PATH)
    ap.add_argument("--reports-dir", type=Path, default=config.REPORTS_DIR)
    ap.add_argument("--model-path", type=Path, default=config.MODEL_PATH)
    ap.add_argument("--fast", action="store_true", help="small grids for a quick check")
    defaults = business.CampaignAssumptions()
    ap.add_argument("--contact-cost", type=float, default=defaults.contact_cost)
    ap.add_argument("--incentive-cost", type=float, default=defaults.incentive_cost)
    ap.add_argument("--save-rate", type=float, default=defaults.save_rate)
    ap.add_argument("--value-months", type=int, default=defaults.value_months)
    ap.add_argument("--gross-margin", type=float, default=defaults.gross_margin)
    args = ap.parse_args()

    assumptions = business.CampaignAssumptions(
        contact_cost=args.contact_cost, incentive_cost=args.incentive_cost,
        save_rate=args.save_rate, value_months=args.value_months, gross_margin=args.gross_margin)
    run(args.data, args.reports_dir, args.model_path, assumptions, fast=args.fast)


if __name__ == "__main__":
    main()
