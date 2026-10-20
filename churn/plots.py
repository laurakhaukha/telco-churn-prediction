"""Figures for the report. All functions save a PNG and close the figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter, PercentFormatter  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

# Palette: categorical slots in fixed order (CVD-validated), plus ink and chrome.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",
    "axes.axisbelow": True,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "legend.frameon": False,
    "legend.labelcolor": INK_2,
    "lines.linewidth": 2,
    "lines.solid_capstyle": "round",
})

def money(v: float) -> str:
    return f"-${abs(v):,.0f}" if v < 0 else f"${v:,.0f}"


DOLLARS = FuncFormatter(lambda v, _: money(v))


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def churn_drivers(df: pd.DataFrame, path: Path) -> None:
    """Churn rate by the four strongest single drivers, with the overall rate as reference."""
    data = df.copy()
    data["Tenure (months)"] = pd.cut(
        data["tenure"], [-1, 6, 12, 24, 48, 72],
        labels=["0-6", "7-12", "13-24", "25-48", "49-72"],
    )
    panels = [
        ("Contract", ["Month-to-month", "One year", "Two year"]),
        ("Tenure (months)", ["0-6", "7-12", "13-24", "25-48", "49-72"]),
        ("InternetService", ["Fiber optic", "DSL", "No"]),
        ("PaymentMethod", ["Electronic check", "Mailed check",
                           "Bank transfer (automatic)", "Credit card (automatic)"]),
    ]
    titles = {"InternetService": "Internet service", "PaymentMethod": "Payment method"}
    overall = data["Churn"].mean()

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (col, order) in zip(axes.ravel(), panels):
        rates = data.groupby(col, observed=True)["Churn"].mean().reindex(order)
        counts = data[col].value_counts().reindex(order)
        labels = [o.replace(" (automatic)", "\n(automatic)") for o in order]
        ypos = np.arange(len(order))[::-1]
        ax.barh(ypos, rates.values, height=0.5, color=SERIES[0])
        ax.set_yticks(ypos, labels)
        for yv, r, n in zip(ypos, rates.values, counts.values):
            ax.text(r + 0.01, yv, f"{r:.0%}", va="center", color=INK, fontsize=9)
            ax.text(0.995, yv, f"n={n:,}", va="center", ha="right", color=MUTED,
                    fontsize=8, transform=ax.get_yaxis_transform())
        ax.axvline(overall, color=INK_2, linewidth=1)
        ax.text(overall + 0.005, ypos.max() + 0.42, f"overall {overall:.0%}",
                color=INK_2, fontsize=8, va="bottom")
        ax.set_xlim(0, 0.68)
        ax.set_ylim(-0.6, ypos.max() + 0.8)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.set_title(titles.get(col, col))
    fig.suptitle("One-month churn rate by customer segment (training data)",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout()
    _save(fig, path)


def roc_pr_curves(y_true, probas: dict[str, np.ndarray], labels: dict[str, str], path: Path) -> None:
    from sklearn.metrics import average_precision_score, roc_auc_score

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
    for i, (name, p) in enumerate(probas.items()):
        fpr, tpr, _ = roc_curve(y_true, p)
        prec, rec, _ = precision_recall_curve(y_true, p)
        ax1.plot(fpr, tpr, color=SERIES[i],
                 label=f"{labels[name]}  (AUC {roc_auc_score(y_true, p):.3f})")
        ax2.plot(rec, prec, color=SERIES[i],
                 label=f"{labels[name]}  (AP {average_precision_score(y_true, p):.3f})")
    ax1.plot([0, 1], [0, 1], color=MUTED, linewidth=1)
    ax1.text(0.62, 0.55, "random", color=MUTED, fontsize=8, rotation=0)
    base = float(np.mean(y_true))
    ax2.axhline(base, color=MUTED, linewidth=1)
    ax2.text(0.99, base + 0.015, f"random ({base:.0%} churn rate)", color=MUTED,
             fontsize=8, ha="right")
    ax1.set(xlabel="False positive rate", ylabel="True positive rate (recall)",
            xlim=(0, 1), ylim=(0, 1.01), title="ROC curve (test set)")
    ax2.set(xlabel="Recall (share of churners found)", ylabel="Precision",
            xlim=(0, 1), ylim=(0, 1.01), title="Precision-recall curve (test set)")
    ax1.legend(loc="lower right", fontsize=9)
    ax2.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    _save(fig, path)


def calibration(y_true, probas: dict[str, np.ndarray], labels: dict[str, str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1)
    ax.text(0.72, 0.66, "perfect", color=MUTED, fontsize=8, rotation=45)
    for i, (name, p) in enumerate(probas.items()):
        frac_pos, mean_pred = calibration_curve(y_true, p, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, color=SERIES[i], marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=labels[name])
    ax.set(xlabel="Predicted churn probability", ylabel="Observed churn rate",
           xlim=(0, 1), ylim=(0, 1), title="Calibration (test set, deciles)")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    _save(fig, path)


def profit_curve(share, cumulative, policy_share, policy_profit, n_customers, path: Path) -> None:
    """Cumulative campaign profit vs. share of customers contacted, per 1,000 customers."""
    scale = 1000 / n_customers
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(share, cumulative * scale, color=SERIES[0], label="Contact in order of model's expected profit")
    ax.plot([0, 1], [0, cumulative[-1] * scale], color=MUTED, linewidth=1.5,
            label="Contact at random")
    ax.axhline(0, color=AXIS, linewidth=1)
    best = int(np.argmax(cumulative))
    ax.plot(policy_share, policy_profit * scale, "o", color=SERIES[0], markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
    ax.annotate(
        f"Model policy: contact {policy_share:.0%}\n{money(policy_profit * scale)} per 1,000 customers",
        xy=(policy_share, policy_profit * scale), xytext=(policy_share + 0.08, policy_profit * scale * 0.55),
        color=INK, fontsize=9, arrowprops={"arrowstyle": "-", "color": INK_2, "linewidth": 1},
    )
    ax.annotate(
        f"Contact everyone: {money(cumulative[-1] * scale)}",
        xy=(1, cumulative[-1] * scale), xytext=(0.78, cumulative[-1] * scale + 0.12 * cumulative[best] * scale),
        color=INK_2, fontsize=9, ha="center",
    )
    ax.set(xlabel="Share of customers contacted", ylabel="Campaign profit per 1,000 customers",
           xlim=(0, 1), title="Retention campaign profit (test set, illustrative assumptions)")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(DOLLARS)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    _save(fig, path)


def permutation_importance_plot(importances: pd.DataFrame, path: Path) -> None:
    """Grouped permutation importance with the spread across repeats as whiskers."""
    data = importances.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(data) + 1.5))
    ypos = np.arange(len(data))
    ax.barh(ypos, data["mean"], height=0.5, color=SERIES[0])
    ax.errorbar(data["mean"], ypos, xerr=data["std"], fmt="none", ecolor=INK_2,
                elinewidth=1, capsize=0)
    pad = data["mean"].max() * 0.02
    for yv, m, sd in zip(ypos, data["mean"], data["std"]):
        ax.text(max(m, 0) + sd + pad, yv, f"{m:.3f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(ypos, data["feature"])
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_xlim(min(0, data["mean"].min() * 1.2), data["mean"].max() * 1.25)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set(xlabel="Drop in test ROC-AUC when the group is shuffled (mean of repeats)",
           title="What the model relies on (grouped permutation importance)")
    fig.tight_layout()
    _save(fig, path)
