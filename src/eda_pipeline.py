"""
Employee Attrition - Milestone 1 EDA pipeline (Task 3 + Task 4).

Reproducibly generates every figure and summary table used in
`reports/EDA_Report.md` from the cleaned dataset produced in Task 1.

Usage (from the repo root):
    python src/eda_pipeline.py

Outputs:
    reports/figures/*.png       static figures (histograms, box plots, heatmaps, ...)
    reports/interactive/*.html  interactive Plotly charts
    reports/tables/*.csv        summary statistics and statistical tests
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend, no GUI needed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "employee_attriton_data_clean.csv"
FIG_DIR = ROOT / "reports" / "figures"
TAB_DIR = ROOT / "reports" / "tables"
HTML_DIR = ROOT / "reports" / "interactive"

TARGET = "Attrition"
NUMERIC_COLS = ["Age", "TenureYears", "PerformanceRating", "MonthlySalary"]
CATEGORICAL_COLS = ["Gender", "JobRole"]
PALETTE = {"No": "#4C72B0", "Yes": "#DD8452"}
ORDER = ["No", "Yes"]

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight"})


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the cleaned dataset and add EDA helper / segment columns."""
    df = pd.read_csv(path)
    return add_eda_features(df)


def add_eda_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary target and segment features used for visual analysis."""
    df = df.copy()
    df["Attrition_Flag"] = (df[TARGET] == "Yes").astype(int)

    df["TenureGroup"] = pd.cut(
        df["TenureYears"],
        bins=[-np.inf, 2, 5, 10, np.inf],
        labels=["0-2 yrs", "3-5 yrs", "6-10 yrs", "10+ yrs"],
    )
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[17, 25, 35, 45, 55, np.inf],
        labels=["18-25", "26-35", "36-45", "46-55", "56+"],
    )
    df["SalaryBand"] = pd.qcut(
        df["MonthlySalary"], q=4, labels=["Low", "Medium", "High", "Very High"]
    )
    df["Salary_Performance_Ratio"] = df["MonthlySalary"] / df["PerformanceRating"]
    return df


def attrition_rate(df: pd.DataFrame, by: str | list[str]) -> pd.DataFrame:
    """Attrition rate (%) and headcount per group."""
    g = df.groupby(by, observed=True)["Attrition_Flag"]
    out = pd.DataFrame({"Employees": g.size(), "Leavers": g.sum()})
    out["AttritionRate_%"] = (out["Leavers"] / out["Employees"] * 100).round(2)
    return out.reset_index()


def _save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG_DIR / f"{name}.png")
    plt.close(fig)
    print(f"  [fig] {name}.png")


# --------------------------------------------------------------------------- #
# Task 4 - Summary statistics & statistical tests
# --------------------------------------------------------------------------- #
def summary_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}

    # Overall descriptive statistics
    desc = df[NUMERIC_COLS].describe().T
    desc["skew"] = df[NUMERIC_COLS].skew()
    desc["kurtosis"] = df[NUMERIC_COLS].kurtosis()
    tables["summary_statistics"] = desc.round(3)

    # Numeric features split by Attrition
    tables["numeric_by_attrition"] = (
        df.groupby(TARGET)[NUMERIC_COLS].agg(["mean", "median", "std"]).round(3).T
    )

    # Target distribution
    vc = df[TARGET].value_counts()
    tables["target_distribution"] = pd.DataFrame(
        {"Count": vc, "Percent": (vc / len(df) * 100).round(2)}
    )

    # Attrition rate for every categorical / segment feature
    seg_frames = []
    for col in CATEGORICAL_COLS + ["PerformanceRating", "TenureGroup", "AgeGroup", "SalaryBand"]:
        t = attrition_rate(df, col).rename(columns={col: "Category"})
        t.insert(0, "Feature", col)
        t["Category"] = t["Category"].astype(str)
        seg_frames.append(t)
    tables["attrition_rate_by_segment"] = pd.concat(seg_frames, ignore_index=True)

    # Outliers (IQR rule)
    rows = []
    for col in NUMERIC_COLS:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = int(((df[col] < lo) | (df[col] > hi)).sum())
        rows.append(
            {"Feature": col, "Q1": q1, "Q3": q3, "IQR": iqr, "LowerFence": lo,
             "UpperFence": hi, "Outliers": n, "Outliers_%": round(n / len(df) * 100, 2)}
        )
    tables["outliers_iqr"] = pd.DataFrame(rows).round(3)

    # Imputation artefacts: share of rows sitting exactly on the most frequent value
    imp_rows = []
    for col in ["Age", "MonthlySalary", "PerformanceRating"]:
        mode_val = df[col].mode()[0]
        n = int((df[col] == mode_val).sum())
        imp_rows.append(
            {"Feature": col, "MostFrequentValue": mode_val, "Rows": n,
             "Rows_%": round(n / len(df) * 100, 2),
             "AttritionRate_%_at_value": round(
                 df.loc[df[col] == mode_val, "Attrition_Flag"].mean() * 100, 2)}
        )
    tables["imputation_spikes"] = pd.DataFrame(imp_rows)

    # Statistical tests (preview of Milestone 2)
    test_rows = []
    stay, left = df[df[TARGET] == "No"], df[df[TARGET] == "Yes"]
    for col in NUMERIC_COLS + ["Salary_Performance_Ratio"]:
        u, p = stats.mannwhitneyu(stay[col], left[col], alternative="two-sided")
        r, p_r = stats.pointbiserialr(df["Attrition_Flag"], df[col])
        test_rows.append(
            {"Feature": col, "Test": "Mann-Whitney U", "Statistic": u, "p_value": p,
             "Mean_Stayed": stay[col].mean(), "Mean_Left": left[col].mean(),
             "PointBiserial_r": r, "CramersV": np.nan}
        )
    for col in CATEGORICAL_COLS + ["PerformanceRating", "TenureGroup", "AgeGroup", "SalaryBand"]:
        ct = pd.crosstab(df[col], df[TARGET])
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
        test_rows.append(
            {"Feature": col, "Test": f"Chi-square (dof={dof})", "Statistic": chi2,
             "p_value": p, "Mean_Stayed": np.nan, "Mean_Left": np.nan,
             "PointBiserial_r": np.nan, "CramersV": v}
        )
    tests = pd.DataFrame(test_rows)
    tests["Significant_(p<0.05)"] = tests["p_value"] < 0.05
    tables["statistical_tests"] = tests.round(4)

    # Correlation of every encoded feature with Attrition
    enc = encoded_frame(df)
    corr = enc.corr()["Attrition_Flag"].drop("Attrition_Flag")
    tables["correlation_with_attrition"] = (
        corr.sort_values(key=np.abs, ascending=False).round(4).to_frame("Pearson_r")
    )
    return tables


def encoded_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric + one-hot encoded frame used for correlation heatmaps."""
    cols = NUMERIC_COLS + ["Salary_Performance_Ratio", "Attrition_Flag"]
    dummies = pd.get_dummies(df[CATEGORICAL_COLS], dtype=int)
    return pd.concat([df[cols], dummies], axis=1)


# --------------------------------------------------------------------------- #
# Task 3 - Visualizations
# --------------------------------------------------------------------------- #
def plot_target(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = sns.countplot(data=df, x=TARGET, order=ORDER, hue=TARGET, palette=PALETTE,
                       legend=False, ax=axes[0])
    for c in ax.containers:
        ax.bar_label(c, fmt="%d")
    ax.set_title("Attrition count")
    vc = df[TARGET].value_counts().reindex(ORDER)
    axes[1].pie(vc, labels=vc.index, autopct="%1.1f%%", startangle=90,
                colors=[PALETTE[k] for k in vc.index], wedgeprops={"edgecolor": "white"})
    axes[1].set_title("Attrition share (class imbalance)")
    fig.suptitle("Target variable distribution", fontweight="bold")
    _save(fig, "01_attrition_distribution")


def plot_histograms(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, col in zip(axes.flat, NUMERIC_COLS):
        discrete = col == "PerformanceRating"
        sns.histplot(data=df, x=col, hue=TARGET, hue_order=ORDER, palette=PALETTE,
                     stat="density", common_norm=False, kde=not discrete,
                     discrete=discrete, element="step", alpha=0.35, ax=ax,
                     bins=40 if not discrete else "auto")
        ax.set_title(f"{col} distribution by Attrition")
    fig.suptitle("Histograms of numerical features (density, normalised per class)",
                 fontweight="bold")
    fig.subplots_adjust(hspace=0.35)
    _save(fig, "02_histograms_numeric")


def plot_boxplots(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, col in zip(axes, NUMERIC_COLS):
        sns.boxplot(data=df, x=TARGET, y=col, order=ORDER, hue=TARGET, palette=PALETTE,
                    legend=False, ax=ax, flierprops={"marker": ".", "alpha": 0.3})
        ax.set_title(col)
    fig.suptitle("Box plots: numerical features vs Attrition (outliers shown as dots)",
                 fontweight="bold")
    _save(fig, "03_boxplots_numeric")

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, col in zip(axes, NUMERIC_COLS):
        sns.violinplot(data=df, x=TARGET, y=col, order=ORDER, hue=TARGET,
                       palette=PALETTE, legend=False, inner="quartile", cut=0, ax=ax)
        ax.set_title(col)
    fig.suptitle("Violin plots: distribution shape vs Attrition", fontweight="bold")
    _save(fig, "04_violin_numeric")


def _rate_bar(df: pd.DataFrame, col: str, ax: plt.Axes, overall: float) -> None:
    t = attrition_rate(df, col)
    t[col] = t[col].astype(str)
    if t[col].nunique() > 3 and col in CATEGORICAL_COLS:
        t = t.sort_values("AttritionRate_%", ascending=False)
    sns.barplot(data=t, x=col, y="AttritionRate_%", color="#DD8452", ax=ax)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.1f%%", fontsize=9)
    ax.axhline(overall, ls="--", c="k", lw=1, label=f"Overall {overall:.1f}%")
    ax.set_ylabel("Attrition rate (%)")
    ax.set_ylim(0, max(t["AttritionRate_%"].max(), overall) * 1.25)
    ax.set_title(f"Attrition rate by {col}")
    ax.legend(loc="upper right", fontsize=8)
    ax.tick_params(axis="x", rotation=20)


def plot_categorical(df: pd.DataFrame) -> None:
    overall = df["Attrition_Flag"].mean() * 100
    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    _rate_bar(df, "JobRole", axes[0], overall)
    _rate_bar(df, "Gender", axes[1], overall)
    _rate_bar(df, "PerformanceRating", axes[2], overall)
    fig.suptitle("Attrition rate by categorical features", fontweight="bold")
    _save(fig, "05_attrition_by_categorical")

    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    _rate_bar(df, "TenureGroup", axes[0], overall)
    _rate_bar(df, "AgeGroup", axes[1], overall)
    _rate_bar(df, "SalaryBand", axes[2], overall)
    fig.suptitle("Attrition rate by engineered segments", fontweight="bold")
    _save(fig, "06_attrition_by_segments")


def plot_heatmaps(df: pd.DataFrame) -> None:
    enc = encoded_frame(df)

    # (a) full correlation matrix
    corr = enc.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                vmin=-1, vmax=1, linewidths=0.5, annot_kws={"size": 8}, ax=ax)
    ax.set_title("Correlation heatmap (numeric + one-hot encoded features)",
                 fontweight="bold")
    _save(fig, "07_correlation_heatmap")

    # (b) correlation with target only
    tc = corr["Attrition_Flag"].drop("Attrition_Flag").sort_values()
    fig, ax = plt.subplots(figsize=(4, 8))
    sns.heatmap(tc.to_frame("Attrition"), annot=True, fmt=".3f", cmap="coolwarm",
                center=0, cbar=False, ax=ax)
    ax.set_title("Correlation with Attrition", fontweight="bold")
    _save(fig, "08_correlation_with_attrition")

    # (c-e) attrition-rate pivots
    pivots = [
        ("JobRole", "TenureGroup", "09_heatmap_jobrole_x_tenure"),
        ("JobRole", "SalaryBand", "10_heatmap_jobrole_x_salary"),
        ("JobRole", "Gender", "11_heatmap_jobrole_x_gender"),
        ("TenureGroup", "PerformanceRating", "12_heatmap_tenure_x_performance"),
        ("SalaryBand", "PerformanceRating", "14_heatmap_salary_x_performance"),
    ]
    for rows, cols, name in pivots:
        pv = df.pivot_table(index=rows, columns=cols, values="Attrition_Flag",
                            aggfunc="mean", observed=True) * 100
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.heatmap(pv, annot=True, fmt=".1f", cmap="YlOrRd", linewidths=0.5,
                    cbar_kws={"label": "Attrition rate (%)"}, ax=ax)
        ax.set_title(f"Attrition rate (%) - {rows} x {cols}", fontweight="bold")
        _save(fig, name)


def plot_pairplot(df: pd.DataFrame) -> None:
    sample = df.sample(n=min(4000, len(df)), random_state=42)
    g = sns.pairplot(sample, vars=["Age", "TenureYears", "MonthlySalary"], hue=TARGET,
                     hue_order=ORDER, palette=PALETTE, corner=True,
                     plot_kws={"alpha": 0.35, "s": 10}, diag_kind="kde")
    g.figure.suptitle("Pairwise relationships (4,000-row sample)", y=1.02,
                      fontweight="bold")
    g.savefig(FIG_DIR / "13_pairplot_sample.png")
    plt.close(g.figure)
    print("  [fig] 13_pairplot_sample.png")


def plot_interactive(df: pd.DataFrame) -> None:
    """Interactive Plotly charts (deliverable: 'Interactive Visualizations')."""
    try:
        import plotly.express as px
    except ImportError:
        print("  [skip] plotly not installed - interactive charts skipped")
        return

    figs = {
        "salary_vs_tenure_scatter": px.scatter(
            df.sample(5000, random_state=42), x="TenureYears", y="MonthlySalary",
            color=TARGET, color_discrete_map=PALETTE, facet_col="JobRole",
            facet_col_wrap=3, opacity=0.5, hover_data=["Age", "PerformanceRating"],
            title="Monthly salary vs tenure by job role (5k sample)"),
        "salary_box_by_role": px.box(
            df, x="JobRole", y="MonthlySalary", color=TARGET,
            color_discrete_map=PALETTE, title="Monthly salary by job role and attrition"),
        "attrition_sunburst": px.sunburst(
            attrition_rate(df, ["JobRole", "TenureGroup"]),
            path=["JobRole", "TenureGroup"], values="Employees",
            color="AttritionRate_%", color_continuous_scale="YlOrRd",
            title="Headcount (size) and attrition rate (colour) by role and tenure"),
        "age_histogram": px.histogram(
            df, x="Age", color=TARGET, color_discrete_map=PALETTE, barmode="overlay",
            histnorm="percent", nbins=40, title="Age distribution by attrition"),
    }
    for name, fig in figs.items():
        fig.write_html(HTML_DIR / f"{name}.html", include_plotlyjs="cdn")
        print(f"  [html] {name}.html")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    for d in (FIG_DIR, TAB_DIR, HTML_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print(f"Loading {DATA_PATH.name} ...")
    df = load_data()
    print(f"  shape={df.shape}, nulls={int(df.isna().sum().sum())}, "
          f"duplicates={int(df.duplicated().sum())}")

    print("Summary statistics ...")
    for name, table in summary_tables(df).items():
        table.to_csv(TAB_DIR / f"{name}.csv")
        print(f"  [tab] {name}.csv")

    print("Figures ...")
    plot_target(df)
    plot_histograms(df)
    plot_boxplots(df)
    plot_categorical(df)
    plot_heatmaps(df)
    plot_pairplot(df)
    plot_interactive(df)
    print("Done.")


if __name__ == "__main__":
    main()
