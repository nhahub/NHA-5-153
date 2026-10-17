"""
Employee Attrition - Milestone 2, Task 1: Statistical Analysis & Hypothesis Testing.

Tests the relationship between employee features (salary, performance rating,
job role, tenure, age, gender) and attrition using:

    * Welch's t-test + Mann-Whitney U         numeric feature  vs attrition (2 groups)
    * Chi-square test of independence         categorical      vs attrition
    * Cochran-Armitage trend test             ordinal (rating) vs attrition
    * One-way / two-way ANOVA + Kruskal-Wallis + Tukey HSD
    * Multivariable logistic regression       all features jointly (adjusted effects)

All p-values are Holm-corrected for multiple testing, and every test reports an
effect size: with n = 49,000 almost any difference is "significant", so the
effect size decides whether it matters in practice.

Usage (from the repo root):
    python src/statistical_analysis.py

Outputs:
    reports/milestone2/tables/*.csv
    reports/milestone2/figures/*.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eda_pipeline import DATA_PATH, add_eda_features  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "milestone2"
FIG_DIR, TAB_DIR = OUT / "figures", OUT / "tables"

ALPHA = 0.05
TARGET = "Attrition"
NUMERIC = ["MonthlySalary", "PerformanceRating", "TenureYears", "Age", "Salary_Performance_Ratio"]
CATEGORICAL = ["JobRole", "Gender", "PerformanceRating", "TenureGroup", "SalaryBand", "AgeGroup"]
PALETTE = {"No": "#4C72B0", "Yes": "#DD8452"}

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight"})


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _save_fig(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG_DIR / f"{name}.png")
    plt.close(fig)
    print(f"  [fig] {name}.png")


def _save_tab(df: pd.DataFrame, name: str, index: bool = False) -> None:
    df.to_csv(TAB_DIR / f"{name}.csv", index=index)
    print(f"  [tab] {name}.csv")


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d using the pooled standard deviation (b - a)."""
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (b.mean() - a.mean()) / pooled


def magnitude_d(d: float) -> str:
    d = abs(d)
    return "negligible" if d < 0.2 else "small" if d < 0.5 else "medium" if d < 0.8 else "large"


def magnitude_v(v: float, dof: int) -> str:
    # Cohen (1988) thresholds for Cramer's V depend on min(r, c) - 1
    small, medium, large = {1: (0.10, 0.30, 0.50), 2: (0.07, 0.21, 0.35)}.get(
        min(dof, 2), (0.06, 0.17, 0.29))
    return ("negligible" if v < small else "small" if v < medium
            else "medium" if v < large else "large")


def magnitude_eta(e: float) -> str:
    return "negligible" if e < 0.01 else "small" if e < 0.06 else "medium" if e < 0.14 else "large"


def holm(p: pd.Series) -> np.ndarray:
    return multipletests(p, alpha=ALPHA, method="holm")[1]


# --------------------------------------------------------------------------- #
# 0. Assumption checks
# --------------------------------------------------------------------------- #
def assumption_checks(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in NUMERIC:
        stay = df.loc[df[TARGET] == "No", col]
        left = df.loc[df[TARGET] == "Yes", col]
        lev_stat, lev_p = stats.levene(stay, left, center="median")
        for grp, s in (("No", stay), ("Yes", left)):
            k2, p_norm = stats.normaltest(s)  # D'Agostino-Pearson K^2
            rows.append({"Feature": col, "Group": grp, "n": len(s),
                         "Skew": s.skew(), "Kurtosis": s.kurtosis(),
                         "Normality_K2": k2, "Normality_p": p_norm,
                         "Normal_(p>0.05)": p_norm > ALPHA,
                         "Levene_W": lev_stat, "Levene_p": lev_p,
                         "EqualVar_(p>0.05)": lev_p > ALPHA})
    out = pd.DataFrame(rows).round(4)
    _save_tab(out, "00_assumption_checks")

    # Q-Q plots
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for j, col in enumerate(NUMERIC[:4]):
        for i, grp in enumerate(["No", "Yes"]):
            s = df.loc[df[TARGET] == grp, col].sample(3000, random_state=42)
            stats.probplot(s, dist="norm", plot=axes[i, j])
            axes[i, j].set_title(f"{col} | Attrition={grp}", fontsize=10)
            axes[i, j].get_lines()[0].set(markersize=2, alpha=0.4, color=PALETTE[grp])
    fig.suptitle("Q-Q plots vs normal distribution (3,000-row samples)", fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, "00_qq_plots")
    return out


# --------------------------------------------------------------------------- #
# 1. Numeric features vs attrition: Welch t-test + Mann-Whitney U
# --------------------------------------------------------------------------- #
def numeric_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in NUMERIC:
        a = df.loc[df[TARGET] == "No", col].to_numpy()
        b = df.loc[df[TARGET] == "Yes", col].to_numpy()
        t = stats.ttest_ind(b, a, equal_var=False)  # Welch
        ci = t.confidence_interval(confidence_level=1 - ALPHA)
        u, p_u = stats.mannwhitneyu(b, a, alternative="two-sided")
        rbc = 2 * u / (len(a) * len(b)) - 1  # rank-biserial correlation (Yes vs No)
        d = cohens_d(a, b)
        rows.append({
            "Feature": col,
            "Mean_Stayed": a.mean(), "Mean_Left": b.mean(),
            "Median_Stayed": np.median(a), "Median_Left": np.median(b),
            "MeanDiff_(Left-Stayed)": b.mean() - a.mean(),
            "CI95_low": ci.low, "CI95_high": ci.high,
            "Welch_t": t.statistic, "Welch_df": t.df, "Welch_p": t.pvalue,
            "MannWhitney_U": u, "MannWhitney_p": p_u,
            "Cohens_d": d, "RankBiserial_r": rbc, "Effect": magnitude_d(d),
        })
    out = pd.DataFrame(rows)
    out["Welch_p_holm"] = holm(out["Welch_p"])
    out["MannWhitney_p_holm"] = holm(out["MannWhitney_p"])
    out["Reject_H0"] = (out["Welch_p_holm"] < ALPHA) & (out["MannWhitney_p_holm"] < ALPHA)
    _save_tab(out.round(6), "01_numeric_ttest_mannwhitney")

    # Effect-size forest plot (Cohen's d with bootstrap-free analytic 95% CI)
    n1 = (df[TARGET] == "No").sum()
    n2 = (df[TARGET] == "Yes").sum()
    se = np.sqrt((n1 + n2) / (n1 * n2) + out["Cohens_d"] ** 2 / (2 * (n1 + n2)))
    plot = out.assign(lo=out["Cohens_d"] - 1.96 * se, hi=out["Cohens_d"] + 1.96 * se)
    plot = plot.sort_values("Cohens_d")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(plot["Cohens_d"], plot["Feature"],
                xerr=[plot["Cohens_d"] - plot["lo"], plot["hi"] - plot["Cohens_d"]],
                fmt="o", color="#DD8452", ecolor="k", capsize=4)
    for x in (-0.2, 0.2):
        ax.axvline(x, ls=":", c="grey")
    ax.axvline(0, c="k", lw=1)
    ax.set_xlabel("Cohen's d  (Left - Stayed)   |   dotted lines = 'small' effect threshold")
    ax.set_title("Effect size of numeric features on attrition (95% CI)", fontweight="bold")
    _save_fig(fig, "01_effect_sizes_cohens_d")
    return out


# --------------------------------------------------------------------------- #
# 2. Categorical features vs attrition: Chi-square + Cramer's V + residuals
# --------------------------------------------------------------------------- #
def chi_square_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows, resid_tables = [], {}
    for col in CATEGORICAL:
        ct = pd.crosstab(df[col], df[TARGET])
        chi2, p, dof, exp = stats.chi2_contingency(ct)
        n = ct.values.sum()
        v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
        # adjusted standardized residuals (|r| > 1.96 -> cell differs from independence)
        row_p = ct.sum(axis=1).to_numpy()[:, None] / n
        col_p = ct.sum(axis=0).to_numpy()[None, :] / n
        adj = (ct.values - exp) / np.sqrt(exp * (1 - row_p) * (1 - col_p))
        resid_tables[col] = pd.DataFrame(adj, index=ct.index.astype(str), columns=ct.columns)
        rates = ct["Yes"] / ct.sum(axis=1) * 100
        rows.append({"Feature": col, "Levels": ct.shape[0], "Chi2": chi2, "dof": dof,
                     "p_value": p, "Cramers_V": v, "Effect": magnitude_v(v, ct.shape[0] - 1),
                     "Min_Expected": exp.min(), "Expected>=5": bool(exp.min() >= 5),
                     "MinRate_%": rates.min(), "MaxRate_%": rates.max(),
                     "HighestRisk_Level": str(rates.idxmax())})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_value"])
    out["Reject_H0"] = out["p_holm"] < ALPHA
    _save_tab(out.round(6), "02_chi_square_tests")

    res = pd.concat({k: v["Yes"] for k, v in resid_tables.items()}).rename("AdjResidual_Yes")
    _save_tab(res.round(3).reset_index().rename(columns={"level_0": "Feature",
                                                         "level_1": "Level"}),
              "02_chi_square_adjusted_residuals")

    # Residual heatmap for the "Yes" column of each table
    fig, axes = plt.subplots(1, len(CATEGORICAL), figsize=(22, 4.8))
    for ax, col in zip(axes, CATEGORICAL):
        r = resid_tables[col][["Yes"]]
        sns.heatmap(r, annot=True, fmt=".1f", cmap="RdBu_r", center=0, vmin=-12, vmax=12,
                    cbar=False, ax=ax)
        p_h = out.loc[out.Feature == col, "p_holm"].iat[0]
        ax.set_title(f"{col}\np(holm)={p_h:.3g}", fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.suptitle("Chi-square adjusted residuals for Attrition = Yes  "
                 "(|r| > 1.96: more/fewer leavers than expected)", fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, "02_chi_square_residuals")
    return out


def trend_test(df: pd.DataFrame) -> pd.DataFrame:
    """Cochran-Armitage (linear-by-linear) trend test for ordinal features."""
    rows = []
    for col, scores in (("PerformanceRating", None), ("TenureGroup", None), ("SalaryBand", None)):
        x = df[col].cat.codes if hasattr(df[col], "cat") else df[col]
        y = df["Attrition_Flag"]
        r = np.corrcoef(x, y)[0, 1]
        m2 = (len(df) - 1) * r ** 2  # ~ chi2(1)
        p = stats.chi2.sf(m2, 1)
        rows.append({"Feature": col, "Trend_r": r, "M2_chi2(1)": m2, "p_value": p,
                     "Direction": "increasing" if r > 0 else "decreasing"})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_value"])
    _save_tab(out.round(6), "03_trend_tests")
    return out


# --------------------------------------------------------------------------- #
# 3. ANOVA
# --------------------------------------------------------------------------- #
def _anova_effects(table: pd.DataFrame) -> pd.DataFrame:
    ss_res = table.loc["Residual", "sum_sq"]
    ms_res = ss_res / table.loc["Residual", "df"]
    ss_tot = table["sum_sq"].sum()
    t = table.copy()
    t["eta_sq"] = t["sum_sq"] / ss_tot
    t["partial_eta_sq"] = t["sum_sq"] / (t["sum_sq"] + ss_res)
    t["omega_sq"] = (t["sum_sq"] - t["df"] * ms_res) / (ss_tot + ms_res)
    t.loc["Residual", ["eta_sq", "partial_eta_sq", "omega_sq"]] = np.nan
    return t


def anova_tests(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    res: dict[str, pd.DataFrame] = {}

    # (a) One-way ANOVAs + Kruskal-Wallis
    designs = [
        ("MonthlySalary", "JobRole", "Is salary different across job roles?"),
        ("MonthlySalary", "PerformanceRating", "Is pay linked to performance?"),
        ("TenureYears", "JobRole", "Is tenure different across job roles?"),
        ("TenureYears", "PerformanceRating", "Is tenure linked to performance?"),
    ]
    rows = []
    for y, g, question in designs:
        groups = [s.to_numpy() for _, s in df.groupby(g, observed=True)[y]]
        f, p = stats.f_oneway(*groups)
        h, p_kw = stats.kruskal(*groups)
        lev, p_lev = stats.levene(*groups, center="median")
        model = smf.ols(f"{y} ~ C({g})", data=df).fit()
        tab = _anova_effects(sm.stats.anova_lm(model, typ=2))
        eta = tab["eta_sq"].iloc[0]
        means = df.groupby(g, observed=True)[y].mean()
        rows.append({"Question": question, "Dependent": y, "Factor": g,
                     "F": f, "p_value": p, "eta_sq": eta, "omega_sq": tab["omega_sq"].iloc[0],
                     "Effect": magnitude_eta(eta), "KruskalWallis_H": h, "KW_p": p_kw,
                     "Levene_p": p_lev, "GroupMean_min": means.min(),
                     "GroupMean_max": means.max()})
    one = pd.DataFrame(rows)
    one["p_holm"] = holm(one["p_value"])
    one["KW_p_holm"] = holm(one["KW_p"])
    res["04_oneway_anova"] = one

    # (b) Two-way ANOVA: does the salary gap between leavers and stayers depend on the role?
    #     And: does the attrition-related salary gap depend on performance?
    for name, formula in (
        ("05_twoway_anova_salary_role_attrition", "MonthlySalary ~ C(JobRole) * C(Attrition)"),
        ("06_twoway_anova_salary_perf_attrition",
         "MonthlySalary ~ C(PerformanceRating) * C(Attrition)"),
    ):
        model = smf.ols(formula, data=df).fit()
        res[name] = _anova_effects(sm.stats.anova_lm(model, typ=2))

    # (c) Tukey HSD post-hoc: salary across performance ratings
    tk = pairwise_tukeyhsd(df["MonthlySalary"], df["PerformanceRating"].astype(str), alpha=ALPHA)
    res["07_tukey_salary_by_performance"] = pd.DataFrame(
        tk.summary().data[1:], columns=tk.summary().data[0])

    for k, v in res.items():
        _save_tab(v.round(6), k, index=k.startswith(("05", "06")))

    # Figures
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    sns.pointplot(data=df, x="JobRole", y="MonthlySalary", hue=TARGET, palette=PALETTE,
                  dodge=0.3, errorbar=("ci", 95), capsize=0.1, ax=axes[0])
    axes[0].set_title("Mean salary (95% CI) by job role and attrition")
    axes[0].tick_params(axis="x", rotation=15)
    sns.pointplot(data=df, x="PerformanceRating", y="MonthlySalary", hue=TARGET,
                  palette=PALETTE, dodge=0.3, errorbar=("ci", 95), capsize=0.1, ax=axes[1])
    axes[1].set_title("Mean salary (95% CI) by performance rating and attrition")
    fig.suptitle("Two-way ANOVA interaction plots (parallel lines = no interaction)",
                 fontweight="bold")
    _save_fig(fig, "03_anova_interaction_plots")

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=df, x="PerformanceRating", y="MonthlySalary", color="#9ecae1",
                flierprops={"marker": ".", "alpha": 0.2}, ax=ax)
    m = df.groupby("PerformanceRating")["MonthlySalary"].mean()
    ax.plot(range(len(m)), m.values, "o-", c="#DD8452", label="mean")
    p = one.loc[(one.Factor == "PerformanceRating") & (one.Dependent == "MonthlySalary"),
                "p_value"].iat[0]
    ax.set_title(f"Salary by performance rating - one-way ANOVA p = {p:.3f}", fontweight="bold")
    ax.legend()
    _save_fig(fig, "04_salary_by_performance_anova")
    return res


# --------------------------------------------------------------------------- #
# 4. Salary vs attrition within each job role (stratified Welch tests)
# --------------------------------------------------------------------------- #
def stratified_salary_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strat in ("JobRole", "PerformanceRating"):
        for level, sub in df.groupby(strat, observed=True):
            a = sub.loc[sub[TARGET] == "No", "MonthlySalary"].to_numpy()
            b = sub.loc[sub[TARGET] == "Yes", "MonthlySalary"].to_numpy()
            t = stats.ttest_ind(b, a, equal_var=False)
            rows.append({"Stratum": strat, "Level": str(level), "n_Stayed": len(a),
                         "n_Left": len(b), "Mean_Stayed": a.mean(), "Mean_Left": b.mean(),
                         "Diff": b.mean() - a.mean(), "Welch_t": t.statistic,
                         "p_value": t.pvalue, "Cohens_d": cohens_d(a, b)})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_value"])
    out["Reject_H0"] = out["p_holm"] < ALPHA
    _save_tab(out.round(6), "08_stratified_salary_tests")
    return out


# --------------------------------------------------------------------------- #
# 5. Multivariable logistic regression (adjusted odds ratios)
# --------------------------------------------------------------------------- #
def logistic_regression(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    d = df.copy()
    d["Salary_1k"] = d["MonthlySalary"] / 1000  # OR per +1,000 salary
    formula = ("Attrition_Flag ~ TenureYears + Salary_1k + PerformanceRating + Age "
               "+ C(JobRole, Treatment('Data Analyst')) + C(Gender, Treatment('Male'))")
    full = smf.logit(formula, data=d).fit(disp=False)

    ors = pd.DataFrame({
        "coef": full.params, "std_err": full.bse, "z": full.tvalues, "p_value": full.pvalues,
        "OddsRatio": np.exp(full.params),
        "OR_CI95_low": np.exp(full.conf_int()[0]), "OR_CI95_high": np.exp(full.conf_int()[1]),
    })
    ors.index = (ors.index.str.replace(r"C\((\w+), Treatment\('[^']+'\)\)\[T\.", r"\1=", regex=True)
                 .str.replace("]", "", regex=False))

    # Likelihood-ratio tests: does each feature block add information?
    blocks = {
        "TenureYears": "TenureYears", "Salary_1k": "Salary_1k",
        "PerformanceRating": "PerformanceRating", "Age": "Age",
        "JobRole": "C(JobRole, Treatment('Data Analyst'))",
        "Gender": "C(Gender, Treatment('Male'))",
    }
    lr_rows = []
    for name, term in blocks.items():
        reduced_formula = formula.replace(f" + {term}", "").replace(f"{term} + ", "")
        reduced = smf.logit(reduced_formula, data=d).fit(disp=False)
        lr = 2 * (full.llf - reduced.llf)
        dof = full.df_model - reduced.df_model
        lr_rows.append({"Feature": name, "LR_chi2": lr, "dof": dof,
                        "p_value": stats.chi2.sf(lr, dof)})
    lr_tab = pd.DataFrame(lr_rows)
    lr_tab["p_holm"] = holm(lr_tab["p_value"])
    lr_tab["Reject_H0"] = lr_tab["p_holm"] < ALPHA

    fit = pd.DataFrame({
        "Metric": ["n", "LogLik", "LogLik_null", "LLR_p", "McFadden_pseudo_R2", "AIC", "BIC"],
        "Value": [full.nobs, full.llf, full.llnull, full.llr_pvalue, full.prsquared,
                  full.aic, full.bic],
    })

    # Forest plot
    p = ors.drop(index="Intercept").sort_values("OddsRatio")
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#DD8452" if pv < ALPHA else "grey" for pv in p["p_value"]]
    ax.errorbar(p["OddsRatio"], p.index,
                xerr=[p["OddsRatio"] - p["OR_CI95_low"], p["OR_CI95_high"] - p["OddsRatio"]],
                fmt="none", ecolor="k", capsize=3)
    ax.scatter(p["OddsRatio"], p.index, c=colors, zorder=3, s=40)
    ax.axvline(1, c="k", lw=1, ls="--")
    ax.set_xlabel("Adjusted odds ratio of leaving (95% CI)   |   orange = p < 0.05")
    ax.set_title("Multivariable logistic regression - adjusted odds ratios", fontweight="bold")
    _save_fig(fig, "05_logistic_odds_ratios")

    out = {"09_logistic_odds_ratios": ors, "10_logistic_lr_tests": lr_tab,
           "11_logistic_model_fit": fit}
    for k, v in out.items():
        _save_tab(v.round(6), k, index=k.startswith("09"))
    (TAB_DIR / "logistic_summary.txt").write_text(str(full.summary()), encoding="utf-8")
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    df = add_eda_features(pd.read_csv(DATA_PATH))
    print(f"Loaded {DATA_PATH.name}: {df.shape}")

    print("0. Assumption checks");               assumption_checks(df)
    print("1. Welch t-test / Mann-Whitney U");   numeric_tests(df)
    print("2. Chi-square tests");                chi_square_tests(df)
    print("   Cochran-Armitage trend tests");    trend_test(df)
    print("3. ANOVA / Kruskal-Wallis / Tukey");  anova_tests(df)
    print("4. Stratified salary tests");         stratified_salary_tests(df)
    print("5. Logistic regression");             logistic_regression(df)
    print("Done.")


if __name__ == "__main__":
    main()
