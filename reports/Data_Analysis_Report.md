# Data Analysis Report: Statistical Analysis & Hypothesis Testing
**Milestone 2 · Task 1 (Draft)** | Project NHA-5-153

> Every table and figure is generated reproducibly by `python src/statistical_analysis.py`
> (outputs in [`milestone2/tables/`](milestone2/tables/) and [`milestone2/figures/`](milestone2/figures/))
> and walked through in [`notebooks/04_statistical_analysis.ipynb`](../notebooks/04_statistical_analysis.ipynb).

---

## 1. Executive Summary

- **Tenure, salary and performance rating are significantly associated with attrition** (all $p_{\text{Holm}} < 0.001$); **job role, gender and age are not**.
- **Tenure is the strongest predictor** ($d = -0.26$, Cramér's $V = 0.11$): leavers have 1.29 fewer years of tenure, and each extra year lowers the adjusted odds of leaving by **6 %**.
- **Leavers earn 204 less per month** on average, and the gap is the same in every job role (two-way ANOVA interaction $p = 0.97$).
- **Leavers are rated higher** (+0.10 points; each rating point raises the odds of leaving by **12 %**).
- **Root-cause finding (one-way ANOVA):** **pay is not linked to performance** ($F = 0.67, p = 0.61$). Rating-5 employees earn the same on average as rating-1 employees. In addition, the salary gap between leavers and stayers **shrinks as rating rises** (two-way ANOVA interaction $p = 0.026$) and disappears at rating 5 ($p = 0.30$). Low performers who leave are underpaid; top performers who leave are not.
- **Job role has zero relationship** with attrition ($\chi^2 = 0.25, p = 0.99$), salary ($F = 0.24, p = 0.92$) or tenure ($F = 0.64, p = 0.63$).
- All features together give a **McFadden pseudo-$R^2$ of only 0.016**, so most of who leaves is driven by factors **not in the dataset** (overtime, satisfaction, manager quality). This sets realistic expectations for Milestone 3 models.

---

## 2. Methodology

- **Data:** `employee_attriton_data_clean.csv` ($n = 49{,}000$; 36,470 stayed, 12,530 left), analysed in **original units** so differences are directly interpretable.
- **Hypotheses:** for each feature $X$, $H_0$: $X$ is not associated with attrition · $H_1$: $X$ is associated with attrition.
- **Significance level:** $\alpha = 0.05$, two-sided, with **Holm step-down correction** across each family of tests to control the family-wise error rate.
- **Why effect sizes are required:** at $n = 49{,}000$, a tiny difference of $d \approx 0.02$ already achieves $p < 0.05$. Every test therefore reports an effect size (**Cohen's $d$**, rank-biserial $r$, **Cramér's $V$**, **$\eta^2$ / $\omega^2$**, or an **adjusted odds ratio**) alongside its $p$-value.

| Question | Parametric test | Non-parametric / robustness check | Effect size |
|---|---|---|---|
| Numeric feature vs attrition (2 groups) | **Welch's $t$-test** (unequal variances) | **Mann-Whitney $U$** | Cohen's $d$, rank-biserial $r$ |
| Categorical feature vs attrition | **Pearson's $\chi^2$** + adjusted residuals | — (all expected counts $\ge 205$) | Cramér's $V$ |
| Ordinal feature vs attrition (monotonic trend) | **Cochran-Armitage trend test** | — | linear-by-linear $r$ |
| Numeric feature across $>2$ groups | **One-way ANOVA** + Tukey HSD | **Kruskal-Wallis $H$** | $\eta^2$, $\omega^2$ |
| Does the salary gap depend on role / rating? | **Two-way ANOVA** (interaction term) | Stratified Welch $t$-tests | partial $\eta^2$ |
| All features jointly (controlling for confounders) | **Multivariable logistic regression** + LR tests | — | Adjusted OR, McFadden $R^2$ |

### 2.1 Assumption checks
(`milestone2/tables/00_assumption_checks.csv`)
- **MonthlySalary** is essentially normal (skew $\approx 0.03$) with equal variances across classes (Levene $p = 0.33$).
- **TenureYears** (skew $\approx 2.0$), **Age** and **PerformanceRating** fail normality and equal-variance checks (Levene $p < 0.002$).
- Using Welch's $t$-test (which drops the equal-variance assumption and is backed by the central limit theorem at $n = 49{,}000$) **and** requiring agreement with Mann-Whitney $U$ / Kruskal-Wallis keeps the conclusions valid despite skewness.

![Q-Q plots](milestone2/figures/00_qq_plots.png)

---

## 3. Numeric Features vs Attrition ($t$-tests & Mann-Whitney $U$)

| Feature | Mean (Stayed) | Mean (Left) | Diff (95 % CI) | Welch $t$ | $p_{\text{Holm}}$ | MWU $p_{\text{Holm}}$ | Cohen's $d$ | Effect | Decision |
|---|---|---|---|---|---|---|---|---|---|
| **TenureYears** | 4.85 | 3.56 | **−1.29** (−1.37, −1.20) | −28.99 | < 0.001 | < 0.001 | **−0.260** | **small** | **Reject $H_0$** |
| **Salary / Performance** | 2,205 | 2,034 | −171 (−195, −147) | −13.87 | < 0.001 | < 0.001 | −0.136 | negligible | **Reject $H_0$** |
| **PerformanceRating** | 3.12 | 3.22 | **+0.10** (+0.08, +0.12) | +10.29 | < 0.001 | < 0.001 | +0.107 | negligible | **Reject $H_0$** |
| **MonthlySalary** | 6,051 | 5,848 | **−204** (−243, −164) | −10.11 | < 0.001 | < 0.001 | −0.104 | negligible | **Reject $H_0$** |
| Age | 36.30 | 36.19 | −0.12 (−0.28, +0.05) | −1.42 | 0.157 | 0.078 | −0.015 | none | Fail to reject |

![Effect sizes](milestone2/figures/01_effect_sizes_cohens_d.png)

**Interpretation:** leavers have **shorter tenure, lower pay, and higher performance ratings** than stayers. Only tenure reaches Cohen's "small" threshold ($|d| \ge 0.20$); salary and performance are statistically solid ($|t| > 10$) but small in magnitude on their own. Age does not differ between groups.

---

## 4. Categorical & Ordinal Features ($\chi^2$ & Trend Tests)

| Feature | $\chi^2$ (dof) | $p_{\text{Holm}}$ | Cramér's $V$ | Effect | Attrition range | Highest-risk level (adj. residual) | Decision |
|---|---|---|---|---|---|---|---|
| **TenureGroup** | 564.81 (3) | < 0.001 | **0.107** | **small** | 14.2 % – 29.2 % | 0–2 yrs (+16.7) | **Reject $H_0$** |
| **PerformanceRating** | 110.62 (4) | < 0.001 | 0.048 | negligible | 20.6 % – 30.2 % | Rating 5 (+7.8) | **Reject $H_0$** |
| **SalaryBand** | 99.58 (3) | < 0.001 | 0.045 | negligible | 22.6 % – 28.1 % | Low (+7.3) | **Reject $H_0$** |
| AgeGroup | 16.79 (4) | 0.006 | 0.019 | negligible | 24.8 % – 27.5 % | 18–25 (+3.0) | Reject $H_0$* |
| Gender | 1.10 (2) | 1.000 | 0.005 | none | 25.5 % – 26.5 % | — (all $|r| \le 1.0$) | **Fail to reject** |
| **JobRole** | **0.25 (4)** | **1.000** | **0.002** | **none** | 25.4 % – 25.7 % | — (all $|r| \le 0.4$) | **Fail to reject** |

*\*AgeGroup loses significance once tenure is controlled for (Section 6).*

![Chi-square residuals](milestone2/figures/02_chi_square_residuals.png)

- **Cochran-Armitage trend tests** confirm that the three ordinal effects are **monotonic**: attrition rises steadily with `PerformanceRating` ($M^2 = 106.3, p < 0.001$) and falls steadily with `TenureGroup` ($M^2 = 526.1, p < 0.001$) and `SalaryBand` ($M^2 = 95.9, p < 0.001$).
- **JobRole and Gender are statistically indistinguishable from uniform attrition.**

---

## 5. ANOVA: Relationships Among Salary, Performance, Job Role & Attrition

### 5.1 One-way ANOVA (and Kruskal-Wallis)

| Question | $F$ | $p_{\text{Holm}}$ | $\eta^2$ | KW $p_{\text{Holm}}$ | Group means (min – max) | Conclusion |
|---|---|---|---|---|---|---|
| Salary across **JobRole** | 0.24 | 1.00 | 0.0000 | 1.00 | 5,988 – 6,012 | Same average pay in all 5 roles |
| Salary across **PerformanceRating** | **0.67** | **1.00** | **0.0001** | **1.00** | 5,945 – 6,008 | **Pay is not linked to performance** |
| Tenure across JobRole | 0.64 | 1.00 | 0.0001 | 1.00 | 4.47 – 4.57 yrs | Same tenure across roles |
| Tenure across PerformanceRating | 0.87 | 1.00 | 0.0001 | 1.00 | 4.51 – 4.70 yrs | Same tenure across ratings |

Tukey HSD (`milestone2/tables/07_tukey_salary_by_performance.csv`) finds **zero significant pairs** among the 10 rating comparisons (all adjusted $p \ge 0.60$).

![Salary by performance](milestone2/figures/04_salary_by_performance_anova.png)

> [!IMPORTANT]
> **Why high performers leave:** the company pays rating-5 employees the same as rating-1 employees (~6,000/month). Combined with the rising attrition rate across ratings, this points to **unrewarded high performance** as a central retention problem.

### 5.2 Two-way ANOVA & stratified salary tests

| Model | Term | $F$ | $p$ | Partial $\eta^2$ | Meaning |
|---|---|---|---|---|---|
| `MonthlySalary ~ JobRole * Attrition` | JobRole | 0.23 | 0.920 | 0.0000 | No role effect |
| | Attrition | 101.26 | < 0.001 | 0.0021 | Leavers earn ~204 less |
| | **JobRole × Attrition** | **0.13** | **0.972** | 0.0000 | **Same salary gap in every role** |
| `MonthlySalary ~ PerformanceRating * Attrition` | PerformanceRating | 0.98 | 0.419 | 0.0001 | No pay-for-performance |
| | Attrition | 102.51 | < 0.001 | 0.0021 | Leavers earn less overall |
| | **Performance × Attrition** | **2.77** | **0.026** | 0.0002 | **Salary gap shrinks at higher ratings** |

Stratified Welch $t$-tests (`08_stratified_salary_tests.csv`) make both interactions concrete:
- **Within each job role:** leavers earn **180 to 222 less** in all five roles (all $p_{\text{Holm}} < 0.001$).
- **Within each performance rating:** the leaver-vs-stayer salary gap shrinks step by step:
  **−346** (rating 1, $p = 0.002$) → **−279** (rating 2) → **−225** (rating 3) → **−152** (rating 4) → **−63 at rating 5 ($p = 0.30$, not significant)**.

![Two-way ANOVA interaction plots](milestone2/figures/03_anova_interaction_plots.png)

---

## 6. Multivariable Logistic Regression (Controlling for Confounders)

Fitting all features together (`Attrition_Flag ~ TenureYears + Salary_1k + PerformanceRating + Age + JobRole + Gender`):

| Feature | Adjusted OR (95 % CI) | Wald $p$ | LR $\chi^2$ (dof) | LR $p_{\text{Holm}}$ | Interpretation (holding other features constant) |
|---|---|---|---|---|---|
| **TenureYears** (+1 yr) | **0.940** (0.935–0.944) | < 0.001 | 696.8 (1) | < 0.001 | **−6.0 %** odds of leaving per year (~−46 % over 10 yrs) |
| **PerformanceRating** (+1 pt) | **1.121** (1.097–1.145) | < 0.001 | 107.8 (1) | < 0.001 | **+12.1 %** odds of leaving per rating point |
| **MonthlySalary** (+1,000) | **0.947** (0.937–0.957) | < 0.001 | 105.0 (1) | < 0.001 | **−5.3 %** odds of leaving per +1,000 salary |
| Age (+1 yr) | 1.002 (0.999–1.004) | 0.212 | 1.56 (1) | 0.636 | No effect once tenure is controlled for |
| JobRole (vs Data Analyst) | 1.006 – 1.016 | ≥ 0.62 | 0.26 (4) | 1.000 | No effect |
| Gender (vs Male) | 1.007 – 1.052 | ≥ 0.33 | 0.97 (2) | 1.000 | No effect |

**Model fit:** $\text{LLR } p < 0.001$, **McFadden pseudo-$R^2 = 0.0163$**, $\text{AIC} = 54{,}828$.

![Adjusted odds ratios](milestone2/figures/05_logistic_odds_ratios.png)

---

## 7. Implications for Milestone 2 (Task 2: Feature Selection / Engineering) & Milestone 3

| Decision | Evidence from this analysis |
|---|---|
| **Keep:** `TenureYears`, `MonthlySalary`, `PerformanceRating` | Significant in every univariate and multivariable test ($p_{\text{Holm}} < 0.001$) |
| **Drop (or RFE-confirm dropping):** `JobRole`, `Gender`, `Age` | Likelihood-ratio $p_{\text{Holm}} \ge 0.64$; dropping `Gender` also avoids demographic bias |
| **Engineer:** `Salary_Performance_Ratio`, `Salary × Performance`, `TenureGroup` | Supported by the significant Performance × Attrition interaction ($p = 0.026$) and the non-linear tenure curve |
| **Expect modest model metrics** | Pseudo-$R^2 = 0.016$ and max $|d| = 0.26$ mean the available features carry a real but limited signal; focus on Recall, F1 and ROC-AUC rather than accuracy |
| **Business action** | Link pay and progression to performance (ANOVA shows no link today) and target onboarding/retention in years 0–2 |
