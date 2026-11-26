"""Streamlit dashboard for HR teams.

    streamlit run src/attrition/dashboard.py

Tabs: single-employee risk check | batch scoring (CSV upload) | monitoring | model card.
Scores locally by default; set ``API_URL`` (e.g. http://api:8000) to score through the REST API instead,
so the same prediction log / metrics are used as for any other client.
"""

from __future__ import annotations

import json
import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from attrition import config
from attrition.config import API_TO_MODEL, FEATURE_LIMITS, MODEL_FEATURES
from attrition.data import read_jsonl
from attrition.predictor import ModelNotFoundError, ModelService

API_URL = os.environ.get("API_URL", "").rstrip("/")
API_KEY = os.environ.get("API_KEY", "")
COLORS = {"Low": "#2e9e5b", "Medium": "#f0a30a", "High": "#d64545"}
ADVICE = {
    "High": "Prioritise a retention conversation soon: review pay position, growth path and workload.",
    "Medium": "Keep on the watch-list; check in at the next review cycle.",
    "Low": "No action needed based on these factors.",
}

st.set_page_config(page_title="Attrition Risk", page_icon=":bar_chart:", layout="wide")


@st.cache_resource(show_spinner=False)
def get_service() -> ModelService | None:
    try:
        return ModelService()
    except ModelNotFoundError:
        return None


def score_records(records: list[dict], explain: bool) -> list[dict]:
    """Returns dicts with attrition_probability / risk_level (/ top_drivers)."""
    if API_URL:
        headers = {"X-API-Key": API_KEY} if API_KEY else {}
        if len(records) == 1:
            r = requests.post(f"{API_URL}/predict", json=records[0], headers=headers, timeout=20)
            r.raise_for_status()
            return [r.json()]
        out = []
        for i in range(0, len(records), 5000):
            r = requests.post(f"{API_URL}/predict/batch", json={"employees": records[i:i + 5000]},
                              headers=headers, timeout=120)
            r.raise_for_status()
            out += r.json()["predictions"]
        return out
    svc = get_service()
    frame = svc.to_frame(records)
    scored = svc.predict_frame(frame)
    rows = []
    for i in range(len(records)):
        row = scored.iloc[i].to_dict()
        if explain:
            row["top_drivers"] = svc.explain(frame.iloc[i])
        rows.append(row)
    return rows


def gauge(prob: float, level: str, thresholds: dict) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100, number={"suffix": "%"},
        title={"text": f"Attrition risk: <b>{level}</b>"},
        gauge={"axis": {"range": [0, 60]}, "bar": {"color": COLORS[level]},
               "steps": [{"range": [0, thresholds["medium"] * 100], "color": "#e6f4ec"},
                         {"range": [thresholds["medium"] * 100, thresholds["high"] * 100], "color": "#fdf0d0"},
                         {"range": [thresholds["high"] * 100, 60], "color": "#f9dada"}]}))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=10))
    return fig


# --------------------------------------------------------------------------------------------
st.title("Employee Attrition Risk")
svc = get_service()
if svc is None and not API_URL:
    st.error("No trained model found. Run `python -m attrition.train` first.")
    st.stop()
card = svc.card if svc else requests.get(f"{API_URL}/model/info", timeout=10).json()
thresholds = card["risk_policy"]["thresholds"]
st.caption(f"Model `{card['model_id']}` ({card['algorithm']}) | test ROC-AUC {card['metrics']['test']['roc_auc']:.2f} | "
           "a **ranking aid** for prioritising conversations, not a decision-maker.")

tab_one, tab_batch, tab_mon, tab_model = st.tabs(["Check an employee", "Batch scoring", "Monitoring", "Model"])

with tab_one:
    c1, c2 = st.columns([1, 1])
    with c1:
        emp_id = st.text_input("Employee ID (optional)", "")
        tenure = st.number_input("Tenure (years)", *FEATURE_LIMITS["TenureYears"][:1], FEATURE_LIMITS["TenureYears"][1],
                                 value=2.0, step=0.5, key="tenure")
        rating = st.select_slider("Performance rating (1-5)", options=[1, 2, 3, 4, 5], value=3, key="rating")
        salary = st.number_input("Monthly salary", 1.0, FEATURE_LIMITS["MonthlySalary"][1], value=6000.0, step=100.0,
                                 key="salary")
        go_btn = st.button("Predict attrition risk", type="primary")
    with c2:
        if go_btn:
            rec = {"employee_id": emp_id or None, "tenure_years": tenure, "performance_rating": int(rating),
                   "monthly_salary": salary}
            res = score_records([rec], explain=True)[0]
            prob, level = res["attrition_probability"], res["risk_level"]
            st.plotly_chart(gauge(prob, level, thresholds), use_container_width=True)
            st.info(ADVICE[level])
            drivers = res.get("top_drivers") or []
            if drivers:
                st.markdown("**What moves this employee's risk** (vs. a typical employee)")
                df_d = pd.DataFrame(drivers)
                fig = go.Figure(go.Bar(x=df_d["impact"] * 100, y=df_d["label"], orientation="h",
                                       marker_color=["#d64545" if v > 0 else "#2e9e5b" for v in df_d["impact"]]))
                fig.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10),
                                  xaxis_title="percentage points of risk")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Enter the employee's details and press **Predict**.")

with tab_batch:
    st.write("Upload a CSV with columns `TenureYears`, `PerformanceRating`, `MonthlySalary` "
             "(an optional `EmployeeID` is kept). Other columns such as Age or Gender are ignored.")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        data = pd.read_csv(up)
        missing = [c for c in MODEL_FEATURES if c not in data.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            bad = ~data[MODEL_FEATURES].apply(lambda s: s.between(*FEATURE_LIMITS[s.name])).all(axis=1)
            if bad.any():
                st.warning(f"{int(bad.sum())} rows have missing/out-of-range values and were skipped.")
            data = data[~bad].reset_index(drop=True)
            recs = [{"tenure_years": r.TenureYears, "performance_rating": int(r.PerformanceRating),
                     "monthly_salary": r.MonthlySalary} for r in data.itertuples()]
            res = pd.DataFrame(score_records(recs, explain=False))
            out = pd.concat([data, res[["attrition_probability", "risk_level"]]], axis=1) \
                .sort_values("attrition_probability", ascending=False)
            mix = out["risk_level"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0)
            m1, m2, m3 = st.columns(3)
            m1.metric("High risk", int(mix["High"])); m2.metric("Medium", int(mix["Medium"])); m3.metric("Low", int(mix["Low"]))
            st.dataframe(out.head(500), use_container_width=True)
            st.download_button("Download scored CSV", out.to_csv(index=False), "scored_employees.csv", "text/csv")

with tab_mon:
    st.write("Latest report from `python -m attrition.monitoring.monitor`.")
    if st.button("Run monitoring now"):
        from attrition.monitoring.monitor import run_monitoring
        run_monitoring(window_hours=168, notify=False)
    if not config.LATEST_REPORT.exists():
        st.info("No monitoring report yet.")
    else:
        rep = json.loads(config.LATEST_REPORT.read_text(encoding="utf-8"))
        badge = {"ok": "OK", "warning": "WARNING", "critical": "CRITICAL"}.get(rep["overall_status"], "OK")
        a, b, c, d = st.columns(4)
        a.metric("Status", badge); b.metric("Predictions", rep["n_predictions"])
        c.metric("Labelled outcomes", rep["n_labelled"]); d.metric("Retraining", "recommended" if rep["recommend_retraining"] else "not needed")
        st.dataframe(pd.DataFrame([{"status": x["status"].upper(), "check": x["check"], "detail": x["message"]}
                                   for x in rep["checks"]]), use_container_width=True, hide_index=True)
        if rep.get("daily"):
            daily = pd.DataFrame(rep["daily"])
            fig = go.Figure()
            fig.add_scatter(x=daily["day"], y=daily["mean_probability"], name="mean predicted risk")
            fig.add_scatter(x=daily["day"], y=daily["high_risk_share"], name="share scored High")
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), title="Daily predicted risk")
            st.plotly_chart(fig, use_container_width=True)
    alerts = read_jsonl(config.ALERT_LOG)
    if alerts:
        st.markdown("**Recent alerts**")
        st.dataframe(pd.DataFrame(alerts[-20:][::-1])[["timestamp", "severity", "check", "message"]],
                     use_container_width=True, hide_index=True)

with tab_model:
    t = card["metrics"]["test"]
    cols = st.columns(4)
    cols[0].metric("ROC-AUC (test)", f"{t['roc_auc']:.3f}")
    cols[1].metric("Lift in top 20%", f"{t['lift_top20']:.2f}x")
    cols[2].metric("Precision in top 20%", f"{t['precision_top20']:.1%}")
    cols[3].metric("Brier score", f"{t['brier']:.3f}")
    st.markdown("**Risk bands**: High = top 20% of scores (probability >= "
                f"{thresholds['high']:.1%}), Medium = next 30% (>= {thresholds['medium']:.1%}), Low = the rest.")
    st.markdown("**Limitations**")
    for item in card["limitations"]:
        st.markdown(f"- {item}")
    with st.expander("Full model card"):
        st.json(card)
