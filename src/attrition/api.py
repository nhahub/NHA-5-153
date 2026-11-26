"""FastAPI service for real-time attrition-risk predictions.

    uvicorn attrition.api:app --host 0.0.0.0 --port 8000

Endpoints
    GET  /health            liveness + whether a model is loaded
    GET  /model/info        model card (version, metrics, thresholds, limitations)
    POST /predict           score one employee (with top risk drivers)
    POST /predict/batch     score up to 5,000 employees
    POST /outcomes          report what actually happened (ground truth) -> enables performance monitoring
    GET  /metrics           Prometheus text exposition (traffic, latency, risk mix, mean risk)
    GET  /monitoring/report latest drift / performance report produced by ``attrition.monitoring``

Every scored employee is appended to a JSONL prediction log (features, probability, model version, request id).
The monitoring job reads that log; /outcomes feedback is joined on ``request_id``.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field

from . import __version__, config
from .config import API_TO_MODEL, FEATURE_LIMITS
from .predictor import ModelNotFoundError, ModelService

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)
MAX_BATCH = 5000


# ---------------------------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------------------------
class EmployeeFeatures(BaseModel):
    """One employee. Unknown fields are rejected so clients notice that e.g. gender/age are not used."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": {"employee_id": "E-1042", "tenure_years": 1.5, "performance_rating": 4,
                                       "monthly_salary": 5200.0}},
    )
    employee_id: str | int | None = Field(None, description="Your own identifier (echoed back, logged).")
    tenure_years: float = Field(..., ge=FEATURE_LIMITS["TenureYears"][0], le=FEATURE_LIMITS["TenureYears"][1])
    performance_rating: int = Field(..., ge=FEATURE_LIMITS["PerformanceRating"][0],
                                    le=FEATURE_LIMITS["PerformanceRating"][1], description="1 (lowest) - 5 (highest)")
    monthly_salary: float = Field(..., gt=FEATURE_LIMITS["MonthlySalary"][0], le=FEATURE_LIMITS["MonthlySalary"][1])


class Driver(BaseModel):
    feature: str
    label: str
    value: float
    typical_value: float
    impact: float = Field(..., description="Change in probability vs. a typical employee on this feature.")
    direction: str


class Prediction(BaseModel):
    request_id: str
    employee_id: str | int | None = None
    attrition_probability: float
    risk_level: Literal["Low", "Medium", "High"]
    model_version: str
    scored_at: datetime
    top_drivers: list[Driver] | None = None


class BatchRequest(BaseModel):
    employees: list[EmployeeFeatures] = Field(..., min_length=1, max_length=MAX_BATCH)


class BatchResponse(BaseModel):
    model_version: str
    count: int
    predictions: list[Prediction]


class Outcome(BaseModel):
    request_id: str
    left: bool = Field(..., description="True if the employee actually left within the observation window.")
    employee_id: str | int | None = None


# ---------------------------------------------------------------------------------------------
# Infrastructure: JSONL logging + in-process metrics
# ---------------------------------------------------------------------------------------------
class JsonlLogger:
    def __init__(self):
        self._lock = threading.Lock()

    def write(self, path, rows: list[dict]) -> None:
        if os.environ.get("DISABLE_PREDICTION_LOGGING", "").lower() in {"1", "true", "yes"}:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = "".join(json.dumps(r, default=str) + "\n" for r in rows)
        with self._lock, open(path, "a", encoding="utf-8") as fh:
            fh.write(payload)


class Metrics:
    """Minimal Prometheus-compatible counters (no extra dependency)."""

    def __init__(self):
        self._lock = threading.Lock()
        self.requests: dict[tuple[str, int], int] = {}
        self.latency_buckets = {b: 0 for b in LATENCY_BUCKETS}
        self.latency_count = 0
        self.latency_sum = 0.0
        self.predictions_by_risk = {"Low": 0, "Medium": 0, "High": 0}
        self.prob_sum = 0.0
        self.prob_count = 0
        self.outcomes = 0
        self.started = time.time()

    def observe_request(self, path: str, status: int, seconds: float) -> None:
        with self._lock:
            self.requests[(path, status)] = self.requests.get((path, status), 0) + 1
            self.latency_count += 1
            self.latency_sum += seconds
            for b in LATENCY_BUCKETS:
                if seconds <= b:
                    self.latency_buckets[b] += 1

    def observe_prediction(self, prob: float, risk: str) -> None:
        with self._lock:
            self.predictions_by_risk[risk] += 1
            self.prob_sum += prob
            self.prob_count += 1

    def render(self, model_version: str) -> str:
        with self._lock:
            lines = [
                "# HELP attrition_http_requests_total HTTP requests by path and status.",
                "# TYPE attrition_http_requests_total counter",
            ]
            for (path, status), n in sorted(self.requests.items()):
                lines.append(f'attrition_http_requests_total{{path="{path}",status="{status}"}} {n}')
            lines += ["# HELP attrition_request_latency_seconds Request latency.",
                      "# TYPE attrition_request_latency_seconds histogram"]
            for b in LATENCY_BUCKETS:
                lines.append(f'attrition_request_latency_seconds_bucket{{le="{b}"}} {self.latency_buckets[b]}')
            lines += [f'attrition_request_latency_seconds_bucket{{le="+Inf"}} {self.latency_count}',
                      f"attrition_request_latency_seconds_sum {self.latency_sum:.6f}",
                      f"attrition_request_latency_seconds_count {self.latency_count}",
                      "# HELP attrition_predictions_total Predictions by risk level.",
                      "# TYPE attrition_predictions_total counter"]
            for lvl, n in self.predictions_by_risk.items():
                lines.append(f'attrition_predictions_total{{risk_level="{lvl}"}} {n}')
            mean = self.prob_sum / self.prob_count if self.prob_count else 0.0
            lines += ["# HELP attrition_mean_predicted_probability Mean predicted attrition probability since start.",
                      "# TYPE attrition_mean_predicted_probability gauge",
                      f"attrition_mean_predicted_probability {mean:.6f}",
                      "# HELP attrition_outcomes_total Ground-truth outcomes received.",
                      "# TYPE attrition_outcomes_total counter",
                      f"attrition_outcomes_total {self.outcomes}",
                      "# HELP attrition_model_info Loaded model version.",
                      "# TYPE attrition_model_info gauge",
                      f'attrition_model_info{{version="{model_version}"}} 1',
                      "# HELP attrition_uptime_seconds Process uptime.",
                      "# TYPE attrition_uptime_seconds gauge",
                      f"attrition_uptime_seconds {time.time() - self.started:.0f}"]
        return "\n".join(lines) + "\n"


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Enforced only when the API_KEY environment variable is set (recommended for any shared deployment)."""
    expected = os.environ.get("API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")


# ---------------------------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------------------------
def create_app(service: ModelService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.service is None:
            try:
                app.state.service = ModelService()
            except ModelNotFoundError as exc:  # start anyway so /health can report the problem
                app.state.load_error = str(exc)
        yield

    app = FastAPI(
        title="Employee Attrition Prediction API",
        version=__version__,
        description="Real-time attrition-risk scoring. The model ranks employees by relative risk "
                    "(test ROC-AUC ~0.58): use it to prioritise conversations, not to make individual decisions.",
        lifespan=lifespan,
    )
    app.state.service = service
    app.state.load_error = None
    app.state.metrics = Metrics()
    app.state.logger = JsonlLogger()

    def get_service(request: Request) -> ModelService:
        svc = request.app.state.service
        if svc is None:
            raise HTTPException(status_code=503, detail=request.app.state.load_error or "Model not loaded.")
        return svc

    @app.middleware("http")
    async def track(request: Request, call_next):
        t0 = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            route = request.scope.get("route")
            path = getattr(route, "path", request.url.path)
            request.app.state.metrics.observe_request(path, status, time.perf_counter() - t0)

    # ------------------------------------------------------------------ helpers
    def _score(svc: ModelService, items: list[EmployeeFeatures], explain: bool, request: Request) -> list[Prediction]:
        now = datetime.now(timezone.utc)
        records = [{k: getattr(it, k) for k in API_TO_MODEL} for it in items]
        X = svc.to_frame(records)
        scored = svc.predict_frame(X)
        out, log_rows = [], []
        for i, it in enumerate(items):
            prob = round(float(scored.iloc[i]["attrition_probability"]), 4)
            risk = str(scored.iloc[i]["risk_level"])
            rid = uuid.uuid4().hex
            drivers = svc.explain(X.iloc[i]) if explain else None
            out.append(Prediction(request_id=rid, employee_id=it.employee_id, attrition_probability=prob,
                                  risk_level=risk, model_version=svc.version, scored_at=now, top_drivers=drivers))
            log_rows.append({"timestamp": now.isoformat(), "request_id": rid, "employee_id": it.employee_id,
                             "model_version": svc.version, **X.iloc[i].to_dict(),
                             "probability": prob, "risk_level": risk})
            request.app.state.metrics.observe_prediction(prob, risk)
        request.app.state.logger.write(config.PREDICTION_LOG, log_rows)
        return out

    # ------------------------------------------------------------------ routes
    @app.get("/health", tags=["ops"])
    def health(request: Request):
        svc = request.app.state.service
        body = {"status": "ok" if svc else "degraded", "model_loaded": svc is not None,
                "model_version": svc.version if svc else None, "api_version": __version__}
        if not svc:
            raise HTTPException(status_code=503, detail=body | {"error": request.app.state.load_error})
        return body

    @app.get("/model/info", tags=["ops"])
    def model_info(svc: ModelService = Depends(get_service)):
        return svc.card

    @app.post("/predict", response_model=Prediction, tags=["prediction"], dependencies=[Depends(require_api_key)])
    def predict(payload: EmployeeFeatures, request: Request, explain: bool = Query(True),
                svc: ModelService = Depends(get_service)):
        return _score(svc, [payload], explain, request)[0]

    @app.post("/predict/batch", response_model=BatchResponse, tags=["prediction"],
              dependencies=[Depends(require_api_key)])
    def predict_batch(payload: BatchRequest, request: Request, explain: bool = Query(False),
                      svc: ModelService = Depends(get_service)):
        preds = _score(svc, payload.employees, explain, request)
        return BatchResponse(model_version=svc.version, count=len(preds), predictions=preds)

    @app.post("/outcomes", status_code=201, tags=["monitoring"], dependencies=[Depends(require_api_key)])
    def record_outcome(payload: Outcome, request: Request):
        request.app.state.logger.write(config.OUTCOME_LOG, [{
            "timestamp": datetime.now(timezone.utc).isoformat(), "request_id": payload.request_id,
            "employee_id": payload.employee_id, "left": payload.left}])
        request.app.state.metrics.outcomes += 1
        return {"status": "recorded", "request_id": payload.request_id}

    @app.get("/metrics", response_class=PlainTextResponse, tags=["ops"])
    def metrics(request: Request):
        svc = request.app.state.service
        return request.app.state.metrics.render(svc.version if svc else "none")

    @app.get("/monitoring/report", tags=["monitoring"])
    def monitoring_report():
        if not config.LATEST_REPORT.exists():
            raise HTTPException(status_code=404, detail="No monitoring report yet. Run `python -m attrition.monitoring.monitor`.")
        return json.loads(config.LATEST_REPORT.read_text(encoding="utf-8"))

    return app


app = create_app()
