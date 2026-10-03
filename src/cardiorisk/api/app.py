"""FastAPI service exposing the heart-disease risk model.

Endpoints
    GET  /             service info
    GET  /health       liveness probe
    GET  /ready        readiness probe (model loaded)
    GET  /model-info   metadata of the loaded model
    POST /predict      score one patient  -> prediction + confidence
    POST /predict/batch score up to 500 patients
    GET  /metrics      Prometheus exposition format
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from cardiorisk import __version__
from cardiorisk.api.monitoring import (
    LATENCY,
    MODEL_INFO,
    PREDICTIONS,
    PROBABILITY,
    REQUESTS,
    VALIDATION_ERRORS,
    Timer,
    configure_logging,
    log_event,
)
from cardiorisk.api.schemas import BatchIn, BatchOut, Patient, PredictionOut
from cardiorisk.predict import RiskModel

logger = configure_logging()
STATE: dict[str, RiskModel] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model = RiskModel()
        STATE["model"] = model
        MODEL_INFO.labels(name=model.name, version=model.version).set(1)
        log_event(logger, "model_loaded", model_name=model.name, model_version=model.version)
    except FileNotFoundError as exc:
        log_event(logger, "model_missing", error=str(exc))
    yield
    STATE.clear()


app = FastAPI(
    title="CardioRisk API",
    description="Predicts heart-disease risk from 13 clinical measurements (UCI Cleveland).",
    version=__version__,
    lifespan=lifespan,
)


@app.middleware("http")
async def observe(request: Request, call_next):
    request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    request.state.request_id = request_id
    with Timer() as t:
        try:
            response = await call_next(request)
            status = response.status_code
        except Exception:
            status = 500
            logger.exception("unhandled_error", extra={"extra_fields": {"request_id": request_id}})
            response = JSONResponse({"detail": "internal error"}, status_code=500)
    route = request.scope.get("route")
    route_path = getattr(route, "path", "unmatched")
    if route_path != "/metrics":
        REQUESTS.labels(request.method, route_path, str(status)).inc()
        LATENCY.labels(route_path).observe(t.elapsed)
        log_event(
            logger, "request", request_id=request_id, method=request.method, route=route_path,
            status=status, latency_ms=round(t.elapsed * 1000, 2),
            client=request.client.host if request.client else None,
        )
    response.headers["x-request-id"] = request_id
    return response


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    for err in exc.errors():
        field = str(err["loc"][-1]) if err.get("loc") else "body"
        VALIDATION_ERRORS.labels(field).inc()
    return JSONResponse(status_code=422, content={"detail": exc.errors()}, headers={})


def _model() -> RiskModel:
    model = STATE.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return model


def _score(patients: list[Patient], request_id: str) -> list[PredictionOut]:
    model = _model()
    preds = model.predict([p.model_dump() for p in patients])
    out = []
    for i, p in enumerate(preds):
        PREDICTIONS.labels(p.label, p.risk_band).inc()
        PROBABILITY.observe(p.probability_disease)
        rid = request_id if len(preds) == 1 else f"{request_id}-{i}"
        log_event(logger, "prediction", request_id=rid, label=p.label,
                  probability_disease=p.probability_disease, risk_band=p.risk_band,
                  model_version=model.version)
        out.append(PredictionOut(**p.__dict__, model_name=model.name,
                                 model_version=model.version, request_id=rid))
    return out


@app.get("/")
def root():
    return {"service": "cardiorisk-api", "version": __version__, "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    _model()
    return {"status": "ready"}


@app.get("/model-info")
def model_info():
    return _model().metadata


@app.post("/predict", response_model=PredictionOut)
def predict(patient: Patient, request: Request):
    return _score([patient], request.state.request_id)[0]


@app.post("/predict/batch", response_model=BatchOut)
def predict_batch(batch: BatchIn, request: Request):
    preds = _score(batch.patients, request.state.request_id)
    return BatchOut(count=len(preds), predictions=preds)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
