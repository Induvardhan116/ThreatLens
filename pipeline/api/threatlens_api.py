from pathlib import Path
import json
import math
import time
from collections import defaultdict, deque
from threading import Lock
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware


# ========================================================================
# THREATLENS - STEP 33
# HARDENED LOCAL INFERENCE API
# ========================================================================

BASE_DIR = Path(r"D:\ThreatLens")

MODEL_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "model"
    / "threatlens_conservative_model.joblib"
)

LOG_DIR = BASE_DIR / "data" / "processed" / "api_logs"
LOG_PATH = LOG_DIR / "threatlens_api.log"

# ------------------------------------------------------------------------
# SECURITY LIMITS
# ------------------------------------------------------------------------

# Maximum HTTP request body size.
MAX_REQUEST_BYTES = 1_000_000

# Maximum number of records accepted by /predict/batch.
MAX_BATCH_SIZE = 100

# Maximum requests allowed from one client within the rate window.
RATE_LIMIT_REQUESTS = 60
RATE_LIMIT_WINDOW_SECONDS = 60

# ------------------------------------------------------------------------
# STARTUP
# ------------------------------------------------------------------------

print("=" * 72)
print("THREATLENS - STEP 33")
print("HARDENED LOCAL INFERENCE API")
print("=" * 72)


# ------------------------------------------------------------------------
# LOAD PACKAGED MODEL
# ------------------------------------------------------------------------

if not MODEL_PATH.exists():
    raise RuntimeError(
        "ThreatLens model artifact is unavailable."
    )

try:
    artifact = joblib.load(MODEL_PATH)
except Exception as exc:
    raise RuntimeError(
        "ThreatLens model artifact could not be loaded."
    ) from exc


if not isinstance(artifact, dict):
    raise RuntimeError(
        "Packaged ThreatLens model must be a dictionary artifact."
    )


required_artifact_keys = [
    "model",
    "features",
    "fill_values",
    "model_name",
    "version",
]

missing_artifact_keys = [
    key
    for key in required_artifact_keys
    if key not in artifact
]

if missing_artifact_keys:
    raise RuntimeError(
        "Packaged model is missing required configuration."
    )


MODEL = artifact["model"]
FEATURES = list(artifact["features"])
FILL_VALUES = artifact["fill_values"]
MODEL_NAME = str(artifact["model_name"])
MODEL_VERSION = str(artifact["version"])


if not hasattr(MODEL, "predict_proba"):
    raise RuntimeError(
        "Packaged model does not provide predict_proba()."
    )


if len(FEATURES) != len(FILL_VALUES):
    raise RuntimeError(
        "Feature configuration is inconsistent."
    )


if not FEATURES:
    raise RuntimeError(
        "Packaged model contains no features."
    )


print()
print(f"Model: {MODEL_NAME}")
print(f"Version: {MODEL_VERSION}")
print(f"Features: {len(FEATURES)}")
print("Model status: READY")


# ------------------------------------------------------------------------
# FASTAPI APPLICATION
# ------------------------------------------------------------------------

app = FastAPI(
    title="ThreatLens API",
    description="Hardened local inference API for the ThreatLens Conservative model.",
    version=MODEL_VERSION,
)


# ------------------------------------------------------------------------
# RATE LIMIT STATE
# ------------------------------------------------------------------------

RATE_LIMIT_LOCK = Lock()

REQUEST_HISTORY = defaultdict(deque)


def get_client_identifier(request: Request) -> str:
    """
    Identify the client without trusting forwarded proxy headers.
    For a local API, the direct socket address is sufficient.
    """

    if request.client is None:
        return "unknown"

    host = request.client.host

    if not host:
        return "unknown"

    return str(host)


def check_rate_limit(request: Request) -> None:
    """
    Simple in-memory fixed-window/sliding-window rate limiter.

    This is intentionally lightweight because ThreatLens is a local
    inference service rather than a distributed production API.
    """

    client_id = get_client_identifier(request)
    now = time.monotonic()

    with RATE_LIMIT_LOCK:

        history = REQUEST_HISTORY[client_id]

        cutoff = now - RATE_LIMIT_WINDOW_SECONDS

        while history and history[0] <= cutoff:
            history.popleft()

        if len(history) >= RATE_LIMIT_REQUESTS:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Rate limit exceeded.",
                    "retry_after_seconds": RATE_LIMIT_WINDOW_SECONDS,
                },
            )

        history.append(now)


# ------------------------------------------------------------------------
# SECURITY MIDDLEWARE
# ------------------------------------------------------------------------

class SecurityMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request: Request, call_next):

        # ------------------------------------------------------------
        # Request-size protection
        # ------------------------------------------------------------

        content_length = request.headers.get("content-length")

        if content_length is not None:

            try:
                declared_length = int(content_length)
            except ValueError:
                return _json_error(
                    400,
                    "Invalid Content-Length header.",
                )

            if declared_length < 0:
                return _json_error(
                    400,
                    "Invalid Content-Length header.",
                )

            if declared_length > MAX_REQUEST_BYTES:
                return _json_error(
                    413,
                    "Request body exceeds the configured size limit.",
                )

        # ------------------------------------------------------------
        # Rate limiting
        # ------------------------------------------------------------

        try:
            check_rate_limit(request)
        except HTTPException as exc:

            return _json_error(
                exc.status_code,
                exc.detail,
            )

        # ------------------------------------------------------------
        # Process request
        # ------------------------------------------------------------

        try:
            response = await call_next(request)

        except HTTPException:
            raise

        except Exception:
            # Never expose internal exception details to clients.
            return _json_error(
                500,
                "Internal server error.",
            )

        # ------------------------------------------------------------
        # Security headers
        # ------------------------------------------------------------

        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"

        # This is an API, not a browser application.
        response.headers["Content-Security-Policy"] = (
            "default-src 'none'; frame-ancestors 'none'"
        )

        return response


def _json_error(status_code: int, detail: Any):
    """
    Create a JSON response without leaking internal application details.
    """

    from starlette.responses import JSONResponse

    return JSONResponse(
        status_code=status_code,
        content={
            "detail": detail,
        },
    )


app.add_middleware(SecurityMiddleware)


# ------------------------------------------------------------------------
# AUDIT LOGGING
# ------------------------------------------------------------------------

def write_audit_event(
    event: str,
    request: Request,
    status_code: int,
    extra: Dict[str, Any] | None = None,
) -> None:

    try:

        LOG_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        client_id = get_client_identifier(request)

        event_record = {
            "timestamp": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(),
            ),
            "event": event,
            "method": request.method,
            "path": request.url.path,
            "client": client_id,
            "status_code": status_code,
        }

        if extra:
            event_record.update(extra)

        with LOG_PATH.open(
            "a",
            encoding="utf-8",
        ) as log_file:

            log_file.write(
                json.dumps(
                    event_record,
                    separators=(",", ":"),
                )
                + "\n"
            )

    except Exception:
        # Logging failure must never break inference.
        pass


# ------------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------------

ALLOWED_RISK_CATEGORIES = [
    "CRITICAL",
    "HIGH",
    "MEDIUM",
    "LOW",
    "MINIMAL",
]


def risk_category(score: float) -> str:
    """
    Authoritative ThreatLens production risk contract.

    CRITICAL >= 0.50
    HIGH     >= 0.20
    MEDIUM   >= 0.05
    LOW      >= 0.01
    MINIMAL  <  0.01
    """

    score = float(score)

    if score >= 0.50:
        return "CRITICAL"

    if score >= 0.20:
        return "HIGH"

    if score >= 0.05:
        return "MEDIUM"

    if score >= 0.01:
        return "LOW"

    return "MINIMAL"


def is_numeric(value: Any) -> bool:
    """
    Return True only for values that can safely be treated as
    finite numeric values.

    None is allowed because the inference pipeline fills missing values.
    """

    if value is None:
        return True

    if isinstance(value, bool):
        return False

    if isinstance(
        value,
        (
            int,
            float,
            np.integer,
            np.floating,
        ),
    ):

        try:
            return math.isfinite(float(value))
        except Exception:
            return False

    return False


def validate_feature_record(
    features: Dict[str, Any],
    record_index: int = 0,
) -> None:

    if not isinstance(features, dict):

        raise HTTPException(
            status_code=422,
            detail={
                "error": "Features must be an object/dictionary.",
                "record_index": record_index,
            },
        )

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in features
    ]

    if missing_features:

        raise HTTPException(
            status_code=422,
            detail={
                "error": "Missing required model features.",
                "record_index": record_index,
                "missing_features": missing_features,
                "required_feature_count": len(FEATURES),
                "supplied_feature_count": len(features),
            },
        )

    for feature in FEATURES:

        value = features.get(feature)

        if not is_numeric(value):

            raise HTTPException(
                status_code=422,
                detail={
                    "error": "Feature value must be numeric or null.",
                    "record_index": record_index,
                    "feature": feature,
                },
            )


def prepare_features(
    features: Dict[str, Any],
) -> pd.DataFrame:

    row = {}

    for feature in FEATURES:

        value = features.get(feature)

        if value is None:
            value = FILL_VALUES.get(
                feature,
                0,
            )

        row[feature] = value

    frame = pd.DataFrame(
        [row],
        columns=FEATURES,
    )

    for feature in FEATURES:

        frame[feature] = pd.to_numeric(
            frame[feature],
            errors="coerce",
        )

        fill_value = FILL_VALUES.get(
            feature,
            0,
        )

        frame[feature] = frame[feature].fillna(
            fill_value
        )

    # Final finite-value protection.
    for feature in FEATURES:

        values = frame[feature].to_numpy(
            dtype=float
        )

        if not np.isfinite(values).all():

            raise HTTPException(
                status_code=422,
                detail={
                    "error": "Feature preparation produced a non-finite value.",
                    "feature": feature,
                },
            )

    return frame


def predict_one(
    features: Dict[str, Any],
    record_index: int = 0,
) -> Dict[str, Any]:

    validate_feature_record(
        features,
        record_index=record_index,
    )

    matrix = prepare_features(features)

    try:
        probabilities = MODEL.predict_proba(matrix)
    except Exception as exc:

        write_audit_event(
            event="prediction_error",
            request=_CURRENT_REQUEST,
            status_code=500,
            extra={
                "record_index": record_index,
            },
        )

        raise HTTPException(
            status_code=500,
            detail={
                "error": "Model inference failed."
            },
        ) from exc

    if (
        probabilities.ndim != 2
        or probabilities.shape[1] < 2
    ):

        raise HTTPException(
            status_code=500,
            detail={
                "error": "Model returned an invalid probability matrix."
            },
        )

    score = float(
        probabilities[0, 1]
    )

    if not math.isfinite(score):

        raise HTTPException(
            status_code=500,
            detail={
                "error": "Model returned a non-finite score."
            },
        )

    score = max(
        0.0,
        min(1.0, score),
    )

    category = risk_category(score)

    return {
        "model": MODEL_NAME,
        "version": MODEL_VERSION,
        "threatlens_score": round(
            score,
            6,
        ),
        "risk_category": category,
    }


# ------------------------------------------------------------------------
# REQUEST CONTEXT
# ------------------------------------------------------------------------

_CURRENT_REQUEST = None


# ------------------------------------------------------------------------
# HEALTH
# ------------------------------------------------------------------------

@app.get("/health")
def health(
    request: Request,
) -> Dict[str, Any]:

    write_audit_event(
        event="health_check",
        request=request,
        status_code=200,
    )

    return {
        "status": "healthy",
        "service": "ThreatLens API",
        "model_loaded": True,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "feature_count": len(FEATURES),
    }


# ------------------------------------------------------------------------
# MODEL METADATA
# ------------------------------------------------------------------------

@app.get("/model")
def model_metadata(
    request: Request,
) -> Dict[str, Any]:

    write_audit_event(
        event="model_metadata",
        request=request,
        status_code=200,
    )

    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "estimator_type": type(MODEL).__name__,
        "feature_count": len(FEATURES),
        "features": FEATURES,
    }


# ------------------------------------------------------------------------
# SINGLE PREDICTION
# ------------------------------------------------------------------------

@app.post("/predict")
def predict(
    payload: Dict[str, Any],
    request: Request,
) -> Dict[str, Any]:

    if not isinstance(payload, dict):

        write_audit_event(
            event="invalid_request",
            request=request,
            status_code=422,
        )

        raise HTTPException(
            status_code=422,
            detail={
                "error": "Request body must be a JSON object."
            },
        )

    features = payload.get("features")

    if features is None:

        write_audit_event(
            event="invalid_request",
            request=request,
            status_code=422,
        )

        raise HTTPException(
            status_code=422,
            detail={
                "error": (
                    "Request must contain a 'features' object."
                )
            },
        )

    global _CURRENT_REQUEST
    _CURRENT_REQUEST = request

    try:

        result = predict_one(
            features,
            record_index=0,
        )

    except HTTPException as exc:

        write_audit_event(
            event="prediction_rejected",
            request=request,
            status_code=exc.status_code,
        )

        raise

    finally:
        _CURRENT_REQUEST = None

    write_audit_event(
        event="prediction",
        request=request,
        status_code=200,
        extra={
            "prediction_type": "single",
            "risk_category": result["risk_category"],
        },
    )

    return result


# ------------------------------------------------------------------------
# BATCH PREDICTION
# ------------------------------------------------------------------------

@app.post("/predict/batch")
def predict_batch(
    payload: Dict[str, Any],
    request: Request,
) -> Dict[str, Any]:

    if not isinstance(payload, dict):

        raise HTTPException(
            status_code=422,
            detail={
                "error": "Request body must be a JSON object."
            },
        )

    records = payload.get("records")

    if records is None:
        records = payload.get("features")

    if records is None:

        raise HTTPException(
            status_code=422,
            detail={
                "error": (
                    "Batch request must contain either "
                    "'records' or 'features'."
                )
            },
        )

    if not isinstance(records, list):

        raise HTTPException(
            status_code=422,
            detail={
                "error": "Batch records must be a list."
            },
        )

    if len(records) == 0:

        raise HTTPException(
            status_code=422,
            detail={
                "error": "Batch records cannot be empty."
            },
        )

    if len(records) > MAX_BATCH_SIZE:

        write_audit_event(
            event="batch_rejected",
            request=request,
            status_code=413,
            extra={
                "record_count": len(records),
            },
        )

        raise HTTPException(
            status_code=413,
            detail={
                "error": "Batch size exceeds the configured limit.",
                "max_batch_size": MAX_BATCH_SIZE,
                "supplied_batch_size": len(records),
            },
        )

    results = []

    global _CURRENT_REQUEST
    _CURRENT_REQUEST = request

    try:

        for index, item in enumerate(records):

            if not isinstance(item, dict):

                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": (
                            "Each batch record must be an object."
                        ),
                        "record_index": index,
                    },
                )

            if "features" in item:
                features = item["features"]
            else:
                features = item

            result = predict_one(
                features,
                record_index=index,
            )

            result["record_index"] = index

            results.append(result)

    except HTTPException as exc:

        write_audit_event(
            event="batch_rejected",
            request=request,
            status_code=exc.status_code,
        )

        raise

    finally:
        _CURRENT_REQUEST = None

    write_audit_event(
        event="batch_prediction",
        request=request,
        status_code=200,
        extra={
            "prediction_type": "batch",
            "record_count": len(results),
        },
    )

    return {
        "model": MODEL_NAME,
        "version": MODEL_VERSION,
        "count": len(results),
        "predictions": results,
    }


# ------------------------------------------------------------------------
# ROOT
# ------------------------------------------------------------------------

@app.get("/")
def root(
    request: Request,
) -> Dict[str, Any]:

    write_audit_event(
        event="root",
        request=request,
        status_code=200,
    )

    return {
        "service": "ThreatLens API",
        "model": MODEL_NAME,
        "version": MODEL_VERSION,
        "status": "ready",
        "endpoints": {
            "health": "GET /health",
            "model": "GET /model",
            "predict": "POST /predict",
            "predict_batch": "POST /predict/batch",
        },
        "security": {
            "request_limit_bytes": MAX_REQUEST_BYTES,
            "max_batch_size": MAX_BATCH_SIZE,
            "rate_limit_requests": RATE_LIMIT_REQUESTS,
            "rate_limit_window_seconds": RATE_LIMIT_WINDOW_SECONDS,
        },
    }


# ------------------------------------------------------------------------
# STARTUP INFORMATION
# ------------------------------------------------------------------------

print()
print("Endpoints:")
print("  GET  /health")
print("  GET  /model")
print("  POST /predict")
print("  POST /predict/batch")
print("  GET  /")
print()
print("Security:")
print(f"  Max request size: {MAX_REQUEST_BYTES} bytes")
print(f"  Max batch size: {MAX_BATCH_SIZE}")
print(
    f"  Rate limit: "
    f"{RATE_LIMIT_REQUESTS} requests/"
    f"{RATE_LIMIT_WINDOW_SECONDS}s"
)
print(f"  Audit log: {LOG_PATH}")
print()
