import os
import json
import time
import uuid
import logging
import traceback
from typing import List, Optional

import joblib
import pandas as pd
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from pydantic import BaseModel, Field, confloat, ValidationError

# ------------------------------
# App initialization & config
# ------------------------------
app = Flask(__name__)
CORS(app)  # allow cross-origin (HF frontends)
app.config["JSON_SORT_KEYS"] = False
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # 1 MB request cap (adjust as needed)

# Logging setup (simple structured logs)
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# ------------------------------
# Model loading (readiness)
# ------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "best_model_superkart.pkl")
MODEL_INFO_PATH = os.environ.get("MODEL_INFO_PATH", "best_model_info.json")

model = None
model_info = {"model_name": "unknown", "version": None}

def load_artifacts():
    global model, model_info
    logger.info("Loading model pipeline from %s", MODEL_PATH)
    model = joblib.load(MODEL_PATH)

    if os.path.exists(MODEL_INFO_PATH):
        with open(MODEL_INFO_PATH, "r") as f:
            model_info = json.load(f)
            # derive a simple version string if not present
            model_info.setdefault("version", str(int(model_info.get("created_at_unix", time.time()))))
    else:
        model_info = {"model_name": "RandomForest (Tuned)", "version": "v1"}

try:
    load_artifacts()
    READY = True
except Exception as e:
    logger.exception("Failed to load model artifacts: %s", e)
    READY = False

# ------------------------------
# Request/Response helpers
# ------------------------------
def new_request_id() -> str:
    return uuid.uuid4().hex

@app.before_request
def assign_request_id():
    g.request_id = request.headers.get("X-Request-ID", new_request_id())
    g.start_ts = time.time()

@app.after_request
def add_response_headers(resp):
    # Include request id in all responses
    resp.headers["X-Request-ID"] = g.get("request_id", new_request_id())
    return resp

def ok(payload: dict, status: int = 200):
    payload["request_id"] = g.request_id
    payload["model_version"] = model_info.get("version")
    return jsonify(payload), status

def err(status: int, code: str, message: str, details: Optional[dict] = None):
    body = {
        "request_id": g.request_id,
        "code": code,
        "message": message
    }
    if details is not None:
        body["details"] = details
    return jsonify(body), status

# ------------------------------
# Pydantic input schema
# (types are permissive; pipeline imputers handle None)
# ------------------------------
class Record(BaseModel):
    Product_Id: Optional[str] = Field(None, description="SKU id")
    Product_Weight: Optional[confloat(ge=0)] = None
    Product_Sugar_Content: Optional[str] = None
    Product_Allocated_Area: Optional[confloat(ge=0, le=1)] = None
    Product_Type: Optional[str] = None
    Product_MRP: Optional[confloat(ge=0)] = None
    Store_Id: Optional[str] = None
    Store_Establishment_Year: Optional[int] = None
    Store_Size: Optional[str] = None
    Store_Location_City_Type: Optional[str] = None
    Store_Type: Optional[str] = None

class BatchRequest(BaseModel):
    # Allow list of records; cap batch size to prevent abuse
    items: List[Record] = Field(..., min_items=1, max_items=1000)

# ------------------------------
# Health & metadata
# ------------------------------
@app.route("/health/live", methods=["GET"])
def health_live():
    return ok({"status": "live"})

@app.route("/health/ready", methods=["GET"])
def health_ready():
    if READY and model is not None:
        return ok({"status": "ready"})
    return err(503, "NOT_READY", "Model artifacts are not loaded.")

@app.route("/v1/metadata", methods=["GET"])
def metadata():
    info = {
        "model_name": model_info.get("model_name"),
        "version": model_info.get("version"),
        "train_rows": model_info.get("train_rows"),
        "train_cols": model_info.get("train_cols"),
        "feature_groups": model_info.get("feature_groups"),
        "metrics_test": model_info.get("metrics_test")
    }
    return ok({"metadata": info})

# ------------------------------
# Prediction endpoint (versioned)
# ------------------------------
@app.route("/v1/predict", methods=["POST"])
def predict_v1():
    if not READY or model is None:
        return err(503, "NOT_READY", "Model artifacts are not loaded.")

    # Content-Type checks
    ctype = request.headers.get("Content-Type", "")
    if "application/json" not in ctype:
        return err(415, "UNSUPPORTED_MEDIA_TYPE", "Content-Type must be application/json.")

    # Parse JSON
    try:
        payload = request.get_json(force=True, silent=False)
    except Exception as e:
        return err(400, "BAD_JSON", "Malformed JSON.", {"exception": str(e)})

    # Accept {items: [...]} or a single object or a raw list
    try:
        if isinstance(payload, dict) and "items" in payload:
            batch = BatchRequest(**payload)
            records = [r.model_dump() for r in batch.items]
        elif isinstance(payload, list):
            batch = BatchRequest(items=[Record(**item) for item in payload])
            records = [r.model_dump() for r in batch.items]
        elif isinstance(payload, dict):
            one = Record(**payload)
            records = [one.model_dump()]
        else:
            return err(422, "INVALID_PAYLOAD", "Expected a JSON object or list of objects.")
    except ValidationError as ve:
        return err(422, "VALIDATION_ERROR", "Input validation failed.", {"errors": json.loads(ve.json())})

    # DataFrame inference
    try:
        df = pd.DataFrame.from_records(records)
        preds = model.predict(df)
        preds = [float(x) for x in preds]
    except Exception as e:
        logger.exception("Prediction failed: %s", e)
        return err(500, "PREDICTION_ERROR", "Model prediction failed.", {"exception": str(e)})

    took_ms = int((time.time() - g.start_ts) * 1000)
    logger.info("pred_ok request_id=%s items=%d took_ms=%d", g.request_id, len(records), took_ms)
    return ok({"count": len(preds), "predictions": preds})

# ------------------------------
# Error handlers (consistent JSON)
# ------------------------------
@app.errorhandler(404)
def not_found(_e):
    return err(404, "NOT_FOUND", "The requested resource was not found.")

@app.errorhandler(405)
def method_not_allowed(_e):
    return err(405, "METHOD_NOT_ALLOWED", "HTTP method not allowed for this endpoint.")

@app.errorhandler(413)
def payload_too_large(_e):
    return err(413, "PAYLOAD_TOO_LARGE", "Request exceeds maximum allowed size.")

@app.errorhandler(500)
def internal_error(_e):
    logger.exception("Unhandled 500: %s", _e)
    return err(500, "INTERNAL_SERVER_ERROR", "An unexpected error occurred.", {"trace": traceback.format_exc()})

# Root: simple router hint
@app.route("/", methods=["GET"])
def root():
    return ok({
        "service": "SuperKart Sales Forecast API",
        "version": model_info.get("version"),
        "endpoints": ["/health/live", "/health/ready", "/v1/metadata", "/v1/predict"]
    })

if __name__ == "__main__":
    # Local debug (Spaces will run via Gunicorn)
    app.run(host="0.0.0.0", port=7860, debug=False)

