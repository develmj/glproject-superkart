import os
import json
import time
import uuid
import logging
import traceback
from typing import List, Optional

import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from pydantic import BaseModel, Field, confloat, ValidationError
from huggingface_hub import hf_hub_download  # for optional Hub download

# ------------------------------
# App initialization & config
# ------------------------------

# --- Writable cache for HF Hub (avoid PermissionError on '/.cache') ---
WRITABLE_CACHE_ROOT = os.environ.get("HF_HOME") or "/tmp/hf_cache"
os.makedirs(WRITABLE_CACHE_ROOT, exist_ok=True)
# Make all huggingface caches point to the same writable place
os.environ.setdefault("HF_HOME", WRITABLE_CACHE_ROOT)
os.environ.setdefault("HF_HUB_CACHE", os.path.join(WRITABLE_CACHE_ROOT, "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(WRITABLE_CACHE_ROOT, "transformers"))

app = Flask(__name__)
CORS(app)  # allow cross-origin (HF Streamlit frontend)
app.config["JSON_SORT_KEYS"] = False
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # 1 MB request cap

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# ------------------------------
# Model paths / env
# ------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "best_model_superkart.pkl")
MODEL_INFO_PATH = os.environ.get("MODEL_INFO_PATH", "best_model_info.json")

# Optional: Hub repo & filename if local file not present
MODEL_REPO_ID = os.environ.get("MODEL_REPO_ID")  # e.g. "your-username/superkart-model"
MODEL_FILENAME = os.environ.get("MODEL_FILENAME", "best_model_superkart.pkl")

model = None
model_info = {"model_name": "unknown", "version": None}
READY = False

# ------------------------------
# Helpers for responses
# ------------------------------
def new_request_id() -> str:
    return uuid.uuid4().hex

@app.before_request
def assign_request_id():
    # Normalize request ID (avoid duplicates joined with commas by proxies)
    raw = request.headers.get("X-Request-ID")
    if raw:
        rid = raw.split(",")[0].strip() or uuid.uuid4().hex
    else:
        rid = uuid.uuid4().hex
    g.request_id = rid
    g.start_ts = time.time()

@app.after_request
def add_response_headers(resp):
    resp.headers["X-Request-ID"] = g.get("request_id", new_request_id())
    return resp

def ok(payload: dict, status: int = 200):
    payload["request_id"] = g.request_id
    payload["model_version"] = model_info.get("version") or "v1"
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

def _safe_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Replicate the training-time feature engineering on incoming payload."""
    df = df.copy()

    # Ensure required raw columns exist (create if missing)
    for col in [
        "Product_Id", "Product_Weight", "Product_Allocated_Area", "Product_MRP",
        "Store_Establishment_Year"
    ]:
        if col not in df.columns:
            df[col] = np.nan

    # Coerce numeric types safely
    num_cols = ["Product_Weight", "Product_Allocated_Area", "Product_MRP", "Store_Establishment_Year"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 1) Store_Age: training used reference 2010 so that 2009 -> 1, 1999 -> 11, 1987 -> 23, etc.
    ref_year = 2010
    df["Store_Age"] = (ref_year - df["Store_Establishment_Year"]).clip(lower=0)

    # 2) Product_Family: first two letters of Product_Id uppercased (e.g., 'FD', 'NC', 'DR', ...)
    df["Product_Family"] = (
        df["Product_Id"].astype(str).str.strip().str[:2].str.upper().replace({"": np.nan})
    )

    # 3) Price_per_Weight: safe divide (avoid div-by-zero)
    denom = df["Product_Weight"].replace({0: np.nan})
    df["Price_per_Weight"] = df["Product_MRP"] / denom

    # 4) Shelf_Investment: Product_Allocated_Area * Product_MRP
    df["Shelf_Investment"] = df["Product_Allocated_Area"] * df["Product_MRP"]

    return df


# ------------------------------
# Pydantic input schema
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
    items: List[Record] = Field(..., min_items=1, max_items=1000)

# ------------------------------
# Model loading (resilient)
# ------------------------------

def load_artifacts():
    """
    Load model artifacts using a resilient strategy:
      1) If MODEL_PATH exists locally, load it.
      2) Else, download from HF Hub into a writable local dir (no symlinks).
      3) Ensure non-null model_info['version'].
    """
    global model, model_info, READY

    logger.info("load_artifacts: starting")
    logger.info("load_artifacts: MODEL_PATH=%s exists=%s", MODEL_PATH, os.path.exists(MODEL_PATH))
    logger.info("load_artifacts: MODEL_REPO_ID=%s MODEL_FILENAME=%s", MODEL_REPO_ID, MODEL_FILENAME)
    logger.info("load_artifacts: HF_HOME=%s HF_HUB_CACHE=%s", os.environ.get("HF_HOME"), os.environ.get("HF_HUB_CACHE"))

    # 1) Local file load
    if os.path.exists(MODEL_PATH):
        logger.info("load_artifacts: loading local model: %s", MODEL_PATH)
        model = joblib.load(MODEL_PATH)
    else:
        # 2) Hub fallback into a writable directory
        if not MODEL_REPO_ID:
            raise RuntimeError(
                "Model not found locally and MODEL_REPO_ID not set. "
                "Set MODEL_REPO_ID secret (and optionally MODEL_FILENAME) or include the .pkl in the repo."
            )
        local_dir = os.path.join(WRITABLE_CACHE_ROOT, "artifacts")
        os.makedirs(local_dir, exist_ok=True)
        logger.info("load_artifacts: downloading from Hub to local_dir=%s", local_dir)
        local_model_path = hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename=MODEL_FILENAME,
            repo_type="model",
            local_dir=local_dir,                 # place file here
            local_dir_use_symlinks=False         # avoid symlink perms issues
        )
        logger.info("load_artifacts: local_model_path=%s", local_model_path)
        model = joblib.load(local_model_path)

    # Try model info (local first, then Hub)
    try:
        if os.path.exists(MODEL_INFO_PATH):
            logger.info("load_artifacts: loading local model info: %s", MODEL_INFO_PATH)
            with open(MODEL_INFO_PATH, "r") as f:
                model_info_local = json.load(f)
            model_info.update(model_info_local)
        elif MODEL_REPO_ID:
            local_dir = os.path.join(WRITABLE_CACHE_ROOT, "artifacts")
            os.makedirs(local_dir, exist_ok=True)
            logger.info("load_artifacts: downloading model info from Hub to %s", local_dir)
            local_info_path = hf_hub_download(
                repo_id=MODEL_REPO_ID,
                filename="best_model_info.json",
                repo_type="model",
                local_dir=local_dir,
                local_dir_use_symlinks=False
            )
            with open(local_info_path, "r") as f:
                model_info_local = json.load(f)
            model_info.update(model_info_local)
    except Exception as e:
        logger.warning("load_artifacts: could not load model info: %s", e)

    # Ensure non-null version
    if not model_info.get("version"):
        ts = model_info.get("created_at_unix") or time.time()
        model_info["version"] = str(int(ts))

    READY = True
    logger.info("load_artifacts: completed OK; version=%s", model_info["version"])


# Try to load on import
try:
    load_artifacts()
except Exception as e:
    READY = False
    logger.exception("Failed to load model artifacts: %s", e)

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
        "version": model_info.get("version") or "v1",
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
    if not READY or model is not None:
        pass
    else:
        return err(503, "NOT_READY", "Model artifacts are not loaded.")

    ctype = request.headers.get("Content-Type", "")
    if "application/json" not in ctype:
        return err(415, "UNSUPPORTED_MEDIA_TYPE", "Content-Type must be application/json.")
    try:
        payload = request.get_json(force=True, silent=False)
    except Exception as e:
        return err(400, "BAD_JSON", "Malformed JSON.", {"exception": str(e)})

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

    # ---- engineer features exactly as in training ----
    try:
        df = pd.DataFrame.from_records(records)
        df = _safe_engineer_features(df)
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

# ------------------------------
# Root endpoint
# ------------------------------
@app.route("/", methods=["GET"])
def root():
    return ok({
        "service": "SuperKart Sales Forecast API",
        "version": model_info.get("version") or "v1",
        "endpoints": ["/health/live", "/health/ready", "/v1/metadata", "/v1/predict"]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
