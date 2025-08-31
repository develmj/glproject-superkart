import os
import time
import json
import math
import copy
import requests
import pandas as pd
import streamlit as st
from typing import List, Dict, Any, Tuple

# -----------------------------
# Page / theme / layout
# -----------------------------
st.set_page_config(page_title="SuperKart Sales Forecast", page_icon="🛒", layout="wide")
st.title("🛒 SuperKart Sales Forecast (Frontend)")

st.markdown(
    """
    This app sends product & store attributes to the **SuperKart Forecast API** and displays predicted revenue.
    - Use the **Form** for a single prediction.
    - Use the **Batch (Table/CSV)** tab for multiple records.
    """
)

# -----------------------------
# Backend URL handling
# -----------------------------
# Priority:
# 1. Streamlit Secret BACKEND_URL (set in HF Space → Settings → Repository secrets)
# 2. Environment variable BACKEND_URL
# 3. Fallback text input (user provided)
default_backend_url = os.environ.get("BACKEND_URL")
backend_url_secret = st.secrets.get("BACKEND_URL") if hasattr(st, "secrets") else None
BACKEND_URL = backend_url_secret or default_backend_url or ""

with st.sidebar:
    st.header("⚙️ Backend Configuration")
    BACKEND_URL = st.text_input(
        "Backend API Base URL",
        value=BACKEND_URL,
        placeholder="e.g., https://<user>-superkart-backend.hf.space",
        help="Point this to your Flask backend Space base URL"
    )
    st.caption("Tip: Prefer setting BACKEND_URL as a Space secret for convenience.")

    # Health checks
    colA, colB = st.columns(2)
    with colA:
        if st.button("Check Live"):
            try:
                r = requests.get(f"{BACKEND_URL}/health/live", timeout=15)
                st.write(r.status_code, r.json())
            except Exception as e:
                st.error(f"Live check failed: {e}")
    with colB:
        if st.button("Check Ready"):
            try:
                r = requests.get(f"{BACKEND_URL}/health/ready", timeout=15)
                st.write(r.status_code, r.json())
            except Exception as e:
                st.error(f"Ready check failed: {e}")

# -----------------------------
# Helper: robust API call with retries and good error display
# -----------------------------
def call_api_predict(backend_base: str, payload: List[Dict[str, Any]], max_retries: int = 3) -> Tuple[bool, Any]:
    """
    Calls POST {backend_base}/v1/predict with retries and returns (ok, data_or_error).
    """
    if not backend_base:
        return False, "Please set the Backend API Base URL in the sidebar."

    url = backend_base.rstrip("/") + "/v1/predict"
    last_err = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, json={"items": payload}, timeout=60, headers={"Content-Type": "application/json"})
            # try to parse json regardless of status to get consistent error body
            try:
                data = resp.json()
            except Exception:
                data = {"raw": resp.text}

            if resp.ok:
                return True, data
            else:
                # Backend returns structured errors: {request_id, code, message, details}
                code = data.get("code", f"HTTP_{resp.status_code}")
                msg = data.get("message", "Unknown error")
                details = data.get("details")
                last_err = f"{code}: {msg}" + (f" | details: {json.dumps(details)[:500]}" if details else "")
        except Exception as e:
            last_err = f"Exception: {e}"

        # backoff
        time.sleep(0.6 * attempt)

    return False, last_err or "Request failed."

# -----------------------------
# Defaults and schema hints
# -----------------------------
PRODUCT_TYPES = [
    "Snack Foods","Dairy","Canned","Soft Drinks","Meat","Frozen Foods",
    "Fruits and Vegetables","Baking Goods","Bread","Breakfast","Seafood",
    "Starchy Foods","Health and Hygiene","Household","Hard Drinks","Others"
]
STORE_IDS = ["OUT001","OUT002","OUT003","OUT004"]
STORE_SIZES = ["Small","Medium","High"]
CITY_TIERS = ["Tier 1","Tier 2","Tier 3"]
STORE_TYPES = ["Supermarket Type1","Supermarket Type2","Departmental Store","Food Mart"]
SUGAR = ["Low Sugar","Regular","No Sugar"]

EXAMPLE_RECORD = {
    "Product_Id": "FDX07",
    "Product_Weight": 12.50,
    "Product_Sugar_Content": "Regular",
    "Product_Allocated_Area": 0.045,
    "Product_Type": "Snack Foods",
    "Product_MRP": 155.0,
    "Store_Id": "OUT002",
    "Store_Establishment_Year": 2005,
    "Store_Size": "Small",
    "Store_Location_City_Type": "Tier 3",
    "Store_Type": "Food Mart"
}

# -----------------------------
# Tabs: Single vs Batch
# -----------------------------
tab_single, tab_batch = st.tabs(["✨ Single Prediction (Form)", "📦 Batch (Table / CSV)"])

with tab_single:
    st.subheader("Single Prediction")
    with st.form("single_form"):
        c1, c2 = st.columns(2)
        with c1:
            Product_Id = st.text_input("Product_Id", EXAMPLE_RECORD["Product_Id"])
            Product_Weight = st.number_input("Product_Weight", min_value=0.0, value=float(EXAMPLE_RECORD["Product_Weight"]), step=0.1)
            Product_Sugar_Content = st.selectbox("Product_Sugar_Content", SUGAR, index=SUGAR.index(EXAMPLE_RECORD["Product_Sugar_Content"]))
            Product_Allocated_Area = st.number_input("Product_Allocated_Area", min_value=0.0, max_value=1.0, value=float(EXAMPLE_RECORD["Product_Allocated_Area"]), step=0.001)
            Product_Type = st.selectbox("Product_Type", PRODUCT_TYPES, index=PRODUCT_TYPES.index(EXAMPLE_RECORD["Product_Type"]))
            Product_MRP = st.number_input("Product_MRP", min_value=0.0, value=float(EXAMPLE_RECORD["Product_MRP"]), step=0.5)
        with c2:
            Store_Id = st.selectbox("Store_Id", STORE_IDS, index=STORE_IDS.index(EXAMPLE_RECORD["Store_Id"]))
            Store_Establishment_Year = st.number_input("Store_Establishment_Year", min_value=1900, max_value=2100, value=int(EXAMPLE_RECORD["Store_Establishment_Year"]), step=1)
            Store_Size = st.selectbox("Store_Size", STORE_SIZES, index=STORE_SIZES.index(EXAMPLE_RECORD["Store_Size"]))
            Store_Location_City_Type = st.selectbox("Store_Location_City_Type", CITY_TIERS, index=CITY_TIERS.index(EXAMPLE_RECORD["Store_Location_City_Type"]))
            Store_Type = st.selectbox("Store_Type", STORE_TYPES, index=STORE_TYPES.index(EXAMPLE_RECORD["Store_Type"]))
        submitted = st.form_submit_button("Predict")

    if submitted:
        payload = [{
            "Product_Id": Product_Id,
            "Product_Weight": Product_Weight,
            "Product_Sugar_Content": Product_Sugar_Content,
            "Product_Allocated_Area": Product_Allocated_Area,
            "Product_Type": Product_Type,
            "Product_MRP": Product_MRP,
            "Store_Id": Store_Id,
            "Store_Establishment_Year": Store_Establishment_Year,
            "Store_Size": Store_Size,
            "Store_Location_City_Type": Store_Location_City_Type,
            "Store_Type": Store_Type
        }]
        with st.spinner("Calling backend..."):
            ok, data = call_api_predict(BACKEND_URL, payload)
        if ok:
            preds = data.get("predictions") or data.get("data") or []
            if preds:
                st.success(f"🧮 Predicted Sales: **{preds[0]:,.2f}**")
            else:
                st.warning("No prediction received.")
        else:
            st.error(data)

with tab_batch:
    st.subheader("Batch Inference")
    st.markdown("Use the editable table or upload a CSV matching the schema below. Click **Predict Batch** to get results and download a CSV of predictions.")

    # CSV uploader
    uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])
    if uploaded:
        try:
            df_in = pd.read_csv(uploaded)
            st.success(f"Loaded {len(df_in)} rows from CSV.")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            df_in = pd.DataFrame([copy.deepcopy(EXAMPLE_RECORD)])
    else:
        # start with sample row
        df_in = pd.DataFrame([copy.deepcopy(EXAMPLE_RECORD)])

    # Editable table
    edited_df = st.data_editor(
        df_in,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True
    )

    # Basic validation hints (lightweight; strict validation is handled by backend)
    def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = ["Product_Weight", "Product_Allocated_Area", "Product_MRP", "Store_Establishment_Year"]
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    edited_df = _coerce_types(edited_df)

    colx, coly = st.columns([1,1])
    with colx:
        if st.button("Predict Batch"):
            records = edited_df.to_dict(orient="records")
            with st.spinner("Calling backend for batch..."):
                ok, data = call_api_predict(BACKEND_URL, records)
            if ok:
                preds = data.get("predictions", [])
                if preds and len(preds) == len(records):
                    out = edited_df.copy()
                    out["Predicted_Sales"] = preds
                    st.success(f"Received {len(preds)} predictions.")
                    st.dataframe(out, use_container_width=True)
                    csv = out.to_csv(index=False).encode("utf-8")
                    st.download_button("Download predictions as CSV", data=csv, file_name="superkart_predictions.csv", mime="text/csv")
                else:
                    st.warning("Received response but the length of predictions did not match inputs.")
            else:
                st.error(data)
    with coly:
        st.download_button("Download sample CSV", data=pd.DataFrame([EXAMPLE_RECORD]).to_csv(index=False), file_name="sample_input.csv", mime="text/csv")

