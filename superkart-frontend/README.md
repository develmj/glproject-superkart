# SuperKart Sales Forecast — Frontend (Streamlit)

This Streamlit app calls the **SuperKart Backend API** and displays predicted sales for single or batch inputs.

## Configure Backend URL

Set the backend Space URL in one of these ways (priority order):
1. **Repository Secret**: Add a secret named `BACKEND_URL` in the Space settings (recommended).
2. **Environment variable**: `BACKEND_URL`.
3. **Sidebar input**: Manually paste the backend URL in the app sidebar.

- Example backend URL: `https://<your-username>-superkart-backend.hf.space`

## Features

- Single prediction via a clean form.
- Batch predictions via editable table or CSV upload.
- Robust API calls with timeout + retries.
- Download predictions as CSV.
- Health checks (live/ready) from the sidebar.

## Run locally (optional)

```bash
pip install -r requirements.txt
export BACKEND_URL="https://<your-username>-superkart-backend.hf.space"
streamlit run app.py
# glproject-superkart
