FROM python:3.10-slim

WORKDIR /app

# System deps (optional but helpful)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Dependencies first (cache layer)
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy artifacts and code
COPY best_model_superkart.pkl best_model_superkart.pkl
COPY best_model_info.json best_model_info.json
COPY app.py app.py

# Expose Flask/Gunicorn port
EXPOSE 7860

# Healthcheck (optional, Spaces usually handles readiness via logs)
# HEALTHCHECK --interval=30s --timeout=5s CMD curl -f http://localhost:7860/health/ready || exit 1

# Start with gunicorn: 2 workers, 2 threads each (adjust for Space resources)
# -k gthread gives you simple threaded workers; for CPU-bound inference this is fine.
CMD ["gunicorn", "-w", "2", "-k", "gthread", "--threads", "2", "-b", "0.0.0.0:7860", "app:app"]

