# SHL Conversational Assessment Recommender — production image.
# Small, single-stage; the normalized catalog is baked in so boot needs no
# network fetch (BUILD_SPEC §12). The container reads config from env vars
# (GROQ_API_KEY etc. supplied by the host), never from a committed .env.

FROM python:3.11-slim

# No .pyc, unbuffered logs (so platform log tailing works).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Install deps first for layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code + the bundled normalized catalog (data/catalog_normalized.json).
COPY app/ ./app/
COPY data/ ./data/

# Documents the port; the real bind uses $PORT (Render/Fly inject it).
EXPOSE 8000

# Shell form so ${PORT} is expanded at runtime. One worker keeps the in-memory
# BM25 index single-copy; the workload is one conversation at a time.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1
