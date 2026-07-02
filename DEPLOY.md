# Deployment

The service is a stateless FastAPI app. The normalized catalog
(`data/catalog_normalized.json`) is committed, so **boot needs no network fetch**.
`/health` does no heavy loading; the BM25 index is built lazily on the first
`/chat` (cold-start safe — the spec allows up to 2 min for the first `/health`).

The only required secret is **`GEMINI_API_KEY`** (a free key from
https://aistudio.google.com/apikey). Everything else has a safe default in
`app/config.py` (see `.env.example` for the full list). The LLM is Google
**Gemini 2.5 Flash** via its OpenAI-compatible endpoint; to swap providers, set
`LLM_BASE_URL`, `LLM_MODEL`, and `LLM_API_KEY_ENV`.

## Option A — Render (recommended, free tier)

### A1. Blueprint (one click)
1. Push this repo to GitHub.
2. Render Dashboard → **New → Blueprint**, point it at the repo. It reads
   [`render.yaml`](render.yaml) (native Python, `healthCheckPath: /health`).
3. Set the **`GEMINI_API_KEY`** env var when prompted (marked `sync: false`, so it
   is never committed). Deploy.

### A2. Manual web service (no blueprint)
1. Render → **New → Web Service** → connect the repo.
2. Environment: **Python 3**. Region/plan: **Free**.
3. **Build command:** `pip install -r requirements.txt`
4. **Start command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1`
5. **Health check path:** `/health`
6. **Environment variables:** add `GEMINI_API_KEY=<your key>`
   (optionally `LLM_MODEL`, `RETRIEVAL_POOL_SIZE`, etc. to override defaults).
7. Create → wait for the first build. Render sets `$PORT` automatically.

`runtime.txt` pins Python 3.11.9; a `Procfile` is also present for
Procfile-based hosts (Heroku/Railway).

## Option B — Docker (Render / Fly / Railway / any container host)

```bash
docker build -t shl-recommender .
docker run -p 8000:8000 -e GEMINI_API_KEY=<your key> shl-recommender
```

The [`Dockerfile`](Dockerfile) is single-stage on `python:3.11-slim`, bakes in the
catalog, and binds `uvicorn` to `$PORT`. On Render, choose **Docker** as the
environment instead of Python and set `GEMINI_API_KEY` in the dashboard.

## Verify after deploy

```bash
# Health
curl https://<your-app>.onrender.com/health
# -> {"status":"ok"}

# Chat
curl -X POST https://<your-app>.onrender.com/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hiring a mid-level Java developer who works with stakeholders"}]}'
```

The first `/chat` after a cold start warms the BM25 index (a few seconds).
Per-call latency is well under the evaluator's 30 s cap.

## Free-tier note (important)

We default to **Gemini 2.5 Flash** because its free tier (15 req/min, **1M
tokens/min**) comfortably absorbs the evaluator replaying many multi-turn traces
+ probes. We originally used Groq, but its free tier throttled us on both
tokens-per-minute (12k for 70b / 6k for 8b) and tokens-per-day (100k for 70b),
causing calls to fall back mid-run. The provider is fully configurable
(`LLM_BASE_URL` / `LLM_MODEL` / `LLM_API_KEY_ENV`) — e.g. set Groq env vars to
switch back.
