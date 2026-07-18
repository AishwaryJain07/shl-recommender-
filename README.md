# SHL Conversational Assessment Recommender

A stateless FastAPI service that turns a vague hiring intent (e.g. *"I'm hiring a
Java developer"*) into a grounded shortlist of SHL assessments through dialogue —
built for the SHL Labs AI Intern take-home assignment.

Live API: **https://shl-recommender-2pul.onrender.com/docs**
Design source of truth: [BUILD_SPEC.md](BUILD_SPEC.md) · Approach write-up: [APPROACH.md](APPROACH.md)

## Status

**Complete and deployed.** The agent handles all four required behaviors —
clarify, recommend, refine, compare — plus refusal of off-topic/legal/injection
input, and never recommends anything outside the scraped SHL catalog.

- Phase 0 — Scaffold & data — **DONE**
- Phase 1 — Retrieval (BM25 + RRF fusion) — **DONE**
- Phase 2 — LLM turn + policy prompt — **DONE**
- Phase 3 — FastAPI service (`/health`, `/chat`) — **DONE**
- Phase 4 — Anti-hallucination hydration, stateless carry-forward, injection guard — **DONE**
- Eval harness (`eval/run_eval.py`) — **DONE**, results below

## How it works

```
guard -> build_query -> retrieve -> inject_anchors -> llm_turn -> hydrate -> validate
```

One LLM call per turn, wrapped so it never returns a 500 and never fabricates a
shortlist:

- **Retrieval:** BM25 (`rank_bm25`) over the 377-item catalog, run per user-turn
  and fused with Reciprocal Rank Fusion (RRF) - this beat a single concatenated
  query and beat dense/hybrid retrieval on measured pool recall (see below).
- **The LLM selects by `entity_id` only** - code hydrates `{name, url,
  test_type}` from the catalog. Hallucinated names or URLs are structurally
  impossible: the model never authors a URL, and any id not in the catalog is
  dropped.
- **Stateless carry-forward:** the API keeps no per-conversation state. Each
  turn re-derives the shortlist by matching catalog names in the last assistant
  reply, so *refine* ("drop REST, add Docker") and confirmation turns work
  correctly across turns.
- **Guardrails:** a deterministic regex pre-filter refuses obvious prompt
  injection before any LLM call; a clarify-loop cap forces a commit after 2
  assistant turns to stay inside the evaluator's 8-turn budget; every failure
  path (bad JSON, timeout, rate limit, malformed request) degrades to a valid,
  honest response instead of a 500 or a fake shortlist.

## Results

Measured with `eval/run_eval.py` against the 10 public conversation traces:

| Retrieval variant | Mean candidate-pool recall |
|---|---|
| BM25, single concatenated query | 0.778 |
| **BM25, per-turn multi-query + RRF fusion + field boosting** | **0.832** |
| Dense only (bge-small) | 0.764 |
| Hybrid BM25 + dense (RRF) | 0.832 (no gain over BM25 alone) |

Dense retrieval didn't help here - the residual recall gap is expert-bundled
companion items (reports, battery extras) that embeddings don't bridge either -
so the deployed system stays BM25-only (no 130MB model in the deploy).

**End-to-end: mean Recall@10 ~= 0.5 on public traces, behavior probes 9/9**
(schema-valid on every response, all URLs in-catalog, <=10 items, vague-> no
premature recommendation, injection->refuse, off-topic->refuse, legal->refuse,
edits honored). Full write-up of what broke and what was fixed is in
[APPROACH.md](APPROACH.md).

## Stack

FastAPI + Pydantic - BM25 (`rank_bm25`) - Gemini 2.5 Flash via its
OpenAI-compatible endpoint (JSON mode) - provider-agnostic, swappable to Groq
or any OpenAI-compatible endpoint via `LLM_BASE_URL` / `LLM_MODEL` /
`LLM_API_KEY_ENV`. Deployed on Render (free tier).

## Layout

```
app/
  main.py         # FastAPI service: GET /health, POST /chat
  agent.py        # per-turn orchestration: guard -> retrieve -> llm_turn -> hydrate
  retrieval.py    # BM25 index, RRF fusion, anchor injection, shortlist reconstruction
  llm.py          # provider-agnostic LLM client (Gemini/Groq, JSON mode, retries)
  prompts.py      # system prompt / policy construction
  catalog.py      # load/normalize catalog, derive test_type, id lookup
  config.py       # all tunables (BUILD_SPEC section 10); env-driven
  schema.py       # pydantic request/response models
data/
  catalog_normalized.json   # 377 SHL assessments, scraped + normalized (committed)
eval/
  traces/                   # the 10 public conversation traces
  parse_traces.py           # extract gold shortlist + user turns from traces
  run_eval.py               # trace-replay harness: Recall@10 + behavior probes
scripts/
  fetch_catalog.py          # download + normalize + print stats
```

## Running locally

```bash
pip install -r requirements.txt
cp .env.example .env        # add your GEMINI_API_KEY
uvicorn app.main:app --reload
```

```bash
curl http://localhost:8000/health
# -> {"status":"ok"}

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hiring a mid-level Java developer who works with stakeholders"}]}'
```

Run the eval harness against a running server:

```bash
python eval/run_eval.py     # uses EVAL_BASE_URL or localhost:8000
```

## Deployment

See [DEPLOY.md](DEPLOY.md) for full instructions (Render Blueprint, manual web
service, or Docker). The only required secret is `GEMINI_API_KEY`
(free at https://aistudio.google.com/apikey); the catalog is pre-normalized and
committed, so boot needs no network fetch.

## AI tools used

Built with Claude Code (Anthropic) for implementation, debugging, and the eval
harness. All design decisions in [APPROACH.md](APPROACH.md) are ones I can
defend in a technical interview.
