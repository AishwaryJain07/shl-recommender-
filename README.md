# SHL Conversational Assessment Recommender

A stateless FastAPI service that recommends SHL assessments through dialogue.
Design source of truth: [BUILD_SPEC.md](BUILD_SPEC.md).

## Status

- **Phase 0 — Scaffold & data — DONE.** Config, catalog fetch/normalize + stats,
  trace parser.
- Phase 1+ — not started.

## Layout

```
app/
  config.py       # all tunables (BUILD_SPEC §10); keys->test_type mapping
  catalog.py      # load/normalize catalog, derive test_type, id lookup, stats
data/
  catalog_normalized.json   # built by fetch_catalog.py (bundled for deploy)
scripts/
  fetch_catalog.py          # download + normalize + print stats
eval/
  traces/                   # the 10 public conversation traces
  parse_traces.py           # extract gold shortlist (last table) + user turns
```

## Phase 0 usage

```bash
pip install -r requirements.txt
python scripts/fetch_catalog.py     # writes data/*.json, prints catalog stats
python eval/parse_traces.py         # writes eval/traces_parsed.json, prints golds
```

Use `--offline` on `fetch_catalog.py` to normalize a previously downloaded raw
file without hitting the network.
