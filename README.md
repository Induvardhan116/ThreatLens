# ThreatLens

ThreatLens is a cybersecurity research and engineering project for
context-aware vulnerability prioritization. The application combines a
vulnerability dashboard, a knowledge graph, evidence retrieval, and
evidence-grounded investigation.

## Current features

- Dashboard for vulnerability totals, risk distribution, and the highest-risk
  records.
- Searchable vulnerability list with risk-category and CISA KEV filters.
- Knowledge graph exploration for CVEs.
- Evidence Engine and AI Investigator lookups backed by the application API.
- Optional, browser-local log analysis for CSV, JSON/NDJSON, `.log`, and `.txt`
  files. Uploaded logs are analyzed in the browser and are not sent to the
  ThreatLens API by this feature.

Log-analysis results are heuristic investigative leads, not confirmed
incidents or proof that activity is malicious. Generated reports can contain
usernames, hostnames, and IP addresses; handle downloaded reports as sensitive
security data.

## Project layout

- `backend/` — FastAPI application API.
- `frontend/` — React and Vite dashboard.
- `ai/` — investigation and model-evaluation integrations.
- `evidence/` — evidence retrieval.
- `knowledge_graph/` — knowledge-graph utilities.
- `pipeline/` — ingestion, normalization, feature, inference, and evaluation
  code.
- `ui/` — Python dashboard.
- `tests/` — Python tests.

## Run locally

1. Install Python dependencies:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

2. Prepare the runtime data files listed under [Deployment](#deployment).
   Source and intermediate datasets are intentionally not included in this
   repository.

3. Start the API:

   ```powershell
   .\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8001
   ```

4. In a second terminal, install and start the frontend:

   ```powershell
   cd frontend
   npm ci
   npm run dev
   ```

The Vite development server proxies `/api` to `http://127.0.0.1:8001`. Set
`VITE_API_PROXY_TARGET` to override that target. Set `VITE_API_BASE_URL` when
the browser should use an absolute API base URL instead.

## Deployment

The frontend and API are separate services: Vercel hosts the Vite frontend,
and the included Render Blueprint deploys the FastAPI application. In Vercel,
import this repository with `frontend` as the Root Directory. The included
`frontend/vercel.json` rewrites client-side routes to `index.html`.

After deploying the API, set Vercel's `VITE_API_BASE_URL` environment variable
to the API origin plus `/api`, for example
`https://threatlens-api.example.com/api`, then redeploy the frontend. Set the
API's `THREATLENS_CORS_ORIGINS` to the exact Vercel production origin (no path),
and add any preview/custom domains that should be allowed, comma-separated.
Do not use `*` for this setting.

### Runtime data required by API features

The source repository excludes datasets. The API needs these five generated
runtime files for its dashboard, vulnerability, graph, evidence, and
investigator features:

- `processed/inference/threatlens_predictions.csv`
- `processed/knowledge_graph/threatlens_graph_nodes.csv`
- `processed/knowledge_graph/threatlens_graph_edges.csv`
- `processed/knowledge_graph/threatlens_graph_summary.json`
- `processed/inference/threatlens_final_audit.json` (used for audit status)

Together these files are about 9.5 MB in the current local dataset; the raw
research datasets and model/evaluation artifacts are not needed by the
deployed API. Provide these files to the API host using its persistent storage
or another private deployment mechanism, preserving the paths above. Set
`THREATLENS_DATA_DIR` to the directory containing `processed/` (for example,
`/var/data` when that is the mounted storage root). Never put credentials or
private data in the public repository.

The Render Blueprint defaults `THREATLENS_DATA_DIR` to `./data` and deploys a
health-checked API. Until the five runtime files are provisioned, `/api/health`
will confirm that the service process is running, but data-backed endpoints
will return explicit errors; `/api/system/status` will report `degraded` and
identify which runtime datasets are unavailable. It reports `ready` only when
all five runtime files are present.
The optional browser-local log analysis does not depend on this data or API.

Render's free service has ephemeral storage; files written there are not a
durable dataset deployment. Choose persistent storage or secure external
provisioning before relying on API features in production. The service can
start without the data files, but a healthy process alone does not mean the
dashboard data is ready.

## Frontend checks

Run from `frontend/`:

```powershell
npm run lint
npm run build
```

## Data, privacy, and licensing

Raw and generated data, model artifacts, virtual environments, frontend
dependencies, build output, and local logs are excluded from Git. Prepare
required datasets separately and ensure you have the right to use or
redistribute any data before sharing it.

The optional log-analysis feature processes selected files in the browser. It
does not upload them to the ThreatLens API or persist them. Findings depend on
the available fields and configured heuristic patterns; validate them against
source telemetry and normal analyst workflows.

No open-source license has been selected for this project yet. Until a license
is added, reuse and redistribution are not granted by this repository.
