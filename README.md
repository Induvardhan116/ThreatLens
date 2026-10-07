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

ThreatLens currently has local data and path assumptions; for the existing
backend configuration, use a Windows checkout at `D:\ThreatLens`.

1. Install Python dependencies:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

2. Prepare the local prediction and graph data files expected by
   `backend/app.py`. These generated and source datasets are intentionally not
   included in this repository. The API reports data-dependent errors until
   the required files are available.

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

Production hosting must route `/api` to the backend and serve the frontend's
`index.html` as the fallback for client-side routes such as `/graph` and
`/vulnerabilities`.

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
