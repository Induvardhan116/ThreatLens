# ThreatLens frontend

The dashboard is a React/Vite client for the read-only ThreatLens Application
API. The frontend does not start or modify the backend.

## Run locally

From the repository root, start the existing FastAPI application:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8001
```

In a second terminal, start the frontend:

```powershell
cd frontend
npm install
npm run dev
```

The Vite development server proxies `/api` requests to
`http://127.0.0.1:8001` by default. This avoids browser CORS issues during
local development. Verify the backend is available at
`http://127.0.0.1:8001/api/health`.

## API configuration

Set `VITE_API_PROXY_TARGET` in a local `.env` file to change the development
proxy target. Set `VITE_API_BASE_URL` when the frontend should call an
absolute API base URL instead of the same-origin `/api` path. For example:

```dotenv
VITE_API_PROXY_TARGET=http://127.0.0.1:8001
# Optional; leave unset to use the Vite /api proxy.
# VITE_API_BASE_URL=https://api.example.test/api
```

For production, deploy this folder to Vercel with `frontend` as the project
Root Directory. Set `VITE_API_BASE_URL` to the deployed API URL ending in
`/api` (for example `https://threatlens-api.example.com/api`) in the Vercel
project's environment variables, then redeploy. `vercel.json` routes
client-side paths such as `/vulnerabilities`, `/graph`, `/research`,
`/investigator`, and `/evidence` back to the app entry point.

The backend must allow the deployed site origin through
`THREATLENS_CORS_ORIGINS`. API-backed dashboard features require the backend's
runtime datasets; see the repository root README for the required files and
data provisioning notes. The optional browser-local log analysis works
without the API.

## Optional local log analysis

The dashboard's **Analyze Your Logs** panel accepts `.csv`, `.json` (including
newline-delimited JSON), `.log`, and `.txt` files up to 25 MB. It parses and
analyzes the selected file in the browser using deterministic pattern checks;
the log contents are not sent to the application API or persisted by this
feature. Analysis output is retained only in the current page until cleared or
the page is closed.

Findings are grouped investigative leads based on available fields, not
confirmed incidents, verified IOCs, or a conclusion that activity is
malicious. The feature is optional and does not replace existing SIEM or
integrated telemetry workflows. Review the generated JSON report as
potentially sensitive because it may include usernames, hosts, and IP
addresses extracted from matched events.

## Available frontend checks

```powershell
npm run lint
npm run build
```
