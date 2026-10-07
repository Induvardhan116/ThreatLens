# ThreatLens

ThreatLens is a cybersecurity research and engineering project for
context-aware vulnerability prioritization. It brings vulnerability risk
information, known-exploited status, knowledge-graph relationships, and
supporting evidence together to help security teams decide what to investigate
first.

> **Live site:** [threat-lens-one-tau.vercel.app](https://threat-lens-one-tau.vercel.app)

## What ThreatLens offers

### Vulnerability intelligence

- Review vulnerability totals, risk distribution, and the highest-priority
  records from a single dashboard.
- Search vulnerabilities and filter by risk category or CISA Known Exploited
  Vulnerabilities (KEV) status.
- Open individual vulnerability records and review their available
  prioritization details.

### Connected context and investigation

- Explore a knowledge graph to see relationships associated with a CVE.
- Use the Evidence Engine to retrieve supporting information for a
  vulnerability.
- Use the AI Investigator to organize evidence into an investigation view.

### Browser-local log analysis

- Analyze CSV, JSON/NDJSON, `.log`, and `.txt` files using browser-side
  heuristic pattern checks.
- Keep selected log contents in the browser; this feature does not upload them
  to the ThreatLens API or persist them.
- Export findings for further review.

Log-analysis findings are investigative leads, not confirmed incidents or
proof that activity is malicious. Validate them against source telemetry and
normal analyst workflows. Reports may contain usernames, hostnames, and IP
addresses; handle them as sensitive security data.

## How it works

ThreatLens combines vulnerability prioritization data with exploitation
signals and related graph context. Analysts can move from an overview to a
specific vulnerability, inspect its relationships, and review supporting
evidence. The log-analysis feature is a separate, optional browser-side
workflow.

Scores, categories, and automated investigation output are decision-support
signals. They should be assessed alongside current advisories, asset exposure,
business impact, and analyst judgment.

## Deployment architecture

The live site is the React and Vite frontend deployed on Vercel. API-backed
features require the separate FastAPI application and its runtime datasets to
be deployed and configured. The source repository includes a Render Blueprint
for the API service.

The backend expects generated prediction and knowledge-graph data files that
are intentionally not included in this repository. Until those files are
provisioned on the API host, the site may load while API-backed dashboard,
graph, evidence, and investigation features remain unavailable. The backend's
`/api/system/status` endpoint reports whether its required data is ready.
Browser-local log analysis does not depend on the API.

Never commit credentials, private deployment configuration, or sensitive
security datasets to the public repository.

## Project structure

| Directory | Purpose |
| --- | --- |
| `frontend/` | React and Vite web application |
| `backend/` | FastAPI application API and deployment configuration |
| `ai/` | Investigation and model-evaluation integrations |
| `evidence/` | Evidence retrieval |
| `knowledge_graph/` | Knowledge-graph utilities |
| `pipeline/` | Data ingestion, normalization, feature processing, inference, and evaluation |
| `ui/` | Python dashboard |
| `tests/` | Automated tests |

## Data, privacy, and licensing

Raw research datasets, generated runtime data, and model artifacts are not
included in the repository. Ensure you have the right to use or redistribute
any data before sharing it.

ThreatLens is provided without an open-source license at this time. Unless a
license is added, no permission to reuse or redistribute the project is
granted by this repository.
