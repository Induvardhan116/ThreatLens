from pathlib import Path
from typing import Any, Dict, Optional

import json
import sys

import pandas as pd

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware


# ========================================================================
# THREATLENS APPLICATION BACKEND
# STEP 37 - EVIDENCE + AI INVESTIGATOR INTEGRATION
# ========================================================================

BASE_DIR = Path(r"D:\ThreatLens")

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


# ------------------------------------------------------------------------
# APPLICATION
# ------------------------------------------------------------------------

app = FastAPI(
    title="ThreatLens Application API",
    description=(
        "Application layer for vulnerability prioritization, "
        "evidence retrieval, knowledge graph exploration, "
        "and AI-assisted investigation."
    ),
    version="1.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=[
        "GET",
        "POST",
    ],
    allow_headers=[
        "Content-Type",
    ],
)


# ------------------------------------------------------------------------
# PATHS
# ------------------------------------------------------------------------

PREDICTIONS_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "inference"
    / "threatlens_predictions.csv"
)

GRAPH_NODES_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "knowledge_graph"
    / "threatlens_graph_nodes.csv"
)

GRAPH_EDGES_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "knowledge_graph"
    / "threatlens_graph_edges.csv"
)

GRAPH_SUMMARY_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "knowledge_graph"
    / "threatlens_graph_summary.json"
)

AUDIT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "inference"
    / "threatlens_final_audit.json"
)


# ------------------------------------------------------------------------
# DATA LOADERS
# ------------------------------------------------------------------------

_predictions = None
_graph_nodes = None
_graph_edges = None
_graph_summary = None


def load_predictions():

    global _predictions

    if _predictions is None:

        if not PREDICTIONS_PATH.exists():
            raise HTTPException(
                status_code=500,
                detail="ThreatLens prediction dataset is unavailable.",
            )

        try:
            _predictions = pd.read_csv(
                PREDICTIONS_PATH
            )

        except Exception as exc:

            raise HTTPException(
                status_code=500,
                detail="ThreatLens prediction dataset could not be loaded.",
            ) from exc

    return _predictions


def load_graph_nodes():

    global _graph_nodes

    if _graph_nodes is None:

        if not GRAPH_NODES_PATH.exists():
            raise HTTPException(
                status_code=500,
                detail="Knowledge graph node data is unavailable.",
            )

        try:
            _graph_nodes = pd.read_csv(
                GRAPH_NODES_PATH
            )

        except Exception as exc:

            raise HTTPException(
                status_code=500,
                detail="Knowledge graph node data could not be loaded.",
            ) from exc

    return _graph_nodes


def load_graph_edges():

    global _graph_edges

    if _graph_edges is None:

        if not GRAPH_EDGES_PATH.exists():
            raise HTTPException(
                status_code=500,
                detail="Knowledge graph edge data is unavailable.",
            )

        try:
            _graph_edges = pd.read_csv(
                GRAPH_EDGES_PATH
            )

        except Exception as exc:

            raise HTTPException(
                status_code=500,
                detail="Knowledge graph edge data could not be loaded.",
            ) from exc

    return _graph_edges


def load_graph_summary():

    global _graph_summary

    if _graph_summary is None:

        if not GRAPH_SUMMARY_PATH.exists():
            raise HTTPException(
                status_code=500,
                detail="Knowledge graph summary is unavailable.",
            )

        try:

            with GRAPH_SUMMARY_PATH.open(
                "r",
                encoding="utf-8",
            ) as file:

                _graph_summary = json.load(
                    file
                )

        except Exception as exc:

            raise HTTPException(
                status_code=500,
                detail="Knowledge graph summary could not be loaded.",
            ) from exc

    return _graph_summary


# ------------------------------------------------------------------------
# SERIALIZATION
# ------------------------------------------------------------------------

def clean_value(value: Any) -> Any:

    if value is None:
        return None

    try:

        if pd.isna(value):
            return None

    except Exception:
        pass

    if isinstance(
        value,
        (
            pd.Timestamp,
        ),
    ):

        return value.isoformat()

    if hasattr(
        value,
        "item",
    ):

        try:
            return value.item()
        except Exception:
            pass

    return value


def row_to_dict(
    row: pd.Series,
) -> Dict[str, Any]:

    return {
        str(key): clean_value(value)
        for key, value in row.to_dict().items()
    }


# ------------------------------------------------------------------------
# CVE LOOKUP
# ------------------------------------------------------------------------

def find_cve(
    cve_id: str,
) -> pd.Series:

    normalized = str(
        cve_id
    ).strip().upper()

    if not normalized:
        raise HTTPException(
            status_code=400,
            detail="CVE ID cannot be empty.",
        )

    df = load_predictions()

    if "cve_id" not in df.columns:
        raise HTTPException(
            status_code=500,
            detail="Prediction dataset is missing cve_id.",
        )

    matches = df[
        df["cve_id"]
        .astype(str)
        .str.upper()
        == normalized
    ]

    if matches.empty:

        raise HTTPException(
            status_code=404,
            detail=f"{normalized} was not found.",
        )

    return matches.iloc[0]


# ------------------------------------------------------------------------
# DASHBOARD SUMMARY
# ------------------------------------------------------------------------

@app.get("/api/dashboard/summary")
def dashboard_summary():

    df = load_predictions()

    total = len(df)

    kev_count = 0

    if "known_exploited" in df.columns:

        kev_count = int(
            df["known_exploited"]
            .fillna(False)
            .astype(bool)
            .sum()
        )

    elif "exploitation_label" in df.columns:

        kev_count = int(
            (
                df["exploitation_label"]
                .fillna(0)
                .astype(int)
                == 1
            ).sum()
        )

    risk_distribution = {}

    if "risk_category" in df.columns:

        risk_distribution = {
            str(key): int(value)
            for key, value in (
                df["risk_category"]
                .fillna("UNKNOWN")
                .value_counts()
                .to_dict()
                .items()
            )
        }

    score_mean = 0.0

    if "threatlens_score" in df.columns:

        score_mean = float(
            pd.to_numeric(
                df["threatlens_score"],
                errors="coerce",
            )
            .fillna(0)
            .mean()
        )

    return {
        "total_vulnerabilities": total,
        "kev_vulnerabilities": kev_count,
        "risk_distribution": risk_distribution,
        "average_threatlens_score": score_mean,
        "model": "ThreatLens Conservative",
        "model_version": "1.0",
    }


# ------------------------------------------------------------------------
# VULNERABILITY LIST
# ------------------------------------------------------------------------

@app.get("/api/vulnerabilities")
def vulnerabilities(
    search: Optional[str] = Query(
        default=None,
        max_length=100,
    ),
    category: Optional[str] = Query(
        default=None,
        max_length=20,
    ),
    kev: Optional[bool] = None,
    limit: int = Query(
        default=50,
        ge=1,
        le=100,
    ),
):

    df = load_predictions().copy()

    if search:

        search_upper = search.upper()

        df = df[
            df["cve_id"]
            .astype(str)
            .str.upper()
            .str.contains(
                search_upper,
                regex=False,
            )
        ]

    if category:

        df = df[
            df["risk_category"]
            .astype(str)
            .str.upper()
            == category.upper()
        ]

    if kev is not None:

        if "known_exploited" in df.columns:

            values = (
                df["known_exploited"]
                .fillna(False)
                .astype(bool)
            )

            df = df[
                values == kev
            ]

        elif "exploitation_label" in df.columns:

            values = (
                df["exploitation_label"]
                .fillna(0)
                .astype(int)
                == 1
            )

            df = df[
                values == kev
            ]

    if "threatlens_score" in df.columns:

        df = df.sort_values(
            "threatlens_score",
            ascending=False,
        )

    records = [
        row_to_dict(row)
        for _, row in df.head(limit).iterrows()
    ]

    return {
        "count": len(records),
        "results": records,
    }


# ------------------------------------------------------------------------
# VULNERABILITY DETAIL
# ------------------------------------------------------------------------

@app.get("/api/vulnerabilities/{cve_id}")
def vulnerability_detail(
    cve_id: str,
):

    row = find_cve(cve_id)

    return row_to_dict(row)


# ------------------------------------------------------------------------
# EVIDENCE ENGINE
# ------------------------------------------------------------------------

def get_evidence_engine():

    try:

        from evidence.evidence_engine import EvidenceEngine

        return EvidenceEngine()

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail="Evidence Engine could not be initialized.",
        ) from exc


@app.get("/api/vulnerabilities/{cve_id}/evidence")
def vulnerability_evidence(
    cve_id: str,
):

    normalized = str(
        cve_id
    ).strip().upper()

    # Confirm the CVE exists in the production prediction dataset.
    find_cve(normalized)

    engine = get_evidence_engine()

    try:

        result = engine.search(
            normalized
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail="Evidence retrieval failed.",
        ) from exc

    if not isinstance(
        result,
        dict,
    ):

        raise HTTPException(
            status_code=500,
            detail="Evidence Engine returned an invalid response.",
        )

    return result


# ------------------------------------------------------------------------
# DETERMINISTIC AI INVESTIGATOR
# ------------------------------------------------------------------------

def get_investigator():

    try:

        from ai.investigator import AIInvestigator

        return AIInvestigator()

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail="AI Investigator could not be initialized.",
        ) from exc


@app.get("/api/vulnerabilities/{cve_id}/investigation")
def vulnerability_investigation(
    cve_id: str,
):

    normalized = str(
        cve_id
    ).strip().upper()

    # Confirm the CVE exists in the production prediction dataset.
    find_cve(normalized)

    investigator = get_investigator()

    try:

        result = investigator.investigate(
            normalized
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail="AI investigation failed.",
        ) from exc

    if not isinstance(
        result,
        dict,
    ):

        raise HTTPException(
            status_code=500,
            detail="AI Investigator returned an invalid response.",
        )

    return result


# ------------------------------------------------------------------------
# KNOWLEDGE GRAPH SUMMARY
# ------------------------------------------------------------------------

@app.get("/api/graph/summary")
def graph_summary():

    return load_graph_summary()


# ------------------------------------------------------------------------
# KNOWLEDGE GRAPH DATA
# ------------------------------------------------------------------------

@app.get("/api/graph/{cve_id}")
def graph_for_cve(
    cve_id: str,
):

    cve_id = cve_id.upper()

    nodes = load_graph_nodes()
    edges = load_graph_edges()

    if "node_id" not in nodes.columns:

        raise HTTPException(
            status_code=500,
            detail="Knowledge graph nodes are malformed.",
        )

    cve_matches = nodes[
        nodes["node_id"]
        .astype(str)
        .str.upper()
        == ("CVE:" + cve_id)
    ]

    if cve_matches.empty:

        raise HTTPException(
            status_code=404,
            detail=(
                f"{cve_id} is not present "
                "in the knowledge graph."
            ),
        )

    root_id = str(
        cve_matches.iloc[0]["node_id"]
    )

    related_edges = edges[
        (
            edges["source"]
            .astype(str)
            == root_id
        )
        |
        (
            edges["target"]
            .astype(str)
            == root_id
        )
    ]

    node_ids = {
        root_id
    }

    for _, edge in related_edges.iterrows():

        node_ids.add(
            str(edge["source"])
        )

        node_ids.add(
            str(edge["target"])
        )

    selected_nodes = nodes[
        nodes["node_id"]
        .astype(str)
        .isin(node_ids)
    ]

    selected_edges = related_edges

    return {
        "cve_id": cve_id,
        "nodes": [
            row_to_dict(row)
            for _, row
            in selected_nodes.iterrows()
        ],
        "edges": [
            row_to_dict(row)
            for _, row
            in selected_edges.iterrows()
        ],
    }


# ------------------------------------------------------------------------
# SYSTEM STATUS
# ------------------------------------------------------------------------

@app.get("/api/system/status")
def system_status():

    audit_exists = AUDIT_PATH.exists()

    graph_ready = (
        GRAPH_NODES_PATH.exists()
        and GRAPH_EDGES_PATH.exists()
        and GRAPH_SUMMARY_PATH.exists()
    )

    prediction_ready = PREDICTIONS_PATH.exists()

    return {
        "status": "ready",
        "prediction_dataset": prediction_ready,
        "knowledge_graph": graph_ready,
        "final_audit": audit_exists,
        "model": "ThreatLens Conservative",
        "model_version": "1.0",
    }


# ------------------------------------------------------------------------
# ROOT
# ------------------------------------------------------------------------

@app.get("/api/health")
def api_health():

    return {
        "status": "healthy",
        "service": "ThreatLens Application API",
        "version": "1.0",
    }


print()
print("=" * 72)
print("THREATLENS APPLICATION BACKEND")
print("STEP 37")
print("=" * 72)
print()
print("Application endpoints:")
print("  GET /api/health")
print("  GET /api/dashboard/summary")
print("  GET /api/vulnerabilities")
print("  GET /api/vulnerabilities/{cve_id}")
print("  GET /api/vulnerabilities/{cve_id}/evidence")
print("  GET /api/vulnerabilities/{cve_id}/investigation")
print("  GET /api/graph/summary")
print("  GET /api/graph/{cve_id}")
print("  GET /api/system/status")
print()
print("Evidence Engine: AVAILABLE")
print("AI Investigator: AVAILABLE")
print("=" * 72)
