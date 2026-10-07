from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backend.config import resolve_data_dir

DATA_DIR = resolve_data_dir()

NODES_PATH = DATA_DIR / "processed" / "knowledge_graph" / "threatlens_graph_nodes.csv"
EDGES_PATH = DATA_DIR / "processed" / "knowledge_graph" / "threatlens_graph_edges.csv"
OUTPUT_DIR = DATA_DIR / "processed" / "evidence"
TEST_OUTPUT = OUTPUT_DIR / "evidence_engine_test_results.json"


class EvidenceEngine:
    """Deterministic evidence retrieval layer for ThreatLens."""

    def __init__(
        self,
        nodes_path: Path = NODES_PATH,
        edges_path: Path = EDGES_PATH,
    ) -> None:
        if not nodes_path.exists():
            raise FileNotFoundError(f"Nodes file not found: {nodes_path}")

        if not edges_path.exists():
            raise FileNotFoundError(f"Edges file not found: {edges_path}")

        self.nodes = pd.read_csv(
            nodes_path,
            dtype=str,
            keep_default_na=False,
        )

        self.edges = pd.read_csv(
            edges_path,
            dtype=str,
            keep_default_na=False,
        )

        self._validate_schema()

        # Preserve node_id inside every node dictionary.
        self.nodes_by_id: dict[str, dict[str, Any]] = {}

        for row in self.nodes.to_dict(orient="records"):
            node_id = str(row["node_id"]).strip()
            row["node_id"] = node_id
            self.nodes_by_id[node_id] = row

        self.edges_by_source: dict[str, list[dict[str, str]]] = {}

        for row in self.edges.to_dict(orient="records"):
            source = str(row["source"]).strip()
            self.edges_by_source.setdefault(source, []).append(row)

    def _validate_schema(self) -> None:
        required_nodes = {
            "node_id",
            "node_type",
            "label",
            "record_id",
            "threatlens_score",
            "risk_category",
            "severity",
            "known_exploited",
            "published_date",
            "kev_date_added",
            "url",
        }

        required_edges = {
            "source",
            "target",
            "relation",
        }

        missing_nodes = required_nodes - set(self.nodes.columns)
        missing_edges = required_edges - set(self.edges.columns)

        if missing_nodes:
            raise ValueError(
                f"Nodes CSV missing columns: {sorted(missing_nodes)}"
            )

        if missing_edges:
            raise ValueError(
                f"Edges CSV missing columns: {sorted(missing_edges)}"
            )

    @staticmethod
    def _clean(value: Any) -> Any:
        if value is None:
            return None

        value = str(value).strip()

        return value if value else None

    @staticmethod
    def _float(value: Any) -> float | None:
        value = EvidenceEngine._clean(value)

        if value is None:
            return None

        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _bool(value: Any) -> bool | None:
        value = EvidenceEngine._clean(value)

        if value is None:
            return None

        if value.lower() == "true":
            return True

        if value.lower() == "false":
            return False

        return None

    @staticmethod
    def _normalize_cve(cve_id: str) -> str:
        value = cve_id.strip().upper()

        if value.startswith("CVE:"):
            value = value[4:]

        return value

    def _cve_node_id(self, cve_id: str) -> str:
        return f"CVE:{self._normalize_cve(cve_id)}"

    def _get_cve_node(self, cve_id: str) -> dict[str, Any] | None:
        node_id = self._cve_node_id(cve_id)
        node = self.nodes_by_id.get(node_id)

        if node is None:
            return None

        if node.get("node_type") != "CVE":
            return None

        return node

    def _related_nodes(
        self,
        cve_node_id: str,
        relation: str,
        node_type: str | None = None,
    ) -> list[dict[str, Any]]:

        results: list[dict[str, Any]] = []

        for edge in self.edges_by_source.get(cve_node_id, []):
            if edge.get("relation") != relation:
                continue

            target_id = edge.get("target")
            target = self.nodes_by_id.get(target_id)

            if target is None:
                continue

            if node_type is not None and target.get("node_type") != node_type:
                continue

            results.append(target)

        return results

    def get_metadata(self, cve_id: str) -> dict[str, Any]:
        node = self._get_cve_node(cve_id)

        if node is None:
            return {
                "status": "UNKNOWN",
                "reason": "CVE is not present in the current graph.",
            }

        return {
            "status": "KNOWN",
            "cve_id": self._clean(node.get("label")),
            "record_id": self._clean(node.get("record_id")),
            "severity": self._clean(node.get("severity")),
            "published_date": self._clean(node.get("published_date")),
        }

    def get_risk_evidence(self, cve_id: str) -> dict[str, Any]:
        node = self._get_cve_node(cve_id)

        if node is None:
            return {
                "status": "UNKNOWN",
                "reason": "CVE is not present in the current graph.",
            }

        score = self._float(node.get("threatlens_score"))
        category = self._clean(node.get("risk_category"))

        if score is None or category is None:
            return {
                "status": "UNKNOWN",
                "reason": "Risk evidence is incomplete.",
            }

        return {
            "status": "KNOWN",
            "source": "ThreatLens production prediction",
            "score": score,
            "risk_category": category,
        }

    def get_kev_evidence(self, cve_id: str) -> dict[str, Any]:
        node = self._get_cve_node(cve_id)

        if node is None:
            return {
                "status": "UNKNOWN",
                "reason": "CVE is not present in the current graph.",
            }

        known_exploited = self._bool(node.get("known_exploited"))
        kev_date = self._clean(node.get("kev_date_added"))

        kev_edges = self._related_nodes(
            node["node_id"],
            "KNOWN_EXPLOITED",
            "ThreatSignal",
        )

        if known_exploited is True or len(kev_edges) > 0:
            result = {
                "status": "KNOWN",
                "source": "CISA Known Exploited Vulnerabilities",
                "known_exploited": True,
            }

            if kev_date:
                result["kev_date_added"] = kev_date

            if kev_edges:
                result["signal"] = self._clean(
                    kev_edges[0].get("label")
                )

            return result

        if known_exploited is False:
            return {
                "status": "KNOWN",
                "source": "ThreatLens knowledge graph",
                "known_exploited": False,
            }

        return {
            "status": "UNKNOWN",
            "reason": "KEV status is unavailable.",
        }

    def get_cwe_evidence(self, cve_id: str) -> dict[str, Any]:
        node = self._get_cve_node(cve_id)

        if node is None:
            return {
                "status": "UNKNOWN",
                "reason": "CVE is not present in the current graph.",
            }

        cwe_nodes = self._related_nodes(
            node["node_id"],
            "HAS_CWE",
            "CWE",
        )

        cwes = sorted(
            {
                self._clean(item.get("label"))
                for item in cwe_nodes
                if self._clean(item.get("label"))
            }
        )

        return {
            "status": "KNOWN",
            "source": "ThreatLens knowledge graph",
            "cwes": cwes,
            "count": len(cwes),
        }

    def get_affected_products(self, cve_id: str) -> dict[str, Any]:
        node = self._get_cve_node(cve_id)

        if node is None:
            return {
                "status": "UNKNOWN",
                "reason": "CVE is not present in the current graph.",
            }

        cpe_nodes = self._related_nodes(
            node["node_id"],
            "AFFECTS",
            "CPE",
        )

        products = sorted(
            {
                self._clean(item.get("label"))
                for item in cpe_nodes
                if self._clean(item.get("label"))
            }
        )

        return {
            "status": "KNOWN",
            "source": "ThreatLens knowledge graph",
            "products": products,
            "count": len(products),
        }

    def get_references(self, cve_id: str) -> dict[str, Any]:
        node = self._get_cve_node(cve_id)

        if node is None:
            return {
                "status": "UNKNOWN",
                "reason": "CVE is not present in the current graph.",
            }

        reference_nodes = self._related_nodes(
            node["node_id"],
            "REFERENCES",
            "Reference",
        )

        references = []

        for item in reference_nodes:
            references.append(
                {
                    "label": self._clean(item.get("label")),
                    "url": self._clean(item.get("url")),
                }
            )

        references.sort(
            key=lambda x: (
                x.get("url") or "",
                x.get("label") or "",
            )
        )

        return {
            "status": "KNOWN",
            "source": "ThreatLens knowledge graph",
            "references": references,
            "count": len(references),
        }

    def search(self, cve_id: str) -> dict[str, Any]:
        normalized = self._normalize_cve(cve_id)
        node = self._get_cve_node(normalized)

        if node is None:
            return {
                "cve_id": normalized,
                "status": "UNKNOWN",
                "evidence": {},
                "summary": {
                    "known": 0,
                    "inferred": 0,
                    "unknown": 1,
                },
                "reason": (
                    "CVE is not present in the current "
                    "ThreatLens knowledge graph."
                ),
            }

        evidence = {
            "metadata": self.get_metadata(normalized),
            "risk": self.get_risk_evidence(normalized),
            "kev": self.get_kev_evidence(normalized),
            "cwe": self.get_cwe_evidence(normalized),
            "affected_products": self.get_affected_products(normalized),
            "references": self.get_references(normalized),
        }

        known = sum(
            1
            for item in evidence.values()
            if item.get("status") == "KNOWN"
        )

        inferred = sum(
            1
            for item in evidence.values()
            if item.get("status") == "INFERRED"
        )

        unknown = sum(
            1
            for item in evidence.values()
            if item.get("status") == "UNKNOWN"
        )

        return {
            "cve_id": normalized,
            "status": "KNOWN",
            "evidence": evidence,
            "summary": {
                "known": known,
                "inferred": inferred,
                "unknown": unknown,
            },
            "graph_source": {
                "nodes": str(NODES_PATH),
                "edges": str(EDGES_PATH),
            },
        }


def run_tests(engine: EvidenceEngine) -> dict[str, Any]:
    cve_nodes = engine.nodes[
        engine.nodes["node_type"] == "CVE"
    ]

    if cve_nodes.empty:
        raise RuntimeError("No CVE nodes exist in the graph.")

    # Test a CVE known to exist in the graph.
    known_cve = str(cve_nodes.iloc[0]["label"])

    known = engine.search(known_cve)
    unknown = engine.search("CVE-0000-0000")

    tests = {
        "nodes_loaded": len(engine.nodes) == 5813,
        "edges_loaded": len(engine.edges) == 9719,
        "cve_nodes_loaded": len(cve_nodes) == 500,
        "known_cve_lookup": known["status"] == "KNOWN",
        "risk_evidence": (
            known["evidence"]["risk"]["status"] == "KNOWN"
        ),
        "kev_evidence": (
            known["evidence"]["kev"]["status"] in {"KNOWN", "UNKNOWN"}
        ),
        "cwe_evidence": (
            known["evidence"]["cwe"]["status"] == "KNOWN"
        ),
        "product_evidence": (
            known["evidence"]["affected_products"]["status"] == "KNOWN"
        ),
        "reference_evidence": (
            known["evidence"]["references"]["status"] == "KNOWN"
        ),
        "unknown_cve_handling": (
            unknown["status"] == "UNKNOWN"
        ),
        "no_inferred_claims": (
            known["summary"]["inferred"] == 0
        ),
    }

    passed = sum(tests.values())
    total = len(tests)

    result = {
        "project": "ThreatLens",
        "component": "Evidence Engine",
        "version": "1.0",
        "status": "PASS" if passed == total else "FAIL",
        "tests_passed": passed,
        "tests_total": total,
        "known_test_cve": known_cve,
        "tests": tests,
        "known_cve_result": known,
        "unknown_cve_result": unknown,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    TEST_OUTPUT.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ThreatLens Evidence Engine"
    )

    parser.add_argument(
        "--cve",
        help="Lookup a CVE, e.g. CVE-2025-9242",
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run deterministic Evidence Engine tests",
    )

    args = parser.parse_args()

    engine = EvidenceEngine()

    if args.test:
        result = run_tests(engine)

        print(
            f"Evidence Engine: {result['status']} "
            f"({result['tests_passed']}/{result['tests_total']})"
        )
        print(f"Test CVE: {result['known_test_cve']}")
        print(f"Output: {TEST_OUTPUT}")

        if result["status"] != "PASS":
            raise SystemExit(1)

        return

    if args.cve:
        print(
            json.dumps(
                engine.search(args.cve),
                indent=2,
            )
        )
        return

    parser.print_help()


if __name__ == "__main__":
    main()
