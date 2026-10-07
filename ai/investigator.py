from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evidence.evidence_engine import EvidenceEngine


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ai_investigator"
)

TEST_OUTPUT = OUTPUT_DIR / "investigator_test_results.json"


class AIInvestigator:
    """
    Evidence-grounded investigator core.

    This component converts Evidence Engine output into a structured
    investigation brief. It does not calculate a new risk score and
    does not invent unsupported security facts.
    """

    VERSION = "1.0"

    def __init__(self, evidence_engine: EvidenceEngine | None = None):
        self.engine = evidence_engine or EvidenceEngine()

    @staticmethod
    def _safe(value: Any) -> Any:
        if value is None:
            return None

        if isinstance(value, str):
            value = value.strip()

        return value if value != "" else None

    def investigate(self, cve_id: str) -> dict[str, Any]:
        evidence_package = self.engine.search(cve_id)

        if evidence_package.get("status") == "UNKNOWN":
            return {
                "investigator_version": self.VERSION,
                "cve_id": evidence_package.get("cve_id"),
                "status": "UNKNOWN",
                "finding": "Insufficient evidence",
                "confidence": "UNKNOWN",
                "facts": [],
                "evidence": evidence_package,
                "limitations": [
                    "The CVE is not present in the current ThreatLens graph."
                ],
            }

        evidence = evidence_package["evidence"]

        metadata = evidence["metadata"]
        risk = evidence["risk"]
        kev = evidence["kev"]
        cwe = evidence["cwe"]
        products = evidence["affected_products"]
        references = evidence["references"]

        facts: list[dict[str, Any]] = []

        # FACT 1 — ThreatLens risk.
        if risk.get("status") == "KNOWN":
            facts.append(
                {
                    "id": "F1",
                    "type": "THREATLENS_RISK",
                    "statement": (
                        f"ThreatLens assigned a risk score of "
                        f"{risk['score']:.6f} and category "
                        f"{risk['risk_category']}."
                    ),
                    "source": risk.get("source"),
                    "evidence_status": "KNOWN",
                }
            )

        # FACT 2 — Severity.
        if metadata.get("severity"):
            facts.append(
                {
                    "id": "F2",
                    "type": "SEVERITY",
                    "statement": (
                        f"The recorded vulnerability severity is "
                        f"{metadata['severity']}."
                    ),
                    "source": "ThreatLens knowledge graph",
                    "evidence_status": "KNOWN",
                }
            )

        # FACT 3 — KEV.
        if kev.get("status") == "KNOWN":
            if kev.get("known_exploited") is True:
                statement = (
                    "The CVE is marked as known exploited in the "
                    "CISA Known Exploited Vulnerabilities evidence "
                    "available to ThreatLens."
                )

                if kev.get("kev_date_added"):
                    statement += (
                        f" The graph records a KEV date of "
                        f"{kev['kev_date_added']}."
                    )

                facts.append(
                    {
                        "id": "F3",
                        "type": "KEV_STATUS",
                        "statement": statement,
                        "source": kev.get("source"),
                        "evidence_status": "KNOWN",
                    }
                )

            elif kev.get("known_exploited") is False:
                facts.append(
                    {
                        "id": "F3",
                        "type": "KEV_STATUS",
                        "statement": (
                            "The current ThreatLens graph does not "
                            "mark this CVE as known exploited."
                        ),
                        "source": kev.get("source"),
                        "evidence_status": "KNOWN",
                    }
                )

        # FACT 4 — CWE.
        cwe_values = cwe.get("cwes", [])

        if cwe_values:
            facts.append(
                {
                    "id": "F4",
                    "type": "CWE",
                    "statement": (
                        "The graph associates this CVE with "
                        + ", ".join(cwe_values)
                        + "."
                    ),
                    "source": cwe.get("source"),
                    "evidence_status": "KNOWN",
                }
            )

        # FACT 5 — Products.
        product_count = products.get("count", 0)

        if product_count:
            facts.append(
                {
                    "id": "F5",
                    "type": "AFFECTED_PRODUCTS",
                    "statement": (
                        f"The graph contains {product_count} affected "
                        "CPE entries for this CVE."
                    ),
                    "source": products.get("source"),
                    "evidence_status": "KNOWN",
                    "count": product_count,
                }
            )

        # FACT 6 — References.
        reference_count = references.get("count", 0)

        facts.append(
            {
                "id": "F6",
                "type": "REFERENCES",
                "statement": (
                    f"The graph contains {reference_count} reference "
                    "entries associated with this CVE."
                ),
                "source": references.get("source"),
                "evidence_status": "KNOWN",
                "count": reference_count,
            }
        )

        # Deterministic finding.
        risk_category = risk.get("risk_category")

        if kev.get("known_exploited") is True:
            finding = (
                f"{risk_category} ThreatLens risk with CISA KEV "
                "exploitation evidence."
            )
        else:
            finding = (
                f"{risk_category} ThreatLens risk based on the "
                "available vulnerability evidence."
            )

        # Confidence is intentionally qualitative and evidence-based.
        # It is NOT a model probability.
        known_count = evidence_package["summary"]["known"]
        total_count = sum(evidence_package["summary"].values())

        if total_count and known_count == total_count:
            confidence = "HIGH_EVIDENCE_COVERAGE"
        elif known_count >= 4:
            confidence = "MODERATE_EVIDENCE_COVERAGE"
        else:
            confidence = "LIMITED_EVIDENCE_COVERAGE"

        limitations = [
            "ThreatLens risk score is reproduced from the production "
            "prediction and is not recalculated by the Investigator.",
            "CISA KEV status is treated as an exploitation signal, "
            "not as proof of every exploitation detail.",
            "CPE entries represent graph-linked affected-product "
            "identifiers; they are not independently validated "
            "installation inventories.",
            "References are evidence pointers and are not treated as "
            "verified claims beyond the graph data.",
        ]

        return {
            "investigator_version": self.VERSION,
            "cve_id": evidence_package["cve_id"],
            "status": "KNOWN",
            "finding": finding,
            "confidence": confidence,
            "risk": {
                "score": risk.get("score"),
                "category": risk.get("risk_category"),
                "source": risk.get("source"),
            },
            "facts": facts,
            "evidence": evidence_package,
            "limitations": limitations,
        }


def validate_investigation(result: dict[str, Any]) -> dict[str, Any]:
    tests: dict[str, bool] = {}

    tests["result_is_dict"] = isinstance(result, dict)

    tests["status_present"] = result.get("status") in {
        "KNOWN",
        "UNKNOWN",
    }

    tests["cve_present"] = bool(result.get("cve_id"))

    tests["risk_present"] = (
        isinstance(result.get("risk"), dict)
        and result["risk"].get("category") is not None
    )

    tests["risk_score_preserved"] = (
        isinstance(result.get("risk"), dict)
        and isinstance(result["risk"].get("score"), (int, float))
        and 0.0 <= float(result["risk"]["score"]) <= 1.0
    )

    tests["facts_are_grounded"] = all(
        fact.get("evidence_status") == "KNOWN"
        for fact in result.get("facts", [])
    )

    tests["no_inferred_facts"] = not any(
        fact.get("evidence_status") == "INFERRED"
        for fact in result.get("facts", [])
    )

    tests["limitations_present"] = (
        isinstance(result.get("limitations"), list)
        and len(result["limitations"]) > 0
    )

    tests["confidence_not_probability"] = (
        result.get("confidence")
        in {
            "HIGH_EVIDENCE_COVERAGE",
            "MODERATE_EVIDENCE_COVERAGE",
            "LIMITED_EVIDENCE_COVERAGE",
            "UNKNOWN",
        }
    )

    passed = sum(tests.values())
    total = len(tests)

    return {
        "status": "PASS" if passed == total else "FAIL",
        "tests_passed": passed,
        "tests_total": total,
        "tests": tests,
    }


def run_tests() -> dict[str, Any]:
    investigator = AIInvestigator()

    known_cve = "CVE-2025-9242"

    known_result = investigator.investigate(known_cve)
    known_validation = validate_investigation(known_result)

    unknown_result = investigator.investigate("CVE-0000-0000")

    unknown_tests = {
        "unknown_status": unknown_result.get("status") == "UNKNOWN",
        "unknown_finding": (
            unknown_result.get("finding") == "Insufficient evidence"
        ),
        "unknown_has_limitations": (
            isinstance(unknown_result.get("limitations"), list)
            and len(unknown_result["limitations"]) > 0
        ),
    }

    all_tests = {
        "known_investigation": known_validation["status"] == "PASS",
        **{
            f"known_{key}": value
            for key, value in known_validation["tests"].items()
        },
        **unknown_tests,
    }

    passed = sum(all_tests.values())
    total = len(all_tests)

    result = {
        "project": "ThreatLens",
        "component": "AI Investigator Core",
        "version": AIInvestigator.VERSION,
        "status": "PASS" if passed == total else "FAIL",
        "tests_passed": passed,
        "tests_total": total,
        "known_test_cve": known_cve,
        "tests": all_tests,
        "known_result": known_result,
        "unknown_result": unknown_result,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    TEST_OUTPUT.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ThreatLens AI Investigator"
    )

    parser.add_argument(
        "--cve",
        help="Investigate a CVE",
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run deterministic investigator tests",
    )

    args = parser.parse_args()

    if args.test:
        result = run_tests()

        print(
            f"AI Investigator: {result['status']} "
            f"({result['tests_passed']}/{result['tests_total']})"
        )

        print(f"Test CVE: {result['known_test_cve']}")
        print(f"Output: {TEST_OUTPUT}")

        if result["status"] != "PASS":
            raise SystemExit(1)

        return

    investigator = AIInvestigator()

    if args.cve:
        result = investigator.investigate(args.cve)
        print(json.dumps(result, indent=2))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
