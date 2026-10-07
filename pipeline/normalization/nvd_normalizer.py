from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pipeline.schema import VulnerabilityRecord


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_NVD_FILE = PROJECT_ROOT / "data" / "raw" / "nvd" / "sample.json"

PROCESSED_NVD_DIR = PROJECT_ROOT / "data" / "processed" / "nvd"
PROCESSED_NVD_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = PROCESSED_NVD_DIR / "sample_normalized.json"


def parse_datetime(value: str | None) -> datetime | None:
    """Convert an NVD timestamp into a Python datetime."""

    if not value:
        return None

    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def get_description(cve: dict[str, Any]) -> str | None:
    """Return the English CVE description when available."""

    descriptions = cve.get("descriptions", [])

    for item in descriptions:
        if item.get("lang") == "en":
            value = item.get("value")

            if isinstance(value, str):
                return value.strip()

    return None


def extract_cvss(cve: dict[str, Any]) -> dict[str, Any]:
    """
    Extract the preferred CVSS metrics.

    NVD records may contain CVSS v4, v3.x, or older metrics.
    We preserve v4 and v3.x separately rather than pretending
    that every vulnerability has the same scoring version.
    """

    result: dict[str, Any] = {
        "cvss_v4_score": None,
        "cvss_v4_vector": None,
        "cvss_v3_score": None,
        "cvss_v3_vector": None,
        "severity": None,
    }

    metrics = cve.get("metrics", {})

    # --------------------------------------------------------------
    # CVSS v4
    # --------------------------------------------------------------

    cvss_v4_entries = metrics.get("cvssMetricV40", [])

    if cvss_v4_entries:
        metric = cvss_v4_entries[0]

        cvss_data = metric.get("cvssData", {})

        result["cvss_v4_score"] = cvss_data.get("baseScore")
        result["cvss_v4_vector"] = cvss_data.get("vectorString")
        result["severity"] = (
            cvss_data.get("baseSeverity")
            or metric.get("baseSeverity")
        )

    # --------------------------------------------------------------
    # CVSS v3.1 / v3.0
    # --------------------------------------------------------------

    if result["cvss_v3_score"] is None:

        for metric_key in ("cvssMetricV31", "cvssMetricV30"):

            entries = metrics.get(metric_key, [])

            if not entries:
                continue

            metric = entries[0]
            cvss_data = metric.get("cvssData", {})

            result["cvss_v3_score"] = cvss_data.get("baseScore")
            result["cvss_v3_vector"] = cvss_data.get("vectorString")

            if result["severity"] is None:
                result["severity"] = (
                    cvss_data.get("baseSeverity")
                    or metric.get("baseSeverity")
                )

            break

    return result


def extract_cwes(cve: dict[str, Any]) -> list[str]:
    """Extract CWE identifiers from problem type information."""

    cwe_ids: list[str] = []

    problem_types = cve.get("weaknesses", [])

    for problem_type in problem_types:

        descriptions = problem_type.get("description", [])

        for description in descriptions:

            value = description.get("value")

            if isinstance(value, str) and value.startswith("CWE-"):
                cwe_ids.append(value)

    return sorted(set(cwe_ids))


def extract_references(cve: dict[str, Any]) -> list[str]:
    """Extract reference URLs."""

    references: list[str] = []

    for reference in cve.get("references", []):

        url = reference.get("url")

        if isinstance(url, str):
            references.append(url)

    return sorted(set(references))


def extract_cpes(cve: dict[str, Any]) -> list[str]:
    """
    Extract CPE names from NVD configurations.

    This intentionally extracts the CPE identifiers without attempting
    to interpret product/version ranges yet. That will be handled by
    a later normalization stage.
    """

    cpes: set[str] = set()

    configurations = cve.get("configurations", [])

    def walk_node(node: dict[str, Any]) -> None:

        for match in node.get("cpeMatch", []):

            criteria = match.get("criteria")

            if isinstance(criteria, str):
                cpes.add(criteria)

        for child in node.get("children", []):
            if isinstance(child, dict):
                walk_node(child)

    for configuration in configurations:

        nodes = configuration.get("nodes", [])

        for node in nodes:

            if isinstance(node, dict):
                walk_node(node)

    return sorted(cpes)


def normalize_cve(cve: dict[str, Any]) -> VulnerabilityRecord:
    """Convert one raw NVD CVE object into a ThreatLens record."""

    cve_id = cve.get("id")

    if not isinstance(cve_id, str):
        raise ValueError("NVD CVE record is missing a valid CVE ID")

    description = get_description(cve)

    cvss = extract_cvss(cve)

    cwe_ids = extract_cwes(cve)

    references = extract_references(cve)

    affected_cpes = extract_cpes(cve)

    published_date = parse_datetime(cve.get("published"))
    modified_date = parse_datetime(cve.get("lastModified"))

    description_length = len(description) if description else 0

    return VulnerabilityRecord(
        cve_id=cve_id,
        description=description,
        published_date=published_date,
        last_modified_date=modified_date,
        cvss_v4_score=cvss["cvss_v4_score"],
        cvss_v4_vector=cvss["cvss_v4_vector"],
        cvss_v3_score=cvss["cvss_v3_score"],
        cvss_v3_vector=cvss["cvss_v3_vector"],
        severity=cvss["severity"],
        cwe_ids=cwe_ids,
        affected_cpes=affected_cpes,
        references=references,
        reference_count=len(references),
        affected_cpe_count=len(affected_cpes),
        cwe_count=len(cwe_ids),
        description_length=description_length,
    )


def load_raw_records() -> list[dict[str, Any]]:
    """Load CVE objects from the raw NVD response."""

    if not RAW_NVD_FILE.exists():
        raise FileNotFoundError(
            f"NVD raw file not found: {RAW_NVD_FILE}"
        )

    with RAW_NVD_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)

    vulnerabilities = data.get("vulnerabilities", [])

    records: list[dict[str, Any]] = []

    for item in vulnerabilities:

        cve = item.get("cve")

        if isinstance(cve, dict):
            records.append(cve)

    return records


def save_normalized_records(
    records: list[VulnerabilityRecord],
) -> None:
    """Save normalized ThreatLens records."""

    serializable = [
        record.model_dump(mode="json")
        for record in records
    ]

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            serializable,
            file,
            indent=2,
            ensure_ascii=False,
        )


def main() -> None:

    print("ThreatLens - NVD normalization")
    print("=" * 45)

    raw_records = load_raw_records()

    print(f"Raw CVE records loaded: {len(raw_records)}")

    normalized_records: list[VulnerabilityRecord] = []

    failures = 0

    for cve in raw_records:

        try:
            record = normalize_cve(cve)
            normalized_records.append(record)

        except Exception as exc:

            failures += 1

            print(
                f"Failed to normalize "
                f"{cve.get('id', 'UNKNOWN')}: {exc}"
            )

    save_normalized_records(normalized_records)

    print()
    print(f"Successfully normalized: {len(normalized_records)}")
    print(f"Normalization failures: {failures}")
    print(f"Output: {OUTPUT_FILE}")

    if normalized_records:

        first = normalized_records[0]

        print()
        print("Example normalized record")
        print("-" * 45)
        print(f"CVE: {first.cve_id}")
        print(f"Description: {first.description}")
        print(f"CVSS v4: {first.cvss_v4_score}")
        print(f"CVSS v3: {first.cvss_v3_score}")
        print(f"Severity: {first.severity}")
        print(f"CWEs: {first.cwe_ids}")
        print(f"CPE count: {first.affected_cpe_count}")
        print(f"Reference count: {first.reference_count}")


if __name__ == "__main__":
    main()
