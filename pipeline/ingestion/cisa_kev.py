from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests


CISA_KEV_URL = (
    "https://www.cisa.gov/sites/default/files/feeds/"
    "known_exploited_vulnerabilities.json"
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "cisa_kev"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

RAW_OUTPUT_FILE = RAW_DATA_DIR / "known_exploited_vulnerabilities.json"


def fetch_kev_catalog(timeout: int = 60) -> dict[str, Any]:
    """Download the official CISA KEV catalog."""

    response = requests.get(
        CISA_KEV_URL,
        timeout=timeout,
    )

    response.raise_for_status()

    data = response.json()

    if not isinstance(data, dict):
        raise ValueError("CISA KEV response is not a JSON object.")

    return data


def validate_catalog(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate and return the vulnerability records."""

    vulnerabilities = data.get("vulnerabilities")

    if not isinstance(vulnerabilities, list):
        raise ValueError(
            "CISA KEV catalog does not contain a valid "
            "'vulnerabilities' list."
        )

    valid_records: list[dict[str, Any]] = []

    for item in vulnerabilities:

        if not isinstance(item, dict):
            continue

        cve_id = item.get("cveID")

        if not isinstance(cve_id, str):
            continue

        if not cve_id.startswith("CVE-"):
            continue

        valid_records.append(item)

    return valid_records


def save_raw_catalog(data: dict[str, Any]) -> None:
    """Save the original CISA response."""

    with RAW_OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )


def main() -> None:

    print("ThreatLens - CISA KEV ingestion")
    print("=" * 45)

    print("Downloading official CISA KEV catalog...")

    data = fetch_kev_catalog()

    records = validate_catalog(data)

    save_raw_catalog(data)

    print()
    print(
        "Catalog version:",
        data.get("catalogVersion"),
    )

    print(
        "Catalog date:",
        data.get("dateReleased"),
    )

    print(
        "Total vulnerabilities:",
        len(records),
    )

    print(
        "Raw catalog saved to:",
        RAW_OUTPUT_FILE,
    )

    if records:

        first = records[0]

        print()
        print("First KEV record")
        print("-" * 45)
        print("CVE:", first.get("cveID"))
        print("Vendor:", first.get("vendorProject"))
        print("Product:", first.get("product"))
        print("Vulnerability:", first.get("vulnerabilityName"))
        print("Date added:", first.get("dateAdded"))
        print("Due date:", first.get("dueDate"))
        print("Ransomware use:", first.get("knownRansomwareCampaignUse"))


if __name__ == "__main__":
    main()
