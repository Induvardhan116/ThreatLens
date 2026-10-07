from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests


NVD_API_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "nvd"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def fetch_nvd_cves(
    start_index: int = 0,
    results_per_page: int = 20,
    timeout: int = 60,
) -> dict[str, Any]:
    """
    Fetch a controlled page of CVE records from NVD API 2.0.

    This function intentionally supports pagination so that the same
    ingestion code can later be used for larger datasets.
    """

    if start_index < 0:
        raise ValueError("start_index must be >= 0")

    if not 1 <= results_per_page <= 2000:
        raise ValueError("results_per_page must be between 1 and 2000")

    params = {
        "startIndex": start_index,
        "resultsPerPage": results_per_page,
    }

    response = requests.get(
        NVD_API_URL,
        params=params,
        timeout=timeout,
    )

    response.raise_for_status()

    return response.json()


def save_raw_response(
    data: dict[str, Any],
    filename: str = "sample.json",
) -> Path:
    """
    Save the raw NVD response exactly as received.
    """

    output_path = RAW_DATA_DIR / filename

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

    return output_path


def extract_cve_records(
    nvd_response: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Extract the raw CVE objects from an NVD API response.

    No normalization happens here. This layer preserves source data.
    """

    vulnerabilities = nvd_response.get("vulnerabilities", [])

    records: list[dict[str, Any]] = []

    for item in vulnerabilities:
        cve = item.get("cve")

        if isinstance(cve, dict):
            records.append(cve)

    return records


def main() -> None:
    print("ThreatLens - NVD ingestion")
    print("=" * 40)

    print("Requesting 20 CVE records from NVD API 2.0...")

    response_data = fetch_nvd_cves(
        start_index=0,
        results_per_page=20,
    )

    output_path = save_raw_response(
        response_data,
        filename="sample.json",
    )

    records = extract_cve_records(response_data)

    print()
    print(f"Total results reported by NVD: {response_data.get('totalResults')}")
    print(f"Records retrieved: {len(records)}")
    print(f"Raw response saved to: {output_path}")

    if records:
        first_cve = records[0]

        print()
        print("First CVE:")
        print(f"  ID: {first_cve.get('id')}")
        print(f"  Published: {first_cve.get('published')}")
        print(f"  Modified: {first_cve.get('lastModified')}")


if __name__ == "__main__":
    main()
