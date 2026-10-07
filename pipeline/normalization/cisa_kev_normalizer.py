from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from pipeline.schema import VulnerabilityRecord


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "cisa_kev"
    / "known_exploited_vulnerabilities.json"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "cisa_kev"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "kev_normalized.json"


def parse_date(value: Any) -> date | None:
    """Parse a CISA date string."""

    if not value:
        return None

    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def load_catalog() -> dict[str, Any]:
    """Load the raw CISA KEV catalog."""

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("Invalid CISA KEV catalog format.")

    return data


def normalize_record(item: dict[str, Any]) -> VulnerabilityRecord:
    """Convert one CISA KEV record to the canonical schema."""

    cve_id = item["cveID"]

    known_ransomware = item.get("knownRansomwareCampaignUse")

    if isinstance(known_ransomware, str):
        known_ransomware = known_ransomware.strip().lower()

        if known_ransomware == "known":
            ransomware_use = True
        elif known_ransomware == "unknown":
            ransomware_use = None
        else:
            ransomware_use = None
    else:
        ransomware_use = None

    return VulnerabilityRecord(
        cve_id=cve_id,
        description=item.get("shortDescription"),
        known_exploited=True,
        kev_date_added=parse_date(item.get("dateAdded")),
        kev_due_date=parse_date(item.get("dueDate")),
        kev_vendor_project=item.get("vendorProject"),
        kev_product=item.get("product"),
        kev_vulnerability_name=item.get("vulnerabilityName"),
        kev_required_action=item.get("requiredAction"),
        ransomware_use=ransomware_use,
    )


def main() -> None:

    print("ThreatLens - CISA KEV normalization")
    print("=" * 45)

    data = load_catalog()

    vulnerabilities = data.get("vulnerabilities", [])

    if not isinstance(vulnerabilities, list):
        raise ValueError(
            "CISA KEV catalog contains no valid vulnerabilities list."
        )

    normalized: list[dict[str, Any]] = []

    failures = 0

    for item in vulnerabilities:

        if not isinstance(item, dict):
            failures += 1
            continue

        try:
            record = normalize_record(item)
            normalized.append(record.model_dump(mode="json"))

        except Exception as exc:
            failures += 1
            print(
                f"Normalization failed for "
                f"{item.get('cveID', 'UNKNOWN')}: {exc}"
            )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            normalized,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("Catalog records:", len(vulnerabilities))
    print("Successfully normalized:", len(normalized))
    print("Normalization failures:", failures)

    print()
    print("Output:", OUTPUT_FILE)

    if normalized:

        first = normalized[0]

        print()
        print("Example normalized record")
        print("-" * 45)
        print("CVE:", first["cve_id"])
        print("Known exploited:", first["known_exploited"])
        print("Vendor:", first["kev_vendor_project"])
        print("Product:", first["kev_product"])
        print("KEV date added:", first["kev_date_added"])
        print("Due date:", first["kev_due_date"])
        print("Ransomware use:", first["ransomware_use"])


if __name__ == "__main__":
    main()
