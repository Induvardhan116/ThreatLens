from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline.schema import VulnerabilityRecord


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "nvd"
    / "research"
    / "nvd_2024_2026.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nvd"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "research_normalized.json"
)


def parse_datetime(
    value: Any,
) -> datetime | None:

    if not value:
        return None

    try:

        parsed = datetime.fromisoformat(
            str(value).replace(
                "Z",
                "+00:00",
            )
        )

        if parsed.tzinfo is None:

            parsed = parsed.replace(
                tzinfo=timezone.utc
            )

        return parsed

    except ValueError:

        return None


def extract_description(
    cve: dict[str, Any],
) -> str | None:

    descriptions = cve.get(
        "descriptions",
        [],
    )

    if not isinstance(
        descriptions,
        list,
    ):
        return None

    english_description = None
    fallback_description = None

    for item in descriptions:

        if not isinstance(
            item,
            dict,
        ):
            continue

        value = item.get(
            "value"
        )

        if not isinstance(
            value,
            str,
        ):
            continue

        lang = item.get(
            "lang"
        )

        if lang == "en":

            english_description = value
            break

        if fallback_description is None:

            fallback_description = value

    return (
        english_description
        or fallback_description
    )


def extract_cvss(
    cve: dict[str, Any],
) -> tuple[
    float | None,
    str | None,
    float | None,
    str | None,
    str | None,
]:

    metrics = cve.get(
        "metrics",
        {}
    )

    if not isinstance(
        metrics,
        dict,
    ):
        return (
            None,
            None,
            None,
            None,
            None,
        )

    cvss_v4_score = None
    cvss_v4_vector = None

    cvss_v3_score = None
    cvss_v3_vector = None

    severity = None

    # Prefer CVSS v4.
    v4_metrics = metrics.get(
        "cvssMetricV40",
        []
    )

    if isinstance(
        v4_metrics,
        list,
    ):

        for metric in v4_metrics:

            if not isinstance(
                metric,
                dict,
            ):
                continue

            cvss_data = metric.get(
                "cvssData",
                {}
            )

            if not isinstance(
                cvss_data,
                dict,
            ):
                continue

            score = cvss_data.get(
                "baseScore"
            )

            vector = cvss_data.get(
                "vectorString"
            )

            if (
                isinstance(score, (int, float))
                and 0 <= score <= 10
            ):

                cvss_v4_score = float(
                    score
                )

            if isinstance(
                vector,
                str,
            ):

                cvss_v4_vector = vector

            severity_value = (
                cvss_data.get(
                    "baseSeverity"
                )
            )

            if isinstance(
                severity_value,
                str,
            ):

                severity = (
                    severity_value.upper()
                )

            if cvss_v4_score is not None:
                break

    # Fall back to CVSS v3.1 / v3.0.
    v3_metrics = metrics.get(
        "cvssMetricV31",
        []
    )

    if not v3_metrics:

        v3_metrics = metrics.get(
            "cvssMetricV30",
            []
        )

    if isinstance(
        v3_metrics,
        list,
    ):

        for metric in v3_metrics:

            if not isinstance(
                metric,
                dict,
            ):
                continue

            cvss_data = metric.get(
                "cvssData",
                {}
            )

            if not isinstance(
                cvss_data,
                dict,
            ):
                continue

            score = cvss_data.get(
                "baseScore"
            )

            vector = cvss_data.get(
                "vectorString"
            )

            if (
                isinstance(score, (int, float))
                and 0 <= score <= 10
            ):

                cvss_v3_score = float(
                    score
                )

            if isinstance(
                vector,
                str,
            ):

                cvss_v3_vector = vector

            if severity is None:

                severity_value = (
                    cvss_data.get(
                        "baseSeverity"
                    )
                )

                if isinstance(
                    severity_value,
                    str,
                ):

                    severity = (
                        severity_value.upper()
                    )

            if cvss_v3_score is not None:
                break

    return (
        cvss_v4_score,
        cvss_v4_vector,
        cvss_v3_score,
        cvss_v3_vector,
        severity,
    )


def extract_cwes(
    cve: dict[str, Any],
) -> list[str]:

    weaknesses = cve.get(
        "weaknesses",
        []
    )

    cwe_ids: set[str] = set()

    if not isinstance(
        weaknesses,
        list,
    ):
        return []

    for weakness in weaknesses:

        if not isinstance(
            weakness,
            dict,
        ):
            continue

        descriptions = weakness.get(
            "description",
            []
        )

        if not isinstance(
            descriptions,
            list,
        ):
            continue

        for item in descriptions:

            if not isinstance(
                item,
                dict,
            ):
                continue

            value = item.get(
                "value"
            )

            if (
                isinstance(value, str)
                and value.startswith("CWE-")
            ):

                cwe_ids.add(
                    value
                )

    return sorted(
        cwe_ids
    )


def extract_references(
    cve: dict[str, Any],
) -> list[str]:

    references = cve.get(
        "references",
        []
    )

    output: set[str] = set()

    if not isinstance(
        references,
        list,
    ):
        return []

    for reference in references:

        if not isinstance(
            reference,
            dict,
        ):
            continue

        url = reference.get(
            "url"
        )

        if isinstance(
            url,
            str,
        ):

            output.add(
                url
            )

    return sorted(
        output
    )


def extract_cpes(
    node: Any,
) -> set[str]:

    results: set[str] = set()

    if not isinstance(
        node,
        dict,
    ):
        return results

    cpe_match = node.get(
        "cpeMatch",
        []
    )

    if isinstance(
        cpe_match,
        list,
    ):

        for match in cpe_match:

            if not isinstance(
                match,
                dict,
            ):
                continue

            criteria = match.get(
                "criteria"
            )

            if isinstance(
                criteria,
                str,
            ):

                results.add(
                    criteria
                )

    children = node.get(
        "children",
        []
    )

    if isinstance(
        children,
        list,
    ):

        for child in children:

            results.update(
                extract_cpes(
                    child
                )
            )

    return results


def extract_affected_cpes(
    cve: dict[str, Any],
) -> list[str]:

    configurations = cve.get(
        "configurations",
        []
    )

    cpes: set[str] = set()

    if not isinstance(
        configurations,
        list,
    ):
        return []

    for configuration in configurations:

        if not isinstance(
            configuration,
            dict,
        ):
            continue

        nodes = configuration.get(
            "nodes",
            []
        )

        if not isinstance(
            nodes,
            list,
        ):
            continue

        for node in nodes:

            cpes.update(
                extract_cpes(
                    node
                )
            )

    return sorted(
        cpes
    )


def calculate_days_since_publication(
    published_date: datetime | None,
) -> int | None:

    if published_date is None:
        return None

    now = datetime.now(
        timezone.utc
    )

    difference = (
        now - published_date
    )

    days = difference.days

    if days < 0:
        return 0

    return days


def normalize_cve(
    item: dict[str, Any],
) -> VulnerabilityRecord:

    cve = item.get(
        "cve"
    )

    if not isinstance(
        cve,
        dict,
    ):

        raise ValueError(
            "Missing CVE object."
        )

    cve_id = cve.get(
        "id"
    )

    if not isinstance(
        cve_id,
        str,
    ):

        raise ValueError(
            "Missing CVE ID."
        )

    description = (
        extract_description(
            cve
        )
    )

    published_date = (
        parse_datetime(
            cve.get(
                "published"
            )
        )
    )

    last_modified_date = (
        parse_datetime(
            cve.get(
                "lastModified"
            )
        )
    )

    (
        cvss_v4_score,
        cvss_v4_vector,
        cvss_v3_score,
        cvss_v3_vector,
        severity,
    ) = extract_cvss(
        cve
    )

    cwe_ids = extract_cwes(
        cve
    )

    references = extract_references(
        cve
    )

    affected_cpes = (
        extract_affected_cpes(
            cve
        )
    )

    description_length = (
        len(description)
        if description
        else 0
    )

    return VulnerabilityRecord(
        cve_id=cve_id,
        description=description,
        published_date=published_date,
        last_modified_date=last_modified_date,
        cvss_v4_score=cvss_v4_score,
        cvss_v4_vector=cvss_v4_vector,
        cvss_v3_score=cvss_v3_score,
        cvss_v3_vector=cvss_v3_vector,
        severity=severity,
        cwe_ids=cwe_ids,
        affected_cpes=affected_cpes,
        references=references,
        reference_count=len(
            references
        ),
        affected_cpe_count=len(
            affected_cpes
        ),
        cwe_count=len(
            cwe_ids
        ),
        description_length=(
            description_length
        ),
        days_since_publication=(
            calculate_days_since_publication(
                published_date
            )
        ),
    )


def main() -> None:

    print(
        "ThreatLens - NVD research "
        "normalization"
    )

    print(
        "=" * 55
    )

    print()
    print(
        "Loading research dataset..."
    )

    with INPUT_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(
            file
        )

    vulnerabilities = data.get(
        "vulnerabilities",
        []
    )

    if not isinstance(
        vulnerabilities,
        list,
    ):

        raise ValueError(
            "Invalid NVD dataset."
        )

    print(
        "Raw NVD records:",
        len(vulnerabilities),
    )

    normalized: list[
        dict[str, Any]
    ] = []

    failures = 0

    for index, item in enumerate(
        vulnerabilities,
        start=1,
    ):

        try:

            record = normalize_cve(
                item
            )

            normalized.append(
                record.model_dump(
                    mode="json"
                )
            )

        except Exception as exc:

            failures += 1

            cve_id = (
                item
                .get("cve", {})
                .get("id", "UNKNOWN")
                if isinstance(
                    item,
                    dict,
                )
                else "UNKNOWN"
            )

            print(
                f"Normalization failed "
                f"for {cve_id}: {exc}"
            )

        if index % 10000 == 0:

            print(
                f"Processed "
                f"{index:,}/"
                f"{len(vulnerabilities):,}"
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
    print(
        "=" * 55
    )

    print(
        "NORMALIZATION COMPLETE"
    )

    print(
        "=" * 55
    )

    print(
        "Raw records:",
        len(vulnerabilities),
    )

    print(
        "Successfully normalized:",
        len(normalized),
    )

    print(
        "Failures:",
        failures,
    )

    print()
    print(
        "Output:",
        OUTPUT_FILE,
    )

    if normalized:

        print()
        print(
            "Example normalized record"
        )

        print(
            "-" * 55
        )

        first = normalized[0]

        print(
            "CVE:",
            first["cve_id"],
        )

        print(
            "Published:",
            first["published_date"],
        )

        print(
            "CVSS v4:",
            first["cvss_v4_score"],
        )

        print(
            "CVSS v3:",
            first["cvss_v3_score"],
        )

        print(
            "Severity:",
            first["severity"],
        )

        print(
            "CWE count:",
            first["cwe_count"],
        )

        print(
            "CPE count:",
            first["affected_cpe_count"],
        )

        print(
            "Reference count:",
            first["reference_count"],
        )


if __name__ == "__main__":
    main()
