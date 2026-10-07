from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pipeline.schema import VulnerabilityRecord


PROJECT_ROOT = Path(__file__).resolve().parents[2]

NVD_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nvd"
    / "research_normalized.json"
)

KEV_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cisa_kev"
    / "kev_normalized.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "enriched"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "threatlens_dataset.json"
)


def load_json(
    path: Path,
) -> Any:

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def main() -> None:

    print(
        "ThreatLens - Research dataset enrichment"
    )

    print(
        "=" * 60
    )

    print()
    print(
        "Loading normalized NVD dataset..."
    )

    nvd_records = load_json(
        NVD_FILE
    )

    print(
        "NVD records:",
        len(nvd_records),
    )

    print()
    print(
        "Loading normalized CISA KEV dataset..."
    )

    kev_records = load_json(
        KEV_FILE
    )

    print(
        "KEV records:",
        len(kev_records),
    )

    kev_by_cve: dict[
        str,
        dict[str, Any],
    ] = {}

    for record in kev_records:

        if not isinstance(
            record,
            dict,
        ):
            continue

        cve_id = record.get(
            "cve_id"
        )

        if isinstance(
            cve_id,
            str,
        ):

            kev_by_cve[
                cve_id
            ] = record

    enriched_records: list[
        dict[str, Any]
    ] = []

    kev_matches = 0
    non_kev_records = 0
    failures = 0

    for index, item in enumerate(
        nvd_records,
        start=1,
    ):

        try:

            nvd_record = (
                VulnerabilityRecord
                .model_validate(item)
            )

            kev = kev_by_cve.get(
                nvd_record.cve_id
            )

            if kev is not None:

                kev_matches += 1

                enriched = (
                    nvd_record.model_copy(
                        update={
                            "known_exploited": True,
                            "kev_date_added": (
                                kev.get(
                                    "kev_date_added"
                                )
                            ),
                            "kev_due_date": (
                                kev.get(
                                    "kev_due_date"
                                )
                            ),
                            "kev_vendor_project": (
                                kev.get(
                                    "kev_vendor_project"
                                )
                            ),
                            "kev_product": (
                                kev.get(
                                    "kev_product"
                                )
                            ),
                            "kev_vulnerability_name": (
                                kev.get(
                                    "kev_vulnerability_name"
                                )
                            ),
                            "kev_required_action": (
                                kev.get(
                                    "kev_required_action"
                                )
                            ),
                            "ransomware_use": (
                                kev.get(
                                    "ransomware_use"
                                )
                            ),
                            "exploitation_label": 1,
                        }
                    )
                )

            else:

                non_kev_records += 1

                enriched = (
                    nvd_record.model_copy(
                        update={
                            "known_exploited": False,
                            "exploitation_label": 0,
                        }
                    )
                )

            enriched_records.append(
                enriched.model_dump(
                    mode="json"
                )
            )

        except Exception as exc:

            failures += 1

            cve_id = (
                item.get(
                    "cve_id",
                    "UNKNOWN"
                )
                if isinstance(
                    item,
                    dict,
                )
                else "UNKNOWN"
            )

            print(
                f"Enrichment failed "
                f"for {cve_id}: {exc}"
            )

        if index % 10000 == 0:

            print(
                f"Processed "
                f"{index:,}/"
                f"{len(nvd_records):,}"
            )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            enriched_records,
            file,
            indent=2,
            ensure_ascii=False,
        )

    kev_total = len(
        kev_records
    )

    kev_outside_cohort = (
        kev_total
        - kev_matches
    )

    print()
    print(
        "=" * 60
    )

    print(
        "THREATLENS DATASET COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        "NVD input records:",
        len(nvd_records),
    )

    print(
        "CISA KEV records:",
        kev_total,
    )

    print(
        "KEV matches inside NVD cohort:",
        kev_matches,
    )

    print(
        "KEV records outside NVD cohort:",
        kev_outside_cohort,
    )

    print(
        "Not in KEV:",
        non_kev_records,
    )

    print(
        "Failures:",
        failures,
    )

    print(
        "Output records:",
        len(enriched_records),
    )

    print()
    print(
        "LABEL DISTRIBUTION"
    )

    print(
        "-" * 60
    )

    positives = sum(
        1
        for record in enriched_records
        if record.get(
            "exploitation_label"
        ) == 1
    )

    negatives = sum(
        1
        for record in enriched_records
        if record.get(
            "exploitation_label"
        ) == 0
    )

    print(
        "Known exploited:",
        positives,
    )

    print(
        "Not known exploited:",
        negatives,
    )

    if enriched_records:

        positive_rate = (
            positives
            / len(enriched_records)
            * 100
        )

        print(
            f"Positive rate: "
            f"{positive_rate:.2f}%"
        )

    print()
    print(
        "Output:"
    )

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()
