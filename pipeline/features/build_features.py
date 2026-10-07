from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "enriched"
    / "threatlens_dataset.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FEATURE_FILE = (
    OUTPUT_DIR
    / "threatlens_features.csv"
)

METADATA_FILE = (
    OUTPUT_DIR
    / "feature_metadata.json"
)


def safe_list_length(value) -> int:
    if isinstance(value, list):
        return len(value)

    return 0


def safe_unique_length(value) -> int:
    if not isinstance(value, list):
        return 0

    return len(
        set(
            str(item)
            for item in value
        )
    )


def safe_text_length(value) -> int:
    if not isinstance(value, str):
        return 0

    return len(value)


def safe_word_count(value) -> int:
    if not isinstance(value, str):
        return 0

    return len(
        value.split()
    )


def safe_upper_count(value) -> int:
    if not isinstance(value, str):
        return 0

    return sum(
        1
        for character in value
        if character.isupper()
    )


def safe_digit_count(value) -> int:
    if not isinstance(value, str):
        return 0

    return sum(
        1
        for character in value
        if character.isdigit()
    )


def safe_special_count(value) -> int:
    if not isinstance(value, str):
        return 0

    return sum(
        1
        for character in value
        if not character.isalnum()
        and not character.isspace()
    )


def build_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    output = pd.DataFrame(
        index=df.index
    )

    # ---------------------------------------------------------
    # TARGET
    # ---------------------------------------------------------

    output["exploitation_label"] = (
        pd.to_numeric(
            df["exploitation_label"],
            errors="coerce",
        )
        .astype("Int64")
    )

    # ---------------------------------------------------------
    # CVSS FEATURES
    # ---------------------------------------------------------

    output["cvss_v3_score"] = (
        pd.to_numeric(
            df["cvss_v3_score"],
            errors="coerce",
        )
    )

    output["cvss_v4_score"] = (
        pd.to_numeric(
            df["cvss_v4_score"],
            errors="coerce",
        )
    )

    output["cvss_v3_available"] = (
        output["cvss_v3_score"]
        .notna()
        .astype(int)
    )

    output["cvss_v4_available"] = (
        output["cvss_v4_score"]
        .notna()
        .astype(int)
    )

    output["cvss_max_score"] = (
        output[
            [
                "cvss_v3_score",
                "cvss_v4_score",
            ]
        ]
        .max(
            axis=1
        )
    )

    output["cvss_score_difference"] = (
        output["cvss_v4_score"]
        - output["cvss_v3_score"]
    )

    # ---------------------------------------------------------
    # SEVERITY
    # ---------------------------------------------------------

    severity_map = {
        "NONE": 0,
        "LOW": 1,
        "MEDIUM": 2,
        "MODERATE": 2,
        "HIGH": 3,
        "CRITICAL": 4,
    }

    severity = (
        df["severity"]
        .fillna("UNKNOWN")
        .astype(str)
        .str.upper()
    )

    output["severity_score"] = (
        severity.map(
            severity_map
        )
        .fillna(0)
        .astype(int)
    )

    output["severity_known"] = (
        severity
        .ne("UNKNOWN")
        .astype(int)
    )

    # ---------------------------------------------------------
    # CWE FEATURES
    # ---------------------------------------------------------

    output["cwe_count"] = (
        pd.to_numeric(
            df["cwe_count"],
            errors="coerce",
        )
        .fillna(0)
    )

    output["cwe_unique_count"] = (
        df["cwe_ids"]
        .apply(
            safe_unique_length
        )
    )

    output["has_cwe"] = (
        output["cwe_count"]
        .gt(0)
        .astype(int)
    )

    # ---------------------------------------------------------
    # AFFECTED PRODUCT / CPE FEATURES
    # ---------------------------------------------------------

    output["affected_cpe_count"] = (
        pd.to_numeric(
            df["affected_cpe_count"],
            errors="coerce",
        )
        .fillna(0)
    )

    output["affected_cpe_unique_count"] = (
        df["affected_cpes"]
        .apply(
            safe_unique_length
        )
    )

    output["has_affected_cpe"] = (
        output["affected_cpe_count"]
        .gt(0)
        .astype(int)
    )

    # ---------------------------------------------------------
    # REFERENCE FEATURES
    # ---------------------------------------------------------

    output["reference_count"] = (
        pd.to_numeric(
            df["reference_count"],
            errors="coerce",
        )
        .fillna(0)
    )

    output["reference_unique_count"] = (
        df["references"]
        .apply(
            safe_unique_length
        )
    )

    output["has_references"] = (
        output["reference_count"]
        .gt(0)
        .astype(int)
    )

    # ---------------------------------------------------------
    # DESCRIPTION FEATURES
    # ---------------------------------------------------------

    description = (
        df["description"]
        .fillna("")
        .astype(str)
    )

    output["description_length"] = (
        description.str.len()
    )

    output["description_word_count"] = (
        description.str.split()
        .str.len()
    )

    output["description_sentence_count"] = (
        description.str.count(
            r"[.!?]"
        )
    )

    output["description_uppercase_count"] = (
        description.apply(
            safe_upper_count
        )
    )

    output["description_digit_count"] = (
        description.apply(
            safe_digit_count
        )
    )

    output["description_special_count"] = (
        description.apply(
            safe_special_count
        )
    )

    output["description_has_remote"] = (
        description
        .str.contains(
            "remote",
            case=False,
            regex=False,
        )
        .astype(int)
    )

    output["description_has_execution"] = (
        description
        .str.contains(
            "execution",
            case=False,
            regex=False,
        )
        .astype(int)
    )

    output["description_has_privilege"] = (
        description
        .str.contains(
            "privilege",
            case=False,
            regex=False,
        )
        .astype(int)
    )

    output["description_has_authentication"] = (
        description
        .str.contains(
            "authentication",
            case=False,
            regex=False,
        )
        .astype(int)
    )

    output["description_has_overflow"] = (
        description
        .str.contains(
            "overflow",
            case=False,
            regex=False,
        )
        .astype(int)
    )

    output["description_has_injection"] = (
        description
        .str.contains(
            "injection",
            case=False,
            regex=False,
        )
        .astype(int)
    )

    output["description_has_bypass"] = (
        description
        .str.contains(
            "bypass",
            case=False,
            regex=False,
        )
        .astype(int)
    )

    # ---------------------------------------------------------
    # TEMPORAL FEATURES
    #
    # We intentionally DO NOT use the existing
    # days_since_publication field.
    #
    # It was calculated relative to the current
    # processing date and is therefore unsafe for
    # historical evaluation.
    # ---------------------------------------------------------

    published = pd.to_datetime(
        df["published_date"],
        errors="coerce",
        utc=True,
    )

    modified = pd.to_datetime(
        df["last_modified_date"],
        errors="coerce",
        utc=True,
    )

    output["publication_year"] = (
        published.dt.year
        .fillna(0)
        .astype(int)
    )

    output["publication_month"] = (
        published.dt.month
        .fillna(0)
        .astype(int)
    )

    output["publication_quarter"] = (
        published.dt.quarter
        .fillna(0)
        .astype(int)
    )

    output["publication_day_of_week"] = (
        published.dt.dayofweek
        .fillna(0)
        .astype(int)
    )

    output["publication_day_of_year"] = (
        published.dt.dayofyear
        .fillna(0)
        .astype(int)
    )

    output["publication_is_weekend"] = (
        published.dt.dayofweek
        .isin([5, 6])
        .fillna(False)
        .astype(int)
    )

    output["modified_after_publication_days"] = (
        (
            modified
            - published
        )
        .dt.total_seconds()
        .div(86400)
        .clip(
            lower=0
        )
        .fillna(0)
    )

    output["has_modified_date"] = (
        modified.notna()
        .astype(int)
    )

    output["publication_date_available"] = (
        published.notna()
        .astype(int)
    )

    # ---------------------------------------------------------
    # ATT&CK FEATURES
    #
    # These are currently structural features only.
    # The actual ATT&CK enrichment will be added later.
    # ---------------------------------------------------------

    output["attack_technique_count"] = (
        pd.to_numeric(
            df["attack_technique_count"],
            errors="coerce",
        )
        .fillna(0)
    )

    output["has_attack_technique"] = (
        output["attack_technique_count"]
        .gt(0)
        .astype(int)
    )

    # ---------------------------------------------------------
    # NUMERICAL CLEANUP
    # ---------------------------------------------------------

    for column in output.columns:

        if column == "exploitation_label":
            continue

        output[column] = pd.to_numeric(
            output[column],
            errors="coerce",
        )

    output = output.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    return output


def main() -> None:

    print()
    print(
        "=" * 70
    )
    print(
        "THREATLENS LEAKAGE-SAFE FEATURE ENGINEERING"
    )
    print(
        "=" * 70
    )
    print()

    print(
        "Loading dataset..."
    )

    with INPUT_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        records = json.load(
            file
        )

    df = pd.DataFrame(
        records
    )

    print(
        f"Input records: {len(df):,}"
    )

    print(
        f"Input columns: {len(df.columns)}"
    )

    print()
    print(
        "Building features..."
    )

    features = build_features(
        df
    )

    print(
        f"Output records: {len(features):,}"
    )

    print(
        f"Output features: "
        f"{len(features.columns) - 1}"
    )

    # ---------------------------------------------------------
    # Explicit leakage verification
    # ---------------------------------------------------------

    forbidden = {
        "known_exploited",
        "kev_date_added",
        "kev_due_date",
        "kev_vendor_project",
        "kev_product",
        "kev_vulnerability_name",
        "kev_required_action",
        "ransomware_use",
        "cve_id",
        "days_since_publication",
    }

    leaked = (
        forbidden
        .intersection(
            features.columns
        )
    )

    if leaked:

        raise RuntimeError(
            "LEAKAGE CHECK FAILED. "
            f"Forbidden fields found: "
            f"{sorted(leaked)}"
        )

    print()
    print(
        "Leakage verification: PASSED"
    )

    # ---------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------

    features.to_csv(
        FEATURE_FILE,
        index=False,
    )

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------

    feature_columns = [
        column
        for column in features.columns
        if column != "exploitation_label"
    ]

    metadata = {
        "project": "ThreatLens",
        "purpose": (
            "Leakage-safe feature matrix for "
            "vulnerability exploitation prioritization."
        ),
        "input_records": int(
            len(df)
        ),
        "output_records": int(
            len(features)
        ),
        "feature_count": len(
            feature_columns
        ),
        "target": "exploitation_label",
        "target_definition": (
            "1 = CVE appears in CISA KEV; "
            "0 = CVE does not appear in CISA KEV."
        ),
        "target_caveat": (
            "CISA KEV is treated as a real-world "
            "exploitation signal, not perfect ground truth."
        ),
        "excluded_fields": sorted(
            forbidden
        ),
        "feature_columns": feature_columns,
        "temporal_design": {
            "existing_days_since_publication_excluded": True,
            "raw_dates_excluded": True,
            "publication_calendar_features": True,
            "modified_after_publication_days": True,
        },
        "ml_note": (
            "The target is highly imbalanced. "
            "Ranking and top-K evaluation should be "
            "primary rather than accuracy."
        ),
    }

    with METADATA_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )

    print()
    print(
        "=" * 70
    )
    print(
        "FEATURE ENGINEERING COMPLETE"
    )
    print(
        "=" * 70
    )

    print()
    print(
        "Feature matrix:"
    )

    print(
        FEATURE_FILE
    )

    print()
    print(
        "Metadata:"
    )

    print(
        METADATA_FILE
    )

    print()
    print(
        "Features:"
    )

    for index, column in enumerate(
        feature_columns,
        start=1,
    ):

        print(
            f"  {index:02d}. {column}"
        )

    print()
    print(
        "Target:"
    )

    print(
        features[
            "exploitation_label"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print(
        "Ready for baseline ranking models."
    )


if __name__ == "__main__":
    main()
