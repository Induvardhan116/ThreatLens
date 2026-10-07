from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

FEATURES_PATH = (
    ROOT
    / "data"
    / "processed"
    / "features"
    / "threatlens_features.csv"
)

ENRICHED_PATH = (
    ROOT
    / "data"
    / "processed"
    / "enriched"
    / "threatlens_dataset.json"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FEATURE_AUDIT_PATH = (
    OUTPUT_DIR
    / "temporal_feature_provenance.csv"
)

TARGET_AUDIT_PATH = (
    OUTPUT_DIR
    / "temporal_target_audit.csv"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "temporal_leakage_audit.json"
)


# ============================================================
# FEATURE PROVENANCE DEFINITIONS
# ============================================================

FEATURE_PROVENANCE = {
    "CVSS": {
        "features": [
            "cvss_v3_score",
            "cvss_v4_score",
            "cvss_v3_available",
            "cvss_v4_available",
            "cvss_max_score",
            "cvss_score_difference",
        ],
        "availability": (
            "Potentially available at publication, but "
            "scores can be revised later."
        ),
        "risk": "REVIEW",
    },

    "Severity": {
        "features": [
            "severity_score",
            "severity_known",
        ],
        "availability": (
            "Usually derived from CVSS/vendor severity "
            "and can change after publication."
        ),
        "risk": "REVIEW",
    },

    "CWE": {
        "features": [
            "cwe_count",
            "cwe_unique_count",
            "has_cwe",
        ],
        "availability": (
            "Often available with vulnerability publication, "
            "but may be enriched later."
        ),
        "risk": "REVIEW",
    },

    "Affected CPE": {
        "features": [
            "affected_cpe_count",
            "affected_cpe_unique_count",
            "has_affected_cpe",
        ],
        "availability": (
            "Product applicability metadata may be present "
            "at publication and may also be updated."
        ),
        "risk": "REVIEW",
    },

    "References": {
        "features": [
            "reference_count",
            "reference_unique_count",
            "has_references",
        ],
        "availability": (
            "References can be added or changed after "
            "initial disclosure."
        ),
        "risk": "HIGH_REVIEW",
    },

    "Description": {
        "features": [
            "description_length",
            "description_word_count",
            "description_sentence_count",
            "description_uppercase_count",
            "description_digit_count",
            "description_special_count",
            "description_has_remote",
            "description_has_execution",
            "description_has_privilege",
            "description_has_authentication",
            "description_has_overflow",
            "description_has_injection",
            "description_has_bypass",
        ],
        "availability": (
            "Initial descriptions may exist at publication, "
            "but descriptions can be revised."
        ),
        "risk": "REVIEW",
    },

    "Publication timing": {
        "features": [
            "publication_year",
            "publication_month",
            "publication_quarter",
            "publication_day_of_week",
            "publication_day_of_year",
            "publication_is_weekend",
        ],
        "availability": (
            "Available once the vulnerability is published."
        ),
        "risk": "LOW",
    },

    "Modification timing": {
        "features": [
            "modified_after_publication_days",
            "has_modified_date",
        ],
        "availability": (
            "Modification information can reflect updates "
            "occurring after initial disclosure."
        ),
        "risk": "HIGH_REVIEW",
    },

    "MITRE ATT&CK": {
        "features": [
            "attack_technique_count",
            "has_attack_technique",
        ],
        "availability": (
            "ATT&CK mappings may be added after "
            "vulnerability disclosure."
        ),
        "risk": "HIGH_REVIEW",
    },
}


# ============================================================
# LOAD DATA
# ============================================================

def load_features() -> pd.DataFrame:

    print("Loading feature dataset...")

    df = pd.read_csv(
        FEATURES_PATH
    )

    print(
        f"Rows loaded: {len(df):,}"
    )

    print(
        f"Columns loaded: {len(df.columns):,}"
    )

    return df


def load_enriched() -> list[dict]:

    print("\nLoading enriched dataset...")

    with open(
        ENRICHED_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    print(
        f"Rows loaded: {len(data):,}"
    )

    return data


# ============================================================
# BASIC VALIDATION
# ============================================================

def validate_alignment(
    features_df: pd.DataFrame,
    enriched_data: list[dict],
):

    print("\nValidating alignment...")

    if len(features_df) != len(enriched_data):

        raise ValueError(
            "Feature and enriched dataset row counts do not match."
        )

    print("Row count alignment: PASS")

    if "cve_id" in features_df.columns:

        feature_ids = (
            features_df["cve_id"]
            .astype(str)
            .to_numpy()
        )

        enriched_ids = np.array(
            [
                str(
                    record.get(
                        "cve_id",
                        "",
                    )
                )
                for record in enriched_data
            ]
        )

        if not np.array_equal(
            feature_ids,
            enriched_ids,
        ):

            raise ValueError(
                "CVE ID alignment failed."
            )

        print("CVE ID alignment: PASS")

    feature_targets = (
        features_df[
            "exploitation_label"
        ]
        .astype(int)
        .to_numpy()
    )

    enriched_targets = np.array(
        [
            int(
                record.get(
                    "exploitation_label",
                    0,
                )
            )
            for record in enriched_data
        ]
    )

    if not np.array_equal(
        feature_targets,
        enriched_targets,
    ):

        raise ValueError(
            "Target alignment failed."
        )

    print("Target alignment: PASS")


# ============================================================
# DATE EXTRACTION
# ============================================================

def build_date_frame(
    enriched_data: list[dict],
) -> pd.DataFrame:

    records = []

    for record in enriched_data:

        published = pd.to_datetime(
            record.get(
                "published_date"
            ),
            errors="coerce",
            utc=True,
        )

        modified = pd.to_datetime(
            record.get(
                "last_modified_date"
            ),
            errors="coerce",
            utc=True,
        )

        kev_added = pd.to_datetime(
            record.get(
                "kev_date_added"
            ),
            errors="coerce",
            utc=True,
        )

        records.append(
            {
                "cve_id": record.get(
                    "cve_id"
                ),
                "published_date": published,
                "modified_date": modified,
                "kev_date_added": kev_added,
                "known_exploited": bool(
                    record.get(
                        "known_exploited",
                        False,
                    )
                ),
            }
        )

    dates = pd.DataFrame(
        records
    )

    for column in [
        "published_date",
        "modified_date",
        "kev_date_added",
    ]:

        dates[column] = (
            dates[column]
            .dt.tz_localize(None)
        )

    return dates


# ============================================================
# FEATURE PROVENANCE AUDIT
# ============================================================

def audit_feature_provenance(
    features_df: pd.DataFrame,
) -> pd.DataFrame:

    print("\nAuditing feature provenance...")

    rows = []

    positive_mask = (
        features_df[
            "exploitation_label"
        ]
        == 1
    )

    negative_mask = (
        features_df[
            "exploitation_label"
        ]
        == 0
    )

    positive_total = int(
        positive_mask.sum()
    )

    negative_total = int(
        negative_mask.sum()
    )

    for category, information in (
        FEATURE_PROVENANCE.items()
    ):

        for feature in information[
            "features"
        ]:

            if feature not in features_df.columns:

                rows.append(
                    {
                        "category": category,
                        "feature": feature,
                        "present_in_dataset": False,
                        "non_null_count": 0,
                        "non_null_percent": 0.0,
                        "positive_non_null_count": 0,
                        "positive_non_null_percent": 0.0,
                        "negative_non_null_count": 0,
                        "negative_non_null_percent": 0.0,
                        "availability_assessment": information[
                            "availability"
                        ],
                        "provenance_risk": information[
                            "risk"
                        ],
                    }
                )

                continue

            values = features_df[
                feature
            ]

            non_null = values.notna()

            non_null_count = int(
                non_null.sum()
            )

            non_null_percent = (
                non_null_count
                / len(values)
                * 100.0
            )

            positive_non_null = int(
                non_null[
                    positive_mask
                ].sum()
            )

            negative_non_null = int(
                non_null[
                    negative_mask
                ].sum()
            )

            positive_percent = (
                positive_non_null
                / positive_total
                * 100.0
                if positive_total
                else 0.0
            )

            negative_percent = (
                negative_non_null
                / negative_total
                * 100.0
                if negative_total
                else 0.0
            )

            rows.append(
                {
                    "category": category,
                    "feature": feature,
                    "present_in_dataset": True,
                    "non_null_count": non_null_count,
                    "non_null_percent": non_null_percent,
                    "positive_non_null_count": positive_non_null,
                    "positive_non_null_percent": positive_percent,
                    "negative_non_null_count": negative_non_null,
                    "negative_non_null_percent": negative_percent,
                    "availability_assessment": information[
                        "availability"
                    ],
                    "provenance_risk": information[
                        "risk"
                    ],
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# TARGET TIMELINE AUDIT
# ============================================================

def audit_target_timeline(
    dates_df: pd.DataFrame,
) -> pd.DataFrame:

    print("\nAuditing KEV target timing...")

    dated = dates_df[
        dates_df[
            "published_date"
        ].notna()
        & dates_df[
            "kev_date_added"
        ].notna()
    ].copy()

    dated[
        "days_publication_to_kev"
    ] = (
        dated[
            "kev_date_added"
        ]
        - dated[
            "published_date"
        ]
    ).dt.total_seconds() / 86400.0

    dated[
        "modified_before_kev"
    ] = (
        dated[
            "modified_date"
        ].notna()
        & (
            dated[
                "modified_date"
            ]
            <= dated[
                "kev_date_added"
            ]
        )
    )

    dated[
        "modified_after_kev"
    ] = (
        dated[
            "modified_date"
        ].notna()
        & (
            dated[
                "modified_date"
            ]
            > dated[
                "kev_date_added"
            ]
        )
    )

    summary_rows = []

    summary_rows.append(
        {
            "metric": (
                "KEV records with publication "
                "and KEV dates"
            ),
            "value": int(
                len(dated)
            ),
        }
    )

    summary_rows.append(
        {
            "metric": (
                "KEV records where KEV date "
                "precedes publication"
            ),
            "value": int(
                (
                    dated[
                        "days_publication_to_kev"
                    ]
                    < 0
                ).sum()
            ),
        }
    )

    summary_rows.append(
        {
            "metric": (
                "KEV records modified "
                "before/on KEV date"
            ),
            "value": int(
                dated[
                    "modified_before_kev"
                ].sum()
            ),
        }
    )

    summary_rows.append(
        {
            "metric": (
                "KEV records modified "
                "after KEV date"
            ),
            "value": int(
                dated[
                    "modified_after_kev"
                ].sum()
            ),
        }
    )

    if not dated.empty:

        median_days = float(
            dated[
                "days_publication_to_kev"
            ].median()
        )

        mean_days = float(
            dated[
                "days_publication_to_kev"
            ].mean()
        )

        min_days = float(
            dated[
                "days_publication_to_kev"
            ].min()
        )

        max_days = float(
            dated[
                "days_publication_to_kev"
            ].max()
        )

    else:

        median_days = 0.0
        mean_days = 0.0
        min_days = 0.0
        max_days = 0.0

    summary_rows.extend(
        [
            {
                "metric": (
                    "Median days publication to KEV"
                ),
                "value": median_days,
            },
            {
                "metric": (
                    "Mean days publication to KEV"
                ),
                "value": mean_days,
            },
            {
                "metric": (
                    "Minimum days publication to KEV"
                ),
                "value": min_days,
            },
            {
                "metric": (
                    "Maximum days publication to KEV"
                ),
                "value": max_days,
            },
        ]
    )

    return pd.DataFrame(
        summary_rows
    )


# ============================================================
# POST-KEV MODIFICATION ANALYSIS
# ============================================================

def analyze_post_kev_modification(
    dates_df: pd.DataFrame,
) -> dict:

    dated = dates_df[
        dates_df[
            "published_date"
        ].notna()
        & dates_df[
            "kev_date_added"
        ].notna()
        & dates_df[
            "modified_date"
        ].notna()
    ].copy()

    if dated.empty:

        return {
            "records_with_all_three_dates": 0,
            "modified_after_kev_count": 0,
            "modified_after_kev_percent": 0.0,
        }

    modified_after = (
        dated[
            "modified_date"
        ]
        > dated[
            "kev_date_added"
        ]
    )

    count = int(
        modified_after.sum()
    )

    percent = (
        count
        / len(dated)
        * 100.0
    )

    return {
        "records_with_all_three_dates": int(
            len(dated)
        ),
        "modified_after_kev_count": count,
        "modified_after_kev_percent": percent,
    }


# ============================================================
# REFERENCE SIGNAL ANALYSIS
# ============================================================

def analyze_reference_signal(
    features_df: pd.DataFrame,
) -> dict:

    print("\nAnalyzing reference signal...")

    positive = features_df[
        features_df[
            "exploitation_label"
        ] == 1
    ]

    negative = features_df[
        features_df[
            "exploitation_label"
        ] == 0
    ]

    result = {}

    for feature in [
        "reference_count",
        "reference_unique_count",
        "has_references",
    ]:

        if feature not in features_df.columns:
            continue

        result[feature] = {
            "positive_mean": float(
                positive[
                    feature
                ].mean()
            ),
            "negative_mean": float(
                negative[
                    feature
                ].mean()
            ),
            "positive_median": float(
                positive[
                    feature
                ].median()
            ),
            "negative_median": float(
                negative[
                    feature
                ].median()
            ),
        }

    return result


# ============================================================
# DESCRIPTION SIGNAL ANALYSIS
# ============================================================

def analyze_description_signal(
    features_df: pd.DataFrame,
) -> dict:

    print("\nAnalyzing description signal...")

    positive = features_df[
        features_df[
            "exploitation_label"
        ] == 1
    ]

    negative = features_df[
        features_df[
            "exploitation_label"
        ] == 0
    ]

    result = {}

    for feature in FEATURE_PROVENANCE[
        "Description"
    ]["features"]:

        if feature not in features_df.columns:
            continue

        result[feature] = {
            "positive_mean": float(
                positive[
                    feature
                ].mean()
            ),
            "negative_mean": float(
                negative[
                    feature
                ].mean()
            ),
            "positive_median": float(
                positive[
                    feature
                ].median()
            ),
            "negative_median": float(
                negative[
                    feature
                ].median()
            ),
        }

    return result


# ============================================================
# GROUP SUMMARY
# ============================================================

def build_group_summary(
    feature_audit: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for category in FEATURE_PROVENANCE:

        subset = feature_audit[
            feature_audit[
                "category"
            ] == category
        ]

        if subset.empty:
            continue

        high_review = int(
            (
                subset[
                    "provenance_risk"
                ]
                == "HIGH_REVIEW"
            ).sum()
        )

        review = int(
            (
                subset[
                    "provenance_risk"
                ]
                == "REVIEW"
            ).sum()
        )

        if high_review > 0:

            overall_risk = "HIGH_REVIEW"

        elif review > 0:

            overall_risk = "REVIEW"

        else:

            overall_risk = "LOW"

        rows.append(
            {
                "category": category,
                "feature_count": int(
                    len(subset)
                ),
                "average_non_null_percent": float(
                    subset[
                        "non_null_percent"
                    ].mean()
                ),
                "high_review_features": high_review,
                "review_features": review,
                "provenance_risk": overall_risk,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print(
        "ThreatLens - Temporal Leakage & Feature Provenance Audit"
    )
    print("=" * 72)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    features_df = load_features()

    enriched_data = load_enriched()

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_alignment(
        features_df,
        enriched_data,
    )

    # --------------------------------------------------------
    # Build timeline
    # --------------------------------------------------------

    dates_df = build_date_frame(
        enriched_data
    )

    print("\n" + "=" * 72)
    print("DATASET TIMELINE")
    print("=" * 72)

    print(
        "Publication dates available: "
        f"{dates_df['published_date'].notna().sum():,}"
    )

    print(
        "Publication dates missing: "
        f"{dates_df['published_date'].isna().sum():,}"
    )

    print(
        "KEV dates available: "
        f"{dates_df['kev_date_added'].notna().sum():,}"
    )

    print(
        "Modified dates available: "
        f"{dates_df['modified_date'].notna().sum():,}"
    )

    # --------------------------------------------------------
    # Feature provenance
    # --------------------------------------------------------

    feature_audit = audit_feature_provenance(
        features_df
    )

    feature_audit.to_csv(
        FEATURE_AUDIT_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Group summary
    # --------------------------------------------------------

    group_summary = build_group_summary(
        feature_audit
    )

    # --------------------------------------------------------
    # Target timeline
    # --------------------------------------------------------

    target_audit = audit_target_timeline(
        dates_df
    )

    target_audit.to_csv(
        TARGET_AUDIT_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Post-KEV modification
    # --------------------------------------------------------

    post_kev_modification = (
        analyze_post_kev_modification(
            dates_df
        )
    )

    # --------------------------------------------------------
    # Strong signal analysis
    # --------------------------------------------------------

    reference_signal = (
        analyze_reference_signal(
            features_df
        )
    )

    description_signal = (
        analyze_description_signal(
            features_df
        )
    )

    # --------------------------------------------------------
    # Risk categories
    # --------------------------------------------------------

    high_review_categories = [
        category
        for category, information
        in FEATURE_PROVENANCE.items()
        if information[
            "risk"
        ] == "HIGH_REVIEW"
    ]

    review_categories = [
        category
        for category, information
        in FEATURE_PROVENANCE.items()
        if information[
            "risk"
        ] == "REVIEW"
    ]

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report = {
        "experiment": (
            "Temporal Leakage & Feature Provenance Audit"
        ),

        "purpose": (
            "Assess whether model features could contain "
            "information that was unavailable at vulnerability "
            "publication or was introduced through later updates."
        ),

        "dataset": {
            "feature_rows": int(
                len(features_df)
            ),
            "published_dates_available": int(
                dates_df[
                    "published_date"
                ].notna().sum()
            ),
            "published_dates_missing": int(
                dates_df[
                    "published_date"
                ].isna().sum()
            ),
            "kev_dates_available": int(
                dates_df[
                    "kev_date_added"
                ].notna().sum()
            ),
            "modified_dates_available": int(
                dates_df[
                    "modified_date"
                ].notna().sum()
            ),
        },

        "feature_provenance": (
            FEATURE_PROVENANCE
        ),

        "high_review_categories": (
            high_review_categories
        ),

        "review_categories": (
            review_categories
        ),

        "target_timeline": {
            "summary": target_audit.to_dict(
                orient="records"
            ),
            "post_kev_modification": (
                post_kev_modification
            ),
        },

        "strong_signal_analysis": {
            "references": reference_signal,
            "description": description_signal,
        },

        "interpretation_rules": [
            (
                "A feature is not automatically considered "
                "leaked because it can change after publication."
            ),
            (
                "A feature should be treated as temporally risky "
                "when its value can reflect information added "
                "after the intended prediction point."
            ),
            (
                "KEV membership is the target and must never "
                "be used as a model feature."
            ),
            (
                "KEV-derived metadata must remain excluded "
                "from the model."
            ),
            (
                "Current days_since_publication must remain "
                "excluded because it depends on the current date."
            ),
            (
                "Raw publication date is used for chronological "
                "splitting, while derived publication timing "
                "features may be model inputs."
            ),
            (
                "Modification timing is considered high review "
                "because later NVD updates may not have existed "
                "at the intended prediction point."
            ),
            (
                "Reference counts are considered high review "
                "because references may accumulate after disclosure."
            ),
            (
                "ATT&CK mappings are high review because mappings "
                "may be added after vulnerability publication."
            ),
        ],

        "important_limitation": (
            "The current dataset contains present-day NVD metadata "
            "and a current CISA KEV snapshot. This audit can identify "
            "potential temporal risk from metadata timestamps, but "
            "cannot prove historical point-in-time feature availability "
            "for every individual vulnerability."
        ),

        "recommended_next_step": (
            "Use the provenance findings to construct a conservative "
            "feature set and rerun chronological ranking evaluation."
        ),
    }

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
        )

    # ========================================================
    # CONSOLE OUTPUT
    # ========================================================

    print("\n" + "=" * 72)
    print("FEATURE PROVENANCE SUMMARY")
    print("=" * 72)

    print(
        group_summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    print("\n" + "=" * 72)
    print("KEV TIMELINE AUDIT")
    print("=" * 72)

    print(
        target_audit.to_string(
            index=False
        )
    )

    print("\n" + "=" * 72)
    print("POST-KEV MODIFICATION ANALYSIS")
    print("=" * 72)

    for key, value in (
        post_kev_modification.items()
    ):

        print(
            f"{key}: {value}"
        )

    print("\n" + "=" * 72)
    print("REFERENCE SIGNAL")
    print("=" * 72)

    for feature, values in (
        reference_signal.items()
    ):

        print(
            f"\n{feature}"
        )

        print(
            "  Positive mean: "
            f"{values['positive_mean']:.4f}"
        )

        print(
            "  Negative mean: "
            f"{values['negative_mean']:.4f}"
        )

        print(
            "  Positive median: "
            f"{values['positive_median']:.4f}"
        )

        print(
            "  Negative median: "
            f"{values['negative_median']:.4f}"
        )

    print("\n" + "=" * 72)
    print("DESCRIPTION SIGNAL")
    print("=" * 72)

    for feature, values in (
        description_signal.items()
    ):

        print(
            f"\n{feature}"
        )

        print(
            "  Positive mean: "
            f"{values['positive_mean']:.4f}"
        )

        print(
            "  Negative mean: "
            f"{values['negative_mean']:.4f}"
        )

        print(
            "  Positive median: "
            f"{values['positive_median']:.4f}"
        )

        print(
            "  Negative median: "
            f"{values['negative_median']:.4f}"
        )

    print("\n" + "=" * 72)
    print("OUTPUT FILES")
    print("=" * 72)

    print(
        FEATURE_AUDIT_PATH
    )

    print(
        TARGET_AUDIT_PATH
    )

    print(
        REPORT_PATH
    )

    print(
        "\nTemporal leakage audit complete."
    )


if __name__ == "__main__":
    main()
