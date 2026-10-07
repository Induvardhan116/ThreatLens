from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score


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

RESULTS_PATH = (
    OUTPUT_DIR
    / "conservative_model_results.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "conservative_model_summary.csv"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "conservative_model_report.json"
)


# ============================================================
# MODEL CONFIGURATION
# Same configuration used in previous experiments.
# ============================================================

MODEL_CONFIG = {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_leaf": 3,
    "class_weight": "balanced_subsample",
    "random_state": 42,
    "n_jobs": -1,
}


# ============================================================
# FEATURES THAT ARE NOT USED
# ============================================================

LEAKAGE_COLUMNS = {
    "exploitation_label",
    "known_exploited",
    "kev_date_added",
    "kev_due_date",
    "kev_vendor_project",
    "kev_product",
    "kev_vulnerability_name",
    "kev_required_action",
    "ransomware_use",
}


MODIFICATION_TIMING_FEATURES = {
    "modified_after_publication_days",
    "has_modified_date",
}


ATTACK_FEATURES = {
    "attack_technique_count",
    "has_attack_technique",
}


REFERENCE_FEATURES = {
    "reference_count",
    "reference_unique_count",
    "has_references",
}


# ============================================================
# DATA LOADING
# ============================================================

def load_features() -> pd.DataFrame:

    print("Loading ThreatLens feature dataset...")

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
        f"Enriched rows loaded: {len(data):,}"
    )

    return data


# ============================================================
# VALIDATION
# ============================================================

def validate_alignment(
    features_df: pd.DataFrame,
    enriched_data: list[dict],
):

    print("\nValidating datasets...")

    if len(features_df) != len(enriched_data):

        raise ValueError(
            "Feature and enriched dataset row counts do not match."
        )

    print("Row count alignment: PASS")

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
# CHRONOLOGICAL SPLIT
# ============================================================

def chronological_split(
    features_df: pd.DataFrame,
    enriched_data: list[dict],
):

    print("\nBuilding chronological split...")

    dates = pd.to_datetime(
        [
            record.get(
                "published_date"
            )
            for record in enriched_data
        ],
        errors="coerce",
        utc=True,
    )

    dates = (
        pd.Series(dates)
        .dt
        .tz_localize(None)
    )

    valid_mask = dates.notna()

    valid_count = int(
        valid_mask.sum()
    )

    missing_count = int(
        (~valid_mask).sum()
    )

    print(
        f"Valid publication dates: {valid_count:,}"
    )

    print(
        f"Missing publication dates: {missing_count:,}"
    )

    valid_indices = np.where(
        valid_mask.to_numpy()
    )[0]

    dated_df = (
        features_df
        .iloc[valid_indices]
        .copy()
    )

    dated_dates = (
        dates
        .iloc[valid_indices]
        .copy()
    )

    dated_df["_published_date"] = (
        dated_dates.to_numpy()
    )

    dated_df = (
        dated_df
        .sort_values(
            "_published_date"
        )
        .reset_index(
            drop=True
        )
    )

    split_date = pd.Timestamp(
        "2026-01-01"
    )

    train_df = dated_df[
        dated_df[
            "_published_date"
        ]
        < split_date
    ].copy()

    test_df = dated_df[
        dated_df[
            "_published_date"
        ]
        >= split_date
    ].copy()

    train_df.pop(
        "_published_date"
    )

    test_df.pop(
        "_published_date"
    )

    print(
        "Rows available for temporal evaluation: "
        f"{len(dated_df):,}"
    )

    print(
        "Excluded from temporal evaluation: "
        f"{missing_count:,}"
    )

    print(
        "Publication range: "
        f"{dated_dates.min().date()} "
        f"to "
        f"{dated_dates.max().date()}"
    )

    print(
        f"\nTraining rows: {len(train_df):,}"
    )

    print(
        f"Test rows: {len(test_df):,}"
    )

    print(
        "Training KEV positives: "
        f"{int(train_df['exploitation_label'].sum()):,}"
    )

    print(
        "Test KEV positives: "
        f"{int(test_df['exploitation_label'].sum()):,}"
    )

    return train_df, test_df


# ============================================================
# FEATURE SELECTION
# ============================================================

def get_all_model_features(
    df: pd.DataFrame,
) -> list[str]:

    return [
        column
        for column in df.columns
        if column not in LEAKAGE_COLUMNS
        and column != "cve_id"
    ]


def remove_features(
    features: list[str],
    remove_set: set[str],
) -> list[str]:

    remaining = [
        feature
        for feature in features
        if feature not in remove_set
    ]

    if not remaining:

        raise ValueError(
            "Feature removal left no model features."
        )

    return remaining


# ============================================================
# MATRIX PREPARATION
# ============================================================

def prepare_matrix(
    df: pd.DataFrame,
    feature_columns: list[str],
):

    X = (
        df[
            feature_columns
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    missing_before = int(
        X.isna()
        .sum()
        .sum()
    )

    X = X.fillna(0.0)

    missing_after = int(
        X.isna()
        .sum()
        .sum()
    )

    if missing_after != 0:

        raise ValueError(
            "Missing values remain after filling."
        )

    y = (
        df[
            "exploitation_label"
        ]
        .astype(int)
    )

    return X, y, missing_before


# ============================================================
# TOP-K METRICS
# ============================================================

def precision_at_k(
    y_true: np.ndarray,
    scores: np.ndarray,
    k: int,
) -> float:

    k = min(
        k,
        len(y_true),
    )

    order = np.argsort(
        -scores
    )[:k]

    return float(
        np.mean(
            y_true[
                order
            ]
        )
    )


def recall_at_k(
    y_true: np.ndarray,
    scores: np.ndarray,
    k: int,
) -> float:

    total_positive = int(
        y_true.sum()
    )

    if total_positive == 0:
        return 0.0

    k = min(
        k,
        len(y_true),
    )

    order = np.argsort(
        -scores
    )[:k]

    return float(
        y_true[
            order
        ].sum()
        / total_positive
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model_name: str,
    y_true: np.ndarray,
    scores: np.ndarray,
) -> dict:

    result = {
        "model": model_name,
        "test_rows": int(
            len(y_true)
        ),
        "test_positives": int(
            y_true.sum()
        ),
        "positive_rate": float(
            y_true.mean()
        ),
        "average_precision": float(
            average_precision_score(
                y_true,
                scores,
            )
        ),
    }

    for k in [
        10,
        50,
        100,
        500,
    ]:

        result[
            f"precision_at_{k}"
        ] = precision_at_k(
            y_true,
            scores,
            k,
        )

        result[
            f"recall_at_{k}"
        ] = recall_at_k(
            y_true,
            scores,
            k,
        )

    return result


# ============================================================
# RANDOM FOREST
# ============================================================

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
):

    model = RandomForestClassifier(
        **MODEL_CONFIG
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


# ============================================================
# CVSS SCORE
# ============================================================

def calculate_cvss_score(
    df: pd.DataFrame,
) -> np.ndarray:

    v4 = (
        pd.to_numeric(
            df[
                "cvss_v4_score"
            ],
            errors="coerce",
        )
        .fillna(0.0)
    )

    v3 = (
        pd.to_numeric(
            df[
                "cvss_v3_score"
            ],
            errors="coerce",
        )
        .fillna(0.0)
    )

    v4_available = (
        pd.to_numeric(
            df[
                "cvss_v4_available"
            ],
            errors="coerce",
        )
        .fillna(0)
        .astype(bool)
    )

    scores = np.where(
        v4_available,
        v4,
        v3,
    )

    return scores.astype(
        float
    )


# ============================================================
# MODEL EXPERIMENT
# ============================================================

def run_model_experiment(
    model_name: str,
    feature_columns: list[str],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_test: np.ndarray,
):

    print("\n" + "=" * 72)
    print(
        f"MODEL: {model_name}"
    )
    print("=" * 72)

    print(
        f"Features used: "
        f"{len(feature_columns)}"
    )

    print(
        "Preparing training matrix..."
    )

    X_train, y_train, train_missing = (
        prepare_matrix(
            train_df,
            feature_columns,
        )
    )

    print(
        "Training missing values before filling: "
        f"{train_missing:,}"
    )

    print(
        "Preparing test matrix..."
    )

    X_test, _, test_missing = (
        prepare_matrix(
            test_df,
            feature_columns,
        )
    )

    print(
        "Test missing values before filling: "
        f"{test_missing:,}"
    )

    print(
        "Training Random Forest..."
    )

    model = train_model(
        X_train,
        y_train,
    )

    print(
        "Random Forest training complete."
    )

    scores = (
        model
        .predict_proba(
            X_test
        )[:, 1]
    )

    result = evaluate(
        model_name,
        y_test,
        scores,
    )

    result[
        "feature_count"
    ] = len(
        feature_columns
    )

    print(
        f"Average Precision: "
        f"{result['average_precision']:.6f}"
    )

    print(
        f"P@100: "
        f"{result['precision_at_100']:.4f}"
    )

    print(
        f"Recall@500: "
        f"{result['recall_at_500']:.4f}"
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print(
        "ThreatLens - Conservative Model Evaluation"
    )
    print("=" * 72)

    # --------------------------------------------------------
    # Load
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
    # Split
    # --------------------------------------------------------

    train_df, test_df = (
        chronological_split(
            features_df,
            enriched_data,
        )
    )

    y_test = (
        test_df[
            "exploitation_label"
        ]
        .astype(int)
        .to_numpy()
    )

    # --------------------------------------------------------
    # All model features
    # --------------------------------------------------------

    all_features = (
        get_all_model_features(
            features_df
        )
    )

    print(
        f"\nAll model features: "
        f"{len(all_features)}"
    )

    # --------------------------------------------------------
    # Conservative feature set
    #
    # Remove:
    #   - modification timing
    #   - MITRE ATT&CK
    #
    # Keep references for this version.
    # --------------------------------------------------------

    conservative_features = (
        remove_features(
            all_features,
            (
                MODIFICATION_TIMING_FEATURES
                | ATTACK_FEATURES
            ),
        )
    )

    # --------------------------------------------------------
    # Conservative + no references
    # --------------------------------------------------------

    conservative_no_refs_features = (
        remove_features(
            conservative_features,
            REFERENCE_FEATURES,
        )
    )

    # --------------------------------------------------------
    # Print feature counts
    # --------------------------------------------------------

    print(
        "\nFeature configuration:"
    )

    print(
        f"All features: "
        f"{len(all_features)}"
    )

    print(
        f"Conservative: "
        f"{len(conservative_features)}"
    )

    print(
        f"Conservative + No References: "
        f"{len(conservative_no_refs_features)}"
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results = []

    # --------------------------------------------------------
    # Model A
    # --------------------------------------------------------

    result = run_model_experiment(
        "ThreatLens - All Features",
        all_features,
        train_df,
        test_df,
        y_test,
    )

    result[
        "removed_groups"
    ] = "None"

    results.append(
        result
    )

    # --------------------------------------------------------
    # Model B
    # --------------------------------------------------------

    result = run_model_experiment(
        "ThreatLens - Conservative",
        conservative_features,
        train_df,
        test_df,
        y_test,
    )

    result[
        "removed_groups"
    ] = (
        "Modification timing; MITRE ATT&CK"
    )

    results.append(
        result
    )

    # --------------------------------------------------------
    # Model C
    # --------------------------------------------------------

    result = run_model_experiment(
        "ThreatLens - Conservative + No References",
        conservative_no_refs_features,
        train_df,
        test_df,
        y_test,
    )

    result[
        "removed_groups"
    ] = (
        "Modification timing; MITRE ATT&CK; References"
    )

    results.append(
        result
    )

    # --------------------------------------------------------
    # Model D - CVSS
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print(
        "MODEL: CVSS-only"
    )
    print("=" * 72)

    cvss_scores = calculate_cvss_score(
        test_df
    )

    cvss_result = evaluate(
        "CVSS-only",
        y_test,
        cvss_scores,
    )

    cvss_result[
        "feature_count"
    ] = 2

    cvss_result[
        "removed_groups"
    ] = (
        "All contextual features"
    )

    results.append(
        cvss_result
    )

    print(
        f"Average Precision: "
        f"{cvss_result['average_precision']:.6f}"
    )

    print(
        f"P@100: "
        f"{cvss_result['precision_at_100']:.4f}"
    )

    print(
        f"Recall@500: "
        f"{cvss_result['recall_at_500']:.4f}"
    )

    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Compare against CVSS
    # --------------------------------------------------------

    cvss_ap = float(
        results_df.loc[
            results_df[
                "model"
            ] == "CVSS-only",
            "average_precision",
        ].iloc[0]
    )

    results_df[
        "ap_improvement_vs_cvss"
    ] = (
        results_df[
            "average_precision"
        ]
        - cvss_ap
    )

    results_df[
        "ap_improvement_percent_vs_cvss"
    ] = (
        results_df[
            "ap_improvement_vs_cvss"
        ]
        / cvss_ap
        * 100.0
    )

    # --------------------------------------------------------
    # Compare against all-feature model
    # --------------------------------------------------------

    full_ap = float(
        results_df.loc[
            results_df[
                "model"
            ]
            == "ThreatLens - All Features",
            "average_precision",
        ].iloc[0]
    )

    results_df[
        "ap_change_vs_all_features"
    ] = (
        results_df[
            "average_precision"
        ]
        - full_ap
    )

    results_df[
        "ap_change_percent_vs_all_features"
    ] = (
        results_df[
            "ap_change_vs_all_features"
        ]
        / full_ap
        * 100.0
    )

    # --------------------------------------------------------
    # Save detailed results
    # --------------------------------------------------------

    results_df.to_csv(
        RESULTS_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary_columns = [
        "model",
        "removed_groups",
        "feature_count",
        "average_precision",
        "precision_at_10",
        "precision_at_50",
        "precision_at_100",
        "precision_at_500",
        "recall_at_10",
        "recall_at_50",
        "recall_at_100",
        "recall_at_500",
        "ap_improvement_percent_vs_cvss",
        "ap_change_percent_vs_all_features",
    ]

    summary_df = (
        results_df[
            summary_columns
        ]
        .sort_values(
            "average_precision",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    summary_df.insert(
        0,
        "rank",
        np.arange(
            1,
            len(summary_df) + 1,
        ),
    )

    summary_df.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Determine whether conservative model beats CVSS
    # --------------------------------------------------------

    conservative_ap = float(
        results_df.loc[
            results_df[
                "model"
            ]
            == "ThreatLens - Conservative",
            "average_precision",
        ].iloc[0]
    )

    conservative_no_refs_ap = float(
        results_df.loc[
            results_df[
                "model"
            ]
            == "ThreatLens - Conservative + No References",
            "average_precision",
        ].iloc[0]
    )

    report = {
        "experiment": (
            "Conservative ThreatLens Model Evaluation"
        ),

        "research_question": (
            "Does ThreatLens retain an advantage over CVSS "
            "after removing potentially post-disclosure "
            "feature groups?"
        ),

        "chronological_split": {
            "train_before": "2026-01-01",
            "test_from": "2026-01-01",
            "training_rows": int(
                len(train_df)
            ),
            "test_rows": int(
                len(test_df)
            ),
            "training_positives": int(
                train_df[
                    "exploitation_label"
                ].sum()
            ),
            "test_positives": int(
                test_df[
                    "exploitation_label"
                ].sum()
            ),
        },

        "models": {
            "all_features": {
                "features": all_features,
                "feature_count": len(
                    all_features
                ),
            },

            "conservative": {
                "features": conservative_features,
                "feature_count": len(
                    conservative_features
                ),
                "removed": [
                    "Modification timing",
                    "MITRE ATT&CK",
                ],
            },

            "conservative_no_references": {
                "features": (
                    conservative_no_refs_features
                ),
                "feature_count": len(
                    conservative_no_refs_features
                ),
                "removed": [
                    "Modification timing",
                    "MITRE ATT&CK",
                    "References",
                ],
            },

            "cvss_only": {
                "features": [
                    "cvss_v3_score",
                    "cvss_v4_score",
                ],
                "feature_count": 2,
            },
        },

        "results": results_df.to_dict(
            orient="records"
        ),

        "key_comparisons": {
            "cvss_average_precision": cvss_ap,
            "all_features_average_precision": full_ap,
            "conservative_average_precision": conservative_ap,
            "conservative_no_references_average_precision": (
                conservative_no_refs_ap
            ),
            "conservative_beats_cvss": (
                conservative_ap > cvss_ap
            ),
            "conservative_no_references_beats_cvss": (
                conservative_no_refs_ap > cvss_ap
            ),
        },

        "leakage_controls": [
            "exploitation_label excluded",
            "known_exploited excluded",
            "KEV metadata excluded",
            "ransomware_use excluded",
            "KEV dates excluded",
            "KEV vendor/project excluded",
            "KEV product excluded",
            "KEV vulnerability name excluded",
            "KEV required action excluded",
            "current days_since_publication excluded",
            "modification timing removed from conservative model",
            "MITRE ATT&CK removed from conservative model",
            "references removed in the strictest model",
            "raw publication date used only for chronological splitting",
        ],

        "interpretation": (
            "The conservative models are designed to test whether "
            "ThreatLens performance remains above CVSS when feature "
            "groups with stronger temporal provenance concerns are "
            "removed."
        ),

        "limitation": (
            "The current experiment uses present-day NVD metadata "
            "and a current CISA KEV snapshot. Removing risky feature "
            "groups reduces temporal concerns but does not create a "
            "perfect historical point-in-time dataset."
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

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 72)
    print(
        "CONSERVATIVE MODEL RESULTS"
    )
    print("=" * 72)

    display_columns = [
        "rank",
        "model",
        "feature_count",
        "average_precision",
        "precision_at_100",
        "recall_at_500",
        "ap_improvement_percent_vs_cvss",
        "ap_change_percent_vs_all_features",
    ]

    print(
        summary_df[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print("\n" + "=" * 72)
    print(
        "OUTPUT FILES"
    )
    print("=" * 72)

    print(
        RESULTS_PATH
    )

    print(
        SUMMARY_PATH
    )

    print(
        REPORT_PATH
    )

    print(
        "\nConservative model evaluation complete."
    )


if __name__ == "__main__":
    main()
