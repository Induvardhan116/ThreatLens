from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance


# ================================================================
# PATHS
# ================================================================

ROOT = Path(__file__).resolve().parents[2]

FEATURES_PATH = (
    ROOT
    / "data"
    / "processed"
    / "features"
    / "threatlens_features.csv"
)

DATASET_PATH = (
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

FEATURE_IMPORTANCE_PATH = (
    OUTPUT_DIR
    / "feature_importance.csv"
)

PERMUTATION_IMPORTANCE_PATH = (
    OUTPUT_DIR
    / "permutation_importance.csv"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "interpretability_report.json"
)


# ================================================================
# MODEL CONFIGURATION
# ================================================================

RANDOM_STATE = 42

N_ESTIMATORS = 300

MAX_DEPTH = 12

MIN_SAMPLES_LEAF = 3

PERMUTATION_REPEATS = 5

MAX_PERMUTATION_SAMPLES = 10000

TRAIN_END = "2026-01-01"


# ================================================================
# LEAKAGE COLUMNS
# ================================================================

LEAKAGE_COLUMNS = {
    "exploitation_label",
}


# ================================================================
# LOAD FEATURES
# ================================================================

def load_features() -> pd.DataFrame:

    print(
        "Loading ThreatLens feature dataset..."
    )

    if not FEATURES_PATH.exists():

        raise FileNotFoundError(
            f"Feature dataset not found:\n"
            f"{FEATURES_PATH}"
        )

    df = pd.read_csv(
        FEATURES_PATH
    )

    print(
        f"Rows loaded: {len(df):,}"
    )

    print(
        f"Columns loaded: {len(df.columns)}"
    )

    return df


# ================================================================
# LOAD ENRICHED DATASET
# ================================================================

def load_dataset() -> pd.DataFrame:

    print(
        "\nLoading enriched dataset..."
    )

    if not DATASET_PATH.exists():

        raise FileNotFoundError(
            f"Enriched dataset not found:\n"
            f"{DATASET_PATH}"
        )

    with DATASET_PATH.open(
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
        f"Enriched rows loaded: "
        f"{len(df):,}"
    )

    required = [
        "cve_id",
        "published_date",
        "exploitation_label",
    ]

    for column in required:

        if column not in df.columns:

            raise ValueError(
                f"Missing required column: "
                f"{column}"
            )

    return df


# ================================================================
# VALIDATE ALIGNMENT
# ================================================================

def validate_alignment(
    features: pd.DataFrame,
    dataset: pd.DataFrame,
) -> None:

    print(
        "\nValidating datasets..."
    )

    if len(features) != len(dataset):

        raise ValueError(
            "Feature and enriched datasets "
            "have different row counts."
        )

    feature_labels = pd.to_numeric(
        features[
            "exploitation_label"
        ],
        errors="coerce",
    ).fillna(0).astype(int)

    dataset_labels = pd.to_numeric(
        dataset[
            "exploitation_label"
        ],
        errors="coerce",
    ).fillna(0).astype(int)

    if not np.array_equal(
        feature_labels.to_numpy(),
        dataset_labels.to_numpy(),
    ):

        raise ValueError(
            "Feature labels and enriched "
            "dataset labels do not align."
        )

    print(
        "Row count alignment: PASS"
    )

    print(
        "Target alignment: PASS"
    )


# ================================================================
# GET MODEL FEATURES
# ================================================================

def get_feature_columns(
    df: pd.DataFrame,
) -> list[str]:

    columns = [
        column
        for column in df.columns
        if column not in LEAKAGE_COLUMNS
    ]

    if not columns:

        raise ValueError(
            "No model features available."
        )

    return columns


# ================================================================
# PREPARE NUMERIC DATA
# ================================================================

def prepare_matrix(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:

    matrix = df[
        feature_columns
    ].copy()

    for column in feature_columns:

        matrix[column] = pd.to_numeric(
            matrix[column],
            errors="coerce",
        )

    missing = int(
        matrix.isna().sum().sum()
    )

    print(
        f"Missing values before filling: "
        f"{missing:,}"
    )

    matrix = matrix.fillna(
        0.0
    )

    remaining = int(
        matrix.isna().sum().sum()
    )

    print(
        f"Missing values after filling: "
        f"{remaining:,}"
    )

    if remaining != 0:

        raise ValueError(
            "Missing values remain after filling."
        )

    return matrix


# ================================================================
# BUILD CHRONOLOGICAL SPLIT
# ================================================================

def chronological_split(
    features: pd.DataFrame,
    dataset: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    print(
        "\nBuilding chronological split..."
    )

    # ------------------------------------------------------------
    # IMPORTANT:
    # Use the actual published_date from the enriched JSON.
    # There is NO reconstruction from publication_year/month/day.
    # ------------------------------------------------------------

    dates = pd.to_datetime(
        dataset[
            "published_date"
        ],
        errors="coerce",
        utc=True,
    )

    # Convert UTC timestamps to timezone-naive timestamps.
    dates = dates.dt.tz_localize(
        None
    )

    missing_dates = int(
        dates.isna().sum()
    )

    print(
        f"Valid publication dates: "
        f"{len(dates) - missing_dates:,}"
    )

    print(
        f"Missing publication dates: "
        f"{missing_dates:,}"
    )

    # ------------------------------------------------------------
    # Keep only records with valid dates for this experiment.
    # ------------------------------------------------------------

    valid_mask = dates.notna()

    working_features = (
        features.loc[
            valid_mask
        ]
        .copy()
        .reset_index(drop=True)
    )

    working_dataset = (
        dataset.loc[
            valid_mask
        ]
        .copy()
        .reset_index(drop=True)
    )

    working_dates = (
        dates.loc[
            valid_mask
        ]
        .reset_index(drop=True)
    )

    print(
        f"Rows available for temporal "
        f"evaluation: {len(working_features):,}"
    )

    print(
        f"Excluded from temporal evaluation: "
        f"{missing_dates:,}"
    )

    # ------------------------------------------------------------
    # Sort by ACTUAL publication date.
    # ------------------------------------------------------------

    order = np.argsort(
        working_dates.to_numpy()
    )

    working_features = (
        working_features
        .iloc[order]
        .reset_index(drop=True)
    )

    working_dataset = (
        working_dataset
        .iloc[order]
        .reset_index(drop=True)
    )

    working_dates = (
        working_dates
        .iloc[order]
        .reset_index(drop=True)
    )

    print(
        f"Publication range: "
        f"{working_dates.min().date()} "
        f"to "
        f"{working_dates.max().date()}"
    )

    # ------------------------------------------------------------
    # Train/test boundary.
    # ------------------------------------------------------------

    train_end = pd.Timestamp(
        TRAIN_END
    )

    train_mask = (
        working_dates
        < train_end
    )

    test_mask = (
        working_dates
        >= train_end
    )

    train_features = (
        working_features.loc[
            train_mask
        ]
        .copy()
        .reset_index(drop=True)
    )

    test_features = (
        working_features.loc[
            test_mask
        ]
        .copy()
        .reset_index(drop=True)
    )

    print(
        f"\nTraining rows: "
        f"{len(train_features):,}"
    )

    print(
        f"Test rows: "
        f"{len(test_features):,}"
    )

    train_positive = int(
        train_features[
            "exploitation_label"
        ].sum()
    )

    test_positive = int(
        test_features[
            "exploitation_label"
        ].sum()
    )

    print(
        f"Training KEV positives: "
        f"{train_positive:,}"
    )

    print(
        f"Test KEV positives: "
        f"{test_positive:,}"
    )

    if len(train_features) == 0:

        raise ValueError(
            "Training dataset is empty."
        )

    if len(test_features) == 0:

        raise ValueError(
            "Test dataset is empty."
        )

    if train_positive == 0:

        raise ValueError(
            "Training dataset has no positive examples."
        )

    if test_positive == 0:

        raise ValueError(
            "Test dataset has no positive examples."
        )

    return (
        train_features,
        test_features,
    )


# ================================================================
# TRAIN RANDOM FOREST
# ================================================================

def train_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestClassifier:

    print(
        "\nTraining Random Forest..."
    )

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        x_train,
        y_train,
    )

    print(
        "Random Forest training complete."
    )

    return model


# ================================================================
# GLOBAL FEATURE IMPORTANCE
# ================================================================

def calculate_global_importance(
    model: RandomForestClassifier,
    feature_columns: list[str],
) -> pd.DataFrame:

    print(
        "\nCalculating global feature importance..."
    )

    result = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": (
                model.feature_importances_
            ),
        }
    )

    result = (
        result
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    result["rank"] = (
        np.arange(
            len(result)
        )
        + 1
    )

    result["importance_percent"] = (
        result[
            "importance"
        ]
        * 100
    )

    result["category"] = (
        result[
            "feature"
        ].apply(
            feature_category
        )
    )

    return result[
        [
            "rank",
            "feature",
            "category",
            "importance",
            "importance_percent",
        ]
    ]


# ================================================================
# PERMUTATION IMPORTANCE
# ================================================================

def calculate_permutation_importance(
    model: RandomForestClassifier,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    feature_columns: list[str],
) -> pd.DataFrame:

    print(
        "\nCalculating permutation importance..."
    )

    # ------------------------------------------------------------
    # Limit computation to 10,000 deterministic test rows.
    # ------------------------------------------------------------

    if len(x_test) > MAX_PERMUTATION_SAMPLES:

        print(
            f"Using {MAX_PERMUTATION_SAMPLES:,} "
            "test samples for permutation analysis."
        )

        rng = np.random.RandomState(
            RANDOM_STATE
        )

        indices = rng.choice(
            len(x_test),
            size=MAX_PERMUTATION_SAMPLES,
            replace=False,
        )

        indices = np.sort(
            indices
        )

        x_perm = x_test.iloc[
            indices
        ].copy()

        y_perm = y_test.iloc[
            indices
        ].copy()

    else:

        x_perm = x_test.copy()

        y_perm = y_test.copy()

    print(
        f"Permutation rows: "
        f"{len(x_perm):,}"
    )

    result = permutation_importance(
        model,
        x_perm,
        y_perm,
        scoring="average_precision",
        n_repeats=PERMUTATION_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    output = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance_mean": (
                result.importances_mean
            ),
            "importance_std": (
                result.importances_std
            ),
        }
    )

    output = (
        output
        .sort_values(
            "importance_mean",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    output["rank"] = (
        np.arange(
            len(output)
        )
        + 1
    )

    output["importance_percent"] = (
        output[
            "importance_mean"
        ]
        * 100
    )

    output["category"] = (
        output[
            "feature"
        ].apply(
            feature_category
        )
    )

    return output[
        [
            "rank",
            "feature",
            "category",
            "importance_mean",
            "importance_std",
            "importance_percent",
        ]
    ]


# ================================================================
# FEATURE CATEGORIES
# ================================================================

def feature_category(
    feature: str,
) -> str:

    if feature.startswith("cvss_"):
        return "CVSS"

    if feature.startswith("severity"):
        return "Severity"

    if feature.startswith("cwe"):
        return "CWE"

    if feature.startswith("affected_cpe"):
        return "Affected CPE"

    if feature.startswith("reference"):
        return "References"

    if feature.startswith("description_"):
        return "Description"

    if feature.startswith("publication_"):
        return "Publication timing"

    if feature.startswith("modified_"):
        return "Modification timing"

    if feature.startswith("attack_"):
        return "MITRE ATT&CK"

    if feature.startswith("has_attack"):
        return "MITRE ATT&CK"

    return "Other"


# ================================================================
# CATEGORY SUMMARY
# ================================================================

def category_summary(
    importance_df: pd.DataFrame,
    importance_column: str,
) -> pd.DataFrame:

    summary = (
        importance_df
        .groupby(
            "category",
            as_index=False,
        )[
            importance_column
        ]
        .sum()
        .sort_values(
            importance_column,
            ascending=False,
        )
        .reset_index(drop=True)
    )

    summary[
        "importance_percent"
    ] = (
        summary[
            importance_column
        ]
        * 100
    )

    return summary


# ================================================================
# BUILD JSON REPORT
# ================================================================

def build_report(
    global_importance: pd.DataFrame,
    permutation_importance_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:

    global_categories = (
        category_summary(
            global_importance,
            "importance",
        )
    )

    permutation_categories = (
        category_summary(
            permutation_importance_df,
            "importance_mean",
        )
    )

    report = {
        "experiment": (
            "ThreatLens Model Interpretability"
        ),

        "model": {
            "algorithm": (
                "RandomForestClassifier"
            ),
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "min_samples_leaf": MIN_SAMPLES_LEAF,
            "class_weight": (
                "balanced_subsample"
            ),
            "random_state": RANDOM_STATE,
        },

        "chronological_evaluation": {
            "train_end": TRAIN_END,
            "train_rows": int(
                len(train_df)
            ),
            "train_positives": int(
                train_df[
                    "exploitation_label"
                ].sum()
            ),
            "test_rows": int(
                len(test_df)
            ),
            "test_positives": int(
                test_df[
                    "exploitation_label"
                ].sum()
            ),
        },

        "feature_count": int(
            len(global_importance)
        ),

        "top_15_global_features": (
            global_importance
            .head(15)
            .to_dict(
                orient="records"
            )
        ),

        "top_15_permutation_features": (
            permutation_importance_df
            .head(15)
            .to_dict(
                orient="records"
            )
        ),

        "global_importance_by_category": (
            global_categories
            .to_dict(
                orient="records"
            )
        ),

        "permutation_importance_by_category": (
            permutation_categories
            .to_dict(
                orient="records"
            )
        ),

        "leakage_controls": {
            "exploitation_label_excluded": True,
            "kev_metadata_excluded": True,
            "cve_identifier_excluded": True,
            "current_days_since_publication_excluded": True,
        },

        "missing_value_handling": {
            "method": "zero_fill",
            "applied_before_model_training": True,
        },

        "interpretation_notes": [
            (
                "Global Random Forest importance measures "
                "feature contribution to tree splitting."
            ),
            (
                "Permutation importance measures the change "
                "in Average Precision after feature shuffling "
                "on chronological test data."
            ),
            (
                "Permutation importance is the preferred "
                "indicator of out-of-sample feature usefulness."
            ),
            (
                "Feature importance does not establish causality."
            ),
            (
                "KEV-derived target and KEV metadata are "
                "excluded from model features."
            ),
            (
                "Publication dates are used only for "
                "chronological splitting."
            ),
        ],
    }

    return report


# ================================================================
# MAIN
# ================================================================

def main() -> None:

    print("=" * 72)
    print(
        "ThreatLens - Model Interpretability"
    )
    print("=" * 72)

    # ------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------

    features = load_features()

    dataset = load_dataset()

    # ------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------

    validate_alignment(
        features,
        dataset,
    )

    feature_columns = (
        get_feature_columns(
            features
        )
    )

    print(
        f"\nModel features: "
        f"{len(feature_columns)}"
    )

    # ------------------------------------------------------------
    # Validate feature types
    # ------------------------------------------------------------

    non_numeric = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(
            features[column]
        )
    ]

    if non_numeric:

        raise ValueError(
            "Non-numeric features found: "
            f"{non_numeric}"
        )

    missing_total = int(
        features[
            feature_columns
        ]
        .isna()
        .sum()
        .sum()
    )

    print(
        f"Total missing feature values: "
        f"{missing_total:,}"
    )

    # ------------------------------------------------------------
    # Chronological split
    # ------------------------------------------------------------

    (
        train_df,
        test_df,
    ) = chronological_split(
        features,
        dataset,
    )

    # ------------------------------------------------------------
    # Prepare matrices
    # ------------------------------------------------------------

    print(
        "\nPreparing training matrix..."
    )

    x_train = prepare_matrix(
        train_df,
        feature_columns,
    )

    print(
        "\nPreparing test matrix..."
    )

    x_test = prepare_matrix(
        test_df,
        feature_columns,
    )

    y_train = train_df[
        "exploitation_label"
    ].astype(int)

    y_test = test_df[
        "exploitation_label"
    ].astype(int)

    # ------------------------------------------------------------
    # Train
    # ------------------------------------------------------------

    model = train_model(
        x_train,
        y_train,
    )

    # ------------------------------------------------------------
    # Global importance
    # ------------------------------------------------------------

    global_importance = (
        calculate_global_importance(
            model,
            feature_columns,
        )
    )

    global_importance.to_csv(
        FEATURE_IMPORTANCE_PATH,
        index=False,
    )

    # ------------------------------------------------------------
    # Permutation importance
    # ------------------------------------------------------------

    permutation_df = (
        calculate_permutation_importance(
            model,
            x_test,
            y_test,
            feature_columns,
        )
    )

    permutation_df.to_csv(
        PERMUTATION_IMPORTANCE_PATH,
        index=False,
    )

    # ------------------------------------------------------------
    # JSON report
    # ------------------------------------------------------------

    report = build_report(
        global_importance,
        permutation_df,
        train_df,
        test_df,
    )

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
        )

    # ============================================================
    # DISPLAY TOP GLOBAL FEATURES
    # ============================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "TOP 15 GLOBAL FEATURE IMPORTANCES"
    )

    print(
        "=" * 72
    )

    print(
        global_importance[
            [
                "rank",
                "feature",
                "category",
                "importance_percent",
            ]
        ]
        .head(15)
        .to_string(
            index=False
        )
    )

    # ============================================================
    # DISPLAY TOP PERMUTATION FEATURES
    # ============================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "TOP 15 PERMUTATION IMPORTANCES"
    )

    print(
        "=" * 72
    )

    print(
        permutation_df[
            [
                "rank",
                "feature",
                "category",
                "importance_percent",
                "importance_std",
            ]
        ]
        .head(15)
        .to_string(
            index=False
        )
    )

    # ============================================================
    # GLOBAL CATEGORY SUMMARY
    # ============================================================

    global_categories = (
        category_summary(
            global_importance,
            "importance",
        )
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "GLOBAL IMPORTANCE BY CATEGORY"
    )

    print(
        "=" * 72
    )

    print(
        global_categories.to_string(
            index=False
        )
    )

    # ============================================================
    # PERMUTATION CATEGORY SUMMARY
    # ============================================================

    permutation_categories = (
        category_summary(
            permutation_df,
            "importance_mean",
        )
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "PERMUTATION IMPORTANCE BY CATEGORY"
    )

    print(
        "=" * 72
    )

    print(
        permutation_categories.to_string(
            index=False
        )
    )

    # ============================================================
    # OUTPUT
    # ============================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "OUTPUT FILES"
    )

    print(
        "=" * 72
    )

    print(
        FEATURE_IMPORTANCE_PATH
    )

    print(
        PERMUTATION_IMPORTANCE_PATH
    )

    print(
        REPORT_PATH
    )

    print(
        "\nInterpretability analysis complete."
    )


if __name__ == "__main__":
    main()