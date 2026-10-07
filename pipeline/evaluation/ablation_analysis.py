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

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = OUTPUT_DIR / "ablation_results.csv"
SUMMARY_PATH = OUTPUT_DIR / "ablation_summary.csv"
REPORT_PATH = OUTPUT_DIR / "ablation_report.json"


# ============================================================
# MODEL CONFIGURATION
# Same configuration used in the previous ThreatLens
# baseline and interpretability experiments.
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
# FEATURE GROUPS
# ============================================================

FEATURE_GROUPS = {
    "CVSS": [
        "cvss_v3_score",
        "cvss_v4_score",
        "cvss_v3_available",
        "cvss_v4_available",
        "cvss_max_score",
        "cvss_score_difference",
    ],

    "Severity": [
        "severity_score",
        "severity_known",
    ],

    "CWE": [
        "cwe_count",
        "cwe_unique_count",
        "has_cwe",
    ],

    "Affected CPE": [
        "affected_cpe_count",
        "affected_cpe_unique_count",
        "has_affected_cpe",
    ],

    "References": [
        "reference_count",
        "reference_unique_count",
        "has_references",
    ],

    "Description": [
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

    "Publication timing": [
        "publication_year",
        "publication_month",
        "publication_quarter",
        "publication_day_of_week",
        "publication_day_of_year",
        "publication_is_weekend",
    ],

    "Modification timing": [
        "modified_after_publication_days",
        "has_modified_date",
    ],

    "MITRE ATT&CK": [
        "attack_technique_count",
        "has_attack_technique",
    ],
}


# ============================================================
# DATA LOADING
# ============================================================

def load_features() -> pd.DataFrame:
    print("Loading ThreatLens feature dataset...")

    df = pd.read_csv(FEATURES_PATH)

    print(f"Rows loaded: {len(df):,}")
    print(f"Columns loaded: {len(df.columns):,}")

    return df


def load_enriched_dataset() -> list[dict]:
    print("\nLoading enriched dataset...")

    with open(ENRICHED_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    print(f"Enriched rows loaded: {len(data):,}")

    return data


# ============================================================
# VALIDATION
# ============================================================

def validate_alignment(
    features_df: pd.DataFrame,
    enriched_data: list[dict],
) -> None:

    print("\nValidating datasets...")

    if len(features_df) != len(enriched_data):
        raise ValueError(
            "Feature dataset and enriched dataset row counts do not match."
        )

    print("Row count alignment: PASS")

    feature_targets = features_df["exploitation_label"].astype(int).to_numpy()

    enriched_targets = np.array(
        [
            int(record.get("exploitation_label", 0))
            for record in enriched_data
        ]
    )

    if not np.array_equal(feature_targets, enriched_targets):
        raise ValueError(
            "Target alignment failed between feature and enriched datasets."
        )

    print("Target alignment: PASS")


# ============================================================
# FEATURE COLUMN SELECTION
# ============================================================

def get_feature_columns(df: pd.DataFrame) -> list[str]:

    columns = [
        column
        for column in df.columns
        if column != "exploitation_label"
    ]

    return columns


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

def chronological_split(
    features_df: pd.DataFrame,
    enriched_data: list[dict],
):
    print("\nBuilding chronological split...")

    dates = pd.to_datetime(
        [record.get("published_date") for record in enriched_data],
        errors="coerce",
        utc=True,
    )

    dates = pd.Series(dates).dt.tz_localize(None)

    valid_mask = dates.notna()

    valid_count = int(valid_mask.sum())
    missing_count = int((~valid_mask).sum())

    print(f"Valid publication dates: {valid_count:,}")
    print(f"Missing publication dates: {missing_count:,}")

    valid_indices = np.where(valid_mask.to_numpy())[0]

    dated_df = features_df.iloc[valid_indices].copy()
    dated_dates = dates.iloc[valid_indices].copy()

    dated_df["_published_date"] = dated_dates.to_numpy()

    dated_df = dated_df.sort_values(
        "_published_date"
    ).reset_index(drop=True)

    split_date = pd.Timestamp("2026-01-01")

    train_df = dated_df[
        dated_df["_published_date"] < split_date
    ].copy()

    test_df = dated_df[
        dated_df["_published_date"] >= split_date
    ].copy()

    train_dates = train_df.pop("_published_date")
    test_dates = test_df.pop("_published_date")

    print(f"Rows available for temporal evaluation: {len(dated_df):,}")
    print(f"Excluded from temporal evaluation: {missing_count:,}")

    print(
        "Publication range: "
        f"{dated_dates.min().date()} to {dated_dates.max().date()}"
    )

    print(f"\nTraining rows: {len(train_df):,}")
    print(f"Test rows: {len(test_df):,}")

    train_positive = int(train_df["exploitation_label"].sum())
    test_positive = int(test_df["exploitation_label"].sum())

    print(f"Training KEV positives: {train_positive:,}")
    print(f"Test KEV positives: {test_positive:,}")

    return train_df, test_df


# ============================================================
# MATRIX PREPARATION
# ============================================================

def prepare_matrix(
    df: pd.DataFrame,
    feature_columns: list[str],
):
    X = df[feature_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    missing_before = int(X.isna().sum().sum())

    X = X.fillna(0.0)

    missing_after = int(X.isna().sum().sum())

    if missing_after != 0:
        raise ValueError(
            "Missing values remain after filling."
        )

    y = df["exploitation_label"].astype(int)

    return X, y, missing_before


# ============================================================
# TOP-K METRICS
# ============================================================

def precision_at_k(
    y_true: np.ndarray,
    scores: np.ndarray,
    k: int,
) -> float:

    k = min(k, len(y_true))

    order = np.argsort(-scores)[:k]

    return float(np.mean(y_true[order]))


def recall_at_k(
    y_true: np.ndarray,
    scores: np.ndarray,
    k: int,
) -> float:

    total_positive = int(y_true.sum())

    if total_positive == 0:
        return 0.0

    k = min(k, len(y_true))

    order = np.argsort(-scores)[:k]

    return float(y_true[order].sum() / total_positive)


def coverage_at_k(
    y_true: np.ndarray,
    scores: np.ndarray,
    k: int,
) -> float:

    return recall_at_k(y_true, scores, k)


# ============================================================
# EVALUATION
# ============================================================

def evaluate_scores(
    model_name: str,
    y_true: np.ndarray,
    scores: np.ndarray,
) -> dict:

    result = {
        "model": model_name,
        "test_rows": int(len(y_true)),
        "test_positives": int(y_true.sum()),
        "positive_rate": float(y_true.mean()),
        "average_precision": float(
            average_precision_score(y_true, scores)
        ),
    }

    for k in [10, 50, 100, 500]:
        result[f"precision_at_{k}"] = precision_at_k(
            y_true,
            scores,
            k,
        )

        result[f"recall_at_{k}"] = recall_at_k(
            y_true,
            scores,
            k,
        )

        result[f"kev_coverage_at_{k}"] = coverage_at_k(
            y_true,
            scores,
            k,
        )

    return result


# ============================================================
# CVSS BASELINE
# ============================================================

def calculate_cvss_score(df: pd.DataFrame) -> np.ndarray:

    v4 = pd.to_numeric(
        df["cvss_v4_score"],
        errors="coerce",
    ).fillna(0.0)

    v3 = pd.to_numeric(
        df["cvss_v3_score"],
        errors="coerce",
    ).fillna(0.0)

    v4_available = (
        pd.to_numeric(
            df["cvss_v4_available"],
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

    return scores.astype(float)


# ============================================================
# FEATURE GROUP REMOVAL
# ============================================================

def features_without_group(
    all_features: list[str],
    group_name: str,
) -> list[str]:

    removed = set(FEATURE_GROUPS[group_name])

    remaining = [
        feature
        for feature in all_features
        if feature not in removed
    ]

    if not remaining:
        raise ValueError(
            f"Removing {group_name} leaves no model features."
        )

    return remaining


# ============================================================
# TRAIN RANDOM FOREST
# ============================================================

def train_random_forest(
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
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("ThreatLens - Ablation & Robustness Analysis")
    print("=" * 72)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    features_df = load_features()
    enriched_data = load_enriched_dataset()

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_alignment(
        features_df,
        enriched_data,
    )

    all_features = get_feature_columns(
        features_df
    )

    print(f"\nModel features: {len(all_features)}")

    # --------------------------------------------------------
    # Check feature groups
    # --------------------------------------------------------

    print("\nValidating feature groups...")

    missing_group_features = []

    for group_name, group_features in FEATURE_GROUPS.items():

        for feature in group_features:

            if feature not in all_features:
                missing_group_features.append(
                    f"{group_name}: {feature}"
                )

    if missing_group_features:

        print("Missing feature-group columns:")

        for item in missing_group_features:
            print(f"  - {item}")

        raise ValueError(
            "Feature group definitions do not match dataset."
        )

    print("Feature group validation: PASS")

    # --------------------------------------------------------
    # Chronological split
    # --------------------------------------------------------

    train_df, test_df = chronological_split(
        features_df,
        enriched_data,
    )

    # --------------------------------------------------------
    # Prepare target
    # --------------------------------------------------------

    y_train = train_df[
        "exploitation_label"
    ].astype(int)

    y_test = test_df[
        "exploitation_label"
    ].astype(int)

    y_test_array = y_test.to_numpy()

    # --------------------------------------------------------
    # Full feature matrix
    # --------------------------------------------------------

    print("\nPreparing full training matrix...")

    X_train_full, _, missing_train = prepare_matrix(
        train_df,
        all_features,
    )

    print(
        f"Training missing values before filling: "
        f"{missing_train:,}"
    )

    print("\nPreparing full test matrix...")

    X_test_full, _, missing_test = prepare_matrix(
        test_df,
        all_features,
    )

    print(
        f"Test missing values before filling: "
        f"{missing_test:,}"
    )

    results = []

    # ========================================================
    # 1. FULL THREATLENS MODEL
    # ========================================================

    print("\n" + "=" * 72)
    print("BASE MODEL: ALL FEATURES")
    print("=" * 72)

    print("Training Random Forest...")

    full_model = train_random_forest(
        X_train_full,
        y_train,
    )

    full_scores = full_model.predict_proba(
        X_test_full
    )[:, 1]

    full_result = evaluate_scores(
        "ThreatLens ML - All Features",
        y_test_array,
        full_scores,
    )

    full_result["removed_group"] = "None"
    full_result["remaining_features"] = len(all_features)

    results.append(full_result)

    print(
        f"Average Precision: "
        f"{full_result['average_precision']:.6f}"
    )

    # ========================================================
    # 2. ABLATION EXPERIMENTS
    # ========================================================

    for group_name in FEATURE_GROUPS:

        print("\n" + "=" * 72)
        print(
            f"ABLATION: WITHOUT {group_name.upper()}"
        )
        print("=" * 72)

        remaining_features = features_without_group(
            all_features,
            group_name,
        )

        print(
            f"Removed features: "
            f"{len(all_features) - len(remaining_features)}"
        )

        print(
            f"Remaining features: "
            f"{len(remaining_features)}"
        )

        X_train, _, missing_train = prepare_matrix(
            train_df,
            remaining_features,
        )

        X_test, _, missing_test = prepare_matrix(
            test_df,
            remaining_features,
        )

        print(
            f"Training missing values filled: "
            f"{missing_train:,}"
        )

        print(
            f"Test missing values filled: "
            f"{missing_test:,}"
        )

        print("Training Random Forest...")

        model = train_random_forest(
            X_train,
            y_train,
        )

        scores = model.predict_proba(
            X_test
        )[:, 1]

        result = evaluate_scores(
            f"ThreatLens ML - Without {group_name}",
            y_test_array,
            scores,
        )

        result["removed_group"] = group_name
        result["remaining_features"] = len(
            remaining_features
        )

        results.append(result)

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

    # ========================================================
    # 3. CVSS-ONLY BASELINE
    # ========================================================

    print("\n" + "=" * 72)
    print("BASELINE: CVSS-ONLY")
    print("=" * 72)

    cvss_scores = calculate_cvss_score(
        test_df
    )

    cvss_result = evaluate_scores(
        "CVSS-only",
        y_test_array,
        cvss_scores,
    )

    cvss_result["removed_group"] = "All contextual features"
    cvss_result["remaining_features"] = 2

    results.append(cvss_result)

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

    # ========================================================
    # RESULTS DATAFRAME
    # ========================================================

    results_df = pd.DataFrame(results)

    results_df = results_df[
        [
            "model",
            "removed_group",
            "remaining_features",
            "test_rows",
            "test_positives",
            "positive_rate",
            "average_precision",
            "precision_at_10",
            "precision_at_50",
            "precision_at_100",
            "precision_at_500",
            "recall_at_10",
            "recall_at_50",
            "recall_at_100",
            "recall_at_500",
            "kev_coverage_at_10",
            "kev_coverage_at_50",
            "kev_coverage_at_100",
            "kev_coverage_at_500",
        ]
    ]

    # ========================================================
    # IMPACT CALCULATION
    # ========================================================

    full_ap = float(
        results_df.loc[
            results_df["model"]
            == "ThreatLens ML - All Features",
            "average_precision",
        ].iloc[0]
    )

    results_df["ap_change_vs_full"] = (
        results_df["average_precision"]
        - full_ap
    )

    results_df["ap_change_percent_vs_full"] = (
        results_df["ap_change_vs_full"]
        / full_ap
        * 100.0
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df.to_csv(
        RESULTS_PATH,
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary_columns = [
        "model",
        "removed_group",
        "remaining_features",
        "average_precision",
        "precision_at_100",
        "precision_at_500",
        "recall_at_100",
        "recall_at_500",
        "kev_coverage_at_100",
        "kev_coverage_at_500",
        "ap_change_vs_full",
        "ap_change_percent_vs_full",
    ]

    summary_df = results_df[
        summary_columns
    ].copy()

    summary_df = summary_df.sort_values(
        "average_precision",
        ascending=False,
    ).reset_index(drop=True)

    summary_df.insert(
        0,
        "rank",
        np.arange(1, len(summary_df) + 1),
    )

    summary_df.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    # ========================================================
    # REPORT
    # ========================================================

    report = {
        "experiment": "ThreatLens Ablation & Robustness Analysis",
        "research_question": (
            "Which feature groups contribute most to "
            "context-aware vulnerability prioritization?"
        ),
        "dataset": {
            "feature_rows": int(len(features_df)),
            "temporal_rows": int(
                len(train_df) + len(test_df)
            ),
            "training_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "training_kev_positives": int(
                y_train.sum()
            ),
            "test_kev_positives": int(
                y_test.sum()
            ),
        },
        "chronological_split": {
            "train_before": "2026-01-01",
            "test_from": "2026-01-01",
            "raw_publication_date_used_for_split": True,
        },
        "model": {
            "type": "RandomForestClassifier",
            "configuration": MODEL_CONFIG,
        },
        "feature_groups": FEATURE_GROUPS,
        "interpretation": {
            "full_model_average_precision": full_ap,
            "best_ablation": (
                summary_df.iloc[1]["model"]
                if len(summary_df) > 1
                else None
            ),
            "note": (
                "Ablation impact is measured by the change in "
                "Average Precision after removing one feature "
                "group while keeping the same chronological "
                "evaluation setup and Random Forest configuration."
            ),
        },
        "leakage_controls": [
            "exploitation_label excluded from model features",
            "known_exploited excluded from model features",
            "KEV metadata excluded from model features",
            "ransomware_use excluded from model features",
            "kev_date_added excluded from model features",
            "kev_due_date excluded from model features",
            "kev_vendor_project excluded from model features",
            "kev_product excluded from model features",
            "kev_vulnerability_name excluded from model features",
            "kev_required_action excluded from model features",
            "current days_since_publication feature excluded",
            "raw publication date used only for chronological splitting",
            "missing publication dates excluded from temporal evaluation",
        ],
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
    # CONSOLE SUMMARY
    # ========================================================

    print("\n")
    print("=" * 72)
    print("ABLATION RESULTS")
    print("=" * 72)

    display_columns = [
        "rank",
        "model",
        "average_precision",
        "precision_at_100",
        "recall_at_500",
        "kev_coverage_at_500",
        "ap_change_percent_vs_full",
    ]

    print(
        summary_df[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\n" + "=" * 72)
    print("OUTPUT FILES")
    print("=" * 72)

    print(RESULTS_PATH)
    print(SUMMARY_PATH)
    print(REPORT_PATH)

    print("\nAblation analysis complete.")


if __name__ == "__main__":
    main()