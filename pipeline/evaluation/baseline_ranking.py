from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
    / "threatlens_features.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULT_FILE = (
    OUTPUT_DIR
    / "baseline_results.json"
)


TARGET = "exploitation_label"


# ------------------------------------------------------------
# Chronological split
#
# Training = older vulnerabilities
# Test     = newer vulnerabilities
#
# This prevents future information from entering training.
# ------------------------------------------------------------

TRAIN_FRACTION = 0.80


# ------------------------------------------------------------
# Ranking cutoffs
# ------------------------------------------------------------

TOP_K_VALUES = [
    10,
    50,
    100,
    500,
]


# ------------------------------------------------------------
# Features that are intentionally NOT ML predictors.
# ------------------------------------------------------------

EXCLUDED_FEATURES = {
    TARGET,
}


# ------------------------------------------------------------
# CVSS baseline
# ------------------------------------------------------------

def build_cvss_score(
    df: pd.DataFrame,
) -> pd.Series:

    v3 = pd.to_numeric(
        df["cvss_v3_score"],
        errors="coerce",
    )

    v4 = pd.to_numeric(
        df["cvss_v4_score"],
        errors="coerce",
    )

    # Prefer CVSS v4 when available.
    # Otherwise use CVSS v3.
    score = v4.fillna(v3)

    return score.fillna(0.0)


# ------------------------------------------------------------
# Contextual rule baseline
# ------------------------------------------------------------

def build_contextual_score(
    df: pd.DataFrame,
) -> pd.Series:

    # Start with CVSS because severity remains
    # an important vulnerability signal.
    cvss = build_cvss_score(
        df
    )

    score = (
        cvss
        / 10.0
    )

    # --------------------------------------------------------
    # Context signals
    #
    # These are intentionally simple and transparent.
    # The purpose is to establish a non-ML baseline.
    # --------------------------------------------------------

    score = (
        score
        + 0.04
        * df["reference_count"].clip(
            upper=10
        )
        / 10.0
    )

    score = (
        score
        + 0.04
        * df["affected_cpe_count"].clip(
            upper=10
        )
        / 10.0
    )

    score = (
        score
        + 0.04
        * df["cwe_count"].clip(
            upper=5
        )
        / 5.0
    )

    score = (
        score
        + 0.04
        * df["description_has_remote"]
    )

    score = (
        score
        + 0.04
        * df["description_has_execution"]
    )

    score = (
        score
        + 0.03
        * df["description_has_privilege"]
    )

    score = (
        score
        + 0.03
        * df["description_has_authentication"]
    )

    score = (
        score
        + 0.03
        * df["description_has_overflow"]
    )

    score = (
        score
        + 0.03
        * df["description_has_injection"]
    )

    score = (
        score
        + 0.03
        * df["description_has_bypass"]
    )

    # Slight signal for vulnerabilities affecting
    # more products/components.
    score = (
        score
        + 0.02
        * np.log1p(
            df["affected_cpe_count"]
        )
    )

    return score


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

def ranking_metrics(
    y_true: pd.Series,
    scores: pd.Series,
) -> dict:

    y = np.asarray(
        y_true,
        dtype=int,
    )

    s = np.asarray(
        scores,
        dtype=float,
    )

    order = np.argsort(
        -s,
        kind="mergesort",
    )

    y_sorted = y[
        order
    ]

    total_positives = int(
        y.sum()
    )

    result = {
        "total_records": int(
            len(y)
        ),
        "total_positive": total_positives,
        "average_precision": float(
            average_precision_score(
                y,
                s,
            )
        ),
    }

    for k in TOP_K_VALUES:

        actual_k = min(
            k,
            len(y_sorted),
        )

        top_k = y_sorted[
            :actual_k
        ]

        positives_at_k = int(
            top_k.sum()
        )

        precision = (
            positives_at_k
            / actual_k
        )

        recall = (
            positives_at_k
            / total_positives
            if total_positives
            else 0.0
        )

        result[
            f"precision_at_{k}"
        ] = float(
            precision
        )

        result[
            f"recall_at_{k}"
        ] = float(
            recall
        )

        result[
            f"kev_coverage_at_{k}"
        ] = float(
            recall
        )

        result[
            f"positives_at_{k}"
        ] = positives_at_k

    return result


# ------------------------------------------------------------
# Feature preparation
# ------------------------------------------------------------

def prepare_ml_features(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    list[str],
]:

    feature_columns = [
        column
        for column in df.columns
        if column not in EXCLUDED_FEATURES
    ]

    # Raw calendar year is kept because it represents
    # publication-era information.
    #
    # Raw dates are not present in this matrix.
    #
    # days_since_publication is also absent by design.
    X = df[
        feature_columns
    ].copy()

    # Convert everything to numeric.
    for column in X.columns:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    return X, feature_columns


# ------------------------------------------------------------
# ML model
# ------------------------------------------------------------

def train_ml_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:

    model = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=12,
                    min_samples_leaf=3,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


# ------------------------------------------------------------
# Main experiment
# ------------------------------------------------------------

def main() -> None:

    print()
    print(
        "=" * 70
    )
    print(
        "THREATLENS BASELINE RANKING EXPERIMENT"
    )
    print(
        "=" * 70
    )
    print()

    print(
        "Loading feature matrix..."
    )

    df = pd.read_csv(
        FEATURE_FILE
    )

    print(
        f"Records: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    # --------------------------------------------------------
    # Convert target
    # --------------------------------------------------------

    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            TARGET
        ]
    ).copy()

    df[TARGET] = (
        df[TARGET]
        .astype(int)
    )

    # --------------------------------------------------------
    # Chronological ordering
    #
    # We use publication_year/month/day-of-year
    # because raw publication timestamps are not part
    # of the feature matrix.
    # --------------------------------------------------------

    sort_columns = [
        "publication_year",
        "publication_month",
        "publication_day_of_year",
    ]

    df = df.sort_values(
        sort_columns,
        kind="mergesort",
    ).reset_index(
        drop=True
    )

    split_index = int(
        len(df)
        * TRAIN_FRACTION
    )

    train_df = df.iloc[
        :split_index
    ].copy()

    test_df = df.iloc[
        split_index:
    ].copy()

    X_train, feature_columns = (
        prepare_ml_features(
            train_df
        )
    )

    X_test = test_df[
        feature_columns
    ].copy()

    y_train = train_df[
        TARGET
    ]

    y_test = test_df[
        TARGET
    ]

    print()
    print(
        "CHRONOLOGICAL SPLIT"
    )

    print(
        f"Training records: "
        f"{len(train_df):,}"
    )

    print(
        f"Training positives: "
        f"{int(y_train.sum()):,}"
    )

    print(
        f"Test records: "
        f"{len(test_df):,}"
    )

    print(
        f"Test positives: "
        f"{int(y_test.sum()):,}"
    )

    print()
    print(
        "Training positive rate: "
        f"{y_train.mean() * 100:.4f}%"
    )

    print(
        "Test positive rate: "
        f"{y_test.mean() * 100:.4f}%"
    )

    # --------------------------------------------------------
    # Baseline 1: CVSS
    # --------------------------------------------------------

    print()
    print(
        "Building CVSS-only ranking..."
    )

    cvss_scores = build_cvss_score(
        test_df
    )

    cvss_result = ranking_metrics(
        y_test,
        cvss_scores,
    )

    # --------------------------------------------------------
    # Baseline 2: Contextual rules
    # --------------------------------------------------------

    print(
        "Building contextual rule ranking..."
    )

    rule_scores = build_contextual_score(
        test_df
    )

    rule_result = ranking_metrics(
        y_test,
        rule_scores,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print(
        "Training ThreatLens ML ranking model..."
    )

    model = train_ml_model(
        X_train,
        y_train,
    )

    ml_scores = model.predict_proba(
        X_test
    )[
        :,
        1
    ]

    ml_result = ranking_metrics(
        y_test,
        ml_scores,
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results = {
        "project": "ThreatLens",
        "experiment": (
            "Chronological vulnerability "
            "prioritization baseline comparison"
        ),
        "dataset": {
            "records": int(
                len(df)
            ),
            "training_records": int(
                len(train_df)
            ),
            "test_records": int(
                len(test_df)
            ),
            "training_positive": int(
                y_train.sum()
            ),
            "test_positive": int(
                y_test.sum()
            ),
        },
        "split": {
            "method": "chronological",
            "train_fraction": TRAIN_FRACTION,
            "sort_columns": sort_columns,
        },
        "target": {
            "name": TARGET,
            "definition": (
                "1 = CVE appears in CISA KEV; "
                "0 = CVE does not appear in CISA KEV."
            ),
            "caveat": (
                "CISA KEV is treated as an exploitation "
                "signal rather than perfect ground truth."
            ),
        },
        "excluded_target_leakage": [
            "known_exploited",
            "kev_date_added",
            "kev_due_date",
            "kev_vendor_project",
            "kev_product",
            "kev_vulnerability_name",
            "kev_required_action",
            "ransomware_use",
            "exploitation_label",
            "cve_id",
            "days_since_publication",
        ],
        "models": {
            "cvss_only": cvss_result,
            "contextual_rules": rule_result,
            "threatlens_ml": ml_result,
        },
        "ml_model": {
            "algorithm": "RandomForestClassifier",
            "n_estimators": 300,
            "max_depth": 12,
            "min_samples_leaf": 3,
            "class_weight": "balanced_subsample",
            "random_state": 42,
        },
        "feature_count": len(
            feature_columns
        ),
        "feature_columns": feature_columns,
    }

    with RESULT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Print concise comparison
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )
    print(
        "BASELINE COMPARISON"
    )
    print(
        "=" * 70
    )

    models = [
        (
            "CVSS-only",
            cvss_result,
        ),
        (
            "Contextual rules",
            rule_result,
        ),
        (
            "ThreatLens ML",
            ml_result,
        ),
    ]

    print()

    header = (
        f"{'Model':<20}"
        f"{'AP':>12}"
        f"{'P@10':>10}"
        f"{'P@50':>10}"
        f"{'P@100':>10}"
        f"{'P@500':>10}"
    )

    print(
        header
    )

    print(
        "-" * 72
    )

    for name, result in models:

        print(
            f"{name:<20}"
            f"{result['average_precision']:>12.6f}"
            f"{result['precision_at_10']:>10.4f}"
            f"{result['precision_at_50']:>10.4f}"
            f"{result['precision_at_100']:>10.4f}"
            f"{result['precision_at_500']:>10.4f}"
        )

    print()
    print(
        "KEV COVERAGE"
    )

    print(
        "-" * 72
    )

    for name, result in models:

        print(
            f"{name:<20}"
            f"@10={result['kev_coverage_at_10']:.4f}  "
            f"@50={result['kev_coverage_at_50']:.4f}  "
            f"@100={result['kev_coverage_at_100']:.4f}  "
            f"@500={result['kev_coverage_at_500']:.4f}"
        )

    print()
    print(
        "RESULT FILE:"
    )

    print(
        RESULT_FILE
    )

    print()
    print(
        "=" * 70
    )

    print(
        "BASELINE EXPERIMENT COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()
