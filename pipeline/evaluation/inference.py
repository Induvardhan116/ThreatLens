"""
========================================================================
THREATLENS - STEP 28
LOCAL INFERENCE / PREDICTION PIPELINE
========================================================================

Loads the packaged ThreatLens Conservative Random Forest artifact.

Step 27 saved the artifact as a dictionary. This script therefore
automatically extracts the trained estimator from the package.

Outputs:


    threatlens_predictions.csv
    threatlens_prediction_summary.csv
    threatlens_inference_report.json
========================================================================
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from pipeline.schema import risk_category

import joblib
import numpy as np
import pandas as pd


# ======================================================================
# PATHS
# ======================================================================

PROJECT_ROOT = Path(r"D:\ThreatLens")

FEATURE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
    / "threatlens_features.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model"
)

MODEL_PATH = (
    MODEL_DIR
    / "threatlens_conservative_model.joblib"
)

FEATURE_LIST_PATH = (
    MODEL_DIR
    / "threatlens_feature_list.json"
)

METADATA_PATH = (
    MODEL_DIR
    / "threatlens_model_metadata.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "inference"
)

PREDICTION_PATH = (
    OUTPUT_DIR
    / "threatlens_predictions.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "threatlens_prediction_summary.csv"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "threatlens_inference_report.json"
)


# ======================================================================
# EXACT STEP 27 FEATURE CONFIGURATION
# ======================================================================

EXPECTED_FEATURES = [
    "cvss_v3_score",
    "cvss_v4_score",
    "cvss_v3_available",
    "cvss_v4_available",
    "cvss_max_score",
    "cvss_score_difference",
    "severity_score",
    "severity_known",
    "cwe_count",
    "cwe_unique_count",
    "has_cwe",
    "affected_cpe_count",
    "affected_cpe_unique_count",
    "has_affected_cpe",
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
    "publication_year",
    "publication_month",
    "publication_quarter",
    "publication_day_of_week",
    "publication_day_of_year",
    "publication_is_weekend",
    "publication_date_available",
    "attack_technique_count",
    "has_attack_technique",
]


# ======================================================================
# BANNER
# ======================================================================

def print_banner() -> None:
    print("=" * 72)
    print("THREATLENS - STEP 28")
    print("LOCAL INFERENCE / PREDICTION PIPELINE")
    print("=" * 72)


# ======================================================================
# LOAD MODEL ARTIFACT
# ======================================================================

def load_model_artifact():
    print()
    print("-" * 72)
    print("LOADING PACKAGED MODEL")
    print("-" * 72)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"ThreatLens model artifact not found:\n{MODEL_PATH}\n\n"
            "Run Step 27 first."
        )

    artifact = joblib.load(MODEL_PATH)

    print(f"Model path: {MODEL_PATH}")
    print(f"Artifact type: {type(artifact).__name__}")

    # --------------------------------------------------------------
    # STEP 27 SAVED A DICTIONARY
    # --------------------------------------------------------------

    if isinstance(artifact, dict):

        print("Packaged artifact is a dictionary.")

        print("Available artifact keys:")

        for key in artifact.keys():
            print(f"  - {key}")

        # Try common estimator keys first.
        possible_model_keys = [
            "model",
            "estimator",
            "classifier",
            "random_forest",
            "rf",
            "trained_model",
            "final_model",
        ]

        model = None
        selected_key = None

        for key in possible_model_keys:

            if key in artifact:

                candidate = artifact[key]

                if hasattr(candidate, "predict_proba"):
                    model = candidate
                    selected_key = key
                    break

        # ----------------------------------------------------------
        # FALLBACK: SEARCH ALL DICTIONARY VALUES
        # ----------------------------------------------------------

        if model is None:

            for key, candidate in artifact.items():

                if hasattr(candidate, "predict_proba"):

                    model = candidate
                    selected_key = key
                    break

        if model is None:

            raise TypeError(
                "The packaged model is a dictionary, but no object "
                "inside the dictionary provides predict_proba().\n\n"
                f"Available keys: {list(artifact.keys())}"
            )

        print()
        print(f"Extracted estimator key: {selected_key}")
        print(f"Estimator type: {type(model).__name__}")

        return model, artifact

    # --------------------------------------------------------------
    # DIRECT ESTIMATOR
    # --------------------------------------------------------------

    if hasattr(artifact, "predict_proba"):

        print("Artifact itself provides predict_proba().")
        print(f"Estimator type: {type(artifact).__name__}")

        return artifact, {
            "model": artifact
        }

    raise TypeError(
        "Unsupported ThreatLens model artifact format."
    )


# ======================================================================
# LOAD FEATURE LIST
# ======================================================================

def load_features() -> list[str]:

    print()
    print("-" * 72)
    print("LOADING MODEL FEATURES")
    print("-" * 72)

    features = None

    if FEATURE_LIST_PATH.exists():

        with FEATURE_LIST_PATH.open(
            "r",
            encoding="utf-8",
        ) as handle:

            data = json.load(handle)

        if isinstance(data, list):

            features = data

        elif isinstance(data, dict):

            for key in [
                "features",
                "feature_list",
                "model_features",
                "final_features",
            ]:

                if isinstance(data.get(key), list):

                    features = data[key]
                    break

    if features is None:

        print(
            "Packaged feature list not directly recognized."
        )

        print(
            "Using exact Step 27 conservative feature configuration."
        )

        features = EXPECTED_FEATURES.copy()

    features = [
        str(feature)
        for feature in features
    ]

    # --------------------------------------------------------------
    # SAFETY CHECK
    # --------------------------------------------------------------

    missing = [
        feature
        for feature in EXPECTED_FEATURES
        if feature not in features
    ]

    extra = [
        feature
        for feature in features
        if feature not in EXPECTED_FEATURES
    ]

    if missing or extra:

        print()
        print(
            "Packaged feature list differs from expected "
            "Step 27 configuration."
        )

        if missing:

            print("Missing:")
            for feature in missing:
                print(f"  - {feature}")

        if extra:

            print("Extra:")
            for feature in extra:
                print(f"  - {feature}")

        print()
        print(
            "Using exact Step 27 feature configuration."
        )

        features = EXPECTED_FEATURES.copy()

    print(
        f"Features used: {len(features)}"
    )

    return features


# ======================================================================
# LOAD DATA
# ======================================================================

from pathlib import Path
import json

import pandas as pd

# Add this path beside FEATURE_PATH in inference.py
ENRICHED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "enriched"
    / "threatlens_dataset.json"
)


def load_features_dataset() -> pd.DataFrame:

    print()
    print("-" * 72)
    print("LOADING THREATLENS FEATURE DATASET")
    print("-" * 72)

    if not FEATURE_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found:\n{FEATURE_PATH}"
        )

    df = pd.read_csv(FEATURE_PATH)

    print(f"Path: {FEATURE_PATH}")
    print(f"Rows loaded: {len(df):,}")
    print(f"Columns loaded: {len(df.columns)}")

    if df.empty:
        raise ValueError(
            "ThreatLens feature dataset is empty."
        )

    # --------------------------------------------------------------
    # RESTORE CVE IDENTIFIERS
    # --------------------------------------------------------------
    if not ENRICHED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Enriched dataset not found:\n{ENRICHED_DATA_PATH}"
        )

    print()
    print("Loading CVE identifiers from enriched dataset...")

    with ENRICHED_DATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        records = json.load(handle)

    enriched = pd.DataFrame(records)

    if "cve_id" not in enriched.columns:
        raise ValueError(
            "Enriched dataset does not contain cve_id."
        )

    if len(enriched) != len(df):
        raise ValueError(
            "Row count mismatch between feature dataset and "
            "enriched dataset: "
            f"features={len(df):,}, "
            f"enriched={len(enriched):,}"
        )

    cve_ids = enriched["cve_id"].astype(str)

    if cve_ids.isna().any():
        raise ValueError(
            "Enriched dataset contains missing cve_id values."
        )

    df.insert(
        0,
        "cve_id",
        cve_ids.to_numpy(),
    )

    print(
        f"CVE identifiers restored: {len(df):,}"
    )

    print(
        f"First CVE: {df['cve_id'].iloc[0]}"
    )

    print(
        f"Last CVE: {df['cve_id'].iloc[-1]}"
    )

    return df


# ======================================================================
# VALIDATE FEATURES
# ======================================================================

def validate_features(
    df: pd.DataFrame,
    features: list[str],
) -> None:

    print()
    print("-" * 72)
    print("VALIDATING INFERENCE FEATURES")
    print("-" * 72)

    missing = [
        feature
        for feature in features
        if feature not in df.columns
    ]

    if missing:

        print("Missing required features:")

        for feature in missing:
            print(f"  - {feature}")

        raise ValueError(
            "Input dataset does not contain all required "
            "ThreatLens features."
        )

    print(
        f"Required features: {len(features)}"
    )

    print(
        "Feature validation: PASS"
    )


# ======================================================================
# PREPARE MATRIX
# ======================================================================

def prepare_matrix(
    df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:

    print()
    print("-" * 72)
    print("PREPARING INFERENCE MATRIX")
    print("-" * 72)

    X = df[
        features
    ].copy()

    for feature in features:

        X[feature] = pd.to_numeric(
            X[feature],
            errors="coerce",
        )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    missing_before = int(
        X.isna().sum().sum()
    )

    print(
        f"Missing values before filling: "
        f"{missing_before:,}"
    )

    # Same conservative handling used by the
    # ThreatLens evaluation pipeline.
    X = X.fillna(0.0)

    missing_after = int(
        X.isna().sum().sum()
    )

    print(
        f"Missing values after filling: "
        f"{missing_after:,}"
    )

    if missing_after != 0:

        raise ValueError(
            "Missing values remain after preprocessing."
        )

    return X


# ======================================================================
# SCORE -> RISK CATEGORY
# ======================================================================

get_risk_category = risk_category


# ======================================================================
# GENERATE SCORES
# ======================================================================

def generate_scores(
    model,
    X: pd.DataFrame,
) -> np.ndarray:

    print()
    print("-" * 72)
    print("GENERATING THREATLENS SCORES")
    print("-" * 72)

    if not hasattr(
        model,
        "predict_proba",
    ):

        raise TypeError(
            f"Estimator type {type(model).__name__} "
            "does not provide predict_proba()."
        )

    probabilities = model.predict_proba(
        X
    )

    probabilities = np.asarray(
        probabilities
    )

    print(
        f"Probability matrix shape: "
        f"{probabilities.shape}"
    )

    if probabilities.ndim != 2:

        raise ValueError(
            "Unexpected probability matrix dimensions."
        )

    if probabilities.shape[1] < 2:

        raise ValueError(
            "Model does not contain a positive class."
        )

    scores = probabilities[
        :,
        1
    ].astype(float)

    if len(scores) != len(X):

        raise ValueError(
            "Number of predictions does not match "
            "number of input rows."
        )

    print(
        f"Scores generated: {len(scores):,}"
    )

    print(
        f"Minimum score: {scores.min():.6f}"
    )

    print(
        f"Maximum score: {scores.max():.6f}"
    )

    print(
        f"Mean score: {scores.mean():.6f}"
    )

    return scores


# ======================================================================
# BUILD PREDICTIONS
# ======================================================================

def build_predictions(
    df: pd.DataFrame,
    scores: np.ndarray,
) -> pd.DataFrame:

    output = pd.DataFrame()

    # --------------------------------------------------------------
    # IDENTIFIER
    # --------------------------------------------------------------

    identifier_columns = [
        "cve_id",
        "CVE_ID",
        "CVE",
        "id",
        "ID",
    ]

    identifier_found = False

    for column in identifier_columns:

        if column in df.columns:

            output[
                "cve_id"
            ] = df[column].astype(str)

            identifier_found = True
            break

    if not identifier_found:

        output[
            "record_id"
        ] = np.arange(
            1,
            len(df) + 1,
        )

    # --------------------------------------------------------------
    # PRESERVE AVAILABLE INFORMATION
    # --------------------------------------------------------------

    for column in [
        "published_date",
        "publication_date",
        "publication_year",
        "publication_month",
        "publication_quarter",
    ]:

        if column in df.columns:

            output[column] = df[column]

    if "exploitation_label" in df.columns:

        output[
            "exploitation_label"
        ] = pd.to_numeric(
            df[
                "exploitation_label"
            ],
            errors="coerce",
        ).fillna(
            0
        ).astype(int)

    # --------------------------------------------------------------
    # MODEL SCORE
    # --------------------------------------------------------------

    output[
        "threatlens_score"
    ] = scores

    output[
        "risk_category"
    ] = [
        get_risk_category(
            float(score)
        )
        for score in scores
    ]

    # --------------------------------------------------------------
    # RANK
    # --------------------------------------------------------------

    output[
        "rank"
    ] = (
        output[
            "threatlens_score"
        ]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(int)
    )

    output = output.sort_values(
        "threatlens_score",
        ascending=False,
    ).reset_index(
        drop=True
    )

    return output


# ======================================================================
# SUMMARY
# ======================================================================

def build_summary(
    predictions: pd.DataFrame,
) -> pd.DataFrame:

    categories = [
        "CRITICAL",
        "HIGH",
        "MEDIUM",
        "LOW",
        "MINIMAL",
    ]

    rows = []

    total = len(predictions)

    for category in categories:

        subset = predictions[
            predictions[
                "risk_category"
            ] == category
        ]

        count = len(subset)

        rows.append(
            {
                "risk_category": category,
                "records": count,
                "percentage": (
                    count / total * 100.0
                    if total
                    else 0.0
                ),
                "mean_score": (
                    float(
                        subset[
                            "threatlens_score"
                        ].mean()
                    )
                    if count
                    else 0.0
                ),
                "maximum_score": (
                    float(
                        subset[
                            "threatlens_score"
                        ].max()
                    )
                    if count
                    else 0.0
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ======================================================================
# REPORT
# ======================================================================

def build_report(
    df: pd.DataFrame,
    predictions: pd.DataFrame,
    summary: pd.DataFrame,
    features: list[str],
    artifact: dict[str, Any],
) -> dict[str, Any]:

    scores = predictions[
        "threatlens_score"
    ]

    report = {
        "step": 28,
        "name": (
            "ThreatLens Local Inference / "
            "Prediction Pipeline"
        ),
        "status": "PASS",
        "model_artifact": str(
            MODEL_PATH
        ),
        "model_type": type(
            artifact
        ).__name__,
        "feature_count": len(
            features
        ),
        "features": features,
        "input": {
            "path": str(
                FEATURE_PATH
            ),
            "rows": int(
                len(df)
            ),
            "columns": int(
                len(df.columns)
            ),
        },
        "prediction": {
            "rows": int(
                len(predictions)
            ),
            "minimum_score": float(
                scores.min()
            ),
            "maximum_score": float(
                scores.max()
            ),
            "mean_score": float(
                scores.mean()
            ),
            "median_score": float(
                scores.median()
            ),
        },
        "prediction_statistics": {
            "records": int(
                len(predictions)
            ),
            "minimum_score": float(
                scores.min()
            ),
            "maximum_score": float(
                scores.max()
            ),
            "mean_score": float(
                scores.mean()
            ),
            "median_score": float(
                scores.median()
            ),
        },
        "risk_categories": {},
        "notes": [
            (
                "ThreatLens score is the positive-class "
                "Random Forest score."
            ),
            (
                "Scores are intended for vulnerability "
                "ranking and triage."
            ),
            (
                "Scores should not be interpreted as "
                "calibrated probabilities."
            ),
            (
                "Inference uses the 36-feature "
                "conservative configuration."
            ),
        ],
    }

    for _, row in summary.iterrows():

        category = str(
            row[
                "risk_category"
            ]
        )

        report[
            "risk_categories"
        ][category] = {
            "records": int(
                row["records"]
            ),
            "percentage": float(
                row["percentage"]
            ),
            "mean_score": float(
                row["mean_score"]
            ),
            "maximum_score": float(
                row["maximum_score"]
            ),
        }

    if "exploitation_label" in predictions.columns:

        positives = int(
            predictions[
                "exploitation_label"
            ].sum()
        )

        report[
            "input_target_distribution"
        ] = {
            "kev_positive": positives,
            "non_kev": int(
                len(predictions)
                - positives
            ),
        }

    return report


# ======================================================================
# DISPLAY TOP RESULTS
# ======================================================================

def display_top_results(
    predictions: pd.DataFrame,
) -> None:

    print()
    print("-" * 72)
    print("TOP 20 THREATLENS RANKINGS")
    print("-" * 72)

    columns = []

    for column in [
        "rank",
        "cve_id",
        "record_id",
        "threatlens_score",
        "risk_category",
        "exploitation_label",
    ]:

        if column in predictions.columns:

            columns.append(
                column
            )

    if not columns:
        return

    display = predictions[
        columns
    ].head(20).copy()

    if "threatlens_score" in display.columns:

        display[
            "threatlens_score"
        ] = display[
            "threatlens_score"
        ].map(
            lambda value:
            f"{value:.6f}"
        )

    print(
        display.to_string(
            index=False
        )
    )


# ======================================================================
# SAVE OUTPUTS
# ======================================================================

def save_outputs(
    predictions: pd.DataFrame,
    summary: pd.DataFrame,
    report: dict[str, Any],
) -> None:

    print()
    print("-" * 72)
    print("SAVING INFERENCE OUTPUTS")
    print("-" * 72)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        PREDICTION_PATH,
        index=False,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            report,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("Saved:")
    print(
        f"  {PREDICTION_PATH}"
    )
    print(
        f"  {SUMMARY_PATH}"
    )
    print(
        f"  {REPORT_PATH}"
    )


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:

    print_banner()

    # --------------------------------------------------------------
    # MODEL
    # --------------------------------------------------------------

    model, artifact = load_model_artifact()

    # --------------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------------

    features = load_features()

    # --------------------------------------------------------------
    # DATA
    # --------------------------------------------------------------

    df = load_features_dataset()

    validate_features(
        df,
        features,
    )

    # --------------------------------------------------------------
    # MATRIX
    # --------------------------------------------------------------

    X = prepare_matrix(
        df,
        features,
    )

    # --------------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------------

    scores = generate_scores(
        model,
        X,
    )

    predictions = build_predictions(
        df,
        scores,
    )

    # --------------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------------

    summary = build_summary(
        predictions
    )

    # --------------------------------------------------------------
    # REPORT
    # --------------------------------------------------------------

    report = build_report(
        df,
        predictions,
        summary,
        features,
        artifact,
    )

    # --------------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------------

    display_top_results(
        predictions
    )

    print()
    print("-" * 72)
    print("RISK CATEGORY SUMMARY")
    print("-" * 72)

    print(
        summary.to_string(
            index=False,
            float_format=lambda value:
            f"{value:.4f}",
        )
    )

    # --------------------------------------------------------------
    # SAVE
    # --------------------------------------------------------------

    save_outputs(
        predictions,
        summary,
        report,
    )

    print()
    print("=" * 72)
    print("STEP 28 COMPLETE")
    print("=" * 72)

    print()
    print(
        "ThreatLens local inference is READY."
    )

    print()
    print("Prediction output:")
    print(
        PREDICTION_PATH
    )

    print()
    print("Summary output:")
    print(
        SUMMARY_PATH
    )

    print()
    print("Report output:")
    print(
        REPORT_PATH
    )


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()
        print(
            "Inference interrupted by user."
        )

        sys.exit(130)

    except Exception as exc:

        print()
        print("=" * 72)
        print("STEP 28 FAILED")
        print("=" * 72)
        print()
        print(
            f"{type(exc).__name__}: {exc}"
        )
        print()

        sys.exit(1)




