from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(".")
MODEL = ROOT / "data/processed/model/threatlens_conservative_model.joblib"
FEATURES = ROOT / "data/processed/features/threatlens_features.csv"
PREDICTIONS = ROOT / "data/processed/inference/threatlens_predictions.csv"
REPORT = ROOT / "data/processed/inference/threatlens_inference_report.json"
TEMPORAL = ROOT / "data/processed/evaluation/temporal_robustness_results.csv"
LEAKAGE = ROOT / "data/processed/evaluation/temporal_leakage_audit.json"
FINAL_VALIDATION = ROOT / "data/processed/evaluation/final_validation_results.csv"

OUTPUT = ROOT / "data/processed/inference/threatlens_final_audit.json"


# ======================================================================
# AUTHORITATIVE RISK CONTRACT
# ======================================================================

def risk_category(score: float) -> str:
    score = float(score)

    if score >= 0.50:
        return "CRITICAL"

    if score >= 0.20:
        return "HIGH"

    if score >= 0.05:
        return "MEDIUM"

    if score >= 0.01:
        return "LOW"

    return "MINIMAL"


EXPECTED_ROWS = 154813
EXPECTED_FEATURES = 36

VALID_CATEGORIES = {
    "CRITICAL",
    "HIGH",
    "MEDIUM",
    "LOW",
    "MINIMAL",
}


# ======================================================================
# HEADER
# ======================================================================

print("=" * 72)
print("THREATLENS - CONSOLIDATED FINAL AUDIT")
print("=" * 72)


results = {}


# ======================================================================
# 1. ARTIFACT AVAILABILITY
# ======================================================================

print()
print("-" * 72)
print("1. ARTIFACT AVAILABILITY")
print("-" * 72)

artifacts = {
    "model": MODEL,
    "features": FEATURES,
    "predictions": PREDICTIONS,
    "inference_report": REPORT,
    "temporal_robustness": TEMPORAL,
    "temporal_leakage_audit": LEAKAGE,
    "final_validation": FINAL_VALIDATION,
}

artifact_results = {}

for name, path in artifacts.items():
    exists = bool(path.exists())
    artifact_results[name] = exists
    print(f"{name:<24} {'PASS' if exists else 'REVIEW'}")


results["artifacts"] = artifact_results


# ======================================================================
# 2. MODEL ARTIFACT
# ======================================================================

print()
print("-" * 72)
print("2. MODEL ARTIFACT")
print("-" * 72)

model_ok = False
model_features = None
model_name = None
model_version = None
model_type = None

if MODEL.exists():
    try:
        artifact = joblib.load(MODEL)

        if isinstance(artifact, dict):
            model_features_list = artifact.get("features", [])
            model_features = len(model_features_list)

            model_name = artifact.get("model_name")
            model_version = artifact.get("version")

            estimator = artifact.get("model")
            model_type = (
                type(estimator).__name__
                if estimator is not None
                else None
            )

            model_ok = bool(
                estimator is not None
                and hasattr(estimator, "predict_proba")
                and model_features == EXPECTED_FEATURES
            )

    except Exception as exc:
        print(f"Model load error: {exc}")

print(f"Model type:             {model_type}")
print(f"Model name:             {model_name}")
print(f"Model version:          {model_version}")
print(f"Model features:         {model_features}")
print(f"Model artifact status:  {'PASS' if model_ok else 'FAIL'}")

results["model"] = {
    "valid": bool(model_ok),
    "name": model_name,
    "version": model_version,
    "type": model_type,
    "features": model_features,
}


# ======================================================================
# 3. FEATURE DATASET
# ======================================================================

print()
print("-" * 72)
print("3. FEATURE DATASET")
print("-" * 72)

feature_ok = False
feature_rows = 0
feature_columns = 0

if FEATURES.exists():
    try:
        feature_df = pd.read_csv(FEATURES)

        feature_rows = len(feature_df)
        feature_columns = len(feature_df.columns)

        feature_ok = bool(
            feature_rows == EXPECTED_ROWS
            and feature_columns == 42
        )

    except Exception as exc:
        print(f"Feature load error: {exc}")
else:
    feature_df = None

print(f"Rows:                   {feature_rows:,}")
print(f"Columns:                {feature_columns}")
print(f"Feature dataset status: {'PASS' if feature_ok else 'FAIL'}")

results["features"] = {
    "rows": int(feature_rows),
    "columns": int(feature_columns),
    "valid": bool(feature_ok),
}


# ======================================================================
# 4. PREDICTION DATASET
# ======================================================================

print()
print("-" * 72)
print("4. PREDICTION DATASET")
print("-" * 72)

prediction_ok = False
prediction_rows = 0
required_columns_ok = False
score_ok = False
category_values_ok = False
record_id_ok = False

if PREDICTIONS.exists():
    try:
        predictions = pd.read_csv(PREDICTIONS)

        prediction_rows = len(predictions)

        required_columns = [
            "record_id",
            "threatlens_score",
            "risk_category",
            "rank",
        ]

        required_columns_ok = bool(
            all(
                column in predictions.columns
                for column in required_columns
            )
        )

        if required_columns_ok:
            scores = pd.to_numeric(
                predictions["threatlens_score"],
                errors="coerce",
            )

            score_ok = bool(
                scores.notna().all()
                and scores.ge(0).all()
                and scores.le(1).all()
            )

            category_values_ok = bool(
                predictions["risk_category"]
                .astype(str)
                .isin(VALID_CATEGORIES)
                .all()
            )

            record_ids = pd.to_numeric(
                predictions["record_id"],
                errors="coerce",
            )

            record_id_ok = bool(
                record_ids.notna().all()
                and record_ids.is_unique
                and record_ids.min() == 1
                and record_ids.max() == EXPECTED_ROWS
            )

        prediction_ok = bool(
            prediction_rows == EXPECTED_ROWS
            and required_columns_ok
            and score_ok
            and category_values_ok
            and record_id_ok
        )

    except Exception as exc:
        print(f"Prediction load error: {exc}")
        predictions = None
else:
    predictions = None

print(f"Rows:                   {prediction_rows:,}")
print(f"Required columns:       {'PASS' if required_columns_ok else 'FAIL'}")
print(f"Score validity:         {'PASS' if score_ok else 'FAIL'}")
print(f"Category values:        {'PASS' if category_values_ok else 'FAIL'}")
print(f"Record IDs:             {'PASS' if record_id_ok else 'FAIL'}")
print(f"Prediction status:      {'PASS' if prediction_ok else 'FAIL'}")

results["predictions"] = {
    "rows": int(prediction_rows),
    "required_columns": bool(required_columns_ok),
    "score_valid": bool(score_ok),
    "category_values_valid": bool(category_values_ok),
    "record_ids_valid": bool(record_id_ok),
    "valid": bool(prediction_ok),
}


# ======================================================================
# 5. CATEGORY CONTRACT
# ======================================================================

print()
print("-" * 72)
print("5. RISK CATEGORY CONTRACT")
print("-" * 72)

category_match = False
category_matches = 0
category_mismatches = 0

if predictions is not None and required_columns_ok and score_ok:

    expected_categories = scores.map(risk_category)

    stored_categories = (
        predictions["risk_category"]
        .astype(str)
    )

    category_matches = int(
        (expected_categories == stored_categories).sum()
    )

    category_mismatches = int(
        (expected_categories != stored_categories).sum()
    )

    category_match = bool(
        category_mismatches == 0
    )

print(
    f"Matches:                "
    f"{category_matches:,}/{prediction_rows:,}"
)

print(
    f"Mismatches:             "
    f"{category_mismatches:,}"
)

print(
    f"Contract status:        "
    f"{'PASS' if category_match else 'FAIL'}"
)

print()
print("AUTHORITATIVE CONTRACT:")
print("  CRITICAL >= 0.50")
print("  HIGH     >= 0.20")
print("  MEDIUM   >= 0.05")
print("  LOW      >= 0.01")
print("  MINIMAL  <  0.01")

results["category_contract"] = {
    "matches": int(category_matches),
    "mismatches": int(category_mismatches),
    "valid": bool(category_match),
    "thresholds": {
        "CRITICAL": ">= 0.50",
        "HIGH": ">= 0.20",
        "MEDIUM": ">= 0.05",
        "LOW": ">= 0.01",
        "MINIMAL": "< 0.01",
    },
}


# ======================================================================
# 6. DISTRIBUTION
# ======================================================================

print()
print("-" * 72)
print("6. RISK DISTRIBUTION")
print("-" * 72)

distribution = {}

if predictions is not None and "risk_category" in predictions.columns:

    counts = (
        predictions["risk_category"]
        .value_counts()
        .reindex(
            [
                "CRITICAL",
                "HIGH",
                "MEDIUM",
                "LOW",
                "MINIMAL",
            ],
            fill_value=0,
        )
    )

    for category, count in counts.items():
        distribution[str(category)] = int(count)
        print(f"{category:<10} {int(count):,}")

results["distribution"] = distribution


# ======================================================================
# 7. INFERENCE REPORT
# ======================================================================

print()
print("-" * 72)
print("7. INFERENCE REPORT")
print("-" * 72)

report_ok = False
report_rows = None

if REPORT.exists():

    try:
        report_data = json.loads(
            REPORT.read_text(
                encoding="utf-8",
                errors="replace",
            )
        )

        report_rows = (
            report_data
            .get("prediction", {})
            .get("rows")
        )

        report_ok = bool(
            report_rows == prediction_rows
        )

    except Exception as exc:
        print(f"Report error: {exc}")

print(f"Report rows:            {report_rows}")
print(f"Expected rows:          {prediction_rows:,}")
print(f"Report status:          {'PASS' if report_ok else 'FAIL'}")

results["inference_report"] = {
    "rows": (
        int(report_rows)
        if report_rows is not None
        else None
    ),
    "consistent": bool(report_ok),
}


# ======================================================================
# 8. TEMPORAL ROBUSTNESS ARTIFACT
# ======================================================================

print()
print("-" * 72)
print("8. TEMPORAL ROBUSTNESS")
print("-" * 72)

temporal_ok = False
temporal_rows = 0

if TEMPORAL.exists():

    try:
        temporal_df = pd.read_csv(TEMPORAL)
        temporal_rows = len(temporal_df)

        temporal_ok = bool(
            temporal_rows > 0
            and "model" in temporal_df.columns
            and "average_precision" in temporal_df.columns
        )

    except Exception as exc:
        print(f"Temporal results error: {exc}")

print(f"Rows:                   {temporal_rows}")
print(f"Temporal artifact:      {'PASS' if temporal_ok else 'REVIEW'}")

results["temporal_robustness"] = {
    "rows": int(temporal_rows),
    "available": bool(temporal_ok),
}


# ======================================================================
# 9. LEAKAGE AUDIT
# ======================================================================

print()
print("-" * 72)
print("9. TEMPORAL LEAKAGE AUDIT")
print("-" * 72)

leakage_ok = False
leakage_status = None

if LEAKAGE.exists():

    try:
        leakage_data = json.loads(
            LEAKAGE.read_text(
                encoding="utf-8",
                errors="replace",
            )
        )

        leakage_status = leakage_data.get("status")

        leakage_ok = bool(
            leakage_status is not None
        )

    except Exception as exc:
        print(f"Leakage audit error: {exc}")

print(f"Audit status:           {leakage_status}")
print(
    f"Leakage artifact:       "
    f"{'PASS' if leakage_ok else 'REVIEW'}"
)

results["temporal_leakage"] = {
    "status": leakage_status,
    "available": bool(leakage_ok),
}


# ======================================================================
# 10. FINAL STATISTICAL VALIDATION
# ======================================================================

print()
print("-" * 72)
print("10. FINAL STATISTICAL VALIDATION")
print("-" * 72)

validation_ok = False
validation_rows = 0

if FINAL_VALIDATION.exists():

    try:
        validation_df = pd.read_csv(
            FINAL_VALIDATION
        )

        validation_rows = len(validation_df)

        validation_ok = bool(
            validation_rows > 0
            and "average_precision" in validation_df.columns
        )

    except Exception as exc:
        print(f"Validation results error: {exc}")

print(f"Rows:                   {validation_rows}")
print(
    f"Validation artifact:   "
    f"{'PASS' if validation_ok else 'REVIEW'}"
)

results["final_validation"] = {
    "rows": int(validation_rows),
    "available": bool(validation_ok),
}


# ======================================================================
# 11. FINAL STATUS
# ======================================================================

final_ok = bool(
    model_ok
    and feature_ok
    and prediction_ok
    and category_match
    and report_ok
)

results["status"] = (
    "PASS"
    if final_ok
    else "REVIEW_REQUIRED"
)


# ======================================================================
# SAVE
# ======================================================================

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT.write_text(
    json.dumps(
        results,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


# ======================================================================
# FINAL OUTPUT
# ======================================================================

print()
print("=" * 72)
print(f"FINAL STATUS: {results['status']}")
print("=" * 72)

print()
print("Production model modified:       NO")
print("Production prediction modified:  NO")
print("Consolidated audit saved:")
print(OUTPUT)
