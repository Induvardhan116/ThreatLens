from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.inspection import permutation_importance


# =====================================================================
# THREATLENS - STEP 26
# FINAL MODEL VALIDATION
# STATISTICAL SIGNIFICANCE & CONFIDENCE ANALYSIS
# =====================================================================

ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = ROOT / "data" / "processed"

FEATURE_PATH = (
    PROCESSED_DIR
    / "features"
    / "threatlens_features.csv"
)

EVAL_DIR = (
    PROCESSED_DIR
    / "evaluation"
)

EVAL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RANDOM_STATE = 42

N_ESTIMATORS = 300

BOOTSTRAP_ITERATIONS = 1000

TOP_K_VALUES = [
    10,
    50,
    100,
    500,
]


# =====================================================================
# DISPLAY HELPERS
# =====================================================================

def banner(title: str) -> None:
    print("=" * 72)
    print(title)
    print("=" * 72)


def section(title: str) -> None:
    print()
    print("-" * 72)
    print(title)
    print("-" * 72)


# =====================================================================
# COLUMN RESOLUTION
# =====================================================================

def resolve_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> str | None:

    mapping = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:

        key = candidate.strip().lower()

        if key in mapping:
            return mapping[key]

    return None


# =====================================================================
# LOAD DATA
# =====================================================================

def load_features() -> pd.DataFrame:

    if not FEATURE_PATH.exists():

        raise FileNotFoundError(
            f"Feature dataset not found:\n{FEATURE_PATH}"
        )

    print("Loading ThreatLens feature dataset...")

    df = pd.read_csv(
        FEATURE_PATH,
        low_memory=False,
    )

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    print(
        f"Rows loaded: {len(df):,}"
    )

    print(
        f"Columns loaded: {len(df.columns)}"
    )

    return df


# =====================================================================
# FEATURE DEFINITIONS
# =====================================================================

def get_feature_configuration(
    df: pd.DataFrame,
) -> tuple[list[str], list[str], str]:

    target_col = resolve_column(
        df,
        [
            "exploitation_label",
            "known_exploited",
        ],
    )

    if target_col is None:

        raise KeyError(
            "Could not locate exploitation_label."
        )

    excluded = {
        target_col
    }

    identifier_candidates = [
        "cve_id",
        "CVE_ID",
        "CVE",
        "id",
        "published_date",
        "last_modified_date",
        "kev_date_added",
        "kev_due_date",
        "kev_vendor_project",
        "kev_product",
        "kev_vulnerability_name",
        "kev_required_action",
        "ransomware_use",
    ]

    for candidate in identifier_candidates:

        actual = resolve_column(
            df,
            [candidate],
        )

        if actual is not None:
            excluded.add(actual)

    numeric_candidates = []

    for column in df.columns:

        if column in excluded:
            continue

        if pd.api.types.is_numeric_dtype(
            df[column]
        ):

            numeric_candidates.append(column)

    if not numeric_candidates:

        raise ValueError(
            "No numeric model features were found."
        )

    # ---------------------------------------------------------------
    # Conservative configuration
    # ---------------------------------------------------------------

    conservative_remove = {
        "reference_count",
        "reference_unique_count",
        "has_references",
    }

    conservative_features = [
        column
        for column in numeric_candidates
        if column not in conservative_remove
    ]

    # Also remove modification timing because the previous
    # provenance audit identified it as HIGH_REVIEW.
    modification_features = {
        "modified_after_publication_days",
        "has_modified_date",
    }

    conservative_features = [
        column
        for column in conservative_features
        if column not in modification_features
    ]

    return (
        numeric_candidates,
        conservative_features,
        target_col,
    )


# =====================================================================
# PREPARE MATRIX
# =====================================================================

def prepare_matrix(
    df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:

    X = df[features].copy()

    for column in features:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    missing = int(
        X.isna().sum().sum()
    )

    print(
        f"Missing values before filling: {missing:,}"
    )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    X = X.fillna(0.0)

    remaining = int(
        X.isna().sum().sum()
    )

    print(
        f"Missing values after filling: {remaining:,}"
    )

    return X


# =====================================================================
# CHRONOLOGICAL SPLIT
# =====================================================================

def build_chronological_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    section(
        "BUILDING CHRONOLOGICAL VALIDATION SPLIT"
    )

    publication_col = resolve_column(
        df,
        [
            "published_date",
            "publication_date",
            "published",
        ],
    )

    # The feature dataset does not contain the raw date.
    # publication_year is therefore used as the available
    # chronological reconstruction signal.

    if publication_col is not None:

        dates = pd.to_datetime(
            df[publication_col],
            errors="coerce",
            utc=True,
        )

    else:

        if "publication_year" not in df.columns:

            raise ValueError(
                "Unable to establish chronological ordering. "
                "No publication date or publication_year exists."
            )

        dates = pd.to_datetime(
            df["publication_year"].astype(str)
            + "-01-01",
            errors="coerce",
            utc=True,
        )

    valid = dates.notna()

    work = df.loc[valid].copy()

    work["_temporal_date"] = dates.loc[
        valid
    ].values

    work = work.sort_values(
        "_temporal_date"
    ).reset_index(
        drop=True
    )

    print(
        f"Rows with valid chronological information: "
        f"{len(work):,}"
    )

    if len(work) < 1000:

        raise ValueError(
            "Too few rows available for chronological evaluation."
        )

    # ---------------------------------------------------------------
    # 70/30 chronological split
    # ---------------------------------------------------------------

    split_index = int(
        len(work) * 0.70
    )

    train = work.iloc[
        :split_index
    ].copy()

    test = work.iloc[
        split_index:
    ].copy()

    print(
        f"Training rows: {len(train):,}"
    )

    print(
        f"Test rows: {len(test):,}"
    )

    print(
        f"Training positives: "
        f"{int(train['exploitation_label'].sum()):,}"
    )

    print(
        f"Test positives: "
        f"{int(test['exploitation_label'].sum()):,}"
    )

    print(
        f"Training date range: "
        f"{train['_temporal_date'].min()} -> "
        f"{train['_temporal_date'].max()}"
    )

    print(
        f"Test date range: "
        f"{test['_temporal_date'].min()} -> "
        f"{test['_temporal_date'].max()}"
    )

    return train, test


# =====================================================================
# MODEL TRAINING
# =====================================================================

def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestClassifier:

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        max_features="sqrt",
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


# =====================================================================
# TOP-K METRICS
# =====================================================================

def ranking_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> dict:

    order = np.argsort(
        -scores
    )

    y_sorted = y_true[
        order
    ]

    total_positive = int(
        y_true.sum()
    )

    results = {}

    for k in TOP_K_VALUES:

        k_actual = min(
            k,
            len(y_sorted),
        )

        selected = y_sorted[
            :k_actual
        ]

        hits = int(
            selected.sum()
        )

        precision = (
            hits / k_actual
            if k_actual
            else 0.0
        )

        recall = (
            hits / total_positive
            if total_positive
            else 0.0
        )

        results[f"precision_at_{k}"] = precision
        results[f"recall_at_{k}"] = recall

    return results


# =====================================================================
# BOOTSTRAP CONFIDENCE INTERVAL
# =====================================================================

def bootstrap_metric(
    y_true: np.ndarray,
    scores: np.ndarray,
    metric: str,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> tuple[float, float, float]:

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    n = len(y_true)

    values = []

    for _ in range(iterations):

        indices = rng.integers(
            0,
            n,
            size=n,
        )

        sample_y = y_true[
            indices
        ]

        sample_scores = scores[
            indices
        ]

        # AP requires both classes.
        if (
            len(np.unique(sample_y))
            < 2
        ):
            continue

        if metric == "average_precision":

            value = average_precision_score(
                sample_y,
                sample_scores,
            )

        else:

            raise ValueError(
                f"Unsupported bootstrap metric: {metric}"
            )

        values.append(
            float(value)
        )

    if not values:

        return (
            float("nan"),
            float("nan"),
            float("nan"),
        )

    values = np.asarray(
        values
    )

    point = float(
        average_precision_score(
            y_true,
            scores,
        )
    )

    lower = float(
        np.percentile(
            values,
            2.5,
        )
    )

    upper = float(
        np.percentile(
            values,
            97.5,
        )
    )

    return (
        point,
        lower,
        upper,
    )


# =====================================================================
# BOOTSTRAP DIFFERENCE
# =====================================================================

def bootstrap_ap_difference(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> dict:

    rng = np.random.default_rng(
        RANDOM_STATE + 1
    )

    n = len(y_true)

    differences = []

    for _ in range(iterations):

        indices = rng.integers(
            0,
            n,
            size=n,
        )

        sample_y = y_true[
            indices
        ]

        if (
            len(np.unique(sample_y))
            < 2
        ):
            continue

        ap_a = average_precision_score(
            sample_y,
            scores_a[
                indices
            ],
        )

        ap_b = average_precision_score(
            sample_y,
            scores_b[
                indices
            ],
        )

        differences.append(
            float(ap_a - ap_b)
        )

    differences = np.asarray(
        differences
    )

    observed = float(
        average_precision_score(
            y_true,
            scores_a,
        )
        -
        average_precision_score(
            y_true,
            scores_b,
        )
    )

    lower = float(
        np.percentile(
            differences,
            2.5,
        )
    )

    upper = float(
        np.percentile(
            differences,
            97.5,
        )
    )

    probability_positive = float(
        np.mean(
            differences > 0
        )
    )

    return {
        "observed_difference": observed,
        "ci_lower": lower,
        "ci_upper": upper,
        "probability_a_better": probability_positive,
        "bootstrap_iterations": int(
            len(differences)
        ),
    }


# =====================================================================
# PERMUTATION TEST
# =====================================================================

def permutation_test_ap_difference(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    iterations: int = 1000,
) -> dict:

    rng = np.random.default_rng(
        RANDOM_STATE + 2
    )

    observed = float(
        average_precision_score(
            y_true,
            scores_a,
        )
        -
        average_precision_score(
            y_true,
            scores_b,
        )
    )

    differences = []

    for _ in range(iterations):

        swap = rng.random(
            len(y_true)
        ) < 0.5

        perm_a = np.where(
            swap,
            scores_b,
            scores_a,
        )

        perm_b = np.where(
            swap,
            scores_a,
            scores_b,
        )

        difference = (
            average_precision_score(
                y_true,
                perm_a,
            )
            -
            average_precision_score(
                y_true,
                perm_b,
            )
        )

        differences.append(
            float(difference)
        )

    differences = np.asarray(
        differences
    )

    p_value = float(
        (
            np.sum(
                np.abs(differences)
                >= abs(observed)
            )
            + 1
        )
        /
        (
            len(differences)
            + 1
        )
    )

    return {
        "observed_difference": observed,
        "permutation_p_value": p_value,
        "permutation_iterations": int(
            iterations
        ),
    }


# =====================================================================
# MODEL EVALUATION
# =====================================================================

def evaluate_model(
    name: str,
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict, np.ndarray]:

    scores = model.predict_proba(
        X_test
    )[:, 1]

    y_true = y_test.to_numpy()

    ap = float(
        average_precision_score(
            y_true,
            scores,
        )
    )

    metrics = {
        "model": name,
        "average_precision": ap,
    }

    metrics.update(
        ranking_metrics(
            y_true,
            scores,
        )
    )

    point, lower, upper = (
        bootstrap_metric(
            y_true,
            scores,
            "average_precision",
        )
    )

    metrics[
        "ap_ci_lower"
    ] = lower

    metrics[
        "ap_ci_upper"
    ] = upper

    metrics[
        "bootstrap_ap_point"
    ] = point

    return (
        metrics,
        scores,
    )


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:

    banner(
        "THREATLENS - STEP 26\n"
        "FINAL MODEL VALIDATION\n"
        "STATISTICAL SIGNIFICANCE & CONFIDENCE ANALYSIS"
    )

    df = load_features()

    section(
        "FEATURE CONFIGURATION"
    )

    (
        all_features,
        conservative_features,
        target_col,
    ) = get_feature_configuration(
        df
    )

    print(
        f"All numeric features: "
        f"{len(all_features)}"
    )

    print(
        f"Conservative features: "
        f"{len(conservative_features)}"
    )

    print(
        "Conservative model removes:"
    )

    print(
        "  - reference_count"
    )

    print(
        "  - reference_unique_count"
    )

    print(
        "  - has_references"
    )

    print(
        "  - modified_after_publication_days"
    )

    print(
        "  - has_modified_date"
    )

    train, test = (
        build_chronological_split(
            df
        )
    )

    y_train = pd.to_numeric(
        train[target_col],
        errors="coerce",
    ).fillna(0).astype(int)

    y_test = pd.to_numeric(
        test[target_col],
        errors="coerce",
    ).fillna(0).astype(int)

    # ================================================================
    # CONSERVATIVE MODEL
    # ================================================================

    section(
        "MODEL 1: THREATLENS CONSERVATIVE"
    )

    X_train_conservative = (
        prepare_matrix(
            train,
            conservative_features,
        )
    )

    X_test_conservative = (
        prepare_matrix(
            test,
            conservative_features,
        )
    )

    print(
        "Training Conservative Random Forest..."
    )

    conservative_model = (
        train_random_forest(
            X_train_conservative,
            y_train,
        )
    )

    print(
        "Training complete."
    )

    conservative_results, conservative_scores = (
        evaluate_model(
            "ThreatLens Conservative",
            conservative_model,
            X_test_conservative,
            y_test,
        )
    )

    print(
        f"Average Precision: "
        f"{conservative_results['average_precision']:.6f}"
    )

    print(
        f"95% Bootstrap CI: "
        f"{conservative_results['ap_ci_lower']:.6f} "
        f"to "
        f"{conservative_results['ap_ci_upper']:.6f}"
    )

    for k in TOP_K_VALUES:

        print(
            f"P@{k}: "
            f"{conservative_results[f'precision_at_{k}']:.4f}  "
            f"Recall@{k}: "
            f"{conservative_results[f'recall_at_{k}']:.4f}"
        )

    # ================================================================
    # CVSS ONLY
    # ================================================================

    section(
        "MODEL 2: CVSS-ONLY"
    )

    cvss_features = [
        feature
        for feature in [
            "cvss_v3_score",
            "cvss_v4_score",
        ]
        if feature in df.columns
    ]

    if not cvss_features:

        raise ValueError(
            "No CVSS features available."
        )

    print(
        f"CVSS features: {cvss_features}"
    )

    X_train_cvss = prepare_matrix(
        train,
        cvss_features,
    )

    X_test_cvss = prepare_matrix(
        test,
        cvss_features,
    )

    print(
        "Training CVSS Random Forest..."
    )

    cvss_model = train_random_forest(
        X_train_cvss,
        y_train,
    )

    print(
        "Training complete."
    )

    cvss_results, cvss_scores = (
        evaluate_model(
            "CVSS-only",
            cvss_model,
            X_test_cvss,
            y_test,
        )
    )

    print(
        f"Average Precision: "
        f"{cvss_results['average_precision']:.6f}"
    )

    print(
        f"95% Bootstrap CI: "
        f"{cvss_results['ap_ci_lower']:.6f} "
        f"to "
        f"{cvss_results['ap_ci_upper']:.6f}"
    )

    for k in TOP_K_VALUES:

        print(
            f"P@{k}: "
            f"{cvss_results[f'precision_at_{k}']:.4f}  "
            f"Recall@{k}: "
            f"{cvss_results[f'recall_at_{k}']:.4f}"
        )

    # ================================================================
    # DIRECT COMPARISON
    # ================================================================

    section(
        "THREATLENS VS CVSS STATISTICAL COMPARISON"
    )

    comparison = bootstrap_ap_difference(
        y_test.to_numpy(),
        conservative_scores,
        cvss_scores,
    )

    print(
        f"Observed AP improvement: "
        f"{comparison['observed_difference']:.6f}"
    )

    print(
        f"95% bootstrap CI: "
        f"{comparison['ci_lower']:.6f} "
        f"to "
        f"{comparison['ci_upper']:.6f}"
    )

    print(
        f"Bootstrap probability ThreatLens > CVSS: "
        f"{comparison['probability_a_better']:.4f}"
    )

    permutation = (
        permutation_test_ap_difference(
            y_test.to_numpy(),
            conservative_scores,
            cvss_scores,
        )
    )

    print(
        f"Permutation p-value: "
        f"{permutation['permutation_p_value']:.6f}"
    )

    # ================================================================
    # RELATIVE IMPROVEMENT
    # ================================================================

    cvss_ap = (
        cvss_results[
            "average_precision"
        ]
    )

    conservative_ap = (
        conservative_results[
            "average_precision"
        ]
    )

    if cvss_ap > 0:

        relative_improvement = (
            (
                conservative_ap
                -
                cvss_ap
            )
            /
            cvss_ap
            * 100.0
        )

    else:

        relative_improvement = np.nan

    print()
    print(
        f"Relative AP improvement over CVSS: "
        f"{relative_improvement:.2f}%"
    )

    # ================================================================
    # FINAL RESULTS TABLE
    # ================================================================

    section(
        "FINAL VALIDATION RESULTS"
    )

    results = pd.DataFrame(
        [
            conservative_results,
            cvss_results,
        ]
    )

    results[
        "ap_improvement_vs_cvss_percent"
    ] = 0.0

    results.loc[
        results["model"]
        == "ThreatLens Conservative",
        "ap_improvement_vs_cvss_percent",
    ] = relative_improvement

    print(
        results[
            [
                "model",
                "average_precision",
                "ap_ci_lower",
                "ap_ci_upper",
                "precision_at_10",
                "precision_at_50",
                "precision_at_100",
                "precision_at_500",
                "recall_at_100",
                "recall_at_500",
                "ap_improvement_vs_cvss_percent",
            ]
        ].to_string(
            index=False
        )
    )

    # ================================================================
    # INTERPRETATION
    # ================================================================

    section(
        "STATISTICAL INTERPRETATION"
    )

    ci_excludes_zero = (
        comparison["ci_lower"] > 0
    )

    statistically_supported = (
        permutation[
            "permutation_p_value"
        ]
        < 0.05
    )

    if (
        ci_excludes_zero
        and statistically_supported
    ):

        conclusion = (
            "The Conservative ThreatLens model shows a "
            "statistically supported improvement over CVSS-only "
            "under this chronological validation split."
        )

    elif ci_excludes_zero:

        conclusion = (
            "The bootstrap confidence interval indicates a positive "
            "ThreatLens-CVSS AP difference, but the permutation test "
            "does not meet the 0.05 threshold."
        )

    else:

        conclusion = (
            "The observed ThreatLens-CVSS improvement is not "
            "statistically conclusive under this validation procedure."
        )

    print(
        conclusion
    )

    print()
    print(
        "Important methodological limitation:"
    )

    print(
        "This significance analysis evaluates one chronological "
        "holdout split. It should not be interpreted as proof of "
        "generalization to every future vulnerability population."
    )

    # ================================================================
    # SAVE RESULTS
    # ================================================================

    results_path = (
        EVAL_DIR
        / "final_validation_results.csv"
    )

    comparison_path = (
        EVAL_DIR
        / "final_validation_comparison.csv"
    )

    results.to_csv(
        results_path,
        index=False,
    )

    comparison_df = pd.DataFrame(
        [
            {
                "comparison": (
                    "ThreatLens Conservative vs CVSS-only"
                ),
                "threatlens_ap": conservative_ap,
                "cvss_ap": cvss_ap,
                "ap_difference": (
                    comparison[
                        "observed_difference"
                    ]
                ),
                "bootstrap_ci_lower": (
                    comparison[
                        "ci_lower"
                    ]
                ),
                "bootstrap_ci_upper": (
                    comparison[
                        "ci_upper"
                    ]
                ),
                "bootstrap_probability_threatlens_better": (
                    comparison[
                        "probability_a_better"
                    ]
                ),
                "permutation_p_value": (
                    permutation[
                        "permutation_p_value"
                    ]
                ),
                "relative_ap_improvement_percent": (
                    relative_improvement
                ),
            }
        ]
    )

    comparison_df.to_csv(
        comparison_path,
        index=False,
    )

    report = {
        "step": 26,
        "title": (
            "Final Model Validation - "
            "Statistical Significance & Confidence Analysis"
        ),
        "random_state": RANDOM_STATE,
        "n_estimators": N_ESTIMATORS,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "validation": {
            "type": "chronological_holdout",
            "training_rows": int(len(train)),
            "test_rows": int(len(test)),
            "training_positive_count": int(
                y_train.sum()
            ),
            "test_positive_count": int(
                y_test.sum()
            ),
        },
        "models": {
            "ThreatLens Conservative": {
                "feature_count": len(
                    conservative_features
                ),
                "average_precision": conservative_ap,
                "ap_ci_lower": conservative_results[
                    "ap_ci_lower"
                ],
                "ap_ci_upper": conservative_results[
                    "ap_ci_upper"
                ],
                "precision_at_100": conservative_results[
                    "precision_at_100"
                ],
                "recall_at_500": conservative_results[
                    "recall_at_500"
                ],
            },
            "CVSS-only": {
                "feature_count": len(
                    cvss_features
                ),
                "average_precision": cvss_ap,
                "ap_ci_lower": cvss_results[
                    "ap_ci_lower"
                ],
                "ap_ci_upper": cvss_results[
                    "ap_ci_upper"
                ],
                "precision_at_100": cvss_results[
                    "precision_at_100"
                ],
                "recall_at_500": cvss_results[
                    "recall_at_500"
                ],
            },
        },
        "comparison": {
            "ap_difference": comparison[
                "observed_difference"
            ],
            "bootstrap_ci_lower": comparison[
                "ci_lower"
            ],
            "bootstrap_ci_upper": comparison[
                "ci_upper"
            ],
            "bootstrap_probability_threatlens_better": comparison[
                "probability_a_better"
            ],
            "permutation_p_value": permutation[
                "permutation_p_value"
            ],
            "relative_ap_improvement_percent": (
                relative_improvement
            ),
        },
        "conclusion": conclusion,
        "limitations": [
            "Single chronological holdout evaluation.",
            "Bootstrap confidence intervals quantify uncertainty "
            "on this test population.",
            "Permutation testing compares ranking performance on "
            "the same test vulnerabilities.",
            "Results do not establish universal future performance.",
            "Current KEV labels represent the available KEV snapshot.",
        ],
        "outputs": {
            "results": str(results_path),
            "comparison": str(comparison_path),
        },
    }

    report_path = (
        EVAL_DIR
        / "final_validation_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            report,
            handle,
            indent=2,
        )

    section(
        "OUTPUT FILES"
    )

    print(
        results_path
    )

    print(
        comparison_path
    )

    print(
        report_path
    )

    banner(
        "STEP 26 COMPLETE"
    )


if __name__ == "__main__":
    main()
