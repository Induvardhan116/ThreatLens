from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score


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

RANDOM_STATE = 42


def precision_at_k(
    y_true: np.ndarray,
    scores: np.ndarray,
    k: int,
) -> float:

    k = min(k, len(y_true))

    if k == 0:
        return 0.0

    order = np.argsort(-scores)[:k]

    return float(
        y_true[order].mean()
    )


def recall_at_k(
    y_true: np.ndarray,
    scores: np.ndarray,
    k: int,
) -> float:

    positives = int(
        y_true.sum()
    )

    if positives == 0:
        return 0.0

    k = min(k, len(y_true))

    order = np.argsort(-scores)[:k]

    return float(
        y_true[order].sum()
        / positives
    )


def build_cvss_score(
    df: pd.DataFrame,
) -> np.ndarray:

    v4 = df[
        "cvss_v4_score"
    ].fillna(0.0)

    v3 = df[
        "cvss_v3_score"
    ].fillna(0.0)

    return np.where(
        df[
            "cvss_v4_available"
        ].astype(bool),
        v4,
        v3,
    ).astype(float)


def build_contextual_score(
    df: pd.DataFrame,
) -> np.ndarray:

    cvss = build_cvss_score(
        df
    )

    reference_component = np.log1p(
        df[
            "reference_count"
        ].fillna(0)
    )

    cpe_component = np.log1p(
        df[
            "affected_cpe_count"
        ].fillna(0)
    )

    cwe_component = np.log1p(
        df[
            "cwe_count"
        ].fillna(0)
    )

    keyword_component = (
        df[
            "description_has_remote"
        ].fillna(0)
        +
        df[
            "description_has_execution"
        ].fillna(0)
        +
        df[
            "description_has_privilege"
        ].fillna(0)
        +
        df[
            "description_has_authentication"
        ].fillna(0)
        +
        df[
            "description_has_overflow"
        ].fillna(0)
        +
        df[
            "description_has_injection"
        ].fillna(0)
        +
        df[
            "description_has_bypass"
        ].fillna(0)
    )

    score = (
        cvss
        + 0.15 * reference_component
        + 0.15 * cpe_component
        + 0.20 * cwe_component
        + 0.10 * keyword_component
    )

    return score.astype(float)


def evaluate_ranking(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> dict:

    return {
        "average_precision": float(
            average_precision_score(
                y_true,
                scores,
            )
        ),
        "p_at_10": precision_at_k(
            y_true,
            scores,
            10,
        ),
        "p_at_50": precision_at_k(
            y_true,
            scores,
            50,
        ),
        "p_at_100": precision_at_k(
            y_true,
            scores,
            100,
        ),
        "p_at_500": precision_at_k(
            y_true,
            scores,
            500,
        ),
        "recall_at_100": recall_at_k(
            y_true,
            scores,
            100,
        ),
        "recall_at_500": recall_at_k(
            y_true,
            scores,
            500,
        ),
        "kev_coverage_at_10": recall_at_k(
            y_true,
            scores,
            10,
        ),
        "kev_coverage_at_50": recall_at_k(
            y_true,
            scores,
            50,
        ),
        "kev_coverage_at_100": recall_at_k(
            y_true,
            scores,
            100,
        ),
        "kev_coverage_at_500": recall_at_k(
            y_true,
            scores,
            500,
        ),
    }


def train_ml(
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestClassifier:

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        x_train,
        y_train,
    )

    return model


def main() -> None:

    print("=" * 72)
    print(
        "ThreatLens - Temporal Robustness Experiment"
    )
    print("=" * 72)

    # ================================================================
    # 1. LOAD FEATURES
    # ================================================================

    print(
        "\nLoading feature dataset..."
    )

    features = pd.read_csv(
        FEATURES_PATH
    )

    print(
        f"Feature rows: {len(features):,}"
    )

    # ================================================================
    # 2. LOAD ENRICHED DATASET
    # ================================================================

    print(
        "\nLoading enriched dataset for publication dates..."
    )

    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:

        records = json.load(f)

    dataset = pd.DataFrame(
        records
    )

    if len(dataset) != len(features):

        raise ValueError(
            "Feature dataset and enriched dataset "
            "have different row counts."
        )

    required_columns = [
        "cve_id",
        "published_date",
        "exploitation_label",
    ]

    for column in required_columns:

        if column not in dataset.columns:

            raise ValueError(
                f"Enriched dataset does not contain "
                f"{column}."
            )

    # ================================================================
    # 3. PARSE DATES
    #
    # IMPORTANT:
    # Convert everything to UTC and then remove timezone information.
    # This makes all datetime comparisons timezone-naive and consistent.
    # ================================================================

    dataset[
        "published_date"
    ] = pd.to_datetime(
        dataset[
            "published_date"
        ],
        errors="coerce",
        utc=True,
    ).dt.tz_localize(None)

    missing_dates = int(
        dataset[
            "published_date"
        ].isna().sum()
    )

    valid_dates = (
        len(dataset)
        - missing_dates
    )

    print(
        f"Records with valid publication dates: "
        f"{valid_dates:,}"
    )

    print(
        f"Records with missing/invalid publication dates: "
        f"{missing_dates:,}"
    )

    # ================================================================
    # 4. VERIFY LABEL ALIGNMENT
    # ================================================================

    labels = pd.to_numeric(
        dataset[
            "exploitation_label"
        ],
        errors="coerce",
    ).fillna(0).astype(int)

    if not labels.isin([0, 1]).all():

        raise ValueError(
            "Invalid exploitation labels detected."
        )

    feature_target = pd.to_numeric(
        features[
            "exploitation_label"
        ],
        errors="coerce",
    ).fillna(0).astype(int)

    if not np.array_equal(
        feature_target.to_numpy(),
        labels.to_numpy(),
    ):

        raise ValueError(
            "Feature target does not align with "
            "enriched dataset labels."
        )

    # ================================================================
    # 5. COMBINE DATA
    # ================================================================

    work = features.copy()

    work[
        "_published_date"
    ] = dataset[
        "published_date"
    ].values

    work[
        "_cve_id"
    ] = dataset[
        "cve_id"
    ].values

    # ================================================================
    # 6. EXCLUDE RECORDS WITHOUT DATES
    # ================================================================

    undated = work[
        work[
            "_published_date"
        ].isna()
    ].copy()

    dated = work[
        work[
            "_published_date"
        ].notna()
    ].copy()

    print(
        "\nExcluded from temporal experiment: "
        f"{len(undated):,}"
    )

    print(
        "Rows available for temporal experiment: "
        f"{len(dated):,}"
    )

    undated_positive = int(
        undated[
            "exploitation_label"
        ].sum()
    )

    dated_positive = int(
        dated[
            "exploitation_label"
        ].sum()
    )

    print(
        "Excluded KEV positives: "
        f"{undated_positive:,}"
    )

    print(
        "Dated KEV positives: "
        f"{dated_positive:,}"
    )

    if len(dated) == 0:

        raise RuntimeError(
            "No records with valid publication dates."
        )

    # ================================================================
    # 7. SORT CHRONOLOGICALLY
    # ================================================================

    work = dated.sort_values(
        "_published_date"
    ).reset_index(
        drop=True
    )

    print(
        "\nPublication range: "
        f"{work['_published_date'].min().date()} "
        "to "
        f"{work['_published_date'].max().date()}"
    )

    print(
        "Temporal evaluation positives: "
        f"{int(work['exploitation_label'].sum()):,}"
    )

    # ================================================================
    # 8. MODEL FEATURES
    # ================================================================

    excluded_columns = {
        "exploitation_label",
        "_published_date",
        "_cve_id",
    }

    feature_columns = [
        column
        for column in work.columns
        if column not in excluded_columns
    ]

    print(
        f"Model features: {len(feature_columns)}"
    )

    # ================================================================
    # 9. CHRONOLOGICAL FOLDS
    # ================================================================

    folds = [
        {
            "name": "2025_H1",
            "train_end": "2025-01-01",
            "test_start": "2025-01-01",
            "test_end": "2025-07-01",
        },
        {
            "name": "2025_H2",
            "train_end": "2025-07-01",
            "test_start": "2025-07-01",
            "test_end": "2026-01-01",
        },
        {
            "name": "2026_H1",
            "train_end": "2026-01-01",
            "test_start": "2026-01-01",
            "test_end": "2026-07-01",
        },
        {
            "name": "2026_H2_partial",
            "train_end": "2026-07-01",
            "test_start": "2026-07-01",
            "test_end": "2026-09-10",
        },
    ]

    results = []

    # ================================================================
    # 10. RUN FOLDS
    # ================================================================

    for fold in folds:

        # ------------------------------------------------------------
        # All timestamps are deliberately timezone-naive here.
        # ------------------------------------------------------------

        train_end = pd.Timestamp(
            fold["train_end"]
        )

        test_start = pd.Timestamp(
            fold["test_start"]
        )

        test_end = pd.Timestamp(
            fold["test_end"]
        )

        train_mask = (
            work[
                "_published_date"
            ]
            < train_end
        )

        test_mask = (
            (
                work[
                    "_published_date"
                ]
                >= test_start
            )
            &
            (
                work[
                    "_published_date"
                ]
                < test_end
            )
        )

        train_df = work.loc[
            train_mask
        ].copy()

        test_df = work.loc[
            test_mask
        ].copy()

        if len(train_df) == 0:

            print(
                f"\nSkipping {fold['name']} "
                "- empty training set."
            )

            continue

        if len(test_df) == 0:

            print(
                f"\nSkipping {fold['name']} "
                "- empty test set."
            )

            continue

        train_positive = int(
            train_df[
                "exploitation_label"
            ].sum()
        )

        test_positive = int(
            test_df[
                "exploitation_label"
            ].sum()
        )

        print(
            "\n"
            + "-"
            * 72
        )

        print(
            f"Fold: {fold['name']}"
        )

        print(
            f"Train period: before "
            f"{fold['train_end']}"
        )

        print(
            f"Test period: "
            f"{fold['test_start']} "
            f"to {fold['test_end']}"
        )

        print(
            f"Train rows: {len(train_df):,}"
        )

        print(
            f"Train KEV positives: "
            f"{train_positive:,}"
        )

        print(
            f"Test rows: {len(test_df):,}"
        )

        print(
            f"Test KEV positives: "
            f"{test_positive:,}"
        )

        if train_positive < 20:

            print(
                "Skipping: training set has "
                "fewer than 20 positive examples."
            )

            continue

        if test_positive < 5:

            print(
                "Skipping: test set has "
                "fewer than 5 positive examples."
            )

            continue

        x_train = train_df[
            feature_columns
        ]

        y_train = train_df[
            "exploitation_label"
        ].astype(int)

        x_test = test_df[
            feature_columns
        ]

        y_test = test_df[
            "exploitation_label"
        ].astype(int)

        y_test_np = y_test.to_numpy()

        # ============================================================
        # CVSS-ONLY
        # ============================================================

        cvss_scores = build_cvss_score(
            test_df
        )

        cvss_metrics = evaluate_ranking(
            y_test_np,
            cvss_scores,
        )

        # ============================================================
        # CONTEXTUAL RULES
        # ============================================================

        contextual_scores = (
            build_contextual_score(
                test_df
            )
        )

        contextual_metrics = (
            evaluate_ranking(
                y_test_np,
                contextual_scores,
            )
        )

        # ============================================================
        # THREATLENS ML
        # ============================================================

        print(
            "\nTraining ThreatLens Random Forest..."
        )

        model = train_ml(
            x_train,
            y_train,
        )

        ml_scores = model.predict_proba(
            x_test
        )[:, 1]

        ml_metrics = evaluate_ranking(
            y_test_np,
            ml_scores,
        )

        # ============================================================
        # STORE RESULTS
        # ============================================================

        models = [
            (
                "CVSS-only",
                cvss_metrics,
            ),
            (
                "Contextual rules",
                contextual_metrics,
            ),
            (
                "ThreatLens ML",
                ml_metrics,
            ),
        ]

        for model_name, metrics in models:

            results.append(
                {
                    "fold": fold["name"],
                    "model": model_name,
                    "train_rows": len(train_df),
                    "train_positives": train_positive,
                    "test_rows": len(test_df),
                    "test_positives": test_positive,
                    **metrics,
                }
            )

        print(
            "\nFold results:"
        )

        print(
            f"CVSS-only        "
            f"AP={cvss_metrics['average_precision']:.6f} "
            f"P@100={cvss_metrics['p_at_100']:.4f} "
            f"R@500={cvss_metrics['recall_at_500']:.4f}"
        )

        print(
            f"Contextual rules  "
            f"AP={contextual_metrics['average_precision']:.6f} "
            f"P@100={contextual_metrics['p_at_100']:.4f} "
            f"R@500={contextual_metrics['recall_at_500']:.4f}"
        )

        print(
            f"ThreatLens ML    "
            f"AP={ml_metrics['average_precision']:.6f} "
            f"P@100={ml_metrics['p_at_100']:.4f} "
            f"R@500={ml_metrics['recall_at_500']:.4f}"
        )

    # ================================================================
    # 11. CHECK RESULTS
    # ================================================================

    if not results:

        raise RuntimeError(
            "No temporal folds were eligible "
            "for evaluation."
        )

    results_df = pd.DataFrame(
        results
    )

    # ================================================================
    # 12. SAVE DETAILED RESULTS
    # ================================================================

    results_path = (
        OUTPUT_DIR
        / "temporal_robustness_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    # ================================================================
    # 13. AGGREGATE RESULTS
    # ================================================================

    metric_columns = [
        "average_precision",
        "p_at_10",
        "p_at_50",
        "p_at_100",
        "p_at_500",
        "recall_at_100",
        "recall_at_500",
        "kev_coverage_at_10",
        "kev_coverage_at_50",
        "kev_coverage_at_100",
        "kev_coverage_at_500",
    ]

    summary_rows = []

    for model_name in [
        "CVSS-only",
        "Contextual rules",
        "ThreatLens ML",
    ]:

        subset = results_df[
            results_df[
                "model"
            ]
            == model_name
        ]

        row = {
            "model": model_name,
            "folds": len(subset),
        }

        for metric in metric_columns:

            row[
                f"{metric}_mean"
            ] = float(
                subset[
                    metric
                ].mean()
            )

            row[
                f"{metric}_std"
            ] = float(
                subset[
                    metric
                ].std(
                    ddof=0
                )
            )

        summary_rows.append(
            row
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        OUTPUT_DIR
        / "temporal_robustness_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    # ================================================================
    # 14. IMPROVEMENT ANALYSIS
    # ================================================================

    improvement_rows = []

    for fold_name in (
        results_df[
            "fold"
        ].unique()
    ):

        fold_df = results_df[
            results_df[
                "fold"
            ]
            == fold_name
        ]

        ml = fold_df[
            fold_df[
                "model"
            ]
            == "ThreatLens ML"
        ].iloc[0]

        cvss = fold_df[
            fold_df[
                "model"
            ]
            == "CVSS-only"
        ].iloc[0]

        rules = fold_df[
            fold_df[
                "model"
            ]
            == "Contextual rules"
        ].iloc[0]

        improvement_rows.append(
            {
                "fold": fold_name,

                "ml_ap": ml[
                    "average_precision"
                ],

                "cvss_ap": cvss[
                    "average_precision"
                ],

                "rules_ap": rules[
                    "average_precision"
                ],

                "ml_minus_cvss_ap": (
                    ml[
                        "average_precision"
                    ]
                    -
                    cvss[
                        "average_precision"
                    ]
                ),

                "ml_minus_rules_ap": (
                    ml[
                        "average_precision"
                    ]
                    -
                    rules[
                        "average_precision"
                    ]
                ),

                "ml_p_at_100": ml[
                    "p_at_100"
                ],

                "cvss_p_at_100": cvss[
                    "p_at_100"
                ],

                "rules_p_at_100": rules[
                    "p_at_100"
                ],

                "ml_recall_at_500": ml[
                    "recall_at_500"
                ],

                "cvss_recall_at_500": cvss[
                    "recall_at_500"
                ],

                "rules_recall_at_500": rules[
                    "recall_at_500"
                ],
            }
        )

    improvement_df = pd.DataFrame(
        improvement_rows
    )

    improvement_path = (
        OUTPUT_DIR
        / "temporal_robustness_improvements.csv"
    )

    improvement_df.to_csv(
        improvement_path,
        index=False,
    )

    # ================================================================
    # 15. FINAL SUMMARY
    # ================================================================

    print(
        "\n"
        + "="
        * 72
    )

    print(
        "TEMPORAL ROBUSTNESS SUMMARY"
    )

    print(
        "="
        * 72
    )

    display_columns = [
        "model",
        "folds",
        "average_precision_mean",
        "p_at_100_mean",
        "p_at_500_mean",
        "recall_at_100_mean",
        "recall_at_500_mean",
        "kev_coverage_at_500_mean",
    ]

    print(
        summary_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\n"
        + "="
        * 72
    )

    print(
        "ML IMPROVEMENT BY FOLD"
    )

    print(
        "="
        * 72
    )

    print(
        improvement_df.to_string(
            index=False
        )
    )

    print(
        "\nSaved files:"
    )

    print(
        results_path
    )

    print(
        summary_path
    )

    print(
        improvement_path
    )

    print(
        "\nEvaluation notes:"
    )

    print(
        f"- {missing_dates:,} records without valid "
        "publication dates were excluded from temporal splitting."
    )

    print(
        "- Those records remain in the original "
        "ThreatLens dataset."
    )

    print(
        "- Date comparisons use timezone-naive UTC-normalized dates."
    )

    print(
        "- No future publication records are used "
        "to train an earlier fold."
    )

    print(
        "- The exploitation label is based on the "
        "current KEV catalog snapshot."
    )

    print(
        "- Therefore this experiment measures "
        "chronological distribution robustness, "
        "not strictly historical as-of-date "
        "KEV availability."
    )


if __name__ == "__main__":
    main()