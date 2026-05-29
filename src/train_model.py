"""Train and evaluate match prediction models for football outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.feature_engineering import FEATURE_COLUMNS, FeatureEngineer
from src.preprocessing import load_fifa_rankings, load_match_data, prepare_training_frame
from src.utils import DEFAULT_METADATA_PATH, DEFAULT_MODEL_PATH, LOGGER, ensure_directory, write_json

try:  # pragma: no cover - optional dependency
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

try:  # pragma: no cover - optional dependency
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None


OUTCOME_TO_CODE = {"home_win": 0, "draw": 1, "away_win": 2}
CODE_TO_OUTCOME = {value: key for key, value in OUTCOME_TO_CODE.items()}


@dataclass
class TrainingResult:
    """Artifacts produced by a full training run."""

    model_path: Path
    metadata_path: Path
    best_model_name: str
    metrics: dict[str, Any]
    feature_engineer_state: dict[str, Any]


def build_training_dataset(matches: pd.DataFrame | None = None, rankings: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.Series, FeatureEngineer]:
    """Create the feature matrix and labels used for supervised learning."""

    match_frame = matches if matches is not None else load_match_data()
    ranking_frame = rankings if rankings is not None else load_fifa_rankings()
    prepared = prepare_training_frame(match_frame, ranking_frame)
    feature_engineer = FeatureEngineer(ranking_frame)
    feature_frame = feature_engineer.fit_transform(prepared)
    X = feature_frame[FEATURE_COLUMNS].copy()
    y = pd.Series(
        np.select(
            [feature_frame["home_win"] == 1, feature_frame["draw"] == 1, feature_frame["away_win"] == 1],
            [OUTCOME_TO_CODE["home_win"], OUTCOME_TO_CODE["draw"], OUTCOME_TO_CODE["away_win"]],
            default=OUTCOME_TO_CODE["draw"],
        ),
        name="outcome",
    )
    return X, y, feature_engineer


def _candidate_models(random_state: int = 42) -> dict[str, Pipeline]:
    """Return the set of candidate estimators for comparison."""

    candidates: dict[str, Pipeline] = {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=None,
                        min_samples_split=4,
                        min_samples_leaf=2,
                        random_state=random_state,
                        class_weight="balanced_subsample",
                    ),
                )
            ]
        ),
    }
    if XGBClassifier is not None:
        candidates["xgboost"] = Pipeline(
            [
                (
                    "model",
                    XGBClassifier(
                        n_estimators=300,
                        learning_rate=0.05,
                        max_depth=4,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="multi:softprob",
                        num_class=3,
                        eval_metric="mlogloss",
                        tree_method="hist",
                        random_state=random_state,
                    ),
                )
            ]
        )
    if LGBMClassifier is not None:
        candidates["lightgbm"] = Pipeline(
            [
                (
                    "model",
                    LGBMClassifier(
                        n_estimators=400,
                        learning_rate=0.04,
                        num_leaves=31,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        random_state=random_state,
                    ),
                )
            ]
        )
    return candidates


def _parameter_grids(random_state: int = 42) -> dict[str, dict[str, list[Any]]]:
    """Return lightweight hyperparameter grids for tuning."""

    grids: dict[str, dict[str, list[Any]]] = {
        "logistic_regression": {
            "model__C": [0.25, 0.5, 1.0, 2.0],
            "model__solver": ["lbfgs"],
        },
        "random_forest": {
            "model__n_estimators": [300, 500],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 4],
            "model__min_samples_leaf": [1, 2],
        },
    }
    if XGBClassifier is not None:
        grids["xgboost"] = {
            "model__n_estimators": [150, 300],
            "model__max_depth": [3, 4, 5],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__subsample": [0.8, 0.9],
            "model__colsample_bytree": [0.8, 0.9],
        }
    if LGBMClassifier is not None:
        grids["lightgbm"] = {
            "model__n_estimators": [200, 400],
            "model__num_leaves": [31, 63],
            "model__learning_rate": [0.03, 0.05],
            "model__subsample": [0.8, 0.9],
            "model__colsample_bytree": [0.8, 0.9],
        }
    return grids


def _tune_model(
    name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> tuple[Pipeline, dict[str, Any]]:
    """Tune one candidate model and return the best fitted estimator plus metadata."""

    grids = _parameter_grids(random_state)
    param_grid = grids.get(name, {})
    if not param_grid:
        pipeline.fit(X_train, y_train)
        return pipeline, {"cv_best_score": None, "best_params": {}, "n_candidates": 1}

    class_counts = y_train.value_counts()
    min_class = int(class_counts.min()) if not class_counts.empty else 2
    n_splits = max(3, min(5, min_class))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="neg_log_loss",
        cv=cv,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )
    search.fit(X_train, y_train)
    summary = {
        "cv_best_score": float(search.best_score_),
        "best_params": search.best_params_,
        "n_candidates": int(len(search.cv_results_["params"])),
    }
    return search.best_estimator_, summary


def train_and_save_model(
    matches: pd.DataFrame | None = None,
    rankings: pd.DataFrame | None = None,
    model_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    random_state: int = 42,
) -> TrainingResult:
    """Train multiple models, evaluate them, and persist the best one."""

    X, y, feature_engineer = build_training_dataset(matches, rankings)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )
    model_scores: dict[str, dict[str, Any]] = {}
    best_name = ""
    best_pipeline: Pipeline | None = None
    best_log_loss = float("inf")
    for name, pipeline in _candidate_models(random_state).items():
        try:
            tuned_pipeline, tuning_summary = _tune_model(name, pipeline, X_train, y_train, random_state=random_state)
            y_pred = tuned_pipeline.predict(X_test)
            y_proba = tuned_pipeline.predict_proba(X_test)
            candidate_metrics = {
                "cv_best_score": tuning_summary.get("cv_best_score"),
                "best_params": tuning_summary.get("best_params", {}),
                "n_candidates": tuning_summary.get("n_candidates", 1),
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "log_loss": float(log_loss(y_test, y_proba, labels=[0, 1, 2])),
                "f1_score": float(f1_score(y_test, y_pred, average="macro")),
                "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1, 2]).tolist(),
                "classification_report": classification_report(y_test, y_pred, labels=[0, 1, 2], target_names=["home_win", "draw", "away_win"], output_dict=True),
            }
            model_scores[name] = candidate_metrics
            if candidate_metrics["log_loss"] < best_log_loss:
                best_name = name
                best_pipeline = tuned_pipeline
                best_log_loss = candidate_metrics["log_loss"]
        except Exception as exc:  # pragma: no cover - optional dependency failures
            LOGGER.exception("Model %s failed during training: %s", name, exc)
    if best_pipeline is None:
        raise RuntimeError("No model could be trained successfully.")
    model_path = Path(model_path or DEFAULT_MODEL_PATH)
    metadata_path = Path(metadata_path or DEFAULT_METADATA_PATH)
    ensure_directory(model_path.parent)
    joblib.dump(best_pipeline, model_path)
    metadata = {
        "best_model_name": best_name,
        "feature_columns": FEATURE_COLUMNS,
        "outcome_mapping": OUTCOME_TO_CODE,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": model_scores,
        "feature_engineer_state": feature_engineer.export_state(),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
    }
    write_json(metadata_path, metadata)
    return TrainingResult(
        model_path=model_path,
        metadata_path=metadata_path,
        best_model_name=best_name,
        metrics=model_scores,
        feature_engineer_state=metadata["feature_engineer_state"],
    )


def load_trained_artifacts(model_path: str | Path | None = None, metadata_path: str | Path | None = None) -> tuple[Pipeline, dict[str, Any]]:
    """Load a saved model and its metadata."""

    model = joblib.load(Path(model_path or DEFAULT_MODEL_PATH))
    from src.utils import read_json

    metadata = read_json(Path(metadata_path or DEFAULT_METADATA_PATH), default={})
    return model, metadata


def main() -> None:
    """Train the model when the module is executed as a script."""

    result = train_and_save_model()
    LOGGER.info("Best model: %s", result.best_model_name)
    LOGGER.info("Saved model to %s", result.model_path)


if __name__ == "__main__":
    main()
