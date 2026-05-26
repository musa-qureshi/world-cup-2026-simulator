"""High-level match prediction utilities built on the trained model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.feature_engineering import FEATURE_COLUMNS, FeatureEngineer
from src.preprocessing import load_fifa_rankings, load_match_data, prepare_training_frame
from src.train_model import CODE_TO_OUTCOME, OUTCOME_TO_CODE, build_training_dataset, load_trained_artifacts, train_and_save_model
from src.utils import DEFAULT_METADATA_PATH, DEFAULT_MODEL_PATH, LOGGER, MatchProbabilities, normalize_probabilities


@dataclass
class MatchPrediction:
    """Prediction output for a single football match."""

    home_team: str
    away_team: str
    probabilities: MatchProbabilities
    expected_home_goals: float
    expected_away_goals: float
    home_elo: float
    away_elo: float
    model_name: str


class MatchPredictor:
    """Load a model and generate outcome probabilities for arbitrary fixtures."""

    def __init__(self, model_path: str | Path | None = None, metadata_path: str | Path | None = None) -> None:
        self.model_path = Path(model_path or DEFAULT_MODEL_PATH)
        self.metadata_path = Path(metadata_path or DEFAULT_METADATA_PATH)
        self.model, self.metadata = self._load_or_train()
        state = self.metadata.get("feature_engineer_state", {})
        if state:
            self.feature_engineer = FeatureEngineer.from_state(state)
        else:
            rankings = load_fifa_rankings()
            _, _, feature_engineer = build_training_dataset(load_match_data(), rankings)
            self.feature_engineer = feature_engineer
        self.feature_columns = list(self.metadata.get("feature_columns", FEATURE_COLUMNS))
        self.model_name = str(self.metadata.get("best_model_name", "unknown"))

    def _load_or_train(self):
        if self.model_path.exists() and self.metadata_path.exists():
            return load_trained_artifacts(self.model_path, self.metadata_path)
        LOGGER.info("Model artifacts not found. Training a fresh model from available data.")
        result = train_and_save_model(model_path=self.model_path, metadata_path=self.metadata_path)
        return load_trained_artifacts(result.model_path, result.metadata_path)

    def predict_match(self, home_team: str, away_team: str, neutral: bool = True, match_date: pd.Timestamp | None = None) -> MatchPrediction:
        """Predict probabilities for a single fixture."""

        feature_frame = self.feature_engineer.transform_single(home_team, away_team, neutral=neutral, match_date=match_date)
        model_input = feature_frame[self.feature_columns]
        probabilities = self.model.predict_proba(model_input)[0]
        normalized = normalize_probabilities(probabilities)
        home_elo = self.feature_engineer.elo_system.get_team_rating(home_team)
        away_elo = self.feature_engineer.elo_system.get_team_rating(away_team)
        home_xg, away_xg = self._estimate_expected_goals(model_input.iloc[0], normalized)
        return MatchPrediction(
            home_team=home_team,
            away_team=away_team,
            probabilities=MatchProbabilities(
                team_a=float(normalized[OUTCOME_TO_CODE["home_win"]]),
                draw=float(normalized[OUTCOME_TO_CODE["draw"]]),
                team_b=float(normalized[OUTCOME_TO_CODE["away_win"]]),
            ),
            expected_home_goals=home_xg,
            expected_away_goals=away_xg,
            home_elo=float(home_elo),
            away_elo=float(away_elo),
            model_name=self.model_name,
        )

    def predict_proba_dict(self, home_team: str, away_team: str, neutral: bool = True, match_date: pd.Timestamp | None = None) -> dict[str, float]:
        """Return a plain dictionary of probabilities for downstream consumers."""

        prediction = self.predict_match(home_team, away_team, neutral=neutral, match_date=match_date)
        return prediction.probabilities.as_dict()

    def update_after_match(self, home_team: str, away_team: str, home_score: int, away_score: int, neutral: bool = True, match_date: pd.Timestamp | None = None) -> None:
        """Update the live feature state with a real match result."""

        match_date = pd.Timestamp(match_date or pd.Timestamp.now()).tz_localize(None)
        self.feature_engineer.elo_system.update_elo(home_team, away_team, home_score, away_score, neutral=neutral)
        update_frame = pd.Series(
            {
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
                "neutral": neutral,
                "tournament": "Live Update",
                "date": match_date,
            }
        )
        self.feature_engineer._update_team_history(home_team, away_team, update_frame, match_date)
        self._save_state()

    def _save_state(self) -> None:
        """Persist the live feature state to the metadata file."""

        self.metadata["feature_engineer_state"] = self.feature_engineer.export_state()
        from src.utils import write_json

        write_json(self.metadata_path, self.metadata)

    @staticmethod
    def _estimate_expected_goals(feature_row: pd.Series, probabilities: np.ndarray) -> tuple[float, float]:
        """Estimate expected goals from matchup features and predicted probabilities."""

        home_elo = float(feature_row["home_elo"])
        away_elo = float(feature_row["away_elo"])
        elo_diff = home_elo - away_elo
        home_form = float(feature_row["home_form_index"])
        away_form = float(feature_row["away_form_index"])
        draw_prob = float(probabilities[OUTCOME_TO_CODE["draw"]])
        home_xg = np.clip(1.15 + 0.55 * (elo_diff / 400.0) + 0.35 * home_form - 0.15 * draw_prob, 0.15, 3.5)
        away_xg = np.clip(1.05 - 0.45 * (elo_diff / 400.0) + 0.35 * away_form - 0.15 * draw_prob, 0.15, 3.5)
        return float(home_xg), float(away_xg)


def load_predictor(model_path: str | Path | None = None, metadata_path: str | Path | None = None) -> MatchPredictor:
    """Convenience loader for the app and simulator layers."""

    return MatchPredictor(model_path=model_path, metadata_path=metadata_path)
