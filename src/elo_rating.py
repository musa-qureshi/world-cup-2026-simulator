"""Custom Elo rating system for football teams."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils import LOGGER, read_json, write_json


@dataclass
class EloRatingSystem:
    """Maintain a dynamic Elo table for all teams in the simulator."""

    initial_rating: float = 1500.0
    k_factor: float = 24.0
    home_advantage: float = 60.0
    ratings: dict[str, float] = field(default_factory=dict)

    def calculate_expected_score(self, team_a: str, team_b: str, neutral: bool = True) -> tuple[float, float]:
        """Compute expected scores for team A and team B."""

        rating_a = self.get_team_rating(team_a)
        rating_b = self.get_team_rating(team_b)
        if not neutral:
            rating_a += self.home_advantage
        expected_a = 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))
        expected_b = 1.0 - expected_a
        return expected_a, expected_b

    def update_elo(self, team_a: str, team_b: str, score_a: float, score_b: float, neutral: bool = True) -> tuple[float, float]:
        """Update ratings after a completed match."""

        expected_a, expected_b = self.calculate_expected_score(team_a, team_b, neutral=neutral)
        if score_a > score_b:
            actual_a, actual_b = 1.0, 0.0
        elif score_b > score_a:
            actual_a, actual_b = 0.0, 1.0
        else:
            actual_a = actual_b = 0.5
        margin_multiplier = max(1.0, abs(score_a - score_b) + 1.0)
        rating_a = self.get_team_rating(team_a) + self.k_factor * margin_multiplier * (actual_a - expected_a)
        rating_b = self.get_team_rating(team_b) + self.k_factor * margin_multiplier * (actual_b - expected_b)
        self.ratings[team_a] = rating_a
        self.ratings[team_b] = rating_b
        return rating_a, rating_b

    def get_team_rating(self, team: str) -> float:
        """Return the current rating for a team, creating an initial entry if needed."""

        return float(self.ratings.get(team, self.initial_rating))

    def bulk_fit(self, matches: pd.DataFrame) -> "EloRatingSystem":
        """Update ratings sequentially over a chronological match table."""

        ordered = matches.sort_values("date") if "date" in matches.columns else matches
        for _, row in ordered.iterrows():
            self.update_elo(
                str(row["home_team"]),
                str(row["away_team"]),
                float(row["home_score"]),
                float(row["away_score"]),
                neutral=bool(row.get("neutral", True)),
            )
        return self

    def to_frame(self) -> pd.DataFrame:
        """Serialize ratings into a dataframe."""

        return pd.DataFrame(sorted(self.ratings.items()), columns=["team", "rating"])

    def save(self, path: str | Path) -> None:
        """Persist the Elo state to disk."""

        payload = {
            "initial_rating": self.initial_rating,
            "k_factor": self.k_factor,
            "home_advantage": self.home_advantage,
            "ratings": self.ratings,
        }
        write_json(Path(path), payload)

    @classmethod
    def load(cls, path: str | Path) -> "EloRatingSystem":
        """Load an Elo state from disk."""

        payload = read_json(Path(path))
        if not payload:
            raise FileNotFoundError(f"Elo state not found: {path}")
        return cls(
            initial_rating=float(payload.get("initial_rating", 1500.0)),
            k_factor=float(payload.get("k_factor", 24.0)),
            home_advantage=float(payload.get("home_advantage", 60.0)),
            ratings={key: float(value) for key, value in payload.get("ratings", {}).items()},
        )


def calculate_expected_score(team_a_rating: float, team_b_rating: float) -> float:
    """Return the expected score for team A given two Elo ratings."""

    return 1.0 / (1.0 + 10 ** ((team_b_rating - team_a_rating) / 400.0))


def update_elo(team_a_rating: float, team_b_rating: float, score_a: float, score_b: float, k_factor: float = 24.0) -> tuple[float, float]:
    """Standalone Elo update helper used by tests and quick calculations."""

    expected_a = calculate_expected_score(team_a_rating, team_b_rating)
    expected_b = 1.0 - expected_a
    if score_a > score_b:
        actual_a, actual_b = 1.0, 0.0
    elif score_b > score_a:
        actual_a, actual_b = 0.0, 1.0
    else:
        actual_a = actual_b = 0.5
    return (
        team_a_rating + k_factor * (actual_a - expected_a),
        team_b_rating + k_factor * (actual_b - expected_b),
    )
