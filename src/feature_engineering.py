"""Football-specific feature engineering for match prediction models."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.elo_rating import EloRatingSystem
from src.utils import LOGGER, safe_divide


FEATURE_COLUMNS = [
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_rank",
    "away_rank",
    "rank_diff",
    "home_recent_goals_for_5",
    "away_recent_goals_for_5",
    "home_recent_goals_against_5",
    "away_recent_goals_against_5",
    "home_recent_points_5",
    "away_recent_points_5",
    "home_recent_win_pct_10",
    "away_recent_win_pct_10",
    "home_recent_goal_diff_10",
    "away_recent_goal_diff_10",
    "home_matches_played",
    "away_matches_played",
    "home_rest_days",
    "away_rest_days",
    "neutral_venue",
    "same_confederation",
    "home_continent_advantage",
    "travel_distance_km",
    "avg_goals_scored_diff",
    "avg_goals_conceded_diff",
    "home_form_index",
    "away_form_index",
    "tournament_experience_diff",
]


@dataclass
class TeamHistory:
    """Rolling state for one team."""

    results_5: deque = field(default_factory=lambda: deque(maxlen=5))
    results_10: deque = field(default_factory=lambda: deque(maxlen=10))
    goals_for_5: deque = field(default_factory=lambda: deque(maxlen=5))
    goals_against_5: deque = field(default_factory=lambda: deque(maxlen=5))
    goals_for_10: deque = field(default_factory=lambda: deque(maxlen=10))
    goals_against_10: deque = field(default_factory=lambda: deque(maxlen=10))
    last_match_date: pd.Timestamp | None = None
    matches_played: int = 0
    tournament_matches: int = 0
    total_goals_for: int = 0
    total_goals_against: int = 0
    total_wins: int = 0
    total_draws: int = 0
    total_losses: int = 0
    continent: str = "Unknown"

    def points(self) -> int:
        return 3 * self.total_wins + self.total_draws

    def recent_points(self, window: deque) -> float:
        points = 0
        for result in window:
            if result == 1:
                points += 3
            elif result == 0:
                points += 1
        return float(points)

    def to_dict(self) -> dict[str, Any]:
        return {
            "results_5": list(self.results_5),
            "results_10": list(self.results_10),
            "goals_for_5": list(self.goals_for_5),
            "goals_against_5": list(self.goals_against_5),
            "goals_for_10": list(self.goals_for_10),
            "goals_against_10": list(self.goals_against_10),
            "last_match_date": self.last_match_date.isoformat() if self.last_match_date is not None else None,
            "matches_played": self.matches_played,
            "tournament_matches": self.tournament_matches,
            "total_goals_for": self.total_goals_for,
            "total_goals_against": self.total_goals_against,
            "total_wins": self.total_wins,
            "total_draws": self.total_draws,
            "total_losses": self.total_losses,
            "continent": self.continent,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TeamHistory":
        history = cls()
        history.results_5.extend(payload.get("results_5", []))
        history.results_10.extend(payload.get("results_10", []))
        history.goals_for_5.extend(payload.get("goals_for_5", []))
        history.goals_against_5.extend(payload.get("goals_against_5", []))
        history.goals_for_10.extend(payload.get("goals_for_10", []))
        history.goals_against_10.extend(payload.get("goals_against_10", []))
        last_match_date = payload.get("last_match_date")
        history.last_match_date = pd.Timestamp(last_match_date) if last_match_date else None
        history.matches_played = int(payload.get("matches_played", 0))
        history.tournament_matches = int(payload.get("tournament_matches", 0))
        history.total_goals_for = int(payload.get("total_goals_for", 0))
        history.total_goals_against = int(payload.get("total_goals_against", 0))
        history.total_wins = int(payload.get("total_wins", 0))
        history.total_draws = int(payload.get("total_draws", 0))
        history.total_losses = int(payload.get("total_losses", 0))
        history.continent = str(payload.get("continent", "Unknown"))
        return history


class FeatureEngineer:
    """Create match-level features from historical results and rankings."""

    def __init__(self, rankings: pd.DataFrame, elo_system: EloRatingSystem | None = None) -> None:
        self.rankings = rankings.copy()
        self.rankings["team"] = self.rankings["team"].astype(str)
        self.rankings = self.rankings.drop_duplicates("team").set_index("team")
        self.elo_system = elo_system or EloRatingSystem()
        self.team_histories: dict[str, TeamHistory] = defaultdict(TeamHistory)

    def fit_transform(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Generate the full training matrix from raw match history."""

        feature_rows: list[dict[str, Any]] = []
        ordered = matches.sort_values("date").reset_index(drop=True).copy()
        ordered["date"] = pd.to_datetime(ordered["date"])
        for _, row in ordered.iterrows():
            home_team = str(row["home_team"])
            away_team = str(row["away_team"])
            date = pd.to_datetime(row["date"])
            home_history = self.team_histories[home_team]
            away_history = self.team_histories[away_team]
            feature_rows.append(self._build_feature_row(row, home_history, away_history, date))
            self._update_team_history(home_team, away_team, row, date)
            self.elo_system.update_elo(home_team, away_team, float(row["home_score"]), float(row["away_score"]), neutral=bool(row.get("neutral", True)))
        features = pd.DataFrame(feature_rows)
        features["home_win"] = (ordered["home_score"] > ordered["away_score"]).astype(int)
        features["draw"] = (ordered["home_score"] == ordered["away_score"]).astype(int)
        features["away_win"] = (ordered["away_score"] > ordered["home_score"]).astype(int)
        return features

    def transform_single(self, home_team: str, away_team: str, neutral: bool = True, match_date: pd.Timestamp | None = None) -> pd.DataFrame:
        """Create a one-row feature frame for an upcoming match."""

        current_date = match_date or pd.Timestamp.now()
        current_date = pd.Timestamp(current_date).tz_localize(None) if pd.Timestamp(current_date).tzinfo is not None else pd.Timestamp(current_date)
        row = pd.Series({"home_team": home_team, "away_team": away_team, "neutral": neutral, "date": current_date})
        home_history = self.team_histories[home_team]
        away_history = self.team_histories[away_team]
        return pd.DataFrame([self._build_feature_row(row, home_history, away_history, pd.Timestamp(row["date"]))])

    def _build_feature_row(self, row: pd.Series, home_history: TeamHistory, away_history: TeamHistory, match_date: pd.Timestamp) -> dict[str, Any]:
        home_team = str(row["home_team"])
        away_team = str(row["away_team"])
        home_rank = self._get_rank(home_team)
        away_rank = self._get_rank(away_team)
        home_elo = self.elo_system.get_team_rating(home_team)
        away_elo = self.elo_system.get_team_rating(away_team)
        home_confed = self._get_confederation(home_team)
        away_confed = self._get_confederation(away_team)
        same_confederation = int(home_confed != "Unknown" and home_confed == away_confed)
        neutral_venue = int(bool(row.get("neutral", True)))
        home_rest_days = self._rest_days(home_history.last_match_date, match_date)
        away_rest_days = self._rest_days(away_history.last_match_date, match_date)
        home_continent_advantage = int(not neutral_venue and home_confed != "Unknown")
        travel_distance_km = self._approximate_travel_distance(home_team, away_team)
        home_recent_points_5 = home_history.recent_points(home_history.results_5)
        away_recent_points_5 = away_history.recent_points(away_history.results_5)
        home_form_index = self._form_index(home_history)
        away_form_index = self._form_index(away_history)
        feature_row = {
            "home_team": home_team,
            "away_team": away_team,
            "date": match_date,
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_diff": home_elo - away_elo,
            "home_rank": home_rank,
            "away_rank": away_rank,
            "rank_diff": away_rank - home_rank,
            "home_recent_goals_for_5": self._mean_deque(home_history.goals_for_5),
            "away_recent_goals_for_5": self._mean_deque(away_history.goals_for_5),
            "home_recent_goals_against_5": self._mean_deque(home_history.goals_against_5),
            "away_recent_goals_against_5": self._mean_deque(away_history.goals_against_5),
            "home_recent_points_5": home_recent_points_5,
            "away_recent_points_5": away_recent_points_5,
            "home_recent_win_pct_10": self._recent_win_percentage(home_history.results_10),
            "away_recent_win_pct_10": self._recent_win_percentage(away_history.results_10),
            "home_recent_goal_diff_10": self._recent_goal_difference(home_history),
            "away_recent_goal_diff_10": self._recent_goal_difference(away_history),
            "home_matches_played": home_history.matches_played,
            "away_matches_played": away_history.matches_played,
            "home_rest_days": home_rest_days,
            "away_rest_days": away_rest_days,
            "neutral_venue": neutral_venue,
            "same_confederation": same_confederation,
            "home_continent_advantage": home_continent_advantage,
            "travel_distance_km": travel_distance_km,
            "avg_goals_scored_diff": self._mean_goals_for(home_history) - self._mean_goals_for(away_history),
            "avg_goals_conceded_diff": self._mean_goals_against(home_history) - self._mean_goals_against(away_history),
            "home_form_index": home_form_index,
            "away_form_index": away_form_index,
            "tournament_experience_diff": home_history.tournament_matches - away_history.tournament_matches,
        }
        return feature_row

    def _update_team_history(self, home_team: str, away_team: str, row: pd.Series, match_date: pd.Timestamp) -> None:
        home_history = self.team_histories[home_team]
        away_history = self.team_histories[away_team]
        home_score = int(row["home_score"])
        away_score = int(row["away_score"])
        self._update_one_history(home_history, home_score, away_score, match_date, row)
        self._update_one_history(away_history, away_score, home_score, match_date, row)

    def _update_one_history(self, history: TeamHistory, scored: int, conceded: int, match_date: pd.Timestamp, row: pd.Series) -> None:
        history.matches_played += 1
        history.total_goals_for += scored
        history.total_goals_against += conceded
        history.last_match_date = match_date
        history.tournament_matches += 1 if str(row.get("tournament", "")).lower() != "friendly" else 0
        if scored > conceded:
            history.total_wins += 1
            history.results_5.append(1)
            history.results_10.append(1)
        elif scored == conceded:
            history.total_draws += 1
            history.results_5.append(0)
            history.results_10.append(0)
        else:
            history.total_losses += 1
            history.results_5.append(-1)
            history.results_10.append(-1)
        history.goals_for_5.append(scored)
        history.goals_against_5.append(conceded)
        history.goals_for_10.append(scored)
        history.goals_against_10.append(conceded)

    def _get_rank(self, team: str) -> float:
        if team in self.rankings.index and "rank" in self.rankings.columns:
            return float(self.rankings.loc[team, "rank"])
        return float(self.rankings["rank"].max() + 25)

    def _get_confederation(self, team: str) -> str:
        if team in self.rankings.index and "confederation" in self.rankings.columns:
            return str(self.rankings.loc[team, "confederation"])
        return "Unknown"

    def _rest_days(self, last_match_date: pd.Timestamp | None, current_date: pd.Timestamp) -> float:
        if last_match_date is None or pd.isna(last_match_date):
            return 14.0
        current_date = pd.Timestamp(current_date)
        last_date = pd.Timestamp(last_match_date)
        if current_date.tzinfo is not None:
            current_date = current_date.tz_localize(None)
        if last_date.tzinfo is not None:
            last_date = last_date.tz_localize(None)
        delta = (current_date - last_date).days
        return float(max(delta, 0))

    def _mean_deque(self, values: deque) -> float:
        return float(np.mean(values)) if values else 0.0

    def _mean_goals_for(self, history: TeamHistory) -> float:
        return safe_divide(history.total_goals_for, history.matches_played, default=1.1)

    def _mean_goals_against(self, history: TeamHistory) -> float:
        return safe_divide(history.total_goals_against, history.matches_played, default=1.1)

    def _recent_win_percentage(self, results: deque) -> float:
        if not results:
            return 0.33
        wins = sum(1 for result in results if result == 1)
        return safe_divide(wins, len(results), default=0.33)

    def _recent_goal_difference(self, history: TeamHistory) -> float:
        if not history.goals_for_10:
            return 0.0
        return float(sum(history.goals_for_10) - sum(history.goals_against_10))

    def _form_index(self, history: TeamHistory) -> float:
        if not history.results_10:
            return 0.0
        weights = np.linspace(0.6, 1.0, len(history.results_10))
        values = np.array([1.0 if result == 1 else 0.5 if result == 0 else 0.0 for result in history.results_10])
        return float(np.average(values, weights=weights))

    def _approximate_travel_distance(self, home_team: str, away_team: str) -> float:
        if home_team == away_team:
            return 0.0
        home_index = abs(hash(home_team)) % 8000
        away_index = abs(hash(away_team)) % 8000
        return float(abs(home_index - away_index) * 0.75)

    def export_state(self) -> dict[str, Any]:
        """Serialize the feature engine state for reuse in prediction."""

        return {
            "rankings": self.rankings.reset_index().to_dict(orient="records"),
            "elo": {
                "initial_rating": self.elo_system.initial_rating,
                "k_factor": self.elo_system.k_factor,
                "home_advantage": self.elo_system.home_advantage,
                "ratings": self.elo_system.ratings,
            },
            "team_histories": {team: history.to_dict() for team, history in self.team_histories.items()},
        }

    @classmethod
    def from_state(cls, payload: dict[str, Any]) -> "FeatureEngineer":
        """Reconstruct a feature engine from serialized state."""

        rankings = pd.DataFrame(payload.get("rankings", []))
        elo_payload = payload.get("elo", {})
        elo_system = EloRatingSystem(
            initial_rating=float(elo_payload.get("initial_rating", 1500.0)),
            k_factor=float(elo_payload.get("k_factor", 24.0)),
            home_advantage=float(elo_payload.get("home_advantage", 60.0)),
            ratings={key: float(value) for key, value in elo_payload.get("ratings", {}).items()},
        )
        feature_engineer = cls(rankings=rankings, elo_system=elo_system)
        feature_engineer.team_histories.update(
            {team: TeamHistory.from_dict(history_payload) for team, history_payload in payload.get("team_histories", {}).items()}
        )
        return feature_engineer
