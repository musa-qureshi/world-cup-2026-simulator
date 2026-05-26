"""Load, clean, and prepare football results data for model training."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.utils import LOGGER, RAW_DIR, ensure_datetime_column, ensure_directory, safe_divide


EXPECTED_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "city",
    "country",
    "neutral",
]


def _require_real_data() -> bool:
    return os.getenv("WORLD_CUP_REQUIRE_REAL_DATA", "").strip().lower() in {"1", "true", "yes"}


def load_match_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load a results dataset from CSV or fall back to a synthetic sample."""

    if path is not None:
        candidate = Path(path)
        if candidate.exists():
            return _standardize_match_frame(pd.read_csv(candidate))

    for candidate in [RAW_DIR / "international_results.csv", RAW_DIR / "matches.csv"]:
        if candidate.exists():
            return _standardize_match_frame(pd.read_csv(candidate))

    if _require_real_data():
        raise FileNotFoundError("No real match dataset found on disk and WORLD_CUP_REQUIRE_REAL_DATA is enabled.")

    LOGGER.warning("No match dataset found on disk. Generating a synthetic training sample.")
    return generate_synthetic_results()


def load_fifa_rankings(path: str | Path | None = None) -> pd.DataFrame:
    """Load FIFA rankings or construct a reasonable fallback table."""

    if path is not None:
        candidate = Path(path)
        if candidate.exists():
            return _standardize_rankings_frame(pd.read_csv(candidate))

    for candidate in [RAW_DIR / "fifa_rankings.csv", RAW_DIR / "rankings.csv"]:
        if candidate.exists():
            return _standardize_rankings_frame(pd.read_csv(candidate))

    if _require_real_data():
        raise FileNotFoundError("No real rankings dataset found on disk and WORLD_CUP_REQUIRE_REAL_DATA is enabled.")

    return generate_synthetic_rankings()


def _standardize_match_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    column_map = {
        "home_team": "home_team",
        "away_team": "away_team",
        "home_score": "home_score",
        "away_score": "away_score",
        "tournament": "tournament",
        "city": "city",
        "country": "country",
        "neutral": "neutral",
    }
    date_candidates = ["date", "match_date", "year"]
    for source, target in column_map.items():
        if source in result.columns and source != target:
            result = result.rename(columns={source: target})
    for candidate in date_candidates:
        if candidate in result.columns:
            result = result.rename(columns={candidate: "date"})
            break
    required = set(EXPECTED_COLUMNS)
    missing = [column for column in required if column not in result.columns]
    if missing:
        raise ValueError(f"Match dataset is missing required columns: {missing}")
    result = ensure_datetime_column(result, "date")
    result["neutral"] = result["neutral"].fillna(True).astype(bool)
    result["home_score"] = pd.to_numeric(result["home_score"], errors="coerce").fillna(0).astype(int)
    result["away_score"] = pd.to_numeric(result["away_score"], errors="coerce").fillna(0).astype(int)
    result = result.sort_values("date").reset_index(drop=True)
    return result[EXPECTED_COLUMNS]


def _standardize_rankings_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    rename_map = {
        "team": "team",
        "country": "team",
        "rank": "rank",
        "fifa_rank": "rank",
        "rating": "rating",
        "points": "rating",
        "confederation": "confederation",
        "continent": "confederation",
    }
    normalized = {}
    for column in result.columns:
        if column in rename_map:
            normalized[column] = rename_map[column]
    result = result.rename(columns=normalized)
    if "team" not in result.columns:
        raise ValueError("Rankings dataset must contain a team column.")
    if "rank" not in result.columns:
        result["rank"] = np.arange(1, len(result) + 1)
    if "rating" not in result.columns:
        result["rating"] = 1500 + (len(result) - result["rank"].rank(method="first")) * 6
    if "confederation" not in result.columns:
        result["confederation"] = "Unknown"
    return result[["team", "rank", "rating", "confederation"]].drop_duplicates("team")


def generate_synthetic_results(n_matches: int = 1500, seed: int = 42) -> pd.DataFrame:
    """Build a realistic synthetic dataset when historical CSVs are unavailable."""

    rng = np.random.default_rng(seed)
    teams = _synthetic_team_names()
    tournaments = ["Friendly", "World Cup", "Qualifiers", "Continental Cup"]
    dates = pd.date_range("2000-01-01", periods=n_matches, freq="D")
    rows: list[dict[str, object]] = []
    team_strength = {team: rng.normal(0.0, 1.0) for team in teams}
    for date in dates:
        home_team, away_team = rng.choice(teams, size=2, replace=False)
        strength_gap = team_strength[home_team] - team_strength[away_team]
        home_goal_expectation = np.clip(1.35 + 0.45 * strength_gap + rng.normal(0, 0.35), 0.1, 4.2)
        away_goal_expectation = np.clip(1.05 - 0.35 * strength_gap + rng.normal(0, 0.3), 0.05, 3.8)
        home_score = int(rng.poisson(home_goal_expectation))
        away_score = int(rng.poisson(away_goal_expectation))
        rows.append(
            {
                "date": date,
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
                "tournament": rng.choice(tournaments, p=[0.36, 0.18, 0.26, 0.20]),
                "city": "Synthetic City",
                "country": "Synthetic Country",
                "neutral": True,
            }
        )
        if home_score > away_score:
            team_strength[home_team] += 0.01
            team_strength[away_team] -= 0.005
        elif away_score > home_score:
            team_strength[away_team] += 0.01
            team_strength[home_team] -= 0.005
    return _standardize_match_frame(pd.DataFrame(rows))


def generate_synthetic_rankings() -> pd.DataFrame:
    """Construct a synthetic FIFA ranking table for demos and tests."""

    teams = _synthetic_team_names()
    rng = np.random.default_rng(42)
    ratings = np.linspace(1850, 1250, len(teams)) + rng.normal(0, 25, len(teams))
    confederations = ["UEFA", "CONMEBOL", "AFC", "CAF", "CONCACAF", "OFC"]
    rows = []
    for index, team in enumerate(teams, start=1):
        rows.append(
            {
                "team": team,
                "rank": index,
                "rating": float(ratings[index - 1]),
                "confederation": confederations[(index - 1) % len(confederations)],
            }
        )
    return pd.DataFrame(rows)


def merge_match_and_ranking_data(matches: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    """Enrich match history with pre-match FIFA ranking information."""

    matches = matches.copy()
    rankings = rankings.copy()
    ranking_map = rankings.set_index("team")
    matches["home_rank"] = matches["home_team"].map(ranking_map["rank"]).fillna(rankings["rank"].max() + 20)
    matches["away_rank"] = matches["away_team"].map(ranking_map["rank"]).fillna(rankings["rank"].max() + 20)
    matches["home_rating"] = matches["home_team"].map(ranking_map["rating"]).fillna(rankings["rating"].mean())
    matches["away_rating"] = matches["away_team"].map(ranking_map["rating"]).fillna(rankings["rating"].mean())
    matches["home_confederation"] = matches["home_team"].map(ranking_map["confederation"]).fillna("Unknown")
    matches["away_confederation"] = matches["away_team"].map(ranking_map["confederation"]).fillna("Unknown")
    return matches


def prepare_training_frame(matches: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    """Return a clean frame with ranking features ready for engineering."""

    merged = merge_match_and_ranking_data(matches, rankings)
    merged["goal_difference"] = merged["home_score"] - merged["away_score"]
    merged["total_goals"] = merged["home_score"] + merged["away_score"]
    merged["home_win"] = (merged["home_score"] > merged["away_score"]).astype(int)
    merged["draw"] = (merged["home_score"] == merged["away_score"]).astype(int)
    merged["away_win"] = (merged["away_score"] > merged["home_score"]).astype(int)
    return merged


def _synthetic_team_names() -> list[str]:
    return [
        "Argentina",
        "Australia",
        "Belgium",
        "Brazil",
        "Canada",
        "Croatia",
        "Denmark",
        "England",
        "France",
        "Germany",
        "Italy",
        "Japan",
        "Mexico",
        "Morocco",
        "Netherlands",
        "Portugal",
        "Senegal",
        "South Korea",
        "Spain",
        "United States",
        "Uruguay",
        "Colombia",
        "Chile",
        "Ecuador",
        "Nigeria",
        "Cameroon",
        "Algeria",
        "Saudi Arabia",
        "Iran",
        "Qatar",
        "Poland",
        "Switzerland",
        "Austria",
        "Turkey",
        "Ukraine",
        "Serbia",
        "Japan B",
        "Egypt",
        "Ghana",
        "Peru",
        "Paraguay",
        "Wales",
        "Scotland",
        "Tunisia",
        "New Zealand",
        "Costa Rica",
        "Honduras",
        "Panama",
    ]
