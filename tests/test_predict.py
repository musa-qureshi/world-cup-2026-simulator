"""Tests for the match prediction layer."""

from src.predict import MatchPredictor


def test_predictor_returns_probabilities() -> None:
    predictor = MatchPredictor()
    prediction = predictor.predict_match("France", "Japan")
    probs = prediction.probabilities.as_dict()
    assert abs(sum(probs.values()) - 1.0) < 1e-6
    assert set(probs.keys()) == {"team_a", "draw", "team_b"}


def test_predictor_updates_live_state() -> None:
    predictor = MatchPredictor()
    before = predictor.feature_engineer.elo_system.get_team_rating("France")
    predictor.update_after_match("France", "Japan", 2, 0)
    after = predictor.feature_engineer.elo_system.get_team_rating("France")
    assert after != before
