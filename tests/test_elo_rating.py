"""Tests for the Elo rating system."""

from src.elo_rating import EloRatingSystem, calculate_expected_score, update_elo


def test_expected_score_is_symmetric() -> None:
    expected_a = calculate_expected_score(1600, 1500)
    expected_b = calculate_expected_score(1500, 1600)
    assert round(expected_a + expected_b, 6) == 1.0


def test_update_elo_moves_winner_up() -> None:
    home, away = update_elo(1500, 1500, 2, 0)
    assert home > 1500
    assert away < 1500


def test_rating_system_updates_and_persists_round_trip(tmp_path) -> None:
    system = EloRatingSystem()
    system.update_elo("France", "Japan", 3, 1)
    path = tmp_path / "elo.json"
    system.save(path)
    loaded = EloRatingSystem.load(path)
    assert loaded.get_team_rating("France") == system.get_team_rating("France")
