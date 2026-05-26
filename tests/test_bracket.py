"""Tests for bracket creation utilities."""

from src.bracket import bracket_to_dataframe, build_bracket


def test_build_bracket_returns_nodes() -> None:
    bracket = build_bracket({"Round 1": [{"home_team": "France", "away_team": "Japan", "winner": "France"}]})
    assert len(bracket) == 1
    assert bracket[0].team_a == "France"


def test_bracket_to_dataframe_contains_expected_columns() -> None:
    bracket = build_bracket({"Round 1": [{"home_team": "France", "away_team": "Japan", "winner": "France"}]})
    frame = bracket_to_dataframe(bracket)
    assert set(["round", "team_a", "team_b", "winner", "probability_team_a", "probability_team_b"]).issubset(frame.columns)
