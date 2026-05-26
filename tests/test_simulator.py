"""Tests for the Monte Carlo tournament simulator."""

from src.predict import MatchPredictor
from src.simulator import WorldCupSimulator


def test_group_stage_produces_standings() -> None:
    simulator = WorldCupSimulator(MatchPredictor())
    group_tables, advancing = simulator.simulate_group_stage()
    assert len(group_tables) == 12
    assert len(advancing) == 32


def test_single_match_has_a_winner_or_draw() -> None:
    simulator = WorldCupSimulator(MatchPredictor())
    outcome = simulator.simulate_match("France", "Japan", knockout=False)
    assert outcome.home_goals >= 0
    assert outcome.away_goals >= 0
