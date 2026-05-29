"""Monte Carlo World Cup tournament simulator."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.predict import MatchPredictor, load_predictor
from src.utils import LOGGER, SimulationSummary, normalize_probabilities, set_random_seed, weighted_choice


@dataclass
class MatchOutcome:
    """Outcome of one simulated match."""

    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    winner: str | None
    decided_by: str


class WorldCupSimulator:
    """Simulate the FIFA World Cup 2026 using model probabilities."""

    def __init__(
        self,
        predictor: MatchPredictor | None = None,
        teams: list[str] | None = None,
        groups: dict[str, list[str]] | None = None,
        seed: int = 42,
    ) -> None:
        self.predictor = predictor or load_predictor()
        self.rng = set_random_seed(seed)
        actual_groups = self._actual_world_cup_2026_groups()
        if groups is not None:
            self.groups = groups
        elif teams is not None:
            self.groups = self._build_groups(teams)
        else:
            self.groups = actual_groups
        if teams is None:
            team_source = [team for group_teams in self.groups.values() for team in group_teams]
            teams = team_source or self._default_world_cup_teams()
        self.teams = teams

    def simulate_many_tournaments(self, n_simulations: int = 10000) -> SimulationSummary:
        """Run a Monte Carlo sweep and aggregate placement probabilities."""

        champion_counts = defaultdict(int)
        semifinal_counts = defaultdict(int)
        final_counts = defaultdict(int)
        elimination_counts = defaultdict(int)
        finish_positions: dict[str, list[int]] = defaultdict(list)
        for _ in range(n_simulations):
            result = self.simulate_tournament()
            champion_counts[result["champion"]] += 1
            for team in result["semifinalists"]:
                semifinal_counts[team] += 1
            for team in result["finalists"]:
                final_counts[team] += 1
            for team in result["group_eliminated"]:
                elimination_counts[team] += 1
            for team, position in result["finishing_positions"].items():
                finish_positions[team].append(position)
        all_teams = sorted(set(self.teams))
        return SimulationSummary(
            champion_probabilities=self._normalize_counts(champion_counts, n_simulations, all_teams),
            semifinal_probabilities=self._normalize_counts(semifinal_counts, n_simulations, all_teams),
            final_probabilities=self._normalize_counts(final_counts, n_simulations, all_teams),
            group_elimination_probabilities=self._normalize_counts(elimination_counts, n_simulations, all_teams),
            average_finishing_position={team: float(np.mean(positions)) if positions else float(len(all_teams)) for team, positions in finish_positions.items()},
        )

    def simulate_tournament(self) -> dict[str, Any]:
        """Simulate one full tournament from groups to final."""

        group_tables, advancing = self.simulate_group_stage()
        round_of_32 = self._build_round_of_32(advancing)
        round_of_16 = self._simulate_knockout_round(round_of_32)
        quarterfinals = self._simulate_knockout_round(round_of_16)
        semifinals = self._simulate_knockout_round(quarterfinals)
        finalists = [match.winner for match in semifinals]
        third_place = self._simulate_third_place_match(semifinals)
        final = self._simulate_knockout_round([(finalists[0], finalists[1])])[0]
        finishing_positions = self._finishing_positions(group_tables, round_of_32, round_of_16, quarterfinals, semifinals, final, third_place)
        return {
            "group_tables": group_tables,
            "advancing_teams": advancing,
            "round_of_32": round_of_32,
            "round_of_16": round_of_16,
            "quarterfinals": quarterfinals,
            "semifinals": semifinals,
            "finalists": finalists,
            "champion": final.winner,
            "runner_up": final.home_team if final.winner == final.away_team else final.away_team,
            "third_place": third_place.winner,
            "semifinalists": [match.home_team for match in semifinals] + [match.away_team for match in semifinals],
            "group_eliminated": [team for team in self.teams if team not in advancing],
            "finishing_positions": finishing_positions,
        }

    def simulate_group_stage(self) -> tuple[dict[str, pd.DataFrame], list[str]]:
        """Simulate the round-robin group stage and select advancing teams."""

        group_tables: dict[str, pd.DataFrame] = {}
        advancing: list[str] = []
        third_place_candidates: list[dict[str, Any]] = []
        for group_name, teams in self.groups.items():
            matches = self._round_robin_pairs(teams)
            table = self._initial_group_table(teams)
            for home_team, away_team in matches:
                outcome = self.simulate_match(home_team, away_team, knockout=False)
                self._apply_group_result(table, outcome)
            table = self._rank_group_table(table, teams)
            group_tables[group_name] = table
            advancing.extend(table.head(2)["team"].tolist())
            third_place_candidates.append(table.iloc[2].to_dict())
        best_third = self._select_best_third_place(third_place_candidates)
        advancing.extend(best_third)
        return group_tables, advancing

    def simulate_match(self, home_team: str, away_team: str, knockout: bool = False) -> MatchOutcome:
        """Simulate a single match using the trained predictor and goal sampling."""

        prediction = self.predictor.predict_match(home_team, away_team, neutral=True)
        probabilities = prediction.probabilities.as_dict()
        home_goals = int(self.rng.poisson(prediction.expected_home_goals))
        away_goals = int(self.rng.poisson(prediction.expected_away_goals))
        winner: str | None = None
        decided_by = "regular_time"
        if home_goals > away_goals:
            winner = home_team
        elif away_goals > home_goals:
            winner = away_team
        elif knockout:
            winner, decided_by = self._resolve_knockout_tie(home_team, away_team, probabilities)
        return MatchOutcome(home_team, away_team, home_goals, away_goals, winner, decided_by)

    def _resolve_knockout_tie(self, home_team: str, away_team: str, probabilities: dict[str, float]) -> tuple[str, str]:
        """Resolve a knockout draw via extra time and penalties."""

        extra_time_home_xg = max(0.05, probabilities["team_a"] * 0.7 + 0.15)
        extra_time_away_xg = max(0.05, probabilities["team_b"] * 0.7 + 0.15)
        extra_home = int(self.rng.poisson(extra_time_home_xg))
        extra_away = int(self.rng.poisson(extra_time_away_xg))
        if extra_home != extra_away:
            return (home_team if extra_home > extra_away else away_team), "extra_time"
        penalty_probability = probabilities["team_a"] / max(probabilities["team_a"] + probabilities["team_b"], 1e-6)
        winner = home_team if self.rng.random() < penalty_probability else away_team
        return winner, "penalties"

    def _simulate_knockout_round(self, pairings: list[tuple[str, str]]) -> list[MatchOutcome]:
        """Simulate a knockout round and return match outcomes."""

        return [self.simulate_match(home_team, away_team, knockout=True) for home_team, away_team in pairings]

    def _simulate_third_place_match(self, semifinals: list[MatchOutcome]) -> MatchOutcome:
        """Simulate the third-place playoff from the semifinal losers."""

        losers = []
        for match in semifinals:
            if match.winner == match.home_team:
                losers.append(match.away_team)
            else:
                losers.append(match.home_team)
        return self.simulate_match(losers[0], losers[1], knockout=False)

    def _build_round_of_32(self, advancing: list[str]) -> list[tuple[str, str]]:
        """Slot 32 teams into a FIFA-style knockout bracket."""

        ordered = list(advancing)
        self.rng.shuffle(ordered)
        return [(ordered[index], ordered[index + 1]) for index in range(0, len(ordered), 2)]

    def _build_groups(self, teams: list[str]) -> dict[str, list[str]]:
        """Split teams into 12 groups of 4."""

        ordered = list(dict.fromkeys(teams))[:48]
        while len(ordered) < 48:
            ordered.append(f"Team {len(ordered) + 1}")
        groups: dict[str, list[str]] = {}
        for index in range(12):
            groups[f"Group {chr(65 + index)}"] = ordered[index * 4 : (index + 1) * 4]
        return groups

    def _actual_world_cup_2026_groups(self) -> dict[str, list[str]]:
        """Return the official 2026 World Cup groups as drawn in December 2025."""

        return {
            "Group A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
            "Group B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
            "Group C": ["Brazil", "Morocco", "Haiti", "Scotland"],
            "Group D": ["United States", "Paraguay", "Australia", "Turkey"],
            "Group E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
            "Group F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
            "Group G": ["Belgium", "Egypt", "Iran", "New Zealand"],
            "Group H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
            "Group I": ["France", "Senegal", "Iraq", "Norway"],
            "Group J": ["Argentina", "Algeria", "Austria", "Jordan"],
            "Group K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
            "Group L": ["England", "Croatia", "Ghana", "Panama"],
        }

    def _default_world_cup_teams(self) -> list[str]:
        """Use the strongest teams from the predictor state when no custom list is provided."""

        rankings = self.predictor.feature_engineer.rankings.reset_index().sort_values("rank")
        teams = rankings["team"].tolist()
        if len(teams) >= 48:
            return teams[:48]
        padding = [f"Team {index}" for index in range(len(teams) + 1, 49)]
        return teams + padding

    def _round_robin_pairs(self, teams: list[str]) -> list[tuple[str, str]]:
        """Return the six group matches for four teams."""

        a, b, c, d = teams
        return [(a, b), (c, d), (a, c), (b, d), (a, d), (b, c)]

    def _initial_group_table(self, teams: list[str]) -> pd.DataFrame:
        """Create an empty standings table for a group."""

        return pd.DataFrame(
            {
                "team": teams,
                "played": 0,
                "wins": 0,
                "draws": 0,
                "losses": 0,
                "goals_for": 0,
                "goals_against": 0,
                "goal_difference": 0,
                "points": 0,
            }
        )

    def _apply_group_result(self, table: pd.DataFrame, outcome: MatchOutcome) -> None:
        """Update a group table from one match result."""

        home_idx = table.index[table["team"] == outcome.home_team][0]
        away_idx = table.index[table["team"] == outcome.away_team][0]
        table.loc[home_idx, "played"] += 1
        table.loc[away_idx, "played"] += 1
        table.loc[home_idx, "goals_for"] += outcome.home_goals
        table.loc[home_idx, "goals_against"] += outcome.away_goals
        table.loc[away_idx, "goals_for"] += outcome.away_goals
        table.loc[away_idx, "goals_against"] += outcome.home_goals
        if outcome.home_goals > outcome.away_goals:
            table.loc[home_idx, ["wins", "points"]] += [1, 3]
            table.loc[away_idx, "losses"] += 1
        elif outcome.home_goals < outcome.away_goals:
            table.loc[away_idx, ["wins", "points"]] += [1, 3]
            table.loc[home_idx, "losses"] += 1
        else:
            table.loc[[home_idx, away_idx], ["draws", "points"]] += [1, 1]
        table["goal_difference"] = table["goals_for"] - table["goals_against"]

    def _rank_group_table(self, table: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
        """Apply FIFA-style standings sorting."""

        return table.sort_values(["points", "goal_difference", "goals_for", "team"], ascending=[False, False, False, True]).reset_index(drop=True)

    def _select_best_third_place(self, candidates: list[dict[str, Any]]) -> list[str]:
        """Pick the best third-place teams across all groups."""

        third_place_frame = pd.DataFrame(candidates)
        if third_place_frame.empty:
            return []
        ranked = third_place_frame.sort_values(["points", "goal_difference", "goals_for", "team"], ascending=[False, False, False, True])
        return ranked.head(8)["team"].tolist()

    def _finishing_positions(
        self,
        group_tables: dict[str, pd.DataFrame],
        round_of_32: list[tuple[str, str]],
        round_of_16: list[MatchOutcome],
        quarterfinals: list[MatchOutcome],
        semifinals: list[MatchOutcome],
        final: MatchOutcome,
        third_place: MatchOutcome,
    ) -> dict[str, int]:
        """Assign approximate finishing positions for simulation summaries."""

        positions: dict[str, int] = {}
        positions[final.winner] = 1
        positions[final.home_team if final.winner == final.away_team else final.away_team] = 2
        positions[third_place.winner] = 3
        third_place_loser = third_place.home_team if third_place.winner == third_place.away_team else third_place.away_team
        positions[third_place_loser] = 4
        for match in semifinals:
            loser = match.away_team if match.winner == match.home_team else match.home_team
            positions.setdefault(loser, 4)
        for match in quarterfinals:
            loser = match.away_team if match.winner == match.home_team else match.home_team
            positions.setdefault(loser, 8)
        for match in round_of_16:
            loser = match.away_team if match.winner == match.home_team else match.home_team
            positions.setdefault(loser, 16)
        for group_table in group_tables.values():
            for _, row in group_table.iterrows():
                positions.setdefault(str(row["team"]), 32 if row.name >= 2 else 16)
        return positions

    def _normalize_counts(self, counts: dict[str, int], total: int, teams: list[str]) -> dict[str, float]:
        return {team: counts.get(team, 0) / total for team in teams}


def simulate_world_cup(
    n_simulations: int = 10000,
    predictor: MatchPredictor | None = None,
    teams: list[str] | None = None,
    groups: dict[str, list[str]] | None = None,
    seed: int = 42,
) -> SimulationSummary:
    """Convenience wrapper used by the app and tests."""

    simulator = WorldCupSimulator(predictor=predictor, teams=teams, groups=groups, seed=seed)
    return simulator.simulate_many_tournaments(n_simulations=n_simulations)
