"""Create interactive bracket structures and charts for the tournament."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import plotly.graph_objects as go


@dataclass
class BracketNode:
    """One node in the tournament bracket."""

    round_name: str
    team_a: str
    team_b: str
    winner: str | None = None
    probability_team_a: float | None = None
    probability_team_b: float | None = None


def build_bracket(rounds: dict[str, list[dict[str, Any]]]) -> list[BracketNode]:
    """Normalize round data into a flat bracket representation."""

    bracket: list[BracketNode] = []
    for round_name, matches in rounds.items():
        for match in matches:
            bracket.append(
                BracketNode(
                    round_name=round_name,
                    team_a=str(match.get("team_a", match.get("home_team", "TBD"))),
                    team_b=str(match.get("team_b", match.get("away_team", "TBD"))),
                    winner=match.get("winner"),
                    probability_team_a=float(match.get("probability_team_a", match.get("home_probability", 0.0))),
                    probability_team_b=float(match.get("probability_team_b", match.get("away_probability", 0.0))),
                )
            )
    return bracket


def create_bracket_figure(bracket: list[BracketNode]) -> go.Figure:
    """Render a simple interactive bracket visualization in Plotly."""

    rounds = []
    for node in bracket:
        if node.round_name not in rounds:
            rounds.append(node.round_name)
    figure = go.Figure()
    for round_index, round_name in enumerate(rounds):
        round_nodes = [node for node in bracket if node.round_name == round_name]
        y_positions = list(range(len(round_nodes)))[::-1]
        for position, node in zip(y_positions, round_nodes):
            figure.add_trace(
                go.Scatter(
                    x=[round_index, round_index],
                    y=[position, position],
                    mode="markers+text",
                    marker=dict(size=12),
                    text=[f"{node.team_a} vs {node.team_b}"],
                    textposition="middle right",
                    name=round_name,
                    showlegend=False,
                    hovertemplate=f"<b>{round_name}</b><br>{node.team_a} vs {node.team_b}<br>Winner: {node.winner or 'TBD'}<extra></extra>",
                )
            )
    figure.update_layout(
        title="Tournament Bracket",
        xaxis=dict(title="Round", tickmode="array", tickvals=list(range(len(rounds))), ticktext=rounds, showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        template="plotly_white",
        height=max(500, 120 * max(1, len(bracket))),
    )
    return figure


def bracket_to_dataframe(bracket: list[BracketNode]):
    """Convert bracket nodes into a tabular structure for display."""

    import pandas as pd

    return pd.DataFrame(
        [
            {
                "round": node.round_name,
                "team_a": node.team_a,
                "team_b": node.team_b,
                "winner": node.winner,
                "probability_team_a": node.probability_team_a,
                "probability_team_b": node.probability_team_b,
            }
            for node in bracket
        ]
    )
