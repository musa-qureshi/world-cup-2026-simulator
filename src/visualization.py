"""Interactive visualizations for the Streamlit dashboard and reports."""

from __future__ import annotations

from typing import Any

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def create_probability_bar_chart(probabilities: dict[str, float], title: str = "Match Probabilities") -> go.Figure:
    """Return a bar chart for outcome probabilities."""

    frame = pd.DataFrame({"outcome": list(probabilities.keys()), "probability": list(probabilities.values())})
    figure = px.bar(frame, x="outcome", y="probability", text="probability", color="outcome", color_discrete_sequence=px.colors.sequential.Tealgrn)
    figure.update_traces(texttemplate="%{text:.1%}", textposition="outside")
    figure.update_layout(title=title, yaxis=dict(tickformat=".0%"), xaxis_title="Outcome", yaxis_title="Probability", template="plotly_white")
    return figure


def create_team_strength_radar(team_metrics: dict[str, float], title: str = "Team Strength Radar") -> go.Figure:
    """Create a radar chart for team strength attributes."""

    categories = list(team_metrics.keys())
    values = list(team_metrics.values())
    figure = go.Figure()
    figure.add_trace(go.Scatterpolar(r=values + [values[0]], theta=categories + [categories[0]], fill="toself", name="Strength"))
    figure.update_layout(title=title, polar=dict(radialaxis=dict(visible=True, range=[0, max(1.0, max(values) * 1.1)])), template="plotly_white")
    return figure


def create_probability_heatmap(probability_table: pd.DataFrame, title: str = "Tournament Probability Heatmap") -> go.Figure:
    """Visualize team tournament probabilities as a heatmap."""

    figure = px.imshow(probability_table, aspect="auto", color_continuous_scale="Tealgrn", title=title)
    figure.update_layout(template="plotly_white")
    return figure


def create_elo_progression_chart(elo_history: pd.DataFrame, title: str = "Elo Progression") -> go.Figure:
    """Plot Elo ratings over time for one or more teams."""

    figure = px.line(elo_history, x="date", y="rating", color="team", title=title)
    figure.update_layout(template="plotly_white")
    return figure


def create_confusion_matrix_chart(matrix: list[list[int]], labels: list[str], title: str = "Confusion Matrix") -> go.Figure:
    """Render a confusion matrix heatmap."""

    figure = px.imshow(matrix, x=labels, y=labels, text_auto=True, color_continuous_scale="Blues", title=title)
    figure.update_layout(template="plotly_white", xaxis_title="Predicted", yaxis_title="Actual")
    return figure


def create_feature_importance_chart(feature_importance: pd.DataFrame, title: str = "Feature Importance") -> go.Figure:
    """Show model feature importance values."""

    figure = px.bar(feature_importance.sort_values("importance"), x="importance", y="feature", orientation="h", title=title)
    figure.update_layout(template="plotly_white")
    return figure
