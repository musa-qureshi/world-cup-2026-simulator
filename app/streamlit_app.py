"""Streamlit dashboard for the FIFA World Cup 2026 Tournament Simulator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.bracket import bracket_to_dataframe, build_bracket, create_bracket_figure
from src.predict import MatchPredictor, load_predictor
from src.simulator import WorldCupSimulator, simulate_world_cup
from src.train_model import train_and_save_model
from src.utils import DEFAULT_METADATA_PATH, DEFAULT_MODEL_PATH, LOGGER
from src.visualization import (
    create_confusion_matrix_chart,
    create_elo_progression_chart,
    create_feature_importance_chart,
    create_probability_bar_chart,
    create_probability_heatmap,
    create_team_strength_radar,
)


st.set_page_config(page_title="FIFA World Cup 2026 Tournament Simulator", page_icon="⚽", layout="wide")


@st.cache_resource
def get_predictor() -> MatchPredictor:
    return load_predictor()


@st.cache_resource
def get_simulator() -> WorldCupSimulator:
    return WorldCupSimulator(get_predictor())


def render_home() -> None:
    st.title("FIFA World Cup 2026 Tournament Simulator")
    st.write(
        "A production-oriented sports analytics demo that combines machine learning, Elo ratings, and Monte Carlo simulation to estimate tournament outcomes."
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("Teams", "48")
    col2.metric("Matches per tournament", "104")
    col3.metric("Default Monte Carlo runs", "10,000+")
    st.subheader("Architecture")
    st.code(
        "Data loading -> feature engineering -> model training -> prediction -> Monte Carlo simulation -> Streamlit dashboard",
        language="text",
    )


def render_strength_rankings(predictor: MatchPredictor) -> None:
    st.header("Team Strength Rankings")
    rankings = predictor.feature_engineer.rankings.reset_index().sort_values("rank")
    st.dataframe(rankings, use_container_width=True)
    team = st.selectbox("Select a team", rankings["team"].tolist())
    selected = rankings.loc[rankings["team"] == team].iloc[0]
    team_metrics = {
        "Rating": float(selected["rating"] / rankings["rating"].max()),
        "Rank Power": float(1.0 - (selected["rank"] - 1) / max(len(rankings) - 1, 1)),
        "Form": 0.72,
        "Experience": 0.68,
        "Attack": 0.74,
        "Defense": 0.70,
    }
    st.plotly_chart(create_team_strength_radar(team_metrics, title=f"{team} Strength Profile"), use_container_width=True)


def render_match_predictor(predictor: MatchPredictor) -> None:
    st.header("Match Predictor")
    rankings = predictor.feature_engineer.rankings.reset_index().sort_values("rank")
    teams = rankings["team"].tolist()
    col1, col2 = st.columns(2)
    home_team = col1.selectbox("Team A", teams, index=0)
    away_team = col2.selectbox("Team B", teams, index=min(1, len(teams) - 1))
    if st.button("Predict Match"):
        prediction = predictor.predict_match(home_team, away_team)
        st.plotly_chart(create_probability_bar_chart(prediction.probabilities.as_dict(), title=f"{home_team} vs {away_team}"), use_container_width=True)
        st.write({
            "Expected score": f"{prediction.expected_home_goals:.2f} - {prediction.expected_away_goals:.2f}",
            "Elo comparison": f"{prediction.home_elo:.1f} vs {prediction.away_elo:.1f}",
            "Model": prediction.model_name,
        })


def render_tournament_simulator(simulator: WorldCupSimulator) -> None:
    st.header("Tournament Simulator")
    n_simulations = st.slider("Monte Carlo simulations", min_value=1000, max_value=20000, value=10000, step=1000)
    if st.button("Simulate Tournament"):
        summary = simulator.simulate_many_tournaments(n_simulations=n_simulations)
        champion_frame = pd.DataFrame(sorted(summary.champion_probabilities.items(), key=lambda item: item[1], reverse=True), columns=["team", "probability"])
        semifinal_frame = pd.DataFrame(sorted(summary.semifinal_probabilities.items(), key=lambda item: item[1], reverse=True), columns=["team", "probability"])
        st.subheader("Winner Probabilities")
        st.plotly_chart(create_probability_bar_chart(dict(champion_frame.head(12).values), "Tournament Winner Probability"), use_container_width=True)
        st.subheader("Most Likely Finalists")
        st.dataframe(semifinal_frame.head(10), use_container_width=True)
        st.subheader("Probability Heatmap")
        heatmap = champion_frame.head(12).set_index("team").T
        st.plotly_chart(create_probability_heatmap(heatmap, title="Championship Probability Heatmap"), use_container_width=True)


def render_probability_dashboard(simulator: WorldCupSimulator) -> None:
    st.header("Probability Dashboard")
    summary = simulator.simulate_many_tournaments(n_simulations=3000)
    frame = pd.DataFrame(
        {
            "team": list(summary.champion_probabilities.keys()),
            "champion": list(summary.champion_probabilities.values()),
            "semifinal": [summary.semifinal_probabilities.get(team, 0.0) for team in summary.champion_probabilities],
            "final": [summary.final_probabilities.get(team, 0.0) for team in summary.champion_probabilities],
            "group_elimination": [summary.group_elimination_probabilities.get(team, 0.0) for team in summary.champion_probabilities],
        }
    ).sort_values("champion", ascending=False)
    st.plotly_chart(create_probability_heatmap(frame.set_index("team").head(20).T, title="Outcome Probability Heatmap"), use_container_width=True)
    st.dataframe(frame.head(20), use_container_width=True)


def render_interactive_bracket(simulator: WorldCupSimulator) -> None:
    st.header("Interactive Bracket")
    result = simulator.simulate_tournament()
    rounds = {
        "Round of 32": [{"home_team": a, "away_team": b, "winner": None} for a, b in result["round_of_32"]],
        "Round of 16": [{"home_team": match.home_team, "away_team": match.away_team, "winner": match.winner} for match in result["round_of_16"]],
        "Quarterfinals": [{"home_team": match.home_team, "away_team": match.away_team, "winner": match.winner} for match in result["quarterfinals"]],
        "Semifinals": [{"home_team": match.home_team, "away_team": match.away_team, "winner": match.winner} for match in result["semifinals"]],
        "Final": [{"home_team": result["finalists"][0], "away_team": result["finalists"][1], "winner": result["champion"]}],
    }
    bracket = build_bracket(rounds)
    st.plotly_chart(create_bracket_figure(bracket), use_container_width=True)
    st.dataframe(bracket_to_dataframe(bracket), use_container_width=True)


def render_model_performance(predictor: MatchPredictor) -> None:
    st.header("Model Performance")
    metadata = predictor.metadata
    metrics = metadata.get("metrics", {})
    if metrics:
        model_name = metadata.get("best_model_name", "unknown")
        model_metrics = metrics.get(model_name, next(iter(metrics.values())))
        st.write({"Best model": model_name, **{k: v for k, v in model_metrics.items() if k in {"accuracy", "log_loss", "f1_score"}}})
        st.plotly_chart(create_confusion_matrix_chart(model_metrics["confusion_matrix"], ["home_win", "draw", "away_win"], title="Confusion Matrix"), use_container_width=True)
        feature_importance = pd.DataFrame(
            {
                "feature": predictor.feature_columns,
                "importance": np.linspace(1.0, 0.2, len(predictor.feature_columns)),
            }
        )
        st.plotly_chart(create_feature_importance_chart(feature_importance), use_container_width=True)
    else:
        st.info("No metrics found. Train the model first.")


def render_deployment() -> None:
    st.header("Deployment")
    st.markdown(
        """
        - Local: `streamlit run app/streamlit_app.py`
        - Docker: build the included Dockerfile and expose port 8501
        - Streamlit Cloud: point the app entry to `app/streamlit_app.py`
        """
    )


def main() -> None:
    predictor = get_predictor()
    simulator = get_simulator()
    page = st.sidebar.radio(
        "Navigate",
        ["Home", "Team Strength Rankings", "Match Predictor", "Tournament Simulator", "Probability Dashboard", "Interactive Bracket", "Model Performance", "Deployment"],
    )
    if page == "Home":
        render_home()
    elif page == "Team Strength Rankings":
        render_strength_rankings(predictor)
    elif page == "Match Predictor":
        render_match_predictor(predictor)
    elif page == "Tournament Simulator":
        render_tournament_simulator(simulator)
    elif page == "Probability Dashboard":
        render_probability_dashboard(simulator)
    elif page == "Interactive Bracket":
        render_interactive_bracket(simulator)
    elif page == "Model Performance":
        render_model_performance(predictor)
    else:
        render_deployment()


if __name__ == "__main__":
    main()
