"""Shared utilities used across the simulator, training, and app layers."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


LOGGER_NAME = "world_cup_simulator"


def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    """Return a configured application logger."""

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


LOGGER = get_logger()


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"
MODELS_DIR = ROOT_DIR / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "match_prediction_model.joblib"
DEFAULT_METADATA_PATH = MODELS_DIR / "match_prediction_metadata.json"


def ensure_directory(path: Path) -> Path:
    """Create a directory if needed and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def set_random_seed(seed: int = 42) -> np.random.Generator:
    """Seed NumPy and return a reusable generator."""

    np.random.seed(seed)
    return np.random.default_rng(seed)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers."""

    if denominator == 0:
        return default
    return float(numerator) / float(denominator)


def sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""

    return float(1.0 / (1.0 + np.exp(-np.clip(x, -60, 60))))


def softmax(values: Sequence[float]) -> np.ndarray:
    """Return a stable softmax distribution."""

    array = np.asarray(values, dtype=float)
    shifted = array - np.max(array)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum()


def normalize_probabilities(probabilities: Sequence[float]) -> np.ndarray:
    """Normalize a probability vector so it sums to 1."""

    array = np.asarray(probabilities, dtype=float)
    total = array.sum()
    if total <= 0:
        return np.full_like(array, 1.0 / len(array))
    return array / total


def read_json(path: Path, default: Any | None = None) -> Any:
    """Read JSON from disk if the file exists."""

    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    """Write a JSON payload to disk."""

    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def ensure_datetime_column(frame: pd.DataFrame, column: str = "date") -> pd.DataFrame:
    """Return a copy with the target column parsed to datetime."""

    result = frame.copy()
    if column in result.columns:
        result[column] = pd.to_datetime(result[column], errors="coerce")
    return result


def flatten_dict(prefix: str, data: dict[str, Any]) -> dict[str, Any]:
    """Flatten a nested dictionary using a prefix."""

    return {f"{prefix}_{key}": value for key, value in data.items()}


@dataclass
class MatchProbabilities:
    """Probabilities for a match outcome from the perspective of team A."""

    team_a: float
    draw: float
    team_b: float

    def normalized(self) -> "MatchProbabilities":
        values = normalize_probabilities([self.team_a, self.draw, self.team_b])
        return MatchProbabilities(*map(float, values))

    def as_dict(self) -> dict[str, float]:
        return asdict(self.normalized())


@dataclass
class SimulationSummary:
    """Aggregated output from many tournament simulations."""

    champion_probabilities: dict[str, float]
    semifinal_probabilities: dict[str, float]
    final_probabilities: dict[str, float]
    group_elimination_probabilities: dict[str, float]
    average_finishing_position: dict[str, float]


def weighted_choice(options: Sequence[Any], probabilities: Sequence[float], rng: np.random.Generator | None = None) -> Any:
    """Sample a single item from a discrete distribution."""

    generator = rng or np.random.default_rng()
    normalized = normalize_probabilities(probabilities)
    index = generator.choice(len(options), p=normalized)
    return options[index]


def chunked(iterable: Sequence[Any], size: int) -> Iterable[list[Any]]:
    """Yield chunks from a sequence."""

    for start in range(0, len(iterable), size):
        yield list(iterable[start : start + size])
