"""Small, interpretable supervised models for System 2 (numpy only, no extra dependency).

  Ridge regression     predicts the next-session return (%) from a FEW standardised features.
  L2 logistic          predicts P(next-session return > 0) for the direction view.

Both are fitted by System 3 on a chronologically earlier TRAINING window only (see
system3_backtesting.experiments) and stored inside a parameter set (`system2.linear`). They are
never activated automatically. Missing feature values are imputed with the training mean (i.e. 0
after standardisation), and the imputation is counted in the reports.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class LinearModel:
    kind: str                    # "ridge" | "logistic"
    features: list[str]
    coef: list[float]            # per standardised feature
    intercept: float
    mean: list[float]
    std: list[float]
    alpha: float
    n_train: int
    train_period: tuple[str, str] = ("", "")

    # ---- inference -------------------------------------------------------------------------------
    def standardise(self, row: dict) -> np.ndarray:
        x = np.array([np.nan if row.get(f) is None else float(row[f]) for f in self.features])
        z = (x - np.array(self.mean)) / np.array(self.std)
        return np.where(np.isnan(z), 0.0, z)

    def decision(self, row: dict) -> float:
        return float(self.intercept + self.standardise(row) @ np.array(self.coef))

    def predict(self, row: dict) -> float:
        d = self.decision(row)
        return 1 / (1 + math.exp(-d)) if self.kind == "logistic" else d

    def contributions(self, row: dict) -> dict[str, float]:
        z = self.standardise(row)
        return {f: round(float(c * v), 5) for f, c, v in zip(self.features, self.coef, z)}

    # ---- (de)serialisation -----------------------------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        d["train_period"] = list(self.train_period)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "LinearModel":
        return cls(**{**d, "train_period": tuple(d.get("train_period", ("", "")))})


def design(rows: list[dict], features: list[str]) -> np.ndarray:
    return np.array([[np.nan if r.get(f) is None else float(r[f]) for f in features] for r in rows],
                    dtype=float).reshape(len(rows), len(features))


def _standardise_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A feature that is missing in every training row gets mean 0 / std 1, i.e. contributes 0."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN columns are handled just below
        mean = np.nanmean(X, axis=0) if X.size else np.zeros(X.shape[1])
        std = np.nanstd(X, axis=0) if X.size else np.ones(X.shape[1])
    mean = np.where(np.isnan(mean), 0.0, mean)
    std = np.where(~np.isfinite(std) | (std < 1e-12), 1.0, std)
    Z = (X - mean) / std
    return np.where(np.isnan(Z), 0.0, Z), mean, std


def fit_ridge(rows: list[dict], y: list[float], features: list[str], alpha: float,
              train_period: tuple[str, str] = ("", "")) -> LinearModel:
    """Closed-form ridge on standardised features: minimise ||y - b0 - Zb||^2 + alpha * n * ||b||^2.
    The intercept is not penalised. With no features this is the training-mean forecast."""
    yv = np.asarray(y, dtype=float)
    n = len(yv)
    if not features:
        return LinearModel("ridge", [], [], float(yv.mean()), [], [], alpha, n, train_period)
    Z, mean, std = _standardise_fit(design(rows, features))
    yc = yv - yv.mean()
    k = Z.shape[1]
    coef = np.linalg.solve(Z.T @ Z + alpha * n * np.eye(k), Z.T @ yc)
    return LinearModel("ridge", list(features), coef.round(8).tolist(), float(yv.mean()),
                       mean.tolist(), std.tolist(), alpha, n, train_period)


def fit_logistic(rows: list[dict], y_up: list[int], features: list[str], alpha: float,
                 train_period: tuple[str, str] = ("", ""), iters: int = 50) -> LinearModel:
    """L2-penalised logistic regression by Newton-Raphson (IRLS). y_up in {0, 1}."""
    yv = np.asarray(y_up, dtype=float)
    n = len(yv)
    if not features:
        p = min(max(yv.mean(), 1e-6), 1 - 1e-6)
        return LinearModel("logistic", [], [], float(math.log(p / (1 - p))), [], [], alpha, n, train_period)
    Z, mean, std = _standardise_fit(design(rows, features))
    X = np.hstack([np.ones((n, 1)), Z])
    w = np.zeros(X.shape[1])
    pen = alpha * n * np.eye(X.shape[1])
    pen[0, 0] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w)))
        g = X.T @ (p - yv) + pen @ w
        H = X.T @ (X * (p * (1 - p))[:, None]) + pen + 1e-9 * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return LinearModel("logistic", list(features), w[1:].round(8).tolist(), float(w[0]), mean.tolist(),
                       std.tolist(), alpha, n, train_period)
