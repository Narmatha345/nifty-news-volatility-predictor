"""Sentiment factor model (factor-v1). Pure function of features + parameters.

  signal z   = (w_company * S_company + w_macro * m_sector * S_macro + w_nifty * beta * S_nifty) / 100
  sigma_h    = daily_vol * sqrt(h)                 h = trading days from previous close to target
  news       = sensitivity * z * sigma_h           (expected move in log-return units)
  drift      = drift_weight * daily_drift * h
  log move   = clip(news + drift, -max_sigma_clip * sigma_h, +max_sigma_clip * sigma_h)
  predicted% = (exp(log move) - 1) * 100
  estimated price = previous_close * (1 + predicted% / 100)          <- model estimate, not a quote

Sentiment scores S are System 1's horizon-specific scores (decayed to the target date), so news
naturally fades for distant horizons while the uncertainty band (sigma_h) widens.

factor-v2 (system2.sentiment_input = "relative") fixes the UP bias found in the May-Sep 2026
backtest. The raw scores S are long averages of headline TONE (company headlines skew positive,
macro negative), so they shifted almost every prediction in the same direction. factor-v2 feeds
RELATIVE sentiment instead (new news minus the entity's own baseline; 0 when nothing new arrived)
and treats a news surprise as a one-off price adjustment:

  news       = sensitivity * (relative signal / 100) * sigma_1      (sigma_1 = daily vol, not sigma_h)
  log move   = clip(news + drift, +/- max_sigma_clip * sigma_h)
so the expected news impact does not grow with the horizon while the uncertainty band still does.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

MODEL_NAME = "factor-v1"
MODEL_NAME_RELATIVE = "factor-v2"
MODEL_NAME_LINEAR = "ridge-v1"


@dataclass
class ModelOutput:
    predicted_movement_pct: float
    estimated_price: float
    direction: str
    uncertainty_1sigma_pct: float
    signal: float
    contributions_pct: dict[str, float]


def predict(previous_close: float, daily_vol: float, daily_drift: float, beta: float, h: int,
            s_company: float, s_macro: float, macro_sens: float, s_nifty: float, p2: dict) -> ModelOutput:
    """`s_*` are raw scores (factor-v1) or relative sentiments (factor-v2), per p2['sentiment_input']."""
    sigma_h = daily_vol * math.sqrt(max(1, h))
    news_sigma = daily_vol if p2.get("sentiment_input", "raw") == "relative" else sigma_h
    k = p2["sensitivity"] * news_sigma / 100.0
    parts = {
        "company_sentiment": k * p2["w_company"] * s_company,
        "macro_sentiment": k * p2["w_macro"] * macro_sens * s_macro,
        "nifty_sentiment": k * p2["w_nifty"] * beta * s_nifty,
        "drift": p2["drift_weight"] * daily_drift * h,
    }
    raw = sum(parts.values())
    cap = p2["max_sigma_clip"] * sigma_h
    log_move = max(-cap, min(cap, raw))
    pct = (math.exp(log_move) - 1.0) * 100.0
    threshold = p2["neutral_threshold_sigma"] * sigma_h * 100.0
    direction = "UP" if pct > threshold else "DOWN" if pct < -threshold else "NEUTRAL"
    signal = (p2["w_company"] * s_company + p2["w_macro"] * macro_sens * s_macro
              + p2["w_nifty"] * beta * s_nifty) / 100.0
    scale = (log_move / raw) if raw else 1.0   # attribute clipping proportionally
    return ModelOutput(round(pct, 4), round(previous_close * (1 + pct / 100.0), 2), direction,
                       round(sigma_h * 100.0, 4), round(signal, 5),
                       {k_: round(v * scale * 100.0, 4) for k_, v in parts.items()})


def direction_for(pct: float, sigma_h_pct: float, p2: dict) -> str:
    threshold = p2["neutral_threshold_sigma"] * sigma_h_pct
    return "UP" if pct > threshold else "DOWN" if pct < -threshold else "NEUTRAL"


def model_name(p2: dict) -> str:
    if p2.get("model", "factor") == "linear":
        return MODEL_NAME_LINEAR
    return MODEL_NAME_RELATIVE if p2.get("sentiment_input", "raw") == "relative" else MODEL_NAME


def model_version(param_version: str, p2: dict | None = None) -> str:
    return f"{model_name(p2 or {})}@{param_version}"
