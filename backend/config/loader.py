"""Loads the YAML reference/config files in backend/config/."""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import yaml

from backend.config.settings import CONFIG_DIR


def _load(name: str) -> dict[str, Any]:
    with open(CONFIG_DIR / name, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@lru_cache
def universe_config() -> dict[str, Any]:
    return _load("universe.yaml")


@lru_cache
def taxonomy_config() -> dict[str, Any]:
    return _load("taxonomy.yaml")


@lru_cache
def macro_sensitivity_config() -> dict[str, Any]:
    return _load("macro_sensitivity.yaml")


@lru_cache
def default_params_config() -> dict[str, Any]:
    return _load("model_params.yaml")
