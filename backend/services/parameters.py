"""Versioned model parameters. Active set is read-only to the tuner; changes need explicit activation."""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.config.loader import default_params_config
from backend.database.models import Parameter, ParameterChange, ParameterSet
from backend.services.timeutil import utcnow

# Bump when model_params.yaml defaults change. Parameter sets are immutable, so new defaults are a
# new version. A new default is seeded ACTIVE only when nothing is active (fresh database);
# otherwise it is stored as a CANDIDATE - no parameter set is ever activated automatically.
DEFAULT_VERSION = "default-v3"

# Values that reproduce the behaviour of parameter sets created before a key existed.
LEGACY_DEFAULTS = {
    "system1": {"price_report_weight": 1.0, "baseline_days": 60, "min_baseline_n": 10},
    "system2": {"model": "factor", "sentiment_input": "raw"},
}


def complete(params: dict[str, Any]) -> dict[str, Any]:
    """Fill keys missing from an older parameter set with their legacy (behaviour-preserving) values."""
    out = copy.deepcopy(params)
    for section, defaults in LEGACY_DEFAULTS.items():
        for k, v in defaults.items():
            out.setdefault(section, {}).setdefault(k, v)
    return out


def default_params() -> dict[str, Any]:
    cfg = copy.deepcopy(default_params_config())
    return {"system1": cfg["system1"], "system2": cfg["system2"]}


def search_space() -> dict[str, list[float]]:
    return dict(default_params_config().get("search_space", {}))


def flatten(params: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in params.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and k != "source_reliability":
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def get_path(params: dict, dotted: str) -> Any:
    node = params
    for part in dotted.split("."):
        node = node[part]
    return node


def set_path(params: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = params
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value


def params_hash(params: dict) -> str:
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:10]


def _write_set(session: Session, version: str, params: dict, status: str, created_by: str,
               parent: str | None, notes: str | None) -> ParameterSet:
    now = utcnow()
    ps = ParameterSet(version=version, status=status, params=params, parent_version=parent,
                      created_at=now, created_by=created_by, notes=notes,
                      activated_at=now if status == "active" else None)
    session.add(ps)
    for name, value in flatten(params).items():
        session.add(Parameter(version=version, parameter_name=name, value=json.dumps(value), created_at=now))
    session.flush()
    return ps


def ensure_default(session: Session) -> None:
    if session.get(ParameterSet, DEFAULT_VERSION) is not None:
        return
    active = session.scalar(select(ParameterSet).where(ParameterSet.status == "active"))
    _write_set(session, DEFAULT_VERSION, default_params(), "active" if active is None else "candidate", "seed",
               active.version if active else None,
               "Seeded from backend/config/model_params.yaml" + ("" if active is None else
               f"; stored as candidate because {active.version} is active - activate explicitly"))


def get_active(session: Session) -> tuple[str, dict]:
    ensure_default(session)
    ps = session.scalar(select(ParameterSet).where(ParameterSet.status == "active"))
    return ps.version, complete(ps.params)


def get_set(session: Session, version: str) -> ParameterSet | None:
    return session.get(ParameterSet, version)


def get_params(session: Session, version: str) -> dict | None:
    ps = session.get(ParameterSet, version)
    return None if ps is None else complete(ps.params)


def create_candidate(session: Session, params: dict, parent_version: str, reason: str,
                     backtest_run_id: str | None, backtest_result: dict) -> ParameterSet:
    """Store a tuned set as CANDIDATE with one ParameterChange row per changed value."""
    parent = session.get(ParameterSet, parent_version)
    version = f"tuned-{params_hash(params)}"
    existing = session.get(ParameterSet, version)
    if existing:
        return existing
    ps = _write_set(session, version, params, "candidate", "tuner", parent_version, reason)
    old_flat, new_flat = flatten(parent.params), flatten(params)
    for name, new in new_flat.items():
        old = old_flat.get(name)
        if old != new:
            session.add(ParameterChange(from_version=parent_version, to_version=version, parameter_name=name,
                                        old_value=json.dumps(old), new_value=json.dumps(new), reason=reason,
                                        backtest_run_id=backtest_run_id, backtest_result=backtest_result,
                                        applied=False, created_at=utcnow()))
    session.flush()
    return ps


def activate(session: Session, version: str) -> ParameterSet:
    """Explicit, human-triggered promotion. Archives the previous active set."""
    target = session.get(ParameterSet, version)
    if target is None:
        raise KeyError(version)
    for ps in session.scalars(select(ParameterSet).where(ParameterSet.status == "active")):
        ps.status = "archived"
    target.status = "active"
    target.activated_at = utcnow()
    for ch in session.scalars(select(ParameterChange).where(ParameterChange.to_version == version)):
        ch.applied = True
    session.flush()
    return target
