"""Shared request helpers / models."""
from __future__ import annotations

from datetime import date, datetime

from fastapi import HTTPException

from backend.services.timeutil import parse_timestamp


def parse_dt(value: str) -> datetime:
    dt = parse_timestamp(value)
    if dt is None:
        raise HTTPException(422, f"invalid datetime {value!r} (use ISO-8601, UTC)")
    return dt


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise HTTPException(422, f"invalid date {value!r} (use YYYY-MM-DD)")
