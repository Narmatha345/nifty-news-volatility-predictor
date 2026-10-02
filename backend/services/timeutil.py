"""Time helpers. Rule: everything internal is naive UTC; convert only at the edges."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from email.utils import parsedate_to_datetime
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
NSE_OPEN = time(9, 15)
NSE_CLOSE = time(15, 30)


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def to_utc_naive(dt: datetime) -> datetime:
    """Aware -> naive UTC. Naive input is assumed to already be UTC."""
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def parse_timestamp(value: str | int | float | datetime | None) -> datetime | None:
    """Parse ISO-8601, RFC-822 (RSS) or epoch seconds into naive UTC. None if unparseable."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return to_utc_naive(value)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc).replace(tzinfo=None)
    text = str(value).strip()
    try:
        iso = text.replace("Z", "+00:00")
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:  # providers that omit offsets document UTC
            return dt
        return to_utc_naive(dt)
    except ValueError:
        pass
    try:
        return to_utc_naive(parsedate_to_datetime(text))
    except (TypeError, ValueError, IndexError):
        return None


@dataclass(frozen=True)
class TimestampInfo:
    """A provider timestamp with what is actually KNOWN about it.

    utc          naive UTC. For date_only values this is 00:00 of that date and must NOT be read as a
                 publication time (see system1_news.availability for how it is used).
    precision    exact (seconds) | minute | hour | date_only | unknown
    timezone     e.g. "UTC", "+05:30", "GMT"; None when not stated
    tz_status    explicit (offset in the value) | provider_documented (provider docs say UTC) | unknown
    raw          the value exactly as received
    """
    utc: datetime | None
    precision: str
    timezone: str | None
    tz_status: str
    raw: str | None


_DATE_ONLY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _precision_of(dt: datetime) -> str:
    if dt.second or dt.microsecond:
        return "exact"
    if dt.minute:
        return "minute"
    return "hour"


def _tz_label(dt: datetime) -> str:
    off = dt.utcoffset()
    if off is None:
        return "unknown"
    if off == timedelta(0):
        return "UTC"
    sign = "+" if off >= timedelta(0) else "-"
    mins = abs(int(off.total_seconds())) // 60
    return f"{sign}{mins // 60:02d}:{mins % 60:02d}"


def parse_timestamp_info(value, provider_timezone: str | None = None) -> TimestampInfo:
    """Parse ISO-8601, RFC-822 (RSS), date-only or epoch values without inventing precision.

    A naive value (no offset) is normalised as UTC only if `provider_timezone` documents it
    ("UTC"); otherwise it is still stored as if UTC but flagged tz_status="unknown" so it can
    never count as verified pre-market information."""
    if value is None or value == "":
        return TimestampInfo(None, "unknown", None, "unknown", None)
    raw = value.isoformat() if isinstance(value, datetime) else str(value).strip()
    if isinstance(value, datetime):
        if value.tzinfo is None:
            st = "provider_documented" if provider_timezone else "unknown"
            return TimestampInfo(value, _precision_of(value), provider_timezone, st, raw)
        return TimestampInfo(to_utc_naive(value), _precision_of(value), _tz_label(value), "explicit", raw)
    if isinstance(value, (int, float)):
        dt = datetime.fromtimestamp(float(value), tz=timezone.utc).replace(tzinfo=None)
        return TimestampInfo(dt, "exact", "UTC", "explicit", raw)
    if _DATE_ONLY.match(raw):
        d = date.fromisoformat(raw)
        return TimestampInfo(datetime.combine(d, time.min), "date_only", provider_timezone,
                             "provider_documented" if provider_timezone else "unknown", raw)
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            st = "provider_documented" if provider_timezone else "unknown"
            return TimestampInfo(dt, _precision_of(dt), provider_timezone, st, raw)
        return TimestampInfo(to_utc_naive(dt), _precision_of(dt), _tz_label(dt), "explicit", raw)
    except ValueError:
        pass
    try:
        dt = parsedate_to_datetime(raw)
    except (TypeError, ValueError, IndexError):
        return TimestampInfo(None, "unknown", None, "unknown", raw)
    if dt.tzinfo is None:   # RFC-822 "-0000" means "UTC, origin unknown"
        return TimestampInfo(dt, _precision_of(dt), None, "unknown", raw)
    label = "GMT" if raw.rstrip().endswith("GMT") else _tz_label(dt)
    return TimestampInfo(to_utc_naive(dt), _precision_of(dt), label, "explicit", raw)


def to_ist_naive(dt_utc: datetime | None) -> datetime | None:
    return None if dt_utc is None else dt_utc.replace(tzinfo=timezone.utc).astimezone(IST).replace(tzinfo=None)


def market_open_utc(d: date) -> datetime:
    """09:15 IST on `d`, as naive UTC. Used as the information cut-off for a day's prediction."""
    return to_utc_naive(datetime.combine(d, NSE_OPEN, tzinfo=IST))


def market_close_utc(d: date) -> datetime:
    return to_utc_naive(datetime.combine(d, NSE_CLOSE, tzinfo=IST))


def ist_date(dt_utc: datetime) -> date:
    return dt_utc.replace(tzinfo=timezone.utc).astimezone(IST).date()


def iso_z(dt: datetime | None) -> str | None:
    return None if dt is None else dt.replace(microsecond=0).isoformat() + "Z"
