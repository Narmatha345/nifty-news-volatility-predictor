"""NSE trading calendar.

Source priority:
  1. `exchange_calendars` XBOM (BSE; NSE and BSE share trading holidays). Holidays for years the
     library has not published yet fall back to weekday-only sessions; such dates are flagged
     `projected=True` so the UI can show the target date may move.
  2. Weekdays only (if the library is not installed).
Historical backtests do not depend on this: they align on the dates actually present in price data.
"""
from __future__ import annotations

import logging
from bisect import bisect_left, bisect_right
from datetime import date, datetime, timedelta
from functools import lru_cache

log = logging.getLogger(__name__)


class TradingCalendar:
    def __init__(self, sessions: list[date], source: str, reliable_until: date):
        self.sessions = sorted(set(sessions))
        self.source = source
        self.reliable_until = reliable_until

    def is_trading_day(self, d: date) -> bool:
        i = bisect_left(self.sessions, d)
        return i < len(self.sessions) and self.sessions[i] == d

    def on_or_before(self, d: date) -> date:
        i = bisect_right(self.sessions, d) - 1
        if i < 0:
            raise ValueError(f"no session on or before {d}")
        return self.sessions[i]

    def on_or_after(self, d: date) -> date:
        i = bisect_left(self.sessions, d)
        if i >= len(self.sessions):
            raise ValueError(f"no session on or after {d}")
        return self.sessions[i]

    def previous_session(self, d: date) -> date:
        return self.on_or_before(d - timedelta(days=1))

    def sessions_between(self, start_exclusive: date, end_inclusive: date) -> int:
        """Number of sessions s with start_exclusive < s <= end_inclusive."""
        return bisect_right(self.sessions, end_inclusive) - bisect_right(self.sessions, start_exclusive)

    def is_projected(self, d: date) -> bool:
        return d > self.reliable_until


def prediction_session(as_of_utc: datetime, cal: TradingCalendar) -> date:
    """First session whose 09:15 IST open is at or after `as_of`.

    News published after a session opens can describe that session's move, so a session that has
    already opened is never predicted. Same rule as the backtest (as_of = open of the session).
    """
    from backend.services.timeutil import ist_date, market_open_utc

    d = cal.on_or_after(ist_date(as_of_utc))
    while market_open_utc(d) < as_of_utc:
        d = cal.on_or_after(d + timedelta(days=1))
    return d


def _weekday_sessions(start: date, end: date) -> list[date]:
    out, d = [], start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def calendar_from_sessions(sessions: list[date], source: str = "price-data") -> TradingCalendar:
    """Calendar built from observed trading dates (used in backtests and tests)."""
    return TradingCalendar(sessions, source, max(sessions) if sessions else date.min)


@lru_cache
def get_calendar(start: date = date(2015, 1, 1), end: date | None = None) -> TradingCalendar:
    end = end or date(date.today().year + 3, 12, 31)
    try:
        import exchange_calendars as xc

        cal = xc.get_calendar("XBOM", start=start.isoformat())
        lib_end = cal.last_session.date()
        sessions = [ts.date() for ts in cal.sessions_in_range(start.isoformat(), lib_end.isoformat())]
        if end > lib_end:
            sessions += _weekday_sessions(lib_end + timedelta(days=1), end)
        return TradingCalendar(sessions, "exchange_calendars:XBOM", lib_end)
    except Exception as exc:  # library missing or failed: degrade loudly, not silently
        log.warning("exchange_calendars unavailable (%s); using weekday-only calendar", exc)
        return TradingCalendar(_weekday_sessions(start, end), "weekdays-only", date.min)
