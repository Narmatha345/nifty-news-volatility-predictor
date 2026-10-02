"""Forecast horizons: today, next 3 month-ends, next 6 quarter-ends.

Month/quarter-end targets are the LAST TRADING SESSION on or before the calendar month/quarter end
(that session's close is the month-end close the market reports).

Quarter conventions (QUARTER_CONVENTION):
  indian_fy  Indian financial year Apr-Mar. Quarter ends Jun 30 (Q1), Sep 30 (Q2), Dec 31 (Q3),
             Mar 31 (Q4). FY is named by the year it ends in: Apr 2026-Mar 2027 = FY27.
  calendar   Jan-Dec. Quarter ends Mar 31 (Q1), Jun 30 (Q2), Sep 30 (Q3), Dec 31 (Q4).
The quarter-END DATES are identical in both conventions; only the labels differ.
"""
from __future__ import annotations

import calendar as _cal
from dataclasses import asdict, dataclass
from datetime import date

from backend.services.calendar import TradingCalendar


@dataclass(frozen=True)
class Horizon:
    horizon_type: str          # today | month_end | quarter_end
    label: str
    calendar_date: date        # nominal month/quarter end (or today)
    target_date: date          # trading session whose close is predicted
    trading_days_ahead: int    # sessions after the previous close up to target (>= 1)
    projected_calendar: bool   # True if exchange holidays for the target year are not yet published

    def to_dict(self) -> dict:
        d = asdict(self)
        d["calendar_date"] = self.calendar_date.isoformat()
        d["target_date"] = self.target_date.isoformat()
        return d


def month_end(year: int, month: int) -> date:
    return date(year, month, _cal.monthrange(year, month)[1])


def _add_months(year: int, month: int, n: int) -> tuple[int, int]:
    idx = year * 12 + (month - 1) + n
    return idx // 12, idx % 12 + 1


def quarter_label(qend: date, convention: str) -> str:
    if convention == "calendar":
        return f"Q{(qend.month - 1) // 3 + 1} {qend.year}"
    fy_q = {6: 1, 9: 2, 12: 3, 3: 4}[qend.month]
    fy_end_year = qend.year + (1 if qend.month >= 4 else 0)
    return f"Q{fy_q} FY{fy_end_year % 100:02d}"


def generate_horizons(today: date, cal: TradingCalendar, convention: str = "indian_fy",
                      n_months: int = 3, n_quarters: int = 6, calendar_today: date | None = None) -> list[Horizon]:
    """`today` is the (IST) date predictions are made for. Its previous close is the base price.
    `calendar_today` (the real current IST date) only affects the label of the first horizon."""
    session_today = today if cal.is_trading_day(today) else cal.on_or_after(today)
    base = cal.previous_session(session_today)
    calendar_today = calendar_today or today

    def make(htype: str, label: str, nominal: date, target: date) -> Horizon:
        return Horizon(htype, label, nominal, target, max(1, cal.sessions_between(base, target)),
                       cal.is_projected(target))

    out = [make("today", "Today" if session_today == calendar_today else f"Next session {session_today}",
                session_today, session_today)]

    # Month-ends strictly after the current session (a month-end equal to today is covered by "today").
    # Loops are bounded: a calendar that ends early yields fewer horizons instead of looping forever.
    y, m, found = today.year, today.month, 0
    for _ in range(n_months + 2):
        if found >= n_months:
            break
        nominal = month_end(y, m)
        target = cal.on_or_before(nominal)
        if target > session_today:
            out.append(make("month_end", nominal.strftime("%b %Y month-end"), nominal, target))
            found += 1
        y, m = _add_months(y, m, 1)

    y, m, found = today.year, today.month, 0
    for _ in range(3 * n_quarters + 3):
        if found >= n_quarters:
            break
        if m % 3 == 0:
            nominal = month_end(y, m)
            target = cal.on_or_before(nominal)
            if target > session_today:
                out.append(make("quarter_end", f"{quarter_label(nominal, convention)} end", nominal, target))
                found += 1
        y, m = _add_months(y, m, 1)
    return out


def horizons_as_dicts(horizons: list[Horizon]) -> list[dict]:
    return [h.to_dict() for h in horizons]
