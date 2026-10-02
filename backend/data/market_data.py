"""Daily OHLCV market data: provider adapters + DB storage + point-in-time reads."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import date, timedelta

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.database.models import MarketData
from backend.services.timeutil import utcnow

log = logging.getLogger(__name__)


class MarketDataError(RuntimeError):
    pass


class MarketDataProvider(ABC):
    name = "base"

    @abstractmethod
    def fetch_daily(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """DataFrame indexed by date with columns open, high, low, close, volume. end inclusive."""


class YFinanceProvider(MarketDataProvider):
    """Yahoo Finance via yfinance. NSE symbols use the `.NS` suffix; NIFTY 50 is `^NSEI`.

    Closes are split/dividend/corporate-action ADJUSTED (auto_adjust=True) so that returns are not
    distorted by ex-dates or demergers. Adjustments apply backwards, so the latest close equals the
    quoted close; older stored closes are restated on each refresh.
    """
    name = "yfinance"

    def fetch_daily(self, symbol, start, end):
        import yfinance as yf

        try:
            df = yf.download(symbol, start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(),
                             interval="1d", auto_adjust=True, progress=False, threads=False)
        except Exception as exc:
            raise MarketDataError(f"yfinance failed for {symbol}: {exc}") from exc
        if df is None or df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        df.index = pd.to_datetime(df.index).date
        return validate_bars(df, symbol)


def validate_bars(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Drop bars with missing / non-positive closes. Never fills gaps."""
    bad = df["close"].isna() | (df["close"] <= 0)
    if bad.any():
        log.warning("%s: dropping %d invalid bars (missing or non-positive close)", symbol, int(bad.sum()))
    return df[~bad]


def store_bars(session: Session, symbol: str, df: pd.DataFrame, provider: str) -> int:
    existing = {d for (d,) in session.execute(
        select(MarketData.date).where(MarketData.ticker == symbol))}
    n = 0
    now = utcnow()
    for d, row in df.iterrows():
        def f(v):
            return None if pd.isna(v) else float(v)
        if d in existing:
            bar = session.scalar(select(MarketData).where(MarketData.ticker == symbol, MarketData.date == d))
            bar.open, bar.high, bar.low, bar.close, bar.volume = (f(row["open"]), f(row["high"]),
                                                                   f(row["low"]), float(row["close"]),
                                                                   f(row["volume"]))
            bar.fetched_at = now
        else:
            session.add(MarketData(ticker=symbol, date=d, open=f(row["open"]), high=f(row["high"]),
                                   low=f(row["low"]), close=float(row["close"]), volume=f(row["volume"]),
                                   provider=provider, fetched_at=now))
            n += 1
    session.flush()
    return n


def refresh(session: Session, symbols: list[str], start: date, end: date | None = None,
            provider: MarketDataProvider | None = None) -> dict[str, dict]:
    provider = provider or YFinanceProvider()
    end = end or date.today()
    out = {}
    for s in symbols:
        try:
            df = provider.fetch_daily(s, start, end)
            out[s] = {"status": "ok", "bars": len(df), "new": store_bars(session, s, df, provider.name)}
        except MarketDataError as exc:
            out[s] = {"status": "error", "error": str(exc)}
        session.commit()
    return out


def load_closes(session: Session, symbol: str, start: date | None = None,
                end: date | None = None) -> pd.Series:
    q = select(MarketData.date, MarketData.close).where(MarketData.ticker == symbol)
    if start:
        q = q.where(MarketData.date >= start)
    if end:
        q = q.where(MarketData.date <= end)
    rows = session.execute(q.order_by(MarketData.date)).all()
    return pd.Series([r[1] for r in rows], index=[r[0] for r in rows], dtype=float, name=symbol)
