"""Stock universe: instrument catalog + DATE-AWARE membership/weights (see config/universe.yaml).

  catalog      `companies` table (symbols, sector, aliases). Not membership.
  membership   `universe_members` table: immutable versions with effective dates, source and status.
For a date d: the CONFIRMED version with the latest effective_from <= d (effective_to empty or >= d).
Versions marked `requires_confirmation` are stored and reported but never used until confirmed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.config.loader import universe_config
from backend.database.models import Company, UniverseMember
from backend.services.timeutil import utcnow
from backend.system1_news.relevance import CompanyRef


def _d(v) -> date | None:
    if v in (None, ""):
        return None
    return v if isinstance(v, date) else date.fromisoformat(str(v))


@dataclass
class UniverseSnapshot:
    on: date
    version: str | None
    members: list[UniverseMember]
    pending: list[str] = field(default_factory=list)   # unconfirmed versions that would apply on `on`

    @property
    def weights(self) -> dict[str, float]:
        return {m.ticker: m.weight or 0.0 for m in self.members}


def seed_universe(session: Session, update: bool = False) -> int:
    """Catalog rows (insert; overwrite with `update`) + universe versions (insert-only: immutable)."""
    cfg = universe_config()
    n = 0
    for c in cfg["companies"]:
        row = session.scalar(select(Company).where(Company.ticker == c["ticker"]))
        if row is not None and not update:
            continue
        row = row or Company(ticker=c["ticker"], nifty_weight=0.0)
        row.name = c["name"]
        row.yahoo_symbol = c["yahoo_symbol"]
        row.sector = c["sector"]
        if c.get("nifty_weight") is not None:          # legacy single-weight format
            row.nifty_weight = float(c["nifty_weight"])
        row.active_from = _d(c.get("active_from")) or date(2000, 1, 1)
        row.active_to = _d(c.get("active_to"))
        row.aliases = c.get("aliases", [c["name"]])
        row.case_sensitive_aliases = c.get("case_sensitive_aliases", [])
        row.exclude_patterns = c.get("exclude_patterns", [])
        session.add(row)
        n += 1
    session.flush()
    seed_versions(session)
    sync_catalog_weights(session)
    return n


def seed_versions(session: Session) -> list[str]:
    """Insert universe versions that are not stored yet. Existing versions are never modified."""
    names = {c["ticker"]: c["name"] for c in universe_config()["companies"]}
    known = {v for (v,) in session.execute(select(UniverseMember.version).distinct())}
    added = []
    for v in universe_config().get("universe_versions", []):
        if v["version"] in known:
            continue
        for m in v["members"]:
            if m["ticker"] not in names:
                raise ValueError(f"universe version {v['version']}: {m['ticker']} is not in the catalog")
            session.add(UniverseMember(
                version=v["version"], ticker=m["ticker"], company=names[m["ticker"]], weight=m.get("weight"),
                weight_note=m.get("note"), effective_from=_d(v["effective_from"]),
                effective_to=_d(v.get("effective_to")), source=" ".join(str(v["source"]).split()),
                source_date=_d(v.get("source_date")), status=v.get("status", "requires_confirmation"),
                confirmed_at=utcnow() if v.get("status") == "confirmed" else None, created_at=utcnow()))
        added.append(v["version"])
    session.flush()
    return added


def _versions(session: Session) -> dict[str, list[UniverseMember]]:
    out: dict[str, list[UniverseMember]] = {}
    for m in session.scalars(select(UniverseMember).order_by(UniverseMember.version, UniverseMember.weight.desc())):
        out.setdefault(m.version, []).append(m)
    return out


def _covers(m: UniverseMember, d: date) -> bool:
    return m.effective_from <= d and (m.effective_to is None or m.effective_to >= d)


def universe_on(session: Session, on: date | None = None) -> UniverseSnapshot:
    on = on or date.today()
    versions = _versions(session)
    covering = {v: ms for v, ms in versions.items() if ms and _covers(ms[0], on)}
    confirmed = [v for v, ms in covering.items() if ms[0].status == "confirmed"]
    best = max(confirmed, key=lambda v: covering[v][0].effective_from, default=None)
    pending = [v for v, ms in covering.items() if ms[0].status != "confirmed"
               and (best is None or ms[0].effective_from >= covering[best][0].effective_from)]
    members = sorted(covering.get(best, []), key=lambda m: -(m.weight or 0))
    return UniverseSnapshot(on, best, members, sorted(pending))


def universe_periods(session: Session, start: date, end: date) -> list[tuple[date, date, UniverseSnapshot]]:
    """Consecutive [from, to] date ranges within [start, end] with a constant effective universe."""
    bounds = sorted({start} | {m.effective_from for ms in _versions(session).values() for m in ms[:1]
                               if start < m.effective_from <= end}
                    | {m.effective_to + timedelta(days=1)
                       for ms in _versions(session).values() for m in ms[:1]
                       if m.effective_to and start <= m.effective_to < end})
    out = []
    for i, b in enumerate(bounds):
        to = (bounds[i + 1] - timedelta(days=1)) if i + 1 < len(bounds) else end
        snap = universe_on(session, b)
        if out and out[-1][2].version == snap.version:
            out[-1] = (out[-1][0], to, out[-1][2])
        else:
            out.append((b, to, snap))
    return out


def confirm_version(session: Session, version: str) -> int:
    rows = list(session.scalars(select(UniverseMember).where(UniverseMember.version == version)))
    if not rows:
        raise KeyError(version)
    for r in rows:
        if r.status != "confirmed":
            r.status, r.confirmed_at = "confirmed", utcnow()
    session.flush()
    sync_catalog_weights(session)
    return len(rows)


def sync_catalog_weights(session: Session, on: date | None = None) -> None:
    """Display-only: mirror today's effective weights onto the catalog rows."""
    snap = universe_on(session, on)
    for m in snap.members:
        c = session.scalar(select(Company).where(Company.ticker == m.ticker))
        if c is not None:
            c.nifty_weight, c.weight_as_of = m.weight or 0.0, m.source_date
            c.weight_source = f"{snap.version}: {m.source}"
    session.flush()


def index_symbol() -> str:
    return universe_config()["index"]["yahoo_symbol"]


def active_companies(session: Session, on: date | None = None) -> list[Company]:
    """Catalog rows of the companies in the effective universe on `on`, by weight."""
    snap = universe_on(session, on)
    by_ticker = {c.ticker: c for c in session.scalars(select(Company))}
    return [by_ticker[m.ticker] for m in snap.members if m.ticker in by_ticker]


def companies_ever_confirmed(session: Session) -> list[Company]:
    """Catalog rows of every company in ANY confirmed universe version. News is analysed against all
    of them, so a company leaving the universe keeps its history (needed by historical backtests)."""
    tickers = {t for (t,) in session.execute(select(UniverseMember.ticker)
                                             .where(UniverseMember.status == "confirmed").distinct())}
    return [c for c in session.scalars(select(Company).order_by(Company.ticker)) if c.ticker in tickers]


def company_infos(session: Session, on: date | None = None):
    """(universe_version, [CompanyInfo]) with the weights effective on `on` (never today's for the past)."""
    from backend.system1_news.pipeline import CompanyInfo
    snap = universe_on(session, on)
    by_ticker = {c.ticker: c for c in session.scalars(select(Company))}
    return snap.version, [CompanyInfo(m.ticker, by_ticker[m.ticker].name, by_ticker[m.ticker].sector, m.weight or 0.0)
                          for m in snap.members if m.ticker in by_ticker]


def universe_report(session: Session, on: date | None = None) -> dict:
    """ACTIVE universe + every stored version (history), with sources and confirmation status."""
    snap = universe_on(session, on)
    versions = _versions(session)

    def rows(ms):
        return [{"company": m.company, "ticker": m.ticker, "weight": m.weight, "effective_from": str(m.effective_from),
                 "effective_to": str(m.effective_to) if m.effective_to else None, "source_date": str(m.source_date)
                 if m.source_date else None, "status": m.status} for m in ms]

    act_set = {m.ticker for m in snap.members}
    diffs = {}
    for v in snap.pending:
        new = {m.ticker for m in versions[v]}
        diffs[v] = {"would_add": sorted(new - act_set), "would_remove": sorted(act_set - new)}
    return {
        "as_of": str(snap.on),
        "active": {"version": snap.version, "source": snap.members[0].source if snap.members else None,
                   "members": rows(snap.members)},
        "requires_confirmation": [{"version": v, "source": versions[v][0].source,
                                   "effective_from": str(versions[v][0].effective_from), **diffs[v],
                                   "members": rows(versions[v])} for v in snap.pending],
        "history": [{"version": v, "status": ms[0].status, "effective_from": str(ms[0].effective_from),
                     "effective_to": str(ms[0].effective_to) if ms[0].effective_to else None,
                     "source": ms[0].source, "members": rows(ms)}
                    for v, ms in sorted(versions.items(), key=lambda kv: kv[1][0].effective_from)],
        "note": "Weights are reference data, not live values. Unconfirmed versions are never used.",
    }


def to_ref(c: Company) -> CompanyRef:
    return CompanyRef(c.ticker, c.name, c.sector, list(c.aliases or []),
                      list(c.case_sensitive_aliases or []), list(c.exclude_patterns or []))
