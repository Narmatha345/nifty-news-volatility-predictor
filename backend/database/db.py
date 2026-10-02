"""Engine/session management."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from backend.config.settings import get_settings
from backend.database.models import Base

_engine: Engine | None = None
_SessionLocal: sessionmaker | None = None


def init_engine(url: str | None = None) -> Engine:
    """(Re)initialise the engine. Tests pass `sqlite://` for an in-memory database."""
    global _engine, _SessionLocal
    url = url or get_settings().database_url
    kwargs: dict = {}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        if url in ("sqlite://", "sqlite:///:memory:"):
            from sqlalchemy.pool import StaticPool
            kwargs["poolclass"] = StaticPool
        else:
            Path(url.replace("sqlite:///", "")).parent.mkdir(parents=True, exist_ok=True)
    _engine = create_engine(url, future=True, **kwargs)
    _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)
    Base.metadata.create_all(_engine)
    _add_missing_columns(_engine)
    return _engine


def _add_missing_columns(engine: Engine) -> None:
    """Minimal forward-only migration: add nullable columns that exist in the models but not in an
    older database file. Never drops or rewrites data."""
    insp = inspect(engine)
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if not insp.has_table(table.name):
                continue
            existing = {c["name"] for c in insp.get_columns(table.name)}
            for col in table.columns:
                if col.name not in existing and col.nullable:
                    ddl = col.type.compile(dialect=engine.dialect)
                    conn.execute(text(f'ALTER TABLE "{table.name}" ADD COLUMN "{col.name}" {ddl}'))


def get_engine() -> Engine:
    return _engine or init_engine()


@contextmanager
def session_scope() -> Iterator[Session]:
    if _SessionLocal is None:
        init_engine()
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency."""
    with session_scope() as s:
        yield s
