"""SQLAlchemy ORM schema. All datetimes are stored as naive UTC (see services.timeutil)."""
from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    JSON, Boolean, Date, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, Index,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Company(Base):
    __tablename__ = "companies"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    ticker: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    yahoo_symbol: Mapped[str] = mapped_column(String(32))
    sector: Mapped[str] = mapped_column(String(64))
    # Instrument CATALOG row. Index membership and weights are date-aware and live in
    # `universe_members`; nifty_weight here is display-only (synced from the active version).
    nifty_weight: Mapped[float] = mapped_column(Float)
    weight_as_of: Mapped[date | None] = mapped_column(Date, nullable=True)
    weight_source: Mapped[str | None] = mapped_column(Text, nullable=True)   # provenance of nifty_weight
    active_from: Mapped[date] = mapped_column(Date)
    active_to: Mapped[date | None] = mapped_column(Date, nullable=True)
    aliases: Mapped[list] = mapped_column(JSON, default=list)
    case_sensitive_aliases: Mapped[list] = mapped_column(JSON, default=list)
    exclude_patterns: Mapped[list] = mapped_column(JSON, default=list)


class NewsArticle(Base):
    __tablename__ = "news"
    id: Mapped[int] = mapped_column(primary_key=True)
    source: Mapped[str] = mapped_column(String(160))          # publisher, e.g. "Reuters"
    provider: Mapped[str] = mapped_column(String(40))         # adapter, e.g. "newsapi"
    title: Mapped[str] = mapped_column(Text)
    url: Mapped[str] = mapped_column(Text)
    url_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    title_fingerprint: Mapped[str] = mapped_column(Text)
    published_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)        # provider description
    article_text: Mapped[str | None] = mapped_column(Text, nullable=True)   # only if the provider supplies it
    provider_article_id: Mapped[str | None] = mapped_column(Text, nullable=True)  # stable id (guid, uuid)
    published_raw: Mapped[str | None] = mapped_column(Text, nullable=True)  # timestamp exactly as received
    published_timezone: Mapped[str | None] = mapped_column(String(40), nullable=True)
    # explicit (offset in the value) | provider_documented | unknown
    timezone_status: Mapped[str | None] = mapped_column(String(24), nullable=True)
    published_at_ist: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)  # naive IST, display/audit
    fetched_at_ist: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    first_usable_session: Mapped[date | None] = mapped_column(Date, nullable=True, index=True)
    query: Mapped[str | None] = mapped_column(Text, nullable=True)
    backfilled: Mapped[bool] = mapped_column(Boolean, default=False)  # fetched after the fact
    # exact | minute | hour | date_only | unknown  (see system1_news.availability)
    timestamp_precision: Mapped[str | None] = mapped_column(String(10), nullable=True)
    # verified_pre_market | published_before_cutoff | collected_after_event | unknown
    availability_status: Mapped[str | None] = mapped_column(String(30), nullable=True, index=True)
    duplicate_of: Mapped[int | None] = mapped_column(ForeignKey("news.id"), nullable=True, index=True)
    raw: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    analyses: Mapped[list["NewsAnalysis"]] = relationship(back_populates="article", cascade="all, delete-orphan")


class NewsAnalysis(Base):
    """One row per (article, affected entity). entity = company ticker or 'MACRO'."""
    __tablename__ = "news_analysis"
    __table_args__ = (UniqueConstraint("news_id", "entity", "engine", name="uq_analysis"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    news_id: Mapped[int] = mapped_column(ForeignKey("news.id"), index=True)
    entity: Mapped[str] = mapped_column(String(32), index=True)
    is_macro: Mapped[bool] = mapped_column(Boolean, default=False)
    category: Mapped[str] = mapped_column(String(40))
    keywords: Mapped[list] = mapped_column(JSON, default=list)
    relevance_score: Mapped[float] = mapped_column(Float)
    sentiment_score: Mapped[float] = mapped_column(Float)     # -100..100
    sentiment_label: Mapped[str] = mapped_column(String(10))
    strength: Mapped[float] = mapped_column(Float)            # 0..1 event strength
    source_reliability: Mapped[float] = mapped_column(Float)
    engine: Mapped[str] = mapped_column(String(40))           # e.g. lexicon-v1
    evidence: Mapped[dict] = mapped_column(JSON, default=dict)  # matched terms, explanation
    analyzed_at: Mapped[datetime] = mapped_column(DateTime)
    article: Mapped[NewsArticle] = relationship(back_populates="analyses")


class SentimentSnapshot(Base):
    """System 1 aggregated output for one as_of time and one horizon target date."""
    __tablename__ = "aggregated_scores"
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[str] = mapped_column(String(40), index=True)
    as_of: Mapped[datetime] = mapped_column(DateTime, index=True)
    horizon_type: Mapped[str] = mapped_column(String(20))     # today | month_end | quarter_end
    horizon_label: Mapped[str] = mapped_column(String(40))
    target_date: Mapped[date] = mapped_column(Date)
    entity: Mapped[str] = mapped_column(String(32), index=True)  # ticker | MACRO | NIFTY_COMPANY | NIFTY
    score: Mapped[float] = mapped_column(Float)
    raw_mean: Mapped[float | None] = mapped_column(Float, nullable=True)
    evidence_weight: Mapped[float] = mapped_column(Float)
    news_count: Mapped[int] = mapped_column(Integer)
    details: Mapped[dict] = mapped_column(JSON, default=dict)
    param_version: Mapped[str] = mapped_column(String(40))
    is_backtest: Mapped[bool] = mapped_column(Boolean, default=False)


class MarketData(Base):
    __tablename__ = "market_data"
    __table_args__ = (UniqueConstraint("ticker", "date", name="uq_market_bar"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    ticker: Mapped[str] = mapped_column(String(32), index=True)  # yahoo symbol
    date: Mapped[date] = mapped_column(Date, index=True)
    open: Mapped[float | None] = mapped_column(Float, nullable=True)
    high: Mapped[float | None] = mapped_column(Float, nullable=True)
    low: Mapped[float | None] = mapped_column(Float, nullable=True)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float | None] = mapped_column(Float, nullable=True)
    provider: Mapped[str] = mapped_column(String(40))
    fetched_at: Mapped[datetime] = mapped_column(DateTime)


class PredictionRun(Base):
    __tablename__ = "prediction_runs"
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    as_of: Mapped[datetime] = mapped_column(DateTime)
    param_version: Mapped[str] = mapped_column(String(40))
    model_version: Mapped[str] = mapped_column(String(80))
    sentiment_run_id: Mapped[str] = mapped_column(String(40))
    is_backtest: Mapped[bool] = mapped_column(Boolean, default=False)
    backtest_run_id: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)
    # Reproducibility metadata (live runs from Phase 3 on; NULL for older runs).
    prediction_cutoff: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)  # UTC, 09:15 IST
    target_session: Mapped[date | None] = mapped_column(Date, nullable=True)
    universe_version: Mapped[str | None] = mapped_column(String(60), nullable=True)
    sentiment_engine: Mapped[str | None] = mapped_column(String(40), nullable=True)
    availability_policy: Mapped[str | None] = mapped_column(String(40), nullable=True)
    data_quality: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (Index("ix_pred_company_target", "ticker", "target_date"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("prediction_runs.id"), index=True)
    ticker: Mapped[str] = mapped_column(String(32))
    company: Mapped[str] = mapped_column(String(120))
    prediction_timestamp: Mapped[datetime] = mapped_column(DateTime)
    horizon_type: Mapped[str] = mapped_column(String(20))
    horizon_label: Mapped[str] = mapped_column(String(40))
    target_date: Mapped[date] = mapped_column(Date)
    trading_days_ahead: Mapped[int] = mapped_column(Integer)
    previous_close_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    previous_close: Mapped[float | None] = mapped_column(Float, nullable=True)
    sentiment_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    predicted_movement: Mapped[float | None] = mapped_column(Float, nullable=True)   # percent
    predicted_price: Mapped[float | None] = mapped_column(Float, nullable=True)      # model estimate
    uncertainty_1sigma_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    predicted_direction: Mapped[str] = mapped_column(String(10))   # UP | DOWN | NEUTRAL | MISSING
    status: Mapped[str] = mapped_column(String(20))                # OK | MISSING_DATA
    missing_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    features: Mapped[dict] = mapped_column(JSON, default=dict)
    contributions: Mapped[dict] = mapped_column(JSON, default=dict)
    evidence_news_ids: Mapped[list] = mapped_column(JSON, default=list)
    model_version: Mapped[str] = mapped_column(String(80))
    param_version: Mapped[str] = mapped_column(String(40))
    # Immutable input snapshot: every news item that fed the prediction with its availability
    # relative to the cut-off, event type, sentiment and engine version.
    news_inputs: Mapped[list | None] = mapped_column(JSON, nullable=True)
    data_quality_flag: Mapped[str | None] = mapped_column(String(20), nullable=True)  # VERIFIED | UNCERTAIN | NO_NEW_NEWS
    actual: Mapped["ActualResult | None"] = relationship(back_populates="prediction", uselist=False)
    run: Mapped["PredictionRun"] = relationship(lazy="joined")
    validations: Mapped[list["LLMValidation"]] = relationship(back_populates="prediction")


class ActualResult(Base):
    __tablename__ = "actual_results"
    id: Mapped[int] = mapped_column(primary_key=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id"), unique=True)
    actual_close_date: Mapped[date] = mapped_column(Date)
    actual_close: Mapped[float] = mapped_column(Float)
    actual_movement: Mapped[float] = mapped_column(Float)      # percent
    error: Mapped[float] = mapped_column(Float)                # predicted - actual (pct points)
    direction_correct: Mapped[bool] = mapped_column(Boolean)
    actual_direction: Mapped[str | None] = mapped_column(String(10), nullable=True)
    abs_error: Mapped[float | None] = mapped_column(Float, nullable=True)
    squared_error: Mapped[float | None] = mapped_column(Float, nullable=True)
    previous_close_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    evaluated_at: Mapped[datetime] = mapped_column(DateTime)
    prediction: Mapped[Prediction] = relationship(back_populates="actual")


class LLMValidation(Base):
    """LLM review of a prediction. Never modifies the prediction itself."""
    __tablename__ = "llm_validations"
    id: Mapped[int] = mapped_column(primary_key=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id"), index=True)
    provider: Mapped[str] = mapped_column(String(40))
    model: Mapped[str] = mapped_column(String(80))
    validation_status: Mapped[str] = mapped_column(String(30))
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    inconsistencies: Mapped[list] = mapped_column(JSON, default=list)
    missing_information: Mapped[list] = mapped_column(JSON, default=list)
    reasoning: Mapped[str] = mapped_column(Text)
    request_payload: Mapped[dict] = mapped_column(JSON)
    raw_response: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Structured review (validation schema v2). `reasoning` holds the reason; `inconsistencies` mirrors
    # contradicting_factors for older readers. No API key or header is ever stored.
    supporting_factors: Mapped[list | None] = mapped_column(JSON, nullable=True)
    contradicting_factors: Mapped[list | None] = mapped_column(JSON, nullable=True)
    data_quality_issues: Mapped[list | None] = mapped_column(JSON, nullable=True)
    cutoff_check: Mapped[str | None] = mapped_column(String(20), nullable=True)    # passed | failed | uncertain
    # ok | not_configured | api_error | invalid_response | leakage_blocked
    request_status: Mapped[str | None] = mapped_column(String(30), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)       # redacted
    created_at: Mapped[datetime] = mapped_column(DateTime)
    prediction: Mapped[Prediction] = relationship(back_populates="validations")


class ParameterSet(Base):
    __tablename__ = "parameter_sets"
    version: Mapped[str] = mapped_column(String(40), primary_key=True)
    status: Mapped[str] = mapped_column(String(20))        # active | candidate | archived
    params: Mapped[dict] = mapped_column(JSON)
    parent_version: Mapped[str | None] = mapped_column(String(40), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    created_by: Mapped[str] = mapped_column(String(40))    # seed | tuner | api
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    activated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class Parameter(Base):
    """Flattened view of a parameter set (one row per parameter) for querying."""
    __tablename__ = "parameters"
    __table_args__ = (UniqueConstraint("version", "parameter_name", name="uq_param"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    version: Mapped[str] = mapped_column(ForeignKey("parameter_sets.version"), index=True)
    parameter_name: Mapped[str] = mapped_column(String(80))
    value: Mapped[str] = mapped_column(Text)   # JSON-encoded
    created_at: Mapped[datetime] = mapped_column(DateTime)


class ParameterChange(Base):
    __tablename__ = "parameter_changes"
    id: Mapped[int] = mapped_column(primary_key=True)
    from_version: Mapped[str] = mapped_column(String(40))
    to_version: Mapped[str] = mapped_column(String(40), index=True)
    parameter_name: Mapped[str] = mapped_column(String(80))
    old_value: Mapped[str] = mapped_column(Text)
    new_value: Mapped[str] = mapped_column(Text)
    reason: Mapped[str] = mapped_column(Text)
    backtest_run_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    backtest_result: Mapped[dict] = mapped_column(JSON, default=dict)
    applied: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime)


class BacktestRun(Base):
    __tablename__ = "backtest_runs"
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    status: Mapped[str] = mapped_column(String(20))       # pending | running | completed | failed
    kind: Mapped[str] = mapped_column(String(20))         # backtest | tune
    created_at: Mapped[datetime] = mapped_column(DateTime)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    start_date: Mapped[date] = mapped_column(Date)
    end_date: Mapped[date] = mapped_column(Date)
    param_version: Mapped[str] = mapped_column(String(40))
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    report_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    report_md: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


class FetchLog(Base):
    """Every provider call, success or failure, for reliability auditing."""
    __tablename__ = "fetch_log"
    id: Mapped[int] = mapped_column(primary_key=True)
    provider: Mapped[str] = mapped_column(String(40))
    query: Mapped[str] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(DateTime)
    status: Mapped[str] = mapped_column(String(20))       # ok | error | skipped
    items: Mapped[int] = mapped_column(Integer, default=0)
    new_items: Mapped[int] = mapped_column(Integer, default=0)
    duplicates: Mapped[int | None] = mapped_column(Integer, nullable=True)
    rejected_future: Mapped[int | None] = mapped_column(Integer, nullable=True)
    rejected_missing_timestamp: Mapped[int | None] = mapped_column(Integer, nullable=True)
    rejected_invalid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


class UniverseMember(Base):
    """Date-aware index membership + weight. One row per (version, ticker). Versions are immutable
    once seeded; only `status` changes (requires_confirmation -> confirmed) through an explicit call.
    For a date d the effective universe is the CONFIRMED version with the latest effective_from <= d
    (and effective_to NULL or >= d). Historical periods therefore keep their own weights."""
    __tablename__ = "universe_members"
    __table_args__ = (UniqueConstraint("version", "ticker", name="uq_universe_member"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    version: Mapped[str] = mapped_column(String(60), index=True)
    ticker: Mapped[str] = mapped_column(String(32))
    company: Mapped[str] = mapped_column(String(120))
    weight: Mapped[float | None] = mapped_column(Float, nullable=True)
    weight_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    effective_from: Mapped[date] = mapped_column(Date)
    effective_to: Mapped[date | None] = mapped_column(Date, nullable=True)
    source: Mapped[str] = mapped_column(Text)
    source_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    status: Mapped[str] = mapped_column(String(30))     # confirmed | requires_confirmation
    confirmed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)
