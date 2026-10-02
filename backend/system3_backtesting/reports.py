"""Human-readable (Markdown) and AI-readable (JSON) backtest / experiment reports.

Every number in a report is computed from the backtest records passed in. Observations are
rule-based statements derived from those numbers - nothing is written by hand or by an LLM.
"""
from __future__ import annotations

from dataclasses import asdict

from backend.system3_backtesting.metrics import (
    DIRECTIONS, EvalRecord, compute_metrics, feature_analysis, grouped_metrics,
)
from backend.system3_backtesting.tuner import TuningResult

SMALL_SAMPLE = 100
DISCLAIMER = ("> Research output. Predicted movements are model estimates, not guaranteed outcomes. "
              "Historical performance does not guarantee future performance.")
BASELINE_LABELS = {"no_change": "No change", "historical_mean": "Historical mean (all stocks)",
                   "company_mean": "Company historical mean", "rolling_mean_20d": "Rolling mean (20 sessions)"}


def _v(m: dict, key: str):
    x = m.get(key)
    return x.get("value") if isinstance(x, dict) else x


def _fmt(x, pct=False, nd=3):
    if x is None:
        return "n/a"
    return f"{x * 100:.1f}%" if pct else f"{x:.{nd}f}"


def observations(records: list[EvalRecord], by_horizon: dict, meta: dict) -> list[str]:
    obs = []
    for h, m in by_horizon.items():
        n = m["n"]
        if n < SMALL_SAMPLE:
            obs.append(f"[{h}] Only {n} samples - treat metrics as indicative, not conclusive.")
        skill = m.get("mae_skill_vs_zero")
        if skill is not None:
            if skill <= 0:
                obs.append(f"[{h}] MAE {_fmt(_v(m, 'mae'))} does not beat the no-change baseline "
                           f"({_fmt(m['baseline_zero_change']['mae'])}); news signal adds no MAE skill here.")
            else:
                obs.append(f"[{h}] MAE improves on the no-change baseline by {skill * 100:.1f}%.")
        bc = m.get("baseline_comparison") or {}
        if bc.get("baselines"):
            lost = [BASELINE_LABELS.get(b, b) for b, v in bc["baselines"].items() if not v["model_beats_baseline_mae"]]
            if lost:
                obs.append(f"[{h}] Model MAE does not beat: {', '.join(lost)} (n={bc['n']}).")
            else:
                obs.append(f"[{h}] Model MAE beats every baseline on n={bc['n']} - check sample size and the "
                           f"out-of-sample experiment before trusting it.")
        sa = m.get("sign_accuracy", {})
        p = sa.get("p_value_vs_coinflip")
        if sa.get("value") is not None and p is not None:
            if p >= 0.05:
                verdict = "not statistically different from"
            else:
                verdict = "statistically BETTER than" if sa["value"] > 0.5 else "statistically WORSE than"
            obs.append(f"[{h}] Sign accuracy {_fmt(sa['value'], pct=True)} over {sa['n']} directional calls is "
                       f"{verdict} a coin flip (p={p:.3f}).")
        neutral_n = m.get("neutral_prediction_accuracy", {}).get("n", 0)
        if n and neutral_n / n > 0.5:
            obs.append(f"[{h}] {neutral_n / n * 100:.0f}% of predictions were NEUTRAL (smaller than the neutral "
                       f"band); directional accuracy counts them wrong whenever the actual move leaves the band, "
                       f"so read sign accuracy for the UP/DOWN calls.")
        ups = m.get("positive_prediction_accuracy", {}).get("n", 0)
        downs = m.get("negative_prediction_accuracy", {}).get("n", 0)
        if ups + downs >= 20 and max(ups, downs) / (ups + downs) > 0.8:
            side = "UP" if ups > downs else "DOWN"
            obs.append(f"[{h}] {max(ups, downs) / (ups + downs) * 100:.0f}% of directional calls were {side} "
                       f"(mean actual move {_fmt(m.get('mean_actual'))}%): a one-sided bias. Check the contribution "
                       f"breakdown in 'Prediction distribution'.")
        if h in ("month_end", "quarter_end"):
            obs.append(f"[{h}] Samples from consecutive days share most of their future path (overlapping "
                       f"windows), so the effective number of independent samples is much smaller than {n}.")
    no_news = sum(r.new_news_count == 0 for r in records)
    if records:
        obs.append(f"{no_news / len(records) * 100:.1f}% of predictions had no NEW company news since the previous "
                   f"cut-off; with relative sentiment those carry no company-news signal.")
    dq = meta.get("data_quality") or {}
    if dq.get("total_news_articles"):
        v = dq["verified_pre_market"]
        obs.append(f"News availability: {v} of {dq['total_news_articles']} articles in the period "
                   f"({v / dq['total_news_articles'] * 100:.1f}%) were demonstrably collected before the cut-off "
                   f"they could be used for. The rest rely on provider timestamps of retrospectively collected "
                   f"news - this backtest is NOT equivalent to a live-news backtest.")
    if meta.get("backfilled_share", 0) > 0:
        obs.append(f"{meta['backfilled_share'] * 100:.1f}% of news was collected after the fact (backfill). "
                   "Backfilled feeds can omit since-deleted articles (survivorship) - live-collected news is "
                   "the gold standard for backtests.")
    obs.append("Historical performance does not guarantee future performance.")
    return obs


def failure_cases(records: list[EvalRecord], news_titles: dict[int, str], n: int = 10) -> list[dict]:
    worst = sorted(records, key=lambda r: -abs(r.error))[:n]
    return [{"as_of_date": r.as_of_date, "ticker": r.ticker, "horizon": r.horizon_label,
             "target_date": r.target_date, "predicted_pct": r.predicted, "actual_pct": r.actual,
             "error_pct_points": round(r.error, 4), "sigma_pct": r.sigma_pct,
             "sentiment_score": r.sentiment_score, "news_count": r.news_count,
             "evidence": [news_titles[i] for i in r.evidence_news_ids[:3] if i in news_titles]}
            for r in worst]


def contribution_summary(records: list[EvalRecord]) -> dict:
    """Mean contribution (pct-pts) and share of positive values per model component: the direct
    answer to 'where does a directional bias come from'."""
    comps: dict[str, list[float]] = {}
    for r in records:
        for k, v in r.contributions:   # already in percentage points for every model
            comps.setdefault(k, []).append(v)
    return {k: {"mean": round(sum(v) / len(v), 5), "share_positive": round(sum(x > 0 for x in v) / len(v), 4),
                "share_negative": round(sum(x < 0 for x in v) / len(v), 4), "n": len(v)}
            for k, v in sorted(comps.items())}


def company_summary(records: list[EvalRecord]) -> dict:
    out = {}
    for t in sorted({r.ticker for r in records}):
        out[t] = {}
        for h in sorted({r.horizon_type for r in records if r.ticker == t}):
            rs = [r for r in records if r.ticker == t and r.horizon_type == h]
            m = compute_metrics(rs)
            out[t][h] = {"n": m["n"], "mae": _v(m, "mae"), "rmse": _v(m, "rmse"),
                         "directional_accuracy": _v(m, "directional_accuracy"),
                         "sign_accuracy": m["sign_accuracy"], "bias": m["bias"],
                         "no_change_mae": m["baseline_zero_change"]["mae"],
                         "predicted": {d: m["prediction_distribution"]["predicted"][d]["count"] for d in DIRECTIONS},
                         "actual": {d: m["prediction_distribution"]["actual"][d]["count"] for d in DIRECTIONS},
                         "confusion_matrix": m["confusion_matrix"]}
    return out


def build_reports(meta: dict, records: list[EvalRecord], news_titles: dict[int, str],
                  tuning: TuningResult | None, parameter_changes: list[dict]) -> tuple[dict, str]:
    overall = compute_metrics(records)
    by_h = grouped_metrics(records, "horizon_type")
    by_c = {t: grouped_metrics([r for r in records if r.ticker == t], "horizon_type")
            for t in sorted({r.ticker for r in records})}
    monthly = {h: grouped_metrics([r for r in records if r.horizon_type == h], "month") for h in by_h}
    today = [r for r in records if r.horizon_type == "today"]
    obs = observations(records, by_h, meta)
    fails = failure_cases(records, news_titles)
    scatter = [{"ticker": r.ticker, "horizon_type": r.horizon_type, "as_of_date": r.as_of_date,
                "predicted": r.predicted, "actual": r.actual} for r in records]

    report = {
        "report_type": "nifty_news_volatility_backtest",
        "backtest_run_id": meta["run_id"],
        "generated_at": meta["generated_at"],
        "backtest_period": {"start": meta["start"], "end": meta["end"]},
        "param_version": meta["param_version"],
        "model_version": meta["model_version"],
        "availability_mode": meta.get("availability_mode", "provider_timestamp"),
        "stocks_tested": meta["tickers"],
        "horizon_types": meta["horizon_types"],
        "number_of_news_events": meta["n_news"],
        "number_of_news_clusters": meta.get("n_clusters"),
        "total_predictions": len(records),
        "data_quality": meta.get("data_quality"),
        "metrics": overall,
        "metrics_by_horizon": by_h,
        "contribution_summary_by_horizon": {h: contribution_summary([r for r in records if r.horizon_type == h])
                                            for h in by_h},
        "feature_analysis_next_session": feature_analysis(today) if today else {},
        "company_results": by_c,
        "company_summary": company_summary(records),
        "monthly_results": monthly,
        "parameter_changes": parameter_changes,
        "tuning": None if tuning is None else {k: v for k, v in asdict(tuning).items() if k != "best_params"}
                  | {"best_params": tuning.best_params},
        "failure_cases": fails,
        "observations": obs,
        "lookahead_controls": {
            "news_cutoff": "effective availability <= 09:15 IST on the prediction date (date-only provider "
                           "timestamps count from the end of that date)",
            "news_availability_mode": meta.get("availability_mode", "provider_timestamp"),
            "relative_sentiment_baseline": "only news available before the previous session's cut-off",
            "price_cutoff": "closes strictly before the prediction date",
            "targets": "actual close on the target session; horizons beyond available data skipped",
            "baselines": "computed from closes strictly before the prediction date",
        },
        "disclaimer": "Model estimates, not guaranteed outcomes. Historical performance does not guarantee "
                      "future performance.",
        "predicted_vs_actual": scatter,
    }
    return report, _markdown(report)


# ---------------------------------------------------------------------------------------------- md
def _metric_table(by_h: dict) -> list[str]:
    lines = ["| Group | N | Directional acc. | Sign acc. (n) | MAE | RMSE | Bias | Corr | "
             "No-change MAE | MAE skill |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for h, m in by_h.items():
        if m["n"] == 0:
            continue
        lines.append(
            f"| {h} | {m['n']} | {_fmt(_v(m, 'directional_accuracy'), True)} | "
            f"{_fmt(m['sign_accuracy']['value'], True)} ({m['sign_accuracy']['n']}) | {_fmt(_v(m, 'mae'))} | "
            f"{_fmt(_v(m, 'rmse'))} | {_fmt(m.get('bias'))} | "
            f"{_fmt(_v(m, 'correlation'))} | {_fmt(m['baseline_zero_change']['mae'])} | "
            f"{_fmt(m.get('mae_skill_vs_zero'), True)} |")
    return lines


def _confusion_lines(cm: dict) -> list[str]:
    lines = ["| Predicted \\ Actual | UP | DOWN | NEUTRAL | Total |", "|---|---|---|---|---|"]
    for p in DIRECTIONS:
        row = cm[p]
        lines.append(f"| {p} | {row['UP']} | {row['DOWN']} | {row['NEUTRAL']} | {sum(row.values())} |")
    tot = {a: sum(cm[p][a] for p in DIRECTIONS) for a in DIRECTIONS}
    lines.append(f"| Total | {tot['UP']} | {tot['DOWN']} | {tot['NEUTRAL']} | {sum(tot.values())} |")
    return lines


def _baseline_lines(bc: dict) -> list[str]:
    if not bc or not bc.get("baselines"):
        return ["Baselines unavailable for these records."]
    m = bc["model"]
    lines = [f"Same {bc['n']} samples for every row.", "",
             "| Forecast | MAE | RMSE | Directional acc. | Sign acc. (n) | Bias | Model MAE skill vs it |",
             "|---|---|---|---|---|---|---|",
             f"| **Model** | {_fmt(m['mae'])} | {_fmt(m['rmse'])} | {_fmt(m['directional_accuracy'], True)} | "
             f"{_fmt(m['sign_accuracy']['value'], True)} ({m['sign_accuracy']['n']}) | {_fmt(m['bias'])} | - |"]
    for b, v in bc["baselines"].items():
        lines.append(f"| {BASELINE_LABELS.get(b, b)} | {_fmt(v['mae'])} | {_fmt(v['rmse'])} | "
                     f"{_fmt(v['directional_accuracy'], True)} | {_fmt(v['sign_accuracy']['value'], True)} "
                     f"({v['sign_accuracy']['n']}) | {_fmt(v['bias'])} | {_fmt(v['model_mae_skill'], True)} |")
    return lines


def _feature_lines(fa: dict) -> list[str]:
    if not fa:
        return ["No next-session records."]
    lines = ["| Feature | N | Missing | Mean | Std | Corr. with return (p) | Rank corr. | Sign agreement (n) | "
             "Top-bottom quintile return | Informative (p<0.05) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for f, v in fa.items():
        if "mean" not in v:
            lines.append(f"| {f} | {v['n']} | {v['missing']} | n/a | n/a | n/a | n/a | n/a | n/a | no |")
            continue
        sa = v["sign_agreement"]
        lines.append(f"| {f} | {v['n']} | {v['missing']} | {_fmt(v['mean'])} | {_fmt(v['std'])} | "
                     f"{_fmt(v['correlation'])} ({_fmt(v['p_value'])}) | {_fmt(v['rank_correlation'])} | "
                     f"{_fmt(sa['value'], True)} ({sa['n']}) | {_fmt(v['top_minus_bottom_quintile_return'])} | "
                     f"{'yes' if v['informative_at_5pct'] else 'no'} |")
    lines += ["", "Correlation is not causation; tests are not corrected for multiple comparisons."]
    return lines


def _data_quality_lines(dq: dict | None) -> list[str]:
    if not dq:
        return ["Not available."]
    st = dq["availability_status"]
    miss = dq["missing_price_sessions_by_ticker"]
    return [
        f"- **Total news articles** (available up to the last cut-off): {dq['total_news_articles']}",
        f"- **Verified pre-market** (collected before the cut-off they can be used for): {st['verified_pre_market']}",
        f"- **Published before cut-off, collected during the session:** {st['published_before_cutoff']}",
        f"- **Post-event** (collected after the session they would predict had closed): {st['collected_after_event']}",
        f"- **Unknown availability** (date-only timestamps, not verified): {st['unknown']}",
        f"- **Coarse (date/hour-only) timestamps:** {dq['coarse_timestamps']}",
        f"- **Duplicate rate:** {_fmt(dq['duplicate_rate'], True)}",
        f"- **Price-report analyses** (headline only restates a price move): {dq['price_report_analyses']} of "
        f"{dq['relevant_analyses_in_period']}",
        f"- **Missing price data:** {sum(miss.values())} ticker-sessions missing of "
        f"{dq['price_sessions'] * len(miss)} (coverage {_fmt(dq['price_coverage'], True)}); "
        f"index missing {dq['missing_index_sessions']}",
    ]


def _markdown(r: dict) -> str:
    m = r["metrics"]
    L = [f"# Backtest report `{r['backtest_run_id']}`", "", DISCLAIMER, "",
         "## Backtest setup", "",
         f"- **Backtest period:** {r['backtest_period']['start']} to {r['backtest_period']['end']}",
         f"- **Stocks tested:** {', '.join(r['stocks_tested'])}",
         f"- **Horizons:** {', '.join(r['horizon_types'])}",
         f"- **News events used:** {r['number_of_news_events']} articles "
         f"({r['number_of_news_clusters']} distinct stories)",
         f"- **News availability mode:** `{r['availability_mode']}`",
         f"- **Predictions evaluated:** {r['total_predictions']}",
         f"- **Parameters / model:** `{r['param_version']}` / `{r['model_version']}`", "",
         "Look-ahead controls: news cut-off is 09:15 IST on the prediction date (date-only timestamps count "
         "from the end of their date); prices are closes strictly before it; outcomes are actual closes on the "
         "target session; baselines use closes before the prediction date only.", "",
         "## Data quality", "", *_data_quality_lines(r.get("data_quality")), ""]
    if m.get("n", 0) == 0:
        L += ["## Overall results", "", f"No evaluable predictions: {m.get('note')}", ""]
        return "\n".join(L)
    L += ["## Overall results", "",
          f"- **Directional accuracy:** {_fmt(_v(m, 'directional_accuracy'), True)} (n={m['n']})",
          f"- **Sign accuracy (UP/DOWN calls only):** {_fmt(m['sign_accuracy']['value'], True)} "
          f"(n={m['sign_accuracy']['n']}, p vs coin flip = {_fmt(m['sign_accuracy'].get('p_value_vs_coinflip'))})",
          f"- **MAE:** {_fmt(_v(m, 'mae'))} pct-pts | **RMSE:** {_fmt(_v(m, 'rmse'))} | "
          f"**Bias (mean predicted - actual):** {_fmt(m.get('bias'))} | **Correlation:** {_fmt(_v(m, 'correlation'))}",
          f"- **No-change baseline MAE:** {_fmt(m['baseline_zero_change']['mae'])} "
          f"(skill {_fmt(m.get('mae_skill_vs_zero'), True)})",
          f"- **Errors within 1-sigma band:** {_fmt(_v(m, 'within_1sigma'), True)}", "",
          "## Prediction distribution", ""]
    for h, mm in r["metrics_by_horizon"].items():
        pdist = mm["prediction_distribution"]
        L += [f"### {h}", "",
              "| | UP | DOWN | NEUTRAL |", "|---|---|---|---|",
              "| Predicted | " + " | ".join(f"{pdist['predicted'][d]['count']} ({_fmt(pdist['predicted'][d]['share'], True)})"
                                         for d in DIRECTIONS) + " |",
              "| Actual | " + " | ".join(f"{pdist['actual'][d]['count']} ({_fmt(pdist['actual'][d]['share'], True)})"
                                      for d in DIRECTIONS) + " |", "",
              "Confusion matrix (rows predicted, columns actual; NEUTRAL = inside the same neutral band):", "",
              *_confusion_lines(mm["confusion_matrix"]), "",
              "Mean contribution per model component (sign shares show any one-sided push):", "",
              "| Component | Mean | Share > 0 | Share < 0 |", "|---|---|---|---|"]
        for k, v in r["contribution_summary_by_horizon"].get(h, {}).items():
            L.append(f"| {k} | {v['mean']:.4f} | {_fmt(v['share_positive'], True)} | {_fmt(v['share_negative'], True)} |")
        L.append("")
    L += ["## Baseline comparison", ""]
    for h, mm in r["metrics_by_horizon"].items():
        L += [f"### {h}", "", *_baseline_lines(mm.get("baseline_comparison")), ""]
    L += ["## Results by horizon", "", *_metric_table(r["metrics_by_horizon"]), "",
          "## Feature analysis (next-session horizon)", "", *_feature_lines(r.get("feature_analysis_next_session")), "",
          "## Company-wise results", "",
          "| Company | Horizon | Predictions | MAE | RMSE | Directional acc. | Sign acc. (n) | Bias | "
          "No-change MAE | Predicted UP/DOWN/NEUTRAL | Actual UP/DOWN/NEUTRAL |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for t, by_h in r["company_summary"].items():
        for h, c in by_h.items():
            L.append(f"| {t} | {h} | {c['n']} | {_fmt(c['mae'])} | {_fmt(c['rmse'])} | "
                     f"{_fmt(c['directional_accuracy'], True)} | {_fmt(c['sign_accuracy']['value'], True)} "
                     f"({c['sign_accuracy']['n']}) | {_fmt(c['bias'])} | {_fmt(c['no_change_mae'])} | "
                     f"{'/'.join(str(c['predicted'][d]) for d in DIRECTIONS)} | "
                     f"{'/'.join(str(c['actual'][d]) for d in DIRECTIONS)} |")
    L += ["", "## Time-period results (by month of the prediction date)", ""]
    for h, by_m in r["monthly_results"].items():
        L += [f"### {h}", "", *_metric_table(by_m), ""]
    L += ["## Parameter changes", ""]
    if r["parameter_changes"]:
        L += ["| Parameter | Old | New | Reason |", "|---|---|---|---|"]
        L += [f"| {c['parameter_name']} | {c['old_value']} | {c['new_value']} | {c['reason']} |"
              for c in r["parameter_changes"]]
    else:
        L.append("No parameter changes proposed in this run.")
    t = r.get("tuning")
    if t:
        L += ["", "### Tuning", "",
              f"- Objective: `{t['objective_name']}` on horizon `{t['tuning_horizon']}`, {t['n_trials']} trials",
              f"- Split: {t.get('split_method', '')}",
              f"- Train (selection): {t['train_dates'][0]} to {t['train_dates'][1]} ({t['train_dates'][2]} dates); "
              f"validation (gate): {t['validation_dates'][0]} to {t['validation_dates'][1]} ({t['validation_dates'][2]} dates)"
              + (f"; untouched test: {t['test_dates'][0]} to {t['test_dates'][1]} ({t['test_dates'][2]} dates)"
                 if t.get("test_dates") and t["test_dates"][2] else "; no test period"),
              f"- Active params - train MAE {_fmt(_v(t['baseline_train'], 'mae'))}, "
              f"validation MAE {_fmt(_v(t['baseline_validation'], 'mae'))}, "
              f"test MAE {_fmt(_v(t.get('baseline_test') or {}, 'mae'))}",
              f"- Best trial - train MAE {_fmt(_v(t['best_train'], 'mae'))}, "
              f"validation MAE {_fmt(_v(t['best_validation'], 'mae'))}, "
              f"test MAE {_fmt(_v(t.get('best_test') or {}, 'mae'))} "
              f"(no-change test MAE {_fmt((t.get('best_test') or {}).get('baseline_zero_change', {}).get('mae'))})",
              f"- **Recommended:** {'yes' if t['recommended'] else 'no'} - {t['reason']}",
              "- Candidates are never activated automatically: use `POST /parameters/{version}/activate`."]
    L += ["", "## Failure cases (largest absolute errors)", "",
          "| Date | Ticker | Horizon | Predicted % | Actual % | Error | Sentiment | News | Top evidence |",
          "|---|---|---|---|---|---|---|---|---|"]
    for f in r["failure_cases"]:
        ev = "; ".join(e[:70] for e in f["evidence"]) or "-"
        L.append(f"| {f['as_of_date']} | {f['ticker']} | {f['horizon']} | {f['predicted_pct']:.2f} | "
                 f"{f['actual_pct']:.2f} | {f['error_pct_points']:.2f} | {f['sentiment_score']:.1f} | "
                 f"{f['news_count']} | {ev.replace('|', '/')} |")
    L += ["", "## Important observations", ""] + [f"- {o}" for o in r["observations"]]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------------- experiment md
def _score_row(name: str, s: dict) -> str:
    if not s or not s.get("n"):
        return f"| {name} | 0 | n/a | n/a | n/a | n/a | n/a | n/a |"
    sa = s["sign_accuracy"]
    return (f"| {name} | {s['n']} | {_fmt(s['mae'])} | {_fmt(s['rmse'])} | {_fmt(s['directional_accuracy'], True)} | "
            f"{_fmt(sa['value'], True)} ({sa['n']}, p={_fmt(sa.get('p_value_vs_coinflip'))}) | {_fmt(s['bias'])} | "
            f"{s['predicted']['UP']}/{s['predicted']['DOWN']}/{s['predicted']['NEUTRAL']} |")


SCORE_HEAD = ["| Forecast | N | MAE | RMSE | Directional acc. | Sign acc. (n, p) | Bias | Pred. UP/DOWN/NEUTRAL |",
              "|---|---|---|---|---|---|---|---|"]


def experiment_markdown(x: dict) -> str:
    P = x["periods"]
    dq = x.get("data_quality")
    L = [f"# Out-of-sample experiment `{x['experiment_id']}` (next-session horizon)", "", DISCLAIMER, "",
         "## Methodology", "",
         f"- **Period:** {x['period']['start']} to {x['period']['end']}; parameters `{x['param_version']}`; "
         f"news availability mode `{x['availability_mode']}`",
         f"- **Split ({x['split_method']}), never shuffled:** "
         + "; ".join(f"{k} {v.get('start')} to {v.get('end')} ({v.get('sessions', 0)} sessions, {v['rows']} rows)"
                     for k, v in P.items()),
         "- **Prediction cut-off:** 09:15 IST on the predicted session; news must be available by then "
         "(date-only timestamps count from the end of their date); closes strictly before the session.",
         "- **Target:** (close[session] - close[previous session]) / close[previous session] x 100, "
         "adjusted closes, NSE sessions present in the price data (weekends/holidays excluded).",
         "- **Baselines** (from closes before the session only): no change; pooled historical mean daily "
         "return of all 10 stocks; the company's historical mean; the company's 20-session rolling mean.",
         "- **Selection:** ridge penalty per feature set and the selected feature set are chosen on VALIDATION; "
         "TEST is scored once afterwards. Walk-forward: expanding monthly window, penalty chosen on the "
         "preceding month.", "",
         "## Data quality", "", *_data_quality_lines(dq), "",
         f"- **Verified-news-only replay:** {x['verified_news_only']['rows']} prediction rows, of which "
         f"{x['verified_news_only']['rows_with_any_new_company_news']} had any new company news that this "
         f"system held before the cut-off.", "",
         "## Before vs after (fixed-formula models)", ""]
    for part in ("train", "validation", "test"):
        if not P[part]["rows"]:
            continue
        L += [f"### {part.capitalize()} ({P[part]['start']} to {P[part]['end']})", "", *SCORE_HEAD]
        for name, ref in x["reference_models"].items():
            L.append(_score_row(name, ref[part]))
        for b, s in x["baselines"][part].items():
            L.append(_score_row(f"baseline: {BASELINE_LABELS.get(b, b)}", s))
        L.append("")
    L += ["## Prediction bias diagnosis (all dates)", ""]
    for name, m in x["diagnostics"].items():
        if not m.get("n"):
            continue
        L += [f"### {name}", "", *_confusion_lines(m["confusion_matrix"]), "",
              f"Mean predicted {_fmt(m['mean_predicted'])}% vs mean actual {_fmt(m['mean_actual'])}%; "
              f"bias {_fmt(m['bias'])}.", ""]
    for name, by_t in x.get("diagnostics_by_company", {}).items():
        L += [f"### Per company: {name}", "",
              "| Company | N | Pred UP/DOWN/NEUTRAL | Actual UP/DOWN/NEUTRAL | UP->UP | UP->DOWN | DOWN->UP | DOWN->DOWN | "
              "Sign acc. (n) | Bias |", "|---|---|---|---|---|---|---|---|---|---|"]
        for t, m in by_t.items():
            if not m.get("n"):
                continue
            pdist, cm = m["prediction_distribution"], m["confusion_matrix"]
            L.append(f"| {t} | {m['n']} | " + "/".join(str(pdist["predicted"][d]["count"]) for d in DIRECTIONS)
                     + " | " + "/".join(str(pdist["actual"][d]["count"]) for d in DIRECTIONS)
                     + f" | {cm['UP']['UP']} | {cm['UP']['DOWN']} | {cm['DOWN']['UP']} | {cm['DOWN']['DOWN']} | "
                       f"{_fmt(m['sign_accuracy']['value'], True)} ({m['sign_accuracy']['n']}) | {_fmt(m['bias'])} |")
        L.append("")
    L += ["## Incremental feature sets (ridge regression)", "",
          "| Feature set | Alpha | Train MAE | Validation MAE | Test MAE | Test sign acc. (n) | Test bias |",
          "|---|---|---|---|---|---|---|"]
    for name, mm in x["models"].items():
        te = mm["test"]
        L.append(f"| {name} | {mm['alpha']} | {_fmt(mm['train']['mae'])} | {_fmt(mm['validation']['mae'])} | "
                 f"{_fmt(te.get('mae'))} | {_fmt((te.get('sign_accuracy') or {}).get('value'), True)} "
                 f"({(te.get('sign_accuracy') or {}).get('n', 0)}) | {_fmt(te.get('bias'))} |")
    nb = {p: x["baselines"][p]["no_change"]["mae"] for p in x["baselines"]}
    L += ["", f"No-change MAE: train {_fmt(nb.get('train'))}, validation {_fmt(nb.get('validation'))}, "
              f"test {_fmt(nb.get('test'))}.", "",
          f"**Selected on validation:** `{x['selected_feature_set']}` ({x['selection_rule']}).", ""]
    sel = x["models"][x["selected_feature_set"]]
    L += ["### Selected model vs baselines", ""]
    for part in ("validation", "test"):
        if P[part]["rows"]:
            L += [f"**{part.capitalize()}**", "", *SCORE_HEAD, _score_row("selected ridge model", sel[part])]
            L += [_score_row(f"baseline: {BASELINE_LABELS.get(b, b)}", s) for b, s in x["baselines"][part].items()]
            L.append("")
    L += ["Is the improvement real? Paired, date-clustered one-sided t test on |model error| - |baseline error| "
          "(negative mean = model better):", "",
          "| Period | Baseline | MAE lower? | Mean loss diff | t | p (improvement) | Significant |",
          "|---|---|---|---|---|---|---|"]
    for part, by_b in x["selected_beats_baselines"].items():
        for b, v in by_b.items():
            L.append(f"| {part} | {BASELINE_LABELS.get(b, b)} | {'yes' if v['mae_lower'] else 'no'} | "
                     f"{_fmt(v['mean_loss_diff'], nd=5)} | {_fmt(v['t'])} | {_fmt(v['p_value_improvement'])} | "
                     f"{'yes' if v['significant'] else 'no'} |")
    L += ["", f"Rule: {x.get('recommendation_rule', '')}", ""]
    L += ["Coefficients (per 1 std of the feature, pct-pts):", "", "| Feature | Coefficient |", "|---|---|"]
    L += [f"| {f} | {c:+.4f} |" for f, c in sel["coefficients"].items()] or ["| (none) | |"]
    L += ["", "## Direction model (L2 logistic, P(return > 0))", "",
          "| Feature set | Validation acc. (n) | Test acc. (n, p vs 50%) | Test majority-class acc. | Test Brier | "
          "Test Brier (base rate) |", "|---|---|---|---|---|---|"]
    for name, mm in x["models"].items():
        dm = mm["direction_model"]
        v, t = dm["validation"], dm["test"]
        L.append(f"| {name} | {_fmt(v.get('accuracy'), True)} ({v.get('n')}) | {_fmt(t.get('accuracy'), True)} "
                 f"({t.get('n')}, p={_fmt(t.get('p_value_vs_coinflip'))}) | {_fmt(t.get('majority_class_accuracy'), True)} | "
                 f"{_fmt(t.get('brier'), nd=4)} | {_fmt(t.get('brier_base_rate'), nd=4)} |")
    L += ["", "## Walk-forward (expanding window, monthly)", ""]
    for label, wf in x["walk_forward"].items():
        L += [f"### {label} feature set", "",
              "| Month | Alpha | Model MAE | No-change MAE | Hist. mean MAE | Company mean MAE | Rolling mean MAE | "
              "Model sign acc. (n) |", "|---|---|---|---|---|---|---|---|"]
        for w in wf:
            b = w["baselines"]
            L.append(f"| {w['month']} | {w['alpha']} | {_fmt(w['model']['mae'])} | {_fmt(b['no_change']['mae'])} | "
                     f"{_fmt(b['historical_mean']['mae'])} | {_fmt(b['company_mean']['mae'])} | "
                     f"{_fmt(b['rolling_mean_20d']['mae'])} | {_fmt(w['model']['sign_accuracy']['value'], True)} "
                     f"({w['model']['sign_accuracy']['n']}) |")
        L.append("")
    L += ["## Feature analysis (TRAIN period only)", "", *_feature_lines(x["feature_analysis"]["train"]), "",
          "## Feature analysis (all dates)", "", *_feature_lines(x["feature_analysis"]["all"]), "",
          "## Recommendation", "", f"- {x['recommendation']}",
          f"- Candidate parameter set: `{x['candidate_created']}`" if x["candidate_created"] else
          "- No parameter set was created or activated.",
          "- Activation is always explicit (`POST /parameters/{version}/activate`)."]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------ audit md
def audit_markdown(a: dict) -> str:
    o = a["overall"]
    pct = (lambda x: _fmt(x, True))
    L = ["# News availability audit", "",
         f"Scope: first usable session {a['generated_for']['start'] or 'all'} to {a['generated_for']['end'] or 'all'}; "
         f"sentiment engine `{a['generated_for']['engine']}`.", "",
         "Only **verified_pre_market** articles (collected by this system before the 09:15 IST cut-off of the "
         "session they are used for) count as VERIFIED. Everything else is UNCERTAIN for live use.", "",
         "## Overall", "",
         f"- **Total articles:** {o['articles']}",
         f"- **Verified pre-market:** {o['status']['verified_pre_market']} ({pct(o['verified_pre_market_share'])})",
         f"- **Unknown availability:** {o['status']['unknown']} ({pct(o['unknown_share'])})",
         f"- **Post-event (collected after the session closed):** {o['status']['collected_after_event']} "
         f"({pct(o['post_event_share'])})",
         f"- **Published before cut-off, collected during the session:** {o['status']['published_before_cutoff']}",
         f"- **Rejected at collection:** future timestamp {o['rejected_future']}, missing timestamp "
         f"{o['rejected_missing_timestamp']} ({o['rejected_note']})",
         f"- **Duplicate rate:** {pct(o['duplicate_rate'])}",
         "- **Timestamp precision:** " + ", ".join(f"{k} {v} ({pct(o['precision_share'][k])})"
                                                  for k, v in o["precision"].items()),
         f"- **Unknown timezone:** {o['timezone_unknown']}", "",
         "## By provider", "",
         "| Provider | Articles | Exact | Minute | Hour | Date-only | Unknown | Verified pre-market | Unknown avail. | "
         "Post-event | Duplicates | Rejected future / missing ts | Timestamp quality |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p, v in a["by_provider"].items():
        ps, rj = v["precision_share"], v["rejected_at_collection"]
        L.append(f"| {p} | {v['articles']} | {pct(ps['exact'])} | {pct(ps['minute'])} | {pct(ps['hour'])} | "
                 f"{pct(ps['date_only'])} | {pct(ps['unknown'])} | {pct(v['verified_pre_market_share'])} | "
                 f"{pct(v['unknown_share'])} | {pct(v['post_event_share'])} | {pct(v['duplicate_rate'])} | "
                 f"{rj.get('future', 0)} / {rj.get('missing_timestamp', 0)} | {v['timestamp_quality']} |")
    L += ["", "## By company / entity", "",
          "| Entity | News | Verified pre-market | Unknown | Post-event | Published before cut-off |",
          "|---|---|---|---|---|---|"]
    for e, v in a["by_company"].items():
        L.append(f"| {e} | {v['total']} | {v['verified_pre_market']} | {v['unknown']} | {v['collected_after_event']} | "
                 f"{v['published_before_cutoff']} |")
    L += ["", "## By trading date (first session each article can be used for)", "",
          "| Session | Verified pre-market | Unknown | Post-event | Published before cut-off | Live prediction generated |",
          "|---|---|---|---|---|---|"]
    for r in a["by_trading_date"]:
        L.append(f"| {r['session']} | {r['verified_pre_market']} | {r['unknown']} | {r['collected_after_event']} | "
                 f"{r['published_before_cutoff']} | {'yes' if r['live_prediction_generated'] else 'no'} |")
    return "\n".join(L) + "\n"


def write_audit_report(a: dict):
    import json
    from datetime import date as _date
    from backend.config.settings import get_settings
    out = get_settings().reports_dir
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"audit-{_date.today().isoformat()}"
    md = stem.with_suffix(".md")
    md.write_text(audit_markdown(a), encoding="utf-8")
    stem.with_suffix(".json").write_text(json.dumps(a, indent=2, default=str), encoding="utf-8")
    return md
