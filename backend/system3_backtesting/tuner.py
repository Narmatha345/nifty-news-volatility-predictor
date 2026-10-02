"""Parameter fine-tuning by seeded random search with a chronological TRAIN / VALIDATION / TEST split.

  * Evaluation dates are split in time order (never shuffled): train < validation < test.
    Default: first 60% of sessions train, next 20% validation, last 20% test; or explicit
    `train_end` / `valid_end` dates (e.g. train May-Jul, validation Aug, test Sep).
  * Each trial samples every parameter in `search_space` uniformly within bounds and is scored on
    TRAIN only. The best train trial is the only candidate.
  * The candidate is then scored on VALIDATION against the current (active) set and the no-change
    forecast.
  * Only after the candidate is fixed is the untouched TEST period evaluated - once, for the
    candidate and the active set. The test result cannot change WHICH parameters were chosen; it
    can only veto the recommendation.
  * RECOMMENDED only if the candidate (a) beats the active set on validation, (b) beats the
    no-change MAE on validation and (c) beats the no-change MAE on test with a statistically
    significant improvement (paired, date-clustered t test, p < 0.05 - see
    metrics.paired_improvement_test). A 0.001 MAE "win" is noise, not skill. A recommended set is stored
    as a `candidate`; it is never activated automatically.
The tuning horizon is the next session, whose target close is the prediction date itself, so a
training sample's outcome never falls inside the validation/test periods.
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from datetime import date

from backend.services.parameters import get_path, set_path
from backend.system3_backtesting.backtester import BacktestData, run_backtest
from backend.system3_backtesting.metrics import compute_metrics, objective, paired_improvement_test


@dataclass
class TuningResult:
    objective_name: str
    tuning_horizon: str
    n_trials: int
    train_dates: tuple[str, str, int]
    validation_dates: tuple[str, str, int]
    baseline_train: dict
    baseline_validation: dict
    best_params: dict
    best_train: dict
    best_validation: dict
    recommended: bool
    reason: str
    trials: list[dict] = field(default_factory=list)
    test_dates: tuple[str, str, int] | None = None
    baseline_test: dict = field(default_factory=dict)
    best_test: dict = field(default_factory=dict)
    split_method: str = ""


def chronological_split(dates: list[date], train_end: date | None = None, valid_end: date | None = None,
                        fractions: tuple[float, float] = (0.6, 0.2)) -> tuple[list, list, list, str]:
    ds = sorted(dates)
    if train_end and valid_end:
        if not train_end < valid_end:
            raise ValueError("train_end must be before valid_end")
        tr = [d for d in ds if d <= train_end]
        va = [d for d in ds if train_end < d <= valid_end]
        te = [d for d in ds if d > valid_end]
        how = f"explicit: train <= {train_end} < validation <= {valid_end} < test"
    else:
        a = int(len(ds) * fractions[0])
        b = int(len(ds) * (fractions[0] + fractions[1]))
        tr, va, te = ds[:a], ds[a:b], ds[b:]
        how = f"chronological fractions {fractions[0]:.0%} / {fractions[1]:.0%} / {1 - sum(fractions):.0%}"
    return tr, va, te, how


def _span(ds: list) -> tuple[str, str, int]:
    return (str(ds[0]), str(ds[-1]), len(ds)) if ds else ("", "", 0)


def _sample(base: dict, space: dict[str, list[float]], rng: random.Random) -> dict:
    p = copy.deepcopy(base)
    for path, (lo, hi) in space.items():
        set_path(p, path, round(rng.uniform(lo, hi), 4))
    return p


def tune(data: BacktestData, dates: list, base_params: dict, space: dict[str, list[float]],
         n_trials: int = 30, objective_name: str = "mae", tuning_horizon: str = "today",
         seed: int = 42, convention: str = "indian_fy", train_end: date | None = None,
         valid_end: date | None = None, fractions: tuple[float, float] = (0.6, 0.2),
         availability_mode: str = "provider_timestamp") -> TuningResult:
    train, valid, test, how = chronological_split(dates, train_end, valid_end, fractions)
    if len(train) < 5 or len(valid) < 3:
        raise ValueError(f"need >= 5 train and >= 3 validation dates to tune, got {len(train)} / {len(valid)}")
    horizons = (tuning_horizon,)

    last_records: dict = {}

    def score(params, ds):
        recs = run_backtest(data, ds, params, horizons, convention, availability_mode)
        last_records["recs"] = recs
        m = compute_metrics(recs)
        return objective(m, objective_name), m

    # 1) selection: TRAIN only
    base_train_obj, base_train = score(base_params, train)
    rng = random.Random(seed)
    best = (base_train_obj, base_params, base_train)
    trials = []
    for i in range(n_trials):
        cand = _sample(base_params, space, rng)
        obj, m = score(cand, train)
        trials.append({"trial": i, "train_objective": obj,
                       "params": {k: get_path(cand, k) for k in space}})
        if obj < best[0]:
            best = (obj, cand, m)
    chosen = best[1]
    # 2) gate: VALIDATION
    base_valid_obj, base_valid = score(base_params, valid)
    best_valid_obj, best_valid = score(chosen, valid)
    zero_valid = best_valid.get("baseline_zero_change", {}).get("mae")
    improved = chosen is not base_params and best_valid_obj < base_valid_obj
    beats_zero_valid = zero_valid is not None and best_valid["mae"]["value"] < zero_valid
    # 3) confirmation: TEST (evaluated once, after the choice is fixed)
    base_test, best_test = {}, {}
    beats_zero_test = False
    if test:
        _, base_test = score(base_params, test)
        _, best_test = score(chosen, test)
        trecs = last_records["recs"]
        sig = paired_improvement_test([r.predicted for r in trecs], [0.0] * len(trecs), [r.actual for r in trecs],
                                      [r.as_of_date for r in trecs])
        best_test["significance_vs_no_change"] = sig
        zero_test = best_test.get("baseline_zero_change", {}).get("mae")
        beats_zero_test = zero_test is not None and best_test.get("n", 0) > 0 \
            and best_test["mae"]["value"] < zero_test and sig["significant"]

    if chosen is base_params:
        reason = "No trial beat the active parameters on the training period."
    elif not improved:
        reason = (f"Best train trial did NOT improve {objective_name} on validation "
                  f"({base_valid_obj:.4f} -> {best_valid_obj:.4f}); likely overfit. Not recommended.")
    elif not beats_zero_valid:
        reason = (f"Best train trial improved {objective_name} on validation ({base_valid_obj:.4f} -> "
                  f"{best_valid_obj:.4f}) but its validation MAE {best_valid['mae']['value']:.4f} does not beat "
                  f"the no-change baseline ({zero_valid:.4f}). Not recommended.")
    elif not test:
        reason = "Passed validation, but there is no untouched test period to confirm it. Not recommended."
    elif not beats_zero_test:
        reason = (f"Passed validation, but on the untouched test period its MAE "
                  f"{best_test.get('mae', {}).get('value')} does not significantly beat the no-change baseline "
                  f"({best_test.get('baseline_zero_change', {}).get('mae')}; paired test p="
                  f"{best_test.get('significance_vs_no_change', {}).get('p_value_improvement')}). Not recommended.")
    else:
        reason = (f"Best train trial improved {objective_name} on validation ({base_valid_obj:.4f} -> "
                  f"{best_valid_obj:.4f}) and beats the no-change MAE on validation and on the untouched "
                  f"test period ({best_test['mae']['value']:.4f} vs {best_test['baseline_zero_change']['mae']:.4f}).")
    recommended = improved and beats_zero_valid and bool(test) and beats_zero_test
    return TuningResult(objective_name, tuning_horizon, n_trials, _span(train), _span(valid),
                        base_train, base_valid, chosen, best[2], best_valid, recommended, reason,
                        sorted(trials, key=lambda t: t["train_objective"])[:10], _span(test),
                        base_test, best_test, how)
