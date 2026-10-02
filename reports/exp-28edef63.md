# Out-of-sample experiment `exp-28edef63` (next-session horizon)

> Research output. Predicted movements are model estimates, not guaranteed outcomes. Historical performance does not guarantee future performance.

## Methodology

- **Period:** 2026-05-01 to 2026-09-30; parameters `default-v3`; news availability mode `provider_timestamp`
- **Split (explicit: train <= 2026-07-31 < validation <= 2026-08-31 < test), never shuffled:** train 2026-05-01 to 2026-07-31 (66 sessions, 660 rows); validation 2026-08-03 to 2026-08-31 (21 sessions, 210 rows); test 2026-09-01 to 2026-09-30 (22 sessions, 220 rows)
- **Prediction cut-off:** 09:15 IST on the predicted session; news must be available by then (date-only timestamps count from the end of their date); closes strictly before the session.
- **Target:** (close[session] - close[previous session]) / close[previous session] x 100, adjusted closes, NSE sessions present in the price data (weekends/holidays excluded).
- **Baselines** (from closes before the session only): no change; pooled historical mean daily return of all 10 stocks; the company's historical mean; the company's 20-session rolling mean.
- **Selection:** ridge penalty per feature set and the selected feature set are chosen on VALIDATION; TEST is scored once afterwards. Walk-forward: expanding monthly window, penalty chosen on the preceding month.

## Data quality

- **Total news articles** (available up to the last cut-off): 17925
- **Verified pre-market** (collected before the cut-off they can be used for): 0
- **Published before cut-off, collected during the session:** 0
- **Post-event** (collected after the session they would predict had closed): 1472
- **Unknown availability** (date-only timestamps, not verified): 16453
- **Coarse (date/hour-only) timestamps:** 0
- **Duplicate rate:** 4.0%
- **Price-report analyses** (headline only restates a price move): 2138 of 16329
- **Missing price data:** 0 ticker-sessions missing of 1308 (coverage 100.0%); index missing 4

- **Verified-news-only replay:** 1090 prediction rows, of which 0 had any new company news that this system held before the cut-off.

## Before vs after (fixed-formula models)

### Train (2026-05-01 to 2026-07-31)

| Forecast | N | MAE | RMSE | Directional acc. | Sign acc. (n, p) | Bias | Pred. UP/DOWN/NEUTRAL |
|---|---|---|---|---|---|---|---|
| factor-v2 relative sentiment (default-v3) | 660 | 1.091 | 1.554 | 26.4% | 49.1% (234, p=0.794) | 0.013 | 143/109/408 |
| previous model factor-v1 (default-v2, lexicon-v2) | 660 | 1.076 | 1.533 | 23.0% | 46.4% (194, p=0.315) | 0.125 | 203/0/457 |
| baseline: No change | 660 | 1.055 | 1.521 | 16.2% | n/a (0, p=n/a) | 0.005 | 0/0/660 |
| baseline: Historical mean (all stocks) | 660 | 1.055 | 1.522 | 16.2% | n/a (0, p=n/a) | 0.005 | 0/0/660 |
| baseline: Company historical mean | 660 | 1.059 | 1.525 | 16.2% | n/a (0, p=n/a) | 0.001 | 0/0/660 |
| baseline: Rolling mean (20 sessions) | 660 | 1.109 | 1.569 | 28.9% | 49.5% (376, p=0.837) | -0.015 | 200/198/262 |

### Validation (2026-08-03 to 2026-08-31)

| Forecast | N | MAE | RMSE | Directional acc. | Sign acc. (n, p) | Bias | Pred. UP/DOWN/NEUTRAL |
|---|---|---|---|---|---|---|---|
| factor-v2 relative sentiment (default-v3) | 210 | 1.019 | 1.348 | 25.7% | 45.1% (102, p=0.322) | -0.005 | 35/68/107 |
| previous model factor-v1 (default-v2, lexicon-v2) | 210 | 1.012 | 1.349 | 23.8% | 43.9% (98, p=0.225) | 0.152 | 98/0/112 |
| baseline: No change | 210 | 0.994 | 1.339 | 8.6% | n/a (0, p=n/a) | 0.041 | 0/0/210 |
| baseline: Historical mean (all stocks) | 210 | 0.995 | 1.340 | 8.6% | n/a (0, p=n/a) | 0.049 | 0/0/210 |
| baseline: Company historical mean | 210 | 0.992 | 1.337 | 11.4% | 66.7% (12, p=0.248) | 0.043 | 9/3/198 |
| baseline: Rolling mean (20 sessions) | 210 | 1.020 | 1.377 | 30.5% | 48.5% (130, p=0.726) | 0.107 | 86/46/78 |

### Test (2026-09-01 to 2026-09-30)

| Forecast | N | MAE | RMSE | Directional acc. | Sign acc. (n, p) | Bias | Pred. UP/DOWN/NEUTRAL |
|---|---|---|---|---|---|---|---|
| factor-v2 relative sentiment (default-v3) | 220 | 1.058 | 1.421 | 28.2% | 49.4% (87, p=0.915) | 0.245 | 30/62/128 |
| previous model factor-v1 (default-v2, lexicon-v2) | 220 | 1.069 | 1.419 | 17.3% | 34.5% (29, p=0.095) | 0.351 | 31/0/189 |
| baseline: No change | 220 | 1.041 | 1.398 | 14.5% | n/a (0, p=n/a) | 0.292 | 0/0/220 |
| baseline: Historical mean (all stocks) | 220 | 1.041 | 1.397 | 14.5% | n/a (0, p=n/a) | 0.288 | 0/0/220 |
| baseline: Company historical mean | 220 | 1.040 | 1.399 | 14.5% | 0.0% (1, p=n/a) | 0.283 | 0/1/219 |
| baseline: Rolling mean (20 sessions) | 220 | 1.050 | 1.408 | 37.3% | 56.7% (141, p=0.110) | 0.083 | 3/147/70 |

## Prediction bias diagnosis (all dates)

### factor-v2 relative sentiment (default-v3)

| Predicted \ Actual | UP | DOWN | NEUTRAL | Total |
|---|---|---|---|---|
| UP | 80 | 102 | 26 | 208 |
| DOWN | 96 | 111 | 32 | 239 |
| NEUTRAL | 253 | 291 | 99 | 643 |
| Total | 429 | 504 | 157 | 1090 |

Mean predicted -0.014% vs mean actual -0.070%; bias 0.056.

### previous model factor-v1 (default-v2, lexicon-v2)

| Predicted \ Actual | UP | DOWN | NEUTRAL | Total |
|---|---|---|---|---|
| UP | 124 | 167 | 41 | 332 |
| DOWN | 0 | 0 | 0 | 0 |
| NEUTRAL | 305 | 337 | 116 | 758 |
| Total | 429 | 504 | 157 | 1090 |

Mean predicted 0.105% vs mean actual -0.070%; bias 0.176.

### Per company: factor-v2 relative sentiment (default-v3)

| Company | N | Pred UP/DOWN/NEUTRAL | Actual UP/DOWN/NEUTRAL | UP->UP | UP->DOWN | DOWN->UP | DOWN->DOWN | Sign acc. (n) | Bias |
|---|---|---|---|---|---|---|---|---|---|
| AXISBANK | 109 | 21/20/68 | 41/49/19 | 6 | 12 | 7 | 9 | 46.2% (39) | -0.018 |
| BHARTIARTL | 109 | 17/25/67 | 47/50/12 | 5 | 9 | 10 | 12 | 47.5% (40) | 0.029 |
| HDFCBANK | 109 | 17/25/67 | 43/48/18 | 6 | 7 | 8 | 13 | 48.7% (39) | 0.022 |
| ICICIBANK | 109 | 20/35/54 | 50/48/11 | 9 | 10 | 11 | 19 | 56.9% (51) | -0.090 |
| INFY | 109 | 18/22/69 | 36/58/15 | 6 | 10 | 10 | 11 | 43.6% (39) | 0.103 |
| ITC | 108 | 14/16/78 | 43/51/14 | 6 | 6 | 6 | 9 | 55.2% (29) | 0.108 |
| KOTAKBANK | 1 | 1/0/0 | 1/0/0 | 1 | 0 | 0 | 0 | 100.0% (1) | -2.231 |
| LT | 109 | 24/35/50 | 43/52/14 | 12 | 10 | 16 | 12 | 48.2% (56) | 0.009 |
| M&M | 1 | 0/0/1 | 0/0/1 | 0 | 0 | 0 | 0 | n/a (0) | -0.127 |
| RELIANCE | 109 | 26/24/59 | 44/53/12 | 10 | 15 | 10 | 11 | 47.9% (48) | 0.159 |
| SBIN | 109 | 19/22/68 | 45/43/21 | 4 | 11 | 12 | 8 | 35.1% (37) | 0.069 |
| TCS | 108 | 31/15/62 | 36/52/20 | 15 | 12 | 6 | 7 | 50.0% (44) | 0.198 |

### Per company: previous model factor-v1 (default-v2, lexicon-v2)

| Company | N | Pred UP/DOWN/NEUTRAL | Actual UP/DOWN/NEUTRAL | UP->UP | UP->DOWN | DOWN->UP | DOWN->DOWN | Sign acc. (n) | Bias |
|---|---|---|---|---|---|---|---|---|---|
| AXISBANK | 109 | 24/0/85 | 41/49/19 | 9 | 13 | 0 | 0 | 45.8% (24) | 0.083 |
| BHARTIARTL | 109 | 50/0/59 | 47/50/12 | 23 | 22 | 0 | 0 | 51.0% (49) | 0.178 |
| HDFCBANK | 109 | 17/0/92 | 43/48/18 | 3 | 9 | 0 | 0 | 25.0% (16) | 0.117 |
| ICICIBANK | 109 | 56/0/53 | 50/48/11 | 21 | 28 | 0 | 0 | 43.4% (53) | 0.081 |
| INFY | 109 | 2/0/107 | 36/58/15 | 0 | 2 | 0 | 0 | 0.0% (2) | 0.170 |
| ITC | 108 | 8/0/100 | 43/51/14 | 2 | 5 | 0 | 0 | 28.6% (7) | 0.184 |
| KOTAKBANK | 1 | 0/0/1 | 1/0/0 | 0 | 0 | 0 | 0 | n/a (0) | -2.748 |
| LT | 109 | 106/0/3 | 43/52/14 | 41 | 52 | 0 | 0 | 47.6% (103) | 0.320 |
| M&M | 1 | 0/0/1 | 0/0/1 | 0 | 0 | 0 | 0 | n/a (0) | -0.102 |
| RELIANCE | 109 | 42/0/67 | 44/53/12 | 18 | 21 | 0 | 0 | 46.3% (41) | 0.276 |
| SBIN | 109 | 13/0/96 | 45/43/21 | 2 | 7 | 0 | 0 | 33.3% (12) | 0.160 |
| TCS | 108 | 14/0/94 | 36/52/20 | 5 | 8 | 0 | 0 | 42.9% (14) | 0.216 |

## Incremental feature sets (ridge regression)

| Feature set | Alpha | Train MAE | Validation MAE | Test MAE | Test sign acc. (n) | Test bias |
|---|---|---|---|---|---|---|
| intercept only (training mean) | 0.0 | 1.055 | 0.993 | 1.040 | n/a (0) | 0.287 |
| + raw sentiment | 10.0 | 1.055 | 0.993 | 1.040 | n/a (0) | 0.287 |
| + relative sentiment | 0.001 | 1.059 | 0.989 | 1.039 | 75.0% (8) | 0.301 |
| + news features | 10.0 | 1.054 | 0.992 | 1.040 | n/a (0) | 0.297 |
| recommended initial set | 0.1 | 1.055 | 0.989 | 1.044 | 58.0% (50) | 0.316 |
| market features only | 10.0 | 1.054 | 0.993 | 1.042 | n/a (0) | 0.290 |
| news + market features | 10.0 | 1.053 | 0.992 | 1.042 | n/a (0) | 0.301 |
| full | 10.0 | 1.051 | 0.992 | 1.047 | 100.0% (1) | 0.325 |

No-change MAE: train 1.055, validation 0.994, test 1.041.

**Selected on validation:** `+ relative sentiment` (lowest VALIDATION MAE among non-trivial feature sets (test not used)).

### Selected model vs baselines

**Validation**

| Forecast | N | MAE | RMSE | Directional acc. | Sign acc. (n, p) | Bias | Pred. UP/DOWN/NEUTRAL |
|---|---|---|---|---|---|---|---|
| selected ridge model | 210 | 0.989 | 1.329 | 10.9% | 54.5% (11, p=0.763) | 0.049 | 5/6/199 |
| baseline: No change | 210 | 0.994 | 1.339 | 8.6% | n/a (0, p=n/a) | 0.041 | 0/0/210 |
| baseline: Historical mean (all stocks) | 210 | 0.995 | 1.340 | 8.6% | n/a (0, p=n/a) | 0.049 | 0/0/210 |
| baseline: Company historical mean | 210 | 0.992 | 1.337 | 11.4% | 66.7% (12, p=0.248) | 0.043 | 9/3/198 |
| baseline: Rolling mean (20 sessions) | 210 | 1.020 | 1.377 | 30.5% | 48.5% (130, p=0.726) | 0.107 | 86/46/78 |

**Test**

| Forecast | N | MAE | RMSE | Directional acc. | Sign acc. (n, p) | Bias | Pred. UP/DOWN/NEUTRAL |
|---|---|---|---|---|---|---|---|
| selected ridge model | 220 | 1.039 | 1.393 | 16.8% | 75.0% (8, p=n/a) | 0.301 | 5/4/211 |
| baseline: No change | 220 | 1.041 | 1.398 | 14.5% | n/a (0, p=n/a) | 0.292 | 0/0/220 |
| baseline: Historical mean (all stocks) | 220 | 1.041 | 1.397 | 14.5% | n/a (0, p=n/a) | 0.288 | 0/0/220 |
| baseline: Company historical mean | 220 | 1.040 | 1.399 | 14.5% | 0.0% (1, p=n/a) | 0.283 | 0/1/219 |
| baseline: Rolling mean (20 sessions) | 220 | 1.050 | 1.408 | 37.3% | 56.7% (141, p=0.110) | 0.083 | 3/147/70 |

Is the improvement real? Paired, date-clustered one-sided t test on |model error| - |baseline error| (negative mean = model better):

| Period | Baseline | MAE lower? | Mean loss diff | t | p (improvement) | Significant |
|---|---|---|---|---|---|---|
| validation | No change | yes | -0.00530 | -1.259 | 0.111 | no |
| validation | Historical mean (all stocks) | yes | -0.00654 | -1.601 | 0.062 | no |
| validation | Company historical mean | yes | -0.00380 | -0.555 | 0.292 | no |
| validation | Rolling mean (20 sessions) | yes | -0.03169 | -1.315 | 0.102 | no |
| test | No change | yes | -0.00205 | -0.476 | 0.320 | no |
| test | Historical mean (all stocks) | yes | -0.00200 | -0.455 | 0.327 | no |
| test | Company historical mean | yes | -0.00143 | -0.226 | 0.412 | no |
| test | Rolling mean (20 sessions) | yes | -0.01139 | -0.465 | 0.323 | no |

Rule: recommend only if the selected model's MAE improvement over EVERY baseline is statistically significant (paired, date-clustered one-sided t test, p < 0.05) on BOTH validation and test

Coefficients (per 1 std of the feature, pct-pts):

| Feature | Coefficient |
|---|---|
| relative_sentiment | +0.6026 |
| sentiment_zscore | -0.6747 |

## Direction model (L2 logistic, P(return > 0))

| Feature set | Validation acc. (n) | Test acc. (n, p vs 50%) | Test majority-class acc. | Test Brier | Test Brier (base rate) |
|---|---|---|---|---|---|
| intercept only (training mean) | 54.8% (210) | 64.1% (220, p=0.000) | 64.1% | 0.2427 | 0.2427 |
| + raw sentiment | 54.8% (210) | 64.1% (220, p=0.000) | 64.1% | 0.2427 | 0.2427 |
| + relative sentiment | 55.2% (210) | 63.6% (220, p=0.000) | 64.1% | 0.2431 | 0.2427 |
| + news features | 58.1% (210) | 59.6% (220, p=0.005) | 64.1% | 0.2444 | 0.2427 |
| recommended initial set | 54.8% (210) | 64.1% (220, p=0.000) | 64.1% | 0.2428 | 0.2427 |
| market features only | 54.8% (210) | 64.1% (220, p=0.000) | 64.1% | 0.2429 | 0.2427 |
| news + market features | 55.7% (210) | 60.0% (220, p=0.003) | 64.1% | 0.2451 | 0.2427 |
| full | 57.1% (210) | 53.2% (220, p=0.345) | 64.1% | 0.2483 | 0.2427 |

## Walk-forward (expanding window, monthly)

### full feature set

| Month | Alpha | Model MAE | No-change MAE | Hist. mean MAE | Company mean MAE | Rolling mean MAE | Model sign acc. (n) |
|---|---|---|---|---|---|---|---|
| 2026-07 | 10.0 | 1.088 | 1.086 | 1.086 | 1.088 | 1.127 | 100.0% (1) |
| 2026-08 | 10.0 | 0.992 | 0.994 | 0.995 | 0.992 | 1.020 | n/a (0) |
| 2026-09 | 10.0 | 1.042 | 1.041 | 1.041 | 1.040 | 1.050 | n/a (0) |

### selected feature set

| Month | Alpha | Model MAE | No-change MAE | Hist. mean MAE | Company mean MAE | Rolling mean MAE | Model sign acc. (n) |
|---|---|---|---|---|---|---|---|
| 2026-07 | 10.0 | 1.089 | 1.086 | 1.086 | 1.088 | 1.127 | n/a (0) |
| 2026-08 | 10.0 | 0.993 | 0.994 | 0.995 | 0.992 | 1.020 | n/a (0) |
| 2026-09 | 0.001 | 1.035 | 1.041 | 1.041 | 1.040 | 1.050 | 69.2% (13) |

## Feature analysis (TRAIN period only)

| Feature | N | Missing | Mean | Std | Corr. with return (p) | Rank corr. | Sign agreement (n) | Top-bottom quintile return | Informative (p<0.05) |
|---|---|---|---|---|---|---|---|---|---|
| raw_sentiment | 660 | 0 | 11.770 | 11.707 | -0.011 (0.778) | -0.035 | 48.8% (629) | -0.037 | no |
| relative_sentiment | 660 | 0 | 2.781 | 24.492 | -0.044 (0.263) | -0.054 | 49.3% (594) | -0.327 | no |
| sentiment_zscore | 660 | 0 | 0.093 | 0.778 | -0.051 (0.192) | -0.055 | 49.3% (594) | -0.278 | no |
| news_count | 660 | 0 | 1.629 | 0.763 | 0.031 (0.426) | 0.054 | 49.3% (594) | 0.099 | no |
| news_relevance | 660 | 0 | 0.654 | 0.164 | 0.018 (0.641) | 0.014 | 49.3% (594) | 0.076 | no |
| news_recency | 660 | 0 | 0.772 | 0.238 | -0.022 (0.569) | -0.040 | 49.3% (594) | n/a | no |
| macro_sentiment | 660 | 0 | -6.876 | 2.254 | -0.088 (0.023) | -0.067 | 50.6% (629) | -0.334 | yes |
| macro_relative | 660 | 0 | -4.638 | 15.209 | -0.038 (0.329) | -0.013 | 48.3% (629) | -0.325 | no |
| nifty_sentiment | 660 | 0 | 11.096 | 4.407 | -0.086 (0.026) | -0.084 | 49.4% (629) | -0.120 | yes |
| nifty_relative | 660 | 0 | 1.938 | 8.725 | -0.100 (0.010) | -0.052 | 46.9% (629) | -0.435 | yes |
| historical_volatility | 660 | 0 | 1.698 | 0.379 | 0.022 (0.570) | -0.018 | 49.4% (629) | 0.059 | no |
| beta | 660 | 0 | 0.951 | 0.248 | 0.003 (0.949) | 0.002 | 49.4% (629) | 0.073 | no |
| previous_return | 660 | 0 | -0.009 | 1.521 | 0.065 (0.093) | 0.070 | 51.7% (598) | 0.231 | no |
| rolling_return_5d | 660 | 0 | -0.118 | 3.408 | -0.079 (0.042) | -0.053 | 46.7% (628) | -0.167 | yes |
| rolling_return_20d | 660 | 0 | -0.451 | 6.218 | -0.051 (0.188) | -0.041 | 51.0% (629) | -0.084 | no |

Correlation is not causation; tests are not corrected for multiple comparisons.

## Feature analysis (all dates)

| Feature | N | Missing | Mean | Std | Corr. with return (p) | Rank corr. | Sign agreement (n) | Top-bottom quintile return | Informative (p<0.05) |
|---|---|---|---|---|---|---|---|---|---|
| raw_sentiment | 1090 | 0 | 11.465 | 12.800 | -0.003 (0.926) | -0.018 | 47.9% (1045) | -0.045 | no |
| relative_sentiment | 1090 | 0 | 0.608 | 24.687 | -0.033 (0.272) | -0.042 | 49.9% (985) | -0.261 | no |
| sentiment_zscore | 1090 | 0 | 0.025 | 0.760 | -0.043 (0.154) | -0.044 | 49.9% (985) | -0.292 | no |
| news_count | 1090 | 0 | 1.670 | 0.760 | 0.006 (0.836) | 0.035 | 46.0% (985) | 0.021 | no |
| news_relevance | 1090 | 0 | 0.654 | 0.165 | -0.009 (0.757) | -0.024 | 46.0% (985) | 0.024 | no |
| news_recency | 1090 | 0 | 0.764 | 0.242 | -0.033 (0.277) | -0.045 | 46.0% (985) | n/a | no |
| macro_sentiment | 1090 | 0 | -7.709 | 2.463 | -0.032 (0.291) | -0.015 | 53.6% (1045) | -0.086 | no |
| macro_relative | 1090 | 0 | -6.320 | 16.082 | -0.034 (0.269) | -0.020 | 49.4% (1045) | -0.221 | no |
| nifty_sentiment | 1090 | 0 | 10.480 | 4.159 | -0.041 (0.181) | -0.017 | 46.4% (1045) | -0.112 | no |
| nifty_relative | 1090 | 0 | 0.352 | 10.213 | -0.086 (0.005) | -0.060 | 45.8% (1045) | -0.388 | yes |
| historical_volatility | 1090 | 0 | 1.579 | 0.419 | 0.022 (0.463) | 0.005 | 46.4% (1045) | 0.087 | no |
| beta | 1090 | 0 | 0.959 | 0.250 | 0.017 (0.568) | 0.031 | 46.4% (1045) | 0.103 | no |
| previous_return | 1090 | 0 | -0.077 | 1.459 | -0.034 (0.256) | -0.044 | 47.1% (1001) | -0.066 | no |
| rolling_return_5d | 1090 | 0 | -0.317 | 3.116 | -0.063 (0.037) | -0.053 | 47.5% (1043) | -0.198 | yes |
| rolling_return_20d | 1090 | 0 | -0.876 | 5.926 | -0.034 (0.269) | -0.020 | 51.6% (1044) | -0.052 | no |

Correlation is not causation; tests are not corrected for multiple comparisons.

## Recommendation

- no candidate: the selected model does not SIGNIFICANTLY beat every baseline on both validation and test
- No parameter set was created or activated.
- Activation is always explicit (`POST /parameters/{version}/activate`).
