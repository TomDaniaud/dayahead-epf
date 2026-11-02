# dayahead-epf

Day-ahead electricity price forecasting (LightGBM, N-BEATS) with walk-forward backtesting and drift-triggered retraining.

[![CI](https://github.com/TomDaniaud/dayahead-epf/actions/workflows/ci.yml/badge.svg)](https://github.com/TomDaniaud/dayahead-epf/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Forecasts the 24 hourly prices of the next day's power auction. The project focuses on three questions:

1. **How accurate is the model, day after day, using only information available before gate closure?** -> walk-forward backtest with no data leakage
2. **Is it really better than a simple benchmark?** -> standard forecast KPIs and a statistical significance test
3. **When should the model be retrained?** -> automatic detection of performance drift that triggers retraining

---

## At a glance

| | |
|---|---|
| **Problem** | Forecast the 24 hourly day-ahead prices of day D, at D-1 before the 12:00 auction gate closure |
| **Data** | ENTSO-E Transparency Platform: day-ahead prices + day-ahead load, wind and solar forecasts (any bidding zone, e.g. FR, DE-LU) |
| **Models** | Naive benchmark · LightGBM · N-BEATS with exogenous inputs (PyTorch) |
| **Evaluation** | Daily walk-forward backtest · MAE, RMSE, rMAE, bias, tail error · Diebold-Mariano test |
| **Monitoring** | Page-Hinkley drift detection on daily errors -> retraining; PSI on input features |
| **Engineering** | Installable package · CLI · 17 unit tests incl. leakage tests · ruff · GitHub Actions CI |

## Quick start

```bash
git clone https://github.com/TomDaniaud/dayahead-epf.git && cd dayahead-epf
pip install -e ".[deep,dev]"
pytest -q          # 17 tests, ~10 s
epf demo           # full benchmark on a synthetic market, no API key needed (~1 min)
```

On real data (free ENTSO-E API token):

```bash
pip install -e ".[entsoe]"
export ENTSOE_API_KEY=...
epf download --country FR --start 2019-01-01 --end 2026-11-01 --out data/fr.csv
epf backtest --data data/fr.csv --models naive,lgbm,nbeats --test-start 2025-01-01 --retrain periodic+drift
```

Each run writes to `results/`: a KPI table, hourly forecasts per model, the retraining log (date, reason, drift statistics) and an error plot.

## Results (synthetic market)

`epf demo` runs on a simulated market - price as a merit-order function of residual load (load - wind - solar), a fuel-cost random walk, price spikes and negative prices - with a **regime break on 21 January 2023** (fuel price jump and steeper merit order, similar to the 2021-22 gas crisis). Test period: July 2022 - January 2024, 2-year rolling training window.

| Model | MAE (€/MWh) | rMAE vs naive | Significantly better than naive? |
|---|---|---|---|
| Naive benchmark | 7.34 | 1.00 | - |
| **LightGBM** | **4.95** | **0.68** | yes, p < 0.001 |
| N-BEATS | 5.18 | 0.71 | yes, p < 0.001 |

Retraining policy comparison (LightGBM):

| Policy | rMAE | Number of retrains |
|---|---|---|
| Never retrain | 0.74 | 0 |
| Every 30 days | 0.68 | 18 |
| **On drift only** | **0.70** | **3** |
| Every 30 days + on drift | 0.67 | 19 |

**Takeaway:** the drift detector fires 1 to 3 days after the regime break, and retraining only on drift keeps most of the benefit of monthly retraining with 6x fewer model fits.

![Daily MAE over the test period, with drift-triggered retrains](results/daily_mae.png)

> These results validate the pipeline, not the models' real-world accuracy. A study on real ENTSO-E data (France and Germany-Luxembourg, 2019-2026, with 2022 as a natural drift test) is the next step.

## Key design decisions

**No look-ahead, by construction and by test.** For day D the model sees prices up to D-1 and the *forecasts* of load, wind and solar for D, which are published before gate closure. Realised load or generation would leak future information. Two unit tests enforce this: changing the prices of day D leaves every input feature of D unchanged, and no training window ever contains a forecasted day. Scalers are refitted on the training window at each retrain.

**Robust to price-level shifts.** Tree models cannot extrapolate beyond the price levels seen in training. With raw prices, a LightGBM model that was not retrained was nearly 7x worse than the naive benchmark once prices left the training range (rMAE 6.9). Expressing the lagged prices and the target relative to the previous day's average price fixed it (rMAE 0.74 without any retraining). The same local normalisation is used in N-BEATS.

**Spikes and negative prices.** Targets go through a median/MAD scaling followed by an inverse hyperbolic sine transform (Uniejewski et al., 2018), which, unlike a log transform, is defined for negative prices and dampens spikes.

**Metrics.** MAE in €/MWh is directly interpretable. rMAE (error relative to the naive benchmark) is scale-free, so it compares across markets and years; sMAPE is reported but unstable when prices are near zero. The 95th percentile of absolute errors tracks tail risk, and the bias shows systematic over- or under-forecasting. The multivariate Diebold-Mariano test (Ziel & Weron, 2018) checks that an improvement is not due to chance.

**Retraining driven by performance.** Each day the realised prices become known after the auction, so the day's error is observed before forecasting the next day. Daily errors, normalised by the error level just after the last retrain, feed a Page-Hinkley test; an alarm triggers retraining (with a minimum gap of 7 days). Feature drift (PSI on residual load) is logged as a diagnostic but does not trigger retraining on its own, because it also fires on normal seasonal changes such as summer solar.

**Market details handled.** Since October 2025 the European day-ahead auction clears in 15-minute periods; data is averaged to hourly. Daylight-saving days are normalised to 24 hours (missing hour interpolated, duplicated hour averaged).

## Repository layout

```
src/epf/
  data.py       ENTSO-E download, 15-min and DST handling, synthetic market
  features.py   leak-free feature table, one row per (day, hour)
  models.py     NaiveWeekly, LightGBMForecaster, NBEATSForecaster - common interface
  backtest.py   walk-forward engine and retraining policies
  drift.py      Page-Hinkley test, performance monitor, PSI
  metrics.py    forecast KPIs, Diebold-Mariano test
  cli.py        `epf` command line
tests/          pytest suite
```

Adding a model means implementing three methods (`prepare`, `fit`, `predict`) and registering it in `models.REGISTRY`.

## Roadmap

- Results on real ENTSO-E data (FR, DE-LU)
- Probabilistic forecasts: quantile LightGBM, conformal intervals, pinball loss
- Gas price (TTF) and cross-border flow features; LightGBM + N-BEATS ensemble
- Hyper-parameter search at each retrain (Optuna)

## References

- Lago, Marcjasz, De Schutter, Weron (2021). Forecasting day-ahead electricity prices: a review of state-of-the-art algorithms, best practices and an open-access benchmark. *Applied Energy*.
- Oreshkin, Carpov, Chapados, Bengio (2020). N-BEATS: neural basis expansion analysis for interpretable time series forecasting. *ICLR*.
- Olivares, Challu, Marcjasz, Weron, Dubrawski (2023). Neural basis expansion analysis with exogenous variables (NBEATSx). *International Journal of Forecasting*.
- Uniejewski, Weron, Ziel (2018). Variance stabilizing transformations for electricity spot price forecasting. *IEEE Transactions on Power Systems*.
- Ziel, Weron (2018). Day-ahead electricity price forecasting with high-dimensional structures. *Energy Economics*.

## Author

Tom Daniaud - final-year M.Eng. student in Data & AI, Polytech Nantes · [LinkedIn](https://www.linkedin.com/in/tom-daniaud) · [GitHub](https://github.com/TomDaniaud)
