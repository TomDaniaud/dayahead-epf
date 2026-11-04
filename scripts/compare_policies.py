"""Compare retraining policies from backtest outputs and write summary tables.

Expects one `epf backtest` run per policy, with forecasts, in results/fr_<policy>/:

    for p in never drift periodic periodic+drift; do
      epf backtest --data data/fr.csv --test-start 2021-01-01 --retrain $p --out results/fr_${p/+/_}
    done
    python scripts/compare_policies.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from epf.metrics import diebold_mariano

RESULTS = Path("results")
OUT = RESULTS / "fr_comparison"
POLICIES = {"never": "fr_never", "drift": "fr_drift", "periodic": "fr_periodic",
            "periodic+drift": "fr_periodic_drift"}
MODELS = ["lgbm", "nbeats"]
PERIODS = {"all": ("2021-01-01", "2026-12-31"), "2021": ("2021-01-01", "2021-12-31"),
           "2022": ("2022-01-01", "2022-12-31"), "2023": ("2023-01-01", "2023-12-31"),
           "2024": ("2024-01-01", "2024-12-31"), "2025-26": ("2025-01-01", "2026-12-31"),
           "crisis": ("2021-07-01", "2023-06-30")}
PAIRS = [("never", "periodic"), ("never", "drift"), ("drift", "periodic"),
         ("periodic+drift", "periodic"), ("periodic", "periodic+drift")]


def load(policy: str, model: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Actual and forecast prices as (n_days, 24) matrices."""
    path = RESULTS / POLICIES[policy] / f"forecasts_{model}.csv"
    f = pd.read_csv(path, index_col=0, parse_dates=True)
    f["day"], f["hour"] = f.index.normalize(), f.index.hour
    return tuple(f.pivot(index="day", columns="hour", values=c) for c in ("actual", "forecast"))


def summary() -> pd.DataFrame:
    rows = []
    for policy, folder in POLICIES.items():
        kpis = pd.read_csv(RESULTS / folder / "kpis.csv", index_col=0)
        for model in MODELS:
            log = pd.read_csv(RESULTS / folder / f"retrains_{model}.csv")
            rows.append({"policy": policy, "model": model,
                         **kpis.loc[model, ["MAE", "RMSE", "rMAE", "bias", "MAE_p95",
                                            "fit_seconds", "DM_p_vs_naive"]].to_dict(),
                         "n_retrains": len(log), "n_drift": int((log.reason == "drift").sum())})
    return pd.DataFrame(rows)


def mae_by_period() -> pd.DataFrame:
    actual, naive = load("periodic", "naive")
    rows = [{"period": name, "policy": "-", "model": "naive",
             "MAE": (naive - actual).abs().loc[a:b].to_numpy().mean()}
            for name, (a, b) in PERIODS.items()]
    for policy in POLICIES:
        for model in MODELS:
            forecast = load(policy, model)[1]
            rows += [{"period": name, "policy": policy, "model": model,
                      "MAE": (forecast - actual).abs().loc[a:b].to_numpy().mean()}
                     for name, (a, b) in PERIODS.items()]
    return pd.DataFrame(rows).round(3)


def dm_tests() -> pd.DataFrame:
    """One-sided: a small p-value means ``better`` is significantly more accurate than ``base``."""
    rows = [{"period": "all", "model": "nbeats vs lgbm", "base": "nbeats", "better": "lgbm",
             "p_value": diebold_mariano(load("periodic", "lgbm")[0], load("periodic", "nbeats")[1],
                                        load("periodic", "lgbm")[1])["p_value"]}]
    for model in MODELS:
        for base, better in PAIRS:
            actual, f1 = load(base, model)
            f2 = load(better, model)[1]
            for name in ("all", "crisis"):
                a, b = PERIODS[name]
                p = diebold_mariano(actual.loc[a:b], f1.loc[a:b], f2.loc[a:b])["p_value"]
                rows.append({"period": name, "model": model, "base": base, "better": better,
                             "p_value": p})
    return pd.DataFrame(rows).round(4)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, table in [("policies", summary()), ("mae_by_period", mae_by_period()),
                        ("dm_tests", dm_tests())]:
        table.to_csv(OUT / f"{name}.csv", index=False)
        print(f"\n{name}\n{table.to_string(index=False)}")
