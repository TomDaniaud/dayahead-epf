"""Command line interface.

    epf download --country FR --start 2019-01-01 --end 2026-11-01 --out data/fr.csv
    epf backtest --data data/fr.csv --models naive,lgbm,nbeats --test-start 2025-01-01
    epf demo                       # synthetic market, no API key needed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from epf.backtest import BacktestConfig, walk_forward
from epf.data import download_entsoe, load_csv, synthetic_market
from epf.metrics import diebold_mariano
from epf.models import get_model


def run(df: pd.DataFrame, models: list[str], cfg: BacktestConfig, out: Path) -> pd.DataFrame:
    out.mkdir(parents=True, exist_ok=True)
    results, rows = {}, []
    for name in models:
        r = walk_forward(get_model(name), df, cfg, verbose=True)
        results[name] = r
        r.to_long().to_csv(out / f"forecasts_{name}.csv")
        pd.DataFrame(r.retrains).to_csv(out / f"retrains_{name}.csv", index=False)
        rows.append({"model": name, **r.kpis})
    table = pd.DataFrame(rows).set_index("model").round(3)
    ref = results.get("naive")
    if ref is not None:
        for name, r in results.items():
            if name != "naive":
                table.loc[name, "DM_p_vs_naive"] = round(
                    diebold_mariano(r.actuals, ref.forecasts, r.forecasts)["p_value"], 4)
    table.to_csv(out / "kpis.csv")
    (out / "config.json").write_text(json.dumps(cfg.__dict__, indent=2, default=str))
    _plot(results, out)
    print(table.to_markdown() if _has_tabulate() else table)
    return table


def _plot(results, out):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(11, 4))
    for name, r in results.items():
        r.daily_mae.rolling(14).mean().plot(ax=ax, label=name, lw=1.4)
        if name != "naive":
            for e in r.retrains:
                if e["reason"] == "drift":
                    ax.axvline(e["day"], color="crimson", ls=":", lw=1)
    ax.set_ylabel("daily MAE, 14-day mean (EUR/MWh)")
    ax.set_title("Walk-forward error (dotted red: drift-triggered retrains)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "daily_mae.png", dpi=130)
    plt.close(fig)


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(prog="epf", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="download ENTSO-E data to CSV")
    d.add_argument("--country", default="FR")
    d.add_argument("--start", required=True)
    d.add_argument("--end", required=True)
    d.add_argument("--out", required=True)

    for name in ("backtest", "demo"):
        b = sub.add_parser(name)
        if name == "backtest":
            b.add_argument("--data", required=True)
            b.add_argument("--test-start", required=True)
        else:
            b.add_argument("--test-start", default="2022-07-01")
        b.add_argument("--test-end")
        b.add_argument("--models", default="naive,lgbm,nbeats")
        b.add_argument("--retrain", default="periodic+drift",
                       choices=["never", "periodic", "drift", "periodic+drift"])
        b.add_argument("--retrain-every", type=int, default=30)
        b.add_argument("--window", default="rolling", choices=["rolling", "expanding"])
        b.add_argument("--window-days", type=int, default=730)
        b.add_argument("--out", default="results")

    a = ap.parse_args(argv)
    if a.cmd == "download":
        df = download_entsoe(a.country, a.start, a.end)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(a.out)
        print(f"saved {len(df)} rows to {a.out}")
        return
    df = load_csv(a.data) if a.cmd == "backtest" else synthetic_market()
    cfg = BacktestConfig(test_start=a.test_start, test_end=a.test_end, retrain=a.retrain,
                         retrain_every=a.retrain_every, window=a.window,
                         window_days=a.window_days)
    run(df, a.models.split(","), cfg, Path(a.out))


if __name__ == "__main__":
    main()
