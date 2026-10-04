"""Hourly backtester that calls the exact same strategy functions as the live bot.

    python backtest.py --days 180                 # download Binance 1h data and run
    python backtest.py --days 180 --sweep         # small robustness grid
    python backtest.py --csv data/closes.csv      # reuse cached data

Costs: every traded dollar pays the taker fee (0.1%) — conservative, since live
rebalances mostly fill as maker (0.05%). Reports the competition metrics:
return, Sharpe, Sortino, Calmar and the composite 0.4*Sortino + 0.3*Sharpe + 0.3*Calmar,
plus results over rolling 14-day windows (the competition length).
"""
from __future__ import annotations

import argparse
import itertools
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np
import pandas as pd
import requests

from bot import strategy
from bot.config import DEFAULT_UNIVERSE, StrategyParams

HOST = "https://data-api.binance.vision"


def download(coins, days: int, cache: str = "data/closes.csv") -> pd.DataFrame:
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    end = int(time.time() * 1000)
    start = end - days * 24 * 3600 * 1000
    series = {}
    for coin in coins:
        rows, cursor = [], start
        while cursor < end:
            try:
                r = requests.get(f"{HOST}/api/v3/klines", params={"symbol": f"{coin}USDT", "interval": "1h",
                                 "startTime": cursor, "limit": 1000}, timeout=15)
                batch = r.json() if r.status_code == 200 else []
            except Exception:
                batch = []
            if not batch:
                break
            rows += batch
            cursor = batch[-1][0] + 3600_000
            time.sleep(0.1)
        if rows:
            series[coin] = pd.Series([float(x[4]) for x in rows], index=pd.to_datetime([x[0] for x in rows], unit="ms"))
            print(f"  {coin}: {len(rows)} bars")
        else:
            print(f"  {coin}: no data")
    df = pd.DataFrame(series).sort_index()
    df = df[~df.index.duplicated()]
    df.to_csv(cache)
    return df


def metrics(equity: pd.Series, periods_per_year: float = 24 * 365) -> dict:
    r = equity.pct_change().dropna()
    if r.empty or r.std() == 0:
        return dict(ret=0.0, sharpe=0.0, sortino=0.0, calmar=0.0, maxdd=0.0, composite=0.0)
    ret = equity.iloc[-1] / equity.iloc[0] - 1
    ann = np.sqrt(periods_per_year)
    sharpe = r.mean() / r.std() * ann
    downside = r[r < 0]
    dstd = np.sqrt((downside ** 2).sum() / len(r)) if len(downside) else 1e-9
    sortino = r.mean() / dstd * ann
    maxdd = float((1 - equity / equity.cummax()).max())
    years = len(r) / periods_per_year
    ann_ret = (1 + ret) ** (1 / years) - 1 if years > 0 and ret > -1 else -1
    calmar = ann_ret / maxdd if maxdd > 1e-9 else 0.0
    comp = 0.4 * sortino + 0.3 * sharpe + 0.3 * calmar
    return dict(ret=ret, sharpe=sharpe, sortino=sortino, calmar=calmar, maxdd=maxdd, composite=comp)


def run(closes: pd.DataFrame, p: StrategyParams, fee: float = 0.001, band: float | None = None) -> tuple[pd.Series, dict]:
    """Mirror of the live loop: stops every hour, full retarget every `rebalance_every_hours`
    (on UTC hours divisible by it, as the live bot does)."""
    band = p.rebalance_band if band is None else band
    closes = closes.ffill(limit=3)
    score, vol = strategy.score_frame(closes, p)
    regime_in = strategy.btc_regime_inputs(closes, score, p)
    logret = np.log(closes).diff()
    rets = closes.pct_change().fillna(0.0)
    warm = max(max(p.lookbacks), p.vol_window, p.regime_ema) + 1
    w = pd.Series(0.0, index=closes.columns)
    equity, gross_equity = 1.0, 1.0
    eq_hist: list = []
    hwm: dict = {}
    entry_bar: dict = {}
    cooldown: dict = {}
    regime_st: dict = {}
    eq_curve, turnover, n_trades, fees_paid, exposure = [], 0.0, 0, 0.0, []
    breakdown = {"entry": 0.0, "exit": 0.0, "resize": 0.0, "stop": 0.0}

    def execute(mask, target, stopped_set=frozenset()):
        nonlocal equity, turnover, n_trades, fees_paid
        traded = (target - w)[mask]
        if traded.empty:
            return
        for c, d in traded.items():
            kind = ("stop" if c in stopped_set else "entry" if w[c] <= 1e-4
                    else "exit" if target[c] <= 1e-9 else "resize")
            breakdown[kind] += abs(d)
        cost = float(traded.abs().sum()) * fee
        fees_paid += cost * equity
        equity *= 1 - cost
        turnover += float(traded.abs().sum())
        n_trades += int((traded.abs() > 1e-6).sum())
        w[mask] = target[mask]

    # numpy views for the hourly hot path (pandas indexing per bar is the bottleneck)
    R, PX = rets.to_numpy(), closes.to_numpy()
    DV = np.nan_to_num(vol.to_numpy() * np.sqrt(24), nan=0.04)
    cols, hours = list(closes.columns), closes.index.hour.to_numpy()

    for i in range(warm, len(closes)):
        wv = w.to_numpy()
        port = float(wv @ R[i])
        equity *= 1 + port
        gross_equity *= 1 + port
        if 1 + port > 0:
            w = pd.Series(wv * (1 + R[i]) / (1 + port), index=w.index)
            wv = w.to_numpy()
        eq_hist.append(equity)

        # trailing stops every hour
        stopped = set()
        live = set()
        for j in np.nonzero(wv > 1e-6)[0]:
            c, px = cols[j], PX[i, j]
            live.add(c)
            if np.isnan(px):
                continue
            hwm[c] = max(hwm.get(c, px), px)
            if px <= hwm[c] * (1 - strategy.trailing_stop_pct(DV[i, j], p)):
                stopped.add(c)
                cooldown[c] = i + p.stop_cooldown_hours
        hwm = {c: v for c, v in hwm.items() if c in live}
        entry_bar = {c: entry_bar.get(c, i) for c in live if w[c] > 1e-4}

        if hours[i] % p.rebalance_every_hours == 0:
            held = {c for c in w.index if w[c] > 1e-4} - stopped
            locked = {c for c in held if i - entry_bar.get(c, i) < p.min_hold_hours}
            excluded = {c for c, until in cooldown.items() if until > i}
            ri = regime_in.iloc[i]
            regime, _, regime_st = strategy.regime_state(float(ri.px), float(ri.ema), float(ri.score), regime_st, p)
            dd = strategy.rolling_drawdown(eq_hist, p.dd_lookback_hours)
            tgt, _ = strategy.target_weights(score.iloc[i], vol.iloc[i], logret.iloc[max(0, i - p.vol_window): i + 1],
                                             held, p, regime, strategy.drawdown_multiplier(dd, p), excluded, locked)
            tgt = tgt.reindex(w.index).fillna(0.0)
            dw = tgt - w
            mask = (dw.abs() >= band) | ((tgt == 0) & (w > 1e-4)) | dw.index.isin(list(stopped))
            execute(mask, tgt, stopped)
        elif stopped:
            execute(w.index.isin(list(stopped)), pd.Series(0.0, index=w.index), stopped)
        exposure.append(float(w.sum()))
        eq_curve.append((closes.index[i], equity))
    eq = pd.Series(dict(eq_curve))
    m = metrics(eq)
    m.update(turnover=turnover, trades=n_trades, gross_ret=gross_equity - 1, fee_drag=fees_paid,
             breakdown={k: round(v, 1) for k, v in breakdown.items()},
             avg_exposure=float(np.mean(exposure)) if exposure else 0.0)
    return eq, m


def _run_metrics(job) -> dict:
    closes, params = job
    return run(closes, params)[1]


def rolling_windows(eq: pd.Series, days: int = 14) -> pd.DataFrame:
    step = days * 24
    out = []
    for s in range(0, len(eq) - step, 24):
        seg = eq.iloc[s: s + step]
        m = metrics(seg)
        out.append(dict(start=seg.index[0], ret=m["ret"], maxdd=m["maxdd"], composite=m["composite"]))
    return pd.DataFrame(out)


def buy_and_hold_btc(closes: pd.DataFrame, start) -> dict:
    btc = closes["BTC"].loc[start:].dropna()
    return metrics(btc / btc.iloc[0])


def fmt(m: dict) -> str:
    return (f"ret {m['ret']*100:7.2f}%  maxDD {m['maxdd']*100:6.2f}%  Sharpe {m['sharpe']:5.2f}  "
            f"Sortino {m['sortino']:5.2f}  Calmar {m['calmar']:6.2f}  composite {m['composite']:6.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    closes = pd.read_csv(a.csv, index_col=0, parse_dates=True) if a.csv else download(DEFAULT_UNIVERSE, a.days)
    closes = closes.dropna(axis=1, thresh=int(len(closes) * 0.7))
    print(f"data: {closes.shape[1]} assets, {len(closes)} hourly bars ({closes.index[0]} -> {closes.index[-1]})")

    p = StrategyParams()
    eq, m = run(closes, p)
    print("\nSTRATEGY        ", fmt(m), f" trades {m['trades']}  turnover {m['turnover']:.1f}x")
    print(f"                  before fees {m['gross_ret']*100:.2f}%  fees paid {m['fee_drag']*100:.2f}% of capital  "
          f"avg exposure {m['avg_exposure']*100:.0f}%")
    print(f"                  turnover by cause (x capital): {m['breakdown']}")
    print("BTC buy & hold  ", fmt(buy_and_hold_btc(closes, eq.index[0])))
    rw = rolling_windows(eq)
    if not rw.empty:
        print(f"\n14-day windows ({len(rw)}): median ret {rw.ret.median()*100:.2f}%  "
              f"positive {(rw.ret > 0).mean()*100:.0f}%  worst {rw.ret.min()*100:.2f}%  "
              f"median maxDD {rw.maxdd.median()*100:.2f}%")
    eq.to_csv("data/backtest_equity.csv")

    if a.sweep:
        print("\nrobustness sweep (top_k, target_vol, max_drawdown, rebalance hours, min hold):")
        grid = list(itertools.product([3, 4, 6], [0.015, 0.02, 0.03], [0.08, 0.12], [4, 8], [0, 24]))
        jobs = [(closes, replace(p, top_k=k, target_daily_vol=tv, max_drawdown=mdd, rebalance_every_hours=every,
                                 min_hold_hours=hold)) for k, tv, mdd, every, hold in grid]
        with ProcessPoolExecutor() as pool:
            results = list(pool.map(_run_metrics, jobs))
        rows = [dict(top_k=k, tvol=tv, maxdd_lim=mdd, every=every, hold=hold,
                     **{x: mm[x] for x in ("ret", "gross_ret", "maxdd", "sharpe", "sortino", "calmar",
                                           "composite", "trades", "avg_exposure")})
                for (k, tv, mdd, every, hold), mm in zip(grid, results)]
        df = pd.DataFrame(rows).sort_values("composite", ascending=False)
        pd.set_option("display.width", 200)
        print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        df.to_csv("data/sweep.csv", index=False)
        print("\nmedian composite by setting (robustness: prefer settings that are good on average,"
              " not the single best row):")
        for col in ("top_k", "tvol", "maxdd_lim", "every", "hold"):
            print(" ", df.groupby(col).composite.median().round(3).to_dict())

    if a.plot:
        import matplotlib.pyplot as plt
        btc = closes["BTC"].loc[eq.index[0]:]
        plt.figure(figsize=(11, 5))
        plt.plot(eq.index, eq.values, label="strategy")
        plt.plot(btc.index, btc / btc.iloc[0], label="BTC buy&hold", alpha=0.6)
        plt.legend(); plt.grid(alpha=0.3); plt.title("Backtest equity (net of 0.1% fees)")
        plt.savefig("data/backtest.png", dpi=120, bbox_inches="tight")
        print("saved data/backtest.png")


if __name__ == "__main__":
    main()
