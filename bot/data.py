"""Market data. Roostoo's ticker only exposes the latest price, so hourly candle
history comes from free public sources, tried in order:

  1. Binance public market data (data-api.binance.vision, then api.binance.com)
  2. OKX public candles
  3. The bot's own record of Roostoo ticker snapshots (data/snapshots.csv)

Signals use these closes; execution always uses the live Roostoo bid/ask.
"""
from __future__ import annotations

import os
import time
from collections import deque

import pandas as pd
import requests

from . import logger

log = logger.get("data")

BINANCE_HOSTS = ("https://data-api.binance.vision", "https://api.binance.com", "https://api1.binance.com")
OKX_HOST = "https://www.okx.com"


def _drop_open_bar(s: pd.Series) -> pd.Series:
    """Remove the still-forming current hour so signals only use closed bars."""
    if s.empty:
        return s
    now_hour = pd.Timestamp.now(tz="UTC").floor("h").tz_localize(None)
    return s[s.index < now_hour]


def fetch_binance(coin: str, bars: int = 500, session: requests.Session | None = None) -> pd.Series:
    session = session or requests
    symbol = f"{coin}USDT"
    for host in BINANCE_HOSTS:
        try:
            r = session.get(f"{host}/api/v3/klines",
                            params={"symbol": symbol, "interval": "1h", "limit": min(bars + 1, 1000)},
                            timeout=10)
            if r.status_code != 200:
                continue
            rows = r.json()
            if not rows:
                continue
            idx = pd.to_datetime([row[0] for row in rows], unit="ms")
            s = pd.Series([float(row[4]) for row in rows], index=idx, name=coin)
            return _drop_open_bar(s)
        except Exception as exc:
            log.debug("binance %s %s failed: %s", host, symbol, exc)
    return pd.Series(dtype=float, name=coin)


def fetch_okx(coin: str, bars: int = 500, session: requests.Session | None = None) -> pd.Series:
    session = session or requests
    inst = f"{coin}-USDT"
    out: list = []
    after = None
    try:
        while len(out) < bars + 1:
            params = {"instId": inst, "bar": "1H", "limit": "100"}
            if after:
                params["after"] = after
            r = session.get(f"{OKX_HOST}/api/v5/market/history-candles", params=params, timeout=10)
            data = r.json().get("data", []) if r.status_code == 200 else []
            if not data:
                break
            out.extend(data)
            after = data[-1][0]
            time.sleep(0.12)
    except Exception as exc:
        log.debug("okx %s failed: %s", inst, exc)
    if not out:
        return pd.Series(dtype=float, name=coin)
    out = [row for row in out if len(row) < 9 or row[8] == "1"]  # confirmed bars only
    idx = pd.to_datetime([int(row[0]) for row in out], unit="ms")
    s = pd.Series([float(row[4]) for row in out], index=idx, name=coin).sort_index()
    s = s[~s.index.duplicated()]
    return _drop_open_bar(s)


class SnapshotStore:
    """Persists Roostoo ticker snapshots; provides hourly closes and short-term
    returns (used by the crash guard) without any external dependency."""

    def __init__(self, data_dir: str = "data", memory_minutes: int = 180):
        os.makedirs(data_dir, exist_ok=True)
        self.path = os.path.join(data_dir, "snapshots.csv")
        self.recent: dict[str, deque] = {}
        self.memory = memory_minutes

    def record(self, prices: dict[str, float]) -> None:
        ts = int(time.time())
        new = not os.path.exists(self.path)
        with open(self.path, "a") as fh:
            if new:
                fh.write("ts,coin,price\n")
            for coin, px in prices.items():
                fh.write(f"{ts},{coin},{px}\n")
                dq = self.recent.setdefault(coin, deque(maxlen=self.memory * 2))
                dq.append((ts, px))

    def return_over(self, coin: str, minutes: int) -> float | None:
        dq = self.recent.get(coin)
        if not dq or len(dq) < 2:
            return None
        now_ts, now_px = dq[-1]
        target = now_ts - minutes * 60
        past = [px for ts, px in dq if ts <= target]
        if not past:
            return None
        return now_px / past[-1] - 1.0

    def hourly_closes(self, coins) -> pd.DataFrame:
        if not os.path.exists(self.path):
            return pd.DataFrame()
        df = pd.read_csv(self.path)
        df = df[df.coin.isin(coins)]
        if df.empty:
            return pd.DataFrame()
        df["dt"] = pd.to_datetime(df.ts, unit="s").dt.floor("h")
        closes = df.sort_values("ts").groupby(["dt", "coin"]).price.last().unstack()
        return closes.iloc[:-1]  # drop current (open) hour


class MarketData:
    def __init__(self, data_dir: str = "data"):
        self.snapshots = SnapshotStore(data_dir)
        self.session = requests.Session()
        self.source_by_coin: dict[str, str] = {}

    def hourly_closes(self, coins, bars: int = 500) -> pd.DataFrame:
        series = {}
        local = None
        for coin in coins:
            s = fetch_binance(coin, bars, self.session)
            src = "binance"
            if len(s) < bars // 2:
                s = fetch_okx(coin, bars, self.session)
                src = "okx"
            if len(s) < bars // 2:
                if local is None:
                    local = self.snapshots.hourly_closes(coins)
                s = local[coin].dropna() if coin in local else pd.Series(dtype=float)
                src = "roostoo_snapshots"
            if len(s):
                series[coin] = s
                self.source_by_coin[coin] = src
        if not series:
            return pd.DataFrame()
        df = pd.DataFrame(series).sort_index()
        return df.ffill(limit=3).tail(bars)
