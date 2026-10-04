"""Bot entry point: `python -m bot.main` (add `--dry-run` to compute and log
decisions without placing orders).

Loop cadence (well within Roostoo's limits — roughly 2-4 requests per minute):
  every LOOP_SECONDS (60s): ticker + balance -> equity log, crash guard, trailing stops
  every hour at HH:REBALANCE_MINUTE: fetch candles, run strategy, rebalance
  every RETRY_MINUTES after a rebalance: re-quote unfilled limit orders
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from dataclasses import asdict, dataclass, field

import pandas as pd

from . import logger, strategy
from .config import BotConfig
from .data import MarketData
from .execution import Executor, parse_rules
from .roostoo_client import RoostooClient


@dataclass
class State:
    peak_equity: float = 0.0
    last_rebalance_hour: str = ""
    last_order_ts: float = 0.0
    targets: dict = field(default_factory=dict)
    attempts: dict = field(default_factory=dict)
    hwm: dict = field(default_factory=dict)            # trailing-stop high-water marks
    cooldown_until: dict = field(default_factory=dict)  # coin -> unix ts
    crash_until: float = 0.0
    daily_vol: dict = field(default_factory=dict)
    equity_hist: list = field(default_factory=list)    # one equity sample per hour (rolling DD)
    equity_hist_hour: str = ""
    regime: dict = field(default_factory=dict)         # hysteresis state of the BTC regime
    entry_ts: dict = field(default_factory=dict)       # coin -> unix ts when position was opened

    @classmethod
    def load(cls, path: str) -> "State":
        if os.path.exists(path):
            try:
                with open(path) as fh:
                    return cls(**json.load(fh))
            except Exception:
                pass
        return cls()

    def save(self, path: str) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(asdict(self), fh, indent=1)
        os.replace(tmp, path)


class Bot:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg
        self.log = logger.setup(cfg.log_dir)
        os.makedirs(cfg.data_dir, exist_ok=True)
        self.state_path = os.path.join(cfg.data_dir, "state.json")
        self.state = State.load(self.state_path)
        self.client = RoostooClient(cfg.api_key, cfg.secret_key, cfg.base_url, cfg.min_request_interval)
        self.client.sync_clock()
        self.rules = parse_rules(self.client.exchange_info())
        self.universe = [c for c in cfg.universe if f"{c}/{cfg.quote}" in self.rules]
        self.exec = Executor(self.client, cfg, self.rules)
        self.md = MarketData(cfg.data_dir)
        self.running = True
        self.log.info("universe (%d): %s | dry_run=%s", len(self.universe), ",".join(self.universe), cfg.dry_run)
        logger.event("decisions", action="startup", universe=self.universe,
                     params=asdict(cfg.strategy), dry_run=cfg.dry_run)

    # --------------------------------------------------------------- snapshot
    def snapshot(self):
        t = self.client.ticker()
        book = {}
        for pair, d in (t.get("Data") or {}).items():
            coin = pair.split("/")[0]
            book[coin] = {"bid": float(d.get("MaxBid") or d["LastPrice"]),
                          "ask": float(d.get("MinAsk") or d["LastPrice"]),
                          "last": float(d["LastPrice"])}
        bal = self.client.balance()
        wallet = bal.get("Wallet") or bal.get("SpotWallet") or {}
        usd = wallet.get(self.cfg.quote, {})
        usd_free = float(usd.get("Free", 0))
        usd_total = usd_free + float(usd.get("Lock", 0))
        holdings, equity = {}, usd_total
        for coin, w in wallet.items():
            if coin == self.cfg.quote:
                continue
            qty = float(w.get("Free", 0)) + float(w.get("Lock", 0))
            if qty > 0 and coin in book:
                holdings[coin] = float(w.get("Free", 0))
                equity += qty * book[coin]["last"]
        return book, holdings, usd_free, usd_total, equity

    # -------------------------------------------------------------- main tick
    def tick(self) -> None:
        now = time.time()
        book, holdings, usd_free, cash, equity = self.snapshot()
        self.md.snapshots.record({c: book[c]["last"] for c in self.universe if c in book})
        st = self.state
        p = self.cfg.strategy
        st.peak_equity = max(st.peak_equity, equity)
        hour_key = time.strftime("%Y-%m-%dT%H", time.gmtime(now))
        if hour_key != st.equity_hist_hour:
            st.equity_hist = (st.equity_hist + [round(equity, 2)])[-(p.dd_lookback_hours + 5):]
            st.equity_hist_hour = hour_key
        dd = strategy.rolling_drawdown(st.equity_hist + [equity], p.dd_lookback_hours)
        exposure = 1 - cash / equity if equity > 0 else 0.0
        logger.equity_row(equity, cash, dd, exposure)

        held = {c for c, q in holdings.items() if q * book[c]["last"] >= self.cfg.min_order_usd}
        urgent = self.risk_checks(book, held, now)
        if urgent:
            self.trade(st.targets, urgent, reason="risk_exit")
            book, holdings, usd_free, cash, equity = self.snapshot()

        gm = time.gmtime(now)
        on_schedule = gm.tm_hour % p.rebalance_every_hours == 0 or not st.last_rebalance_hour
        if hour_key != st.last_rebalance_hour and gm.tm_min >= self.cfg.rebalance_minute and on_schedule:
            self.rebalance(held, dd, now)
            st.last_rebalance_hour = hour_key
        elif st.attempts and now - st.last_order_ts >= self.cfg.retry_minutes * 60:
            self.trade(st.targets, set(), reason="requote")
        st.save(self.state_path)

    def risk_checks(self, book: dict, held: set, now: float) -> set:
        st, p = self.state, self.cfg.strategy
        urgent = set()
        btc_1h = self.md.snapshots.return_over("BTC", 60)
        if btc_1h is not None and btc_1h <= p.crash_btc_1h and now >= st.crash_until:
            st.crash_until = now + p.crash_cooldown_hours * 3600
            st.targets = {}
            urgent |= held
            logger.event("decisions", action="crash_guard", btc_1h=btc_1h, flatten=sorted(held))
            self.log.warning("CRASH GUARD: BTC %.2f%% in 1h -> flatten %s", btc_1h * 100, sorted(held))
        for coin in list(st.hwm):
            if coin not in held:
                st.hwm.pop(coin)
        st.entry_ts = {c: st.entry_ts.get(c, now) for c in held}
        for coin in held:
            px = book[coin]["last"]
            st.hwm[coin] = max(st.hwm.get(coin, px), px)
            stop = strategy.trailing_stop_pct(st.daily_vol.get(coin, 0.04), p)
            if px <= st.hwm[coin] * (1 - stop) and coin not in urgent:
                urgent.add(coin)
                st.targets.pop(coin, None)
                st.cooldown_until[coin] = now + p.stop_cooldown_hours * 3600
                logger.event("decisions", action="trailing_stop", coin=coin, price=px,
                             hwm=st.hwm[coin], stop_pct=stop)
                self.log.warning("STOP %s at %.6g (hwm %.6g, stop %.1f%%)", coin, px, st.hwm[coin], stop * 100)
        return urgent

    def rebalance(self, held: set, dd: float, now: float) -> None:
        st, p = self.state, self.cfg.strategy
        closes = self.md.hourly_closes(self.universe, self.cfg.history_bars)
        min_bars = max(max(p.lookbacks), p.vol_window // 2) + 2
        if closes.empty or len(closes) < min_bars or "BTC" not in closes:
            logger.event("decisions", action="skip", reason="insufficient_history",
                         bars=len(closes), sources=self.md.source_by_coin)
            self.log.warning("insufficient history (%d bars) — holding current positions", len(closes))
            return
        closes = closes.dropna(axis=1, thresh=min_bars)
        excluded = {c for c, until in st.cooldown_until.items() if until > now}
        st.cooldown_until = {c: u for c, u in st.cooldown_until.items() if u > now}
        if now < st.crash_until:
            w, info = pd.Series(dtype=float), {"regime": "crash_cooldown"}
        else:
            locked = {c for c in held if now - st.entry_ts.get(c, now) < p.min_hold_hours * 3600}
            w, info = strategy.decide(closes, held, p, dd, excluded, st.regime, locked)
            info["locked"] = sorted(locked)
            st.regime = info.pop("regime_state", st.regime)
        st.daily_vol = info.pop("daily_vol", st.daily_vol)
        st.targets = {k: round(float(v), 5) for k, v in w.items() if v > 0}
        logger.event("decisions", action="rebalance", drawdown=round(dd, 4), held=sorted(held),
                     excluded=sorted(excluded), targets=st.targets, **info)
        self.log.info("rebalance regime=%s targets=%s", info.get("regime"), st.targets)
        self.trade(st.targets, set(), reason="rebalance")

    def trade(self, targets: dict, urgent: set, reason: str) -> None:
        self.exec.cancel_all()
        book, holdings, usd_free, _, equity = self.snapshot()
        self.exec.rebalance(pd.Series(targets, dtype=float), holdings, usd_free, equity, book,
                            self.state.attempts, urgent, reason)
        self.state.last_order_ts = time.time()

    # ------------------------------------------------------------------- loop
    def run(self) -> None:
        signal.signal(signal.SIGTERM, self._stop)
        signal.signal(signal.SIGINT, self._stop)
        while self.running:
            start = time.time()
            try:
                self.tick()
            except Exception as exc:
                self.log.exception("tick failed: %s", exc)
                logger.event("decisions", action="error", error=str(exc))
            time.sleep(max(1.0, self.cfg.loop_seconds - (time.time() - start)))
        self.state.save(self.state_path)
        self.log.info("bot stopped")

    def _stop(self, *_):
        self.running = False


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Roostoo regime-gated momentum bot")
    ap.add_argument("--dry-run", action="store_true", help="log decisions, place no orders")
    args = ap.parse_args(argv)
    cfg = BotConfig()
    if args.dry_run:
        cfg.dry_run = True
    if not cfg.api_key or not cfg.secret_key:
        sys.exit("ROOSTOO_API_KEY / ROOSTOO_SECRET_KEY not set (see .env.example)")
    Bot(cfg).run()


if __name__ == "__main__":
    main()
