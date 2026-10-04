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
from .notify import Telegram, position_changes
from .roostoo_client import RoostooClient


@dataclass
class Snapshot:
    book: dict
    holdings: dict       # coin -> free spot qty
    shorts: dict         # coin -> {qty, collateral, value, entry}
    usd_free: float
    cash: float
    equity: float
    long_value: float
    short_notional: float
    positions: dict = field(default_factory=dict)   # coin -> signed qty (long incl. locked; short < 0)


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
    lwm: dict = field(default_factory=dict)            # trailing-stop low-water marks (shorts)
    squeeze_until: float = 0.0                         # no new shorts until then
    shorts_disabled: bool = False                      # exchange said shorts are not allowed
    entry_px: dict = field(default_factory=dict)       # coin -> price when position was opened (take-profit)
    start_equity: float = 0.0                          # first equity seen (daily summary)
    last_summary_day: str = ""

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
        # dry runs keep separate state so they never affect the live bot
        self.state_path = os.path.join(cfg.data_dir, "state_dry_run.json" if cfg.dry_run else "state.json")
        self.state = State.load(self.state_path)
        self.client = RoostooClient(cfg.api_key, cfg.secret_key, cfg.base_url, cfg.min_request_interval)
        self.client.sync_clock()
        self.rules = parse_rules(self.client.exchange_info())
        self.universe = [c for c in cfg.universe if f"{c}/{cfg.quote}" in self.rules]
        self.exec = Executor(self.client, cfg, self.rules)
        self.md = MarketData(cfg.data_dir)
        if self.state.shorts_disabled:
            cfg.strategy.allow_shorts = False
        self.notify = Telegram(cfg.telegram_token, cfg.telegram_chat_id,
                               prefix=f"[{cfg.bot_name}{' DRY' if cfg.dry_run else ''}] ")
        self._prev_positions: dict | None = None
        self.running = True
        self.log.info("universe (%d): %s | dry_run=%s", len(self.universe), ",".join(self.universe), cfg.dry_run)
        logger.event("decisions", action="startup", universe=self.universe,
                     params=asdict(cfg.strategy), dry_run=cfg.dry_run)

    # --------------------------------------------------------------- snapshot
    def snapshot(self) -> Snapshot:
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
        cash = usd_free + float(usd.get("Lock", 0))
        holdings, long_value, positions = {}, 0.0, {}
        for coin, w in wallet.items():
            if coin == self.cfg.quote:
                continue
            qty = float(w.get("Free", 0)) + float(w.get("Lock", 0))
            if qty > 0 and coin in book:
                holdings[coin] = float(w.get("Free", 0))
                long_value += qty * book[coin]["last"]
                positions[coin] = qty
        # Short collateral leaves the wallet (short_close returns "collateral + PnL - fee"
        # to it), so each open short adds PositionValue = collateral + unrealised PnL.
        shorts, short_value, short_notional = {}, 0.0, 0.0
        # always read positions, so shorts opened earlier are still valued and managed
        try:
            res = self.client.short_positions()
            for pos in res.get("Positions") or []:
                coin = pos["Pair"].split("/")[0]
                qty = float(pos.get("ShortQty", 0))
                coll = float(pos.get("Collateral", 0))
                value = float(pos.get("PositionValue", coll + float(pos.get("UnrealizedPNL", 0))))
                positions[coin] = positions.get(coin, 0.0) - qty
                shorts[coin] = {"qty": qty, "collateral": coll, "value": value,
                                "entry": float(pos.get("EntryPrice", 0))}
                short_value += value
                short_notional += qty * book.get(coin, {}).get("last", float(pos.get("CurrentPrice", 0)))
        except Exception as exc:
            self.log.warning("short_positions failed: %s", exc)
        equity = cash + long_value + short_value
        return Snapshot(book, holdings, shorts, usd_free, cash, equity, long_value, short_notional, positions)

    # -------------------------------------------------------------- main tick
    def tick(self) -> None:
        now = time.time()
        snap = self.snapshot()
        book, equity = snap.book, snap.equity
        self.md.snapshots.record({c: book[c]["last"] for c in self.universe if c in book})
        st = self.state
        p = self.cfg.strategy
        self.report_fills(snap)
        if not st.start_equity:
            st.start_equity = equity
        st.peak_equity = max(st.peak_equity, equity)
        hour_key = time.strftime("%Y-%m-%dT%H", time.gmtime(now))
        if hour_key != st.equity_hist_hour:
            st.equity_hist = (st.equity_hist + [round(equity, 2)])[-(p.dd_lookback_hours + 5):]
            st.equity_hist_hour = hour_key
        dd = strategy.rolling_drawdown(st.equity_hist + [equity], p.dd_lookback_hours)
        gross = (snap.long_value + snap.short_notional) / equity if equity > 0 else 0.0
        logger.equity_row(equity, snap.cash, dd, gross)
        self.daily_summary(snap, dd, gross)

        held = {c for c, q in snap.holdings.items() if q * book[c]["last"] >= self.cfg.min_order_usd}
        held_short = set(snap.shorts)
        urgent = self.risk_checks(book, held, held_short, now)
        if urgent:
            self.trade(st.targets, urgent, reason="risk_exit")
            snap = self.snapshot()
            held = {c for c, q in snap.holdings.items() if q * snap.book[c]["last"] >= self.cfg.min_order_usd}
            held_short = set(snap.shorts)

        gm = time.gmtime(now)
        on_schedule = gm.tm_hour % p.rebalance_every_hours == 0 or not st.last_rebalance_hour
        if hour_key != st.last_rebalance_hour and gm.tm_min >= self.cfg.rebalance_minute and on_schedule:
            self.rebalance(held, dd, now, held_short)
            st.last_rebalance_hour = hour_key
        elif st.attempts and now - st.last_order_ts >= self.cfg.retry_minutes * 60:
            self.trade(st.targets, set(), reason="requote")
        st.save(self.state_path)

    # ------------------------------------------------------------ notifications
    def report_fills(self, snap: Snapshot) -> None:
        """Telegram on every fill, detected from the account itself (covers limit orders that
        fill between loops). The first snapshot after startup only sets the baseline."""
        prices = {c: b["last"] for c, b in snap.book.items()}
        if self._prev_positions is None:
            held = ", ".join(f"{'SHORT' if q < 0 else 'LONG'} {c}" for c, q in sorted(snap.positions.items())
                             if abs(q) * prices.get(c, 0) >= 25) or "none (all cash)"
            self.notify.send(f"✅ Bot started. Equity ${snap.equity:,.0f}. Positions: {held}")
        else:
            msgs = position_changes(self._prev_positions, snap.positions, prices, self.state.entry_px)
            if msgs:
                self.notify.send("\n".join(msgs) + f"\nEquity ${snap.equity:,.0f}")
        self._prev_positions = dict(snap.positions)

    def daily_summary(self, snap: Snapshot, dd: float, gross: float) -> None:
        day = time.strftime("%Y-%m-%d", time.gmtime())
        st = self.state
        if day == st.last_summary_day:
            return
        first = not st.last_summary_day
        st.last_summary_day = day
        if first:
            return  # no summary on the very first loop; the startup message covers it
        ret = snap.equity / st.start_equity - 1 if st.start_equity else 0.0
        pos = ", ".join(f"{'S' if q < 0 else 'L'} {c}" for c, q in sorted(snap.positions.items())
                        if abs(q) * snap.book.get(c, {}).get("last", 0) >= 25) or "cash"
        self.notify.send(f"📊 Daily summary {day}\nEquity ${snap.equity:,.0f} ({ret * 100:+.2f}% since start)\n"
                         f"Drawdown {dd * 100:.2f}% | gross exposure {gross * 100:.0f}%\nPositions: {pos}")

    def risk_checks(self, book: dict, held: set, held_short: set, now: float) -> set:
        st, p = self.state, self.cfg.strategy
        urgent = set()
        btc_1h = self.md.snapshots.return_over("BTC", 60)
        if btc_1h is not None and btc_1h <= p.crash_btc_1h and now >= st.crash_until and held:
            st.crash_until = now + p.crash_cooldown_hours * 3600
            st.targets = {c: w for c, w in st.targets.items() if w < 0}   # keep shorts
            urgent |= held
            logger.event("decisions", action="crash_guard", btc_1h=btc_1h, flatten=sorted(held))
            self.log.warning("CRASH GUARD: BTC %.2f%% in 1h -> close longs %s", btc_1h * 100, sorted(held))
            self.notify.send(f"🚨 Crash guard: BTC {btc_1h * 100:.1f}% in 1h, closing longs {sorted(held)}")
        if btc_1h is not None and btc_1h >= p.squeeze_btc_1h and now >= st.squeeze_until and held_short:
            st.squeeze_until = now + p.crash_cooldown_hours * 3600
            st.targets = {c: w for c, w in st.targets.items() if w > 0}   # keep longs
            urgent |= held_short
            logger.event("decisions", action="squeeze_guard", btc_1h=btc_1h, cover=sorted(held_short))
            self.log.warning("SQUEEZE GUARD: BTC +%.2f%% in 1h -> cover shorts %s", btc_1h * 100, sorted(held_short))
            self.notify.send(f"🚨 Squeeze guard: BTC +{btc_1h * 100:.1f}% in 1h, covering shorts {sorted(held_short)}")

        st.hwm = {c: v for c, v in st.hwm.items() if c in held}
        st.lwm = {c: v for c, v in st.lwm.items() if c in held_short}
        st.entry_ts = {c: st.entry_ts.get(c, now) for c in held | held_short}
        st.entry_px = {c: st.entry_px.get(c, book[c]["last"]) for c in held | held_short if c in book}
        for coin in held | held_short:
            if coin in urgent or coin not in book:
                continue
            px = book[coin]["last"]
            stop = strategy.trailing_stop_pct(st.daily_vol.get(coin, 0.04), p)
            if coin in held:
                st.hwm[coin] = max(st.hwm.get(coin, px), px)
                hit, mark, side = px <= st.hwm[coin] * (1 - stop), st.hwm[coin], "long"
            else:
                st.lwm[coin] = min(st.lwm.get(coin, px), px)
                hit, mark, side = px >= st.lwm[coin] * (1 + stop), st.lwm[coin], "short"
            gain = (px / st.entry_px[coin] - 1) * (1 if side == "long" else -1) if coin in st.entry_px else 0.0
            tp = strategy.take_profit_pct(st.daily_vol.get(coin, 0.04), p)
            if not hit and gain >= tp:
                urgent.add(coin)
                st.targets.pop(coin, None)
                st.cooldown_until[coin] = now + p.take_profit_cooldown_hours * 3600
                logger.event("decisions", action="take_profit", side=side, coin=coin, price=px,
                             entry=st.entry_px[coin], gain=round(gain, 4), tp_pct=round(tp, 4))
                self.log.info("TAKE PROFIT %s %s +%.2f%% (target %.2f%%)", side, coin, gain * 100, tp * 100)
                self.notify.send(f"💰 Take-profit {side} {coin}: +{gain * 100:.2f}%")
            if hit:
                urgent.add(coin)
                st.targets.pop(coin, None)
                st.cooldown_until[coin] = now + p.stop_cooldown_hours * 3600
                logger.event("decisions", action="trailing_stop", side=side, coin=coin, price=px,
                             mark=mark, stop_pct=stop)
                self.log.warning("STOP %s %s at %.6g (mark %.6g, stop %.1f%%)", side, coin, px, mark, stop * 100)
                self.notify.send(f"🛑 Trailing stop {side} {coin} at {px:.6g} ({stop * 100:.1f}% from {mark:.6g})")
        return urgent

    def rebalance(self, held: set, dd: float, now: float, held_short: set | None = None) -> None:
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
            held_short = held_short or set()
            locked = {c for c in held | held_short if now - st.entry_ts.get(c, now) < p.min_hold_hours * 3600}
            w, info = strategy.decide(closes, held, p, dd, excluded, st.regime, locked, held_short)
            info["locked"] = sorted(locked)
            st.regime = info.pop("regime_state", st.regime)
        if now < st.squeeze_until:
            w = w[w > 0]
            info["squeeze_cooldown"] = True
        st.daily_vol = info.pop("daily_vol", st.daily_vol)
        st.targets = {k: round(float(v), 5) for k, v in w.items() if abs(v) > 0}
        logger.event("decisions", action="rebalance", drawdown=round(dd, 4), held=sorted(held),
                     excluded=sorted(excluded), targets=st.targets, **info)
        self.log.info("rebalance regime=%s targets=%s", info.get("regime"), st.targets)
        if st.targets != getattr(self, "_last_notified_targets", None):
            self._last_notified_targets = dict(st.targets)
            tg = ", ".join(f"{c} {w * 100:+.0f}%" for c, w in sorted(st.targets.items(), key=lambda x: -abs(x[1])))
            self.notify.send(f"🔄 Rebalance ({info.get('regime')}): {tg or 'all cash'}")
        self.trade(st.targets, set(), reason="rebalance")

    def trade(self, targets: dict, urgent: set, reason: str) -> None:
        self.exec.cancel_all()
        snap = self.snapshot()
        self.exec.rebalance(pd.Series(targets, dtype=float), snap.holdings, snap.usd_free, snap.equity,
                            snap.book, self.state.attempts, urgent, reason, snap.shorts)
        self.state.last_order_ts = time.time()
        if self.exec.shorts_disabled and not self.state.shorts_disabled:
            self.state.shorts_disabled = True
            self.cfg.strategy.allow_shorts = False
            logger.event("decisions", action="shorts_disabled", reason="exchange rejected short orders")
            self.log.warning("exchange does not allow shorts — continuing long/cash only")
            self.notify.send("ℹ️ Exchange rejected shorts: continuing long/cash only")

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
                self.notify.error(type(exc).__name__ + str(exc)[:60], f"{type(exc).__name__}: {str(exc)[:300]}")
            time.sleep(max(1.0, self.cfg.loop_seconds - (time.time() - start)))
        self.state.save(self.state_path)
        self.log.info("bot stopped")
        self.notify.send("⏹️ Bot stopped (shutdown signal). Positions stay open on the exchange.", sync=True)

    def _stop(self, *_):
        self.running = False


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Roostoo regime-switching long/short momentum bot")
    ap.add_argument("--dry-run", action="store_true", help="log decisions, place no orders")
    args = ap.parse_args(argv)
    cfg = BotConfig()
    if args.dry_run:
        cfg.dry_run = True
    if not cfg.api_key or not cfg.secret_key:
        sys.exit("ROOSTOO_API_KEY / ROOSTOO_SECRET_KEY not set (see .env.example)")
    try:
        Bot(cfg).run()
    except Exception as exc:
        # last-resort alert if the bot dies outside the per-loop error handling
        Telegram(cfg.telegram_token, cfg.telegram_chat_id, prefix=f"[{cfg.bot_name}] ").send(
            f"💥 Bot crashed and exited: {type(exc).__name__}: {str(exc)[:300]}", sync=True)
        raise


if __name__ == "__main__":
    main()