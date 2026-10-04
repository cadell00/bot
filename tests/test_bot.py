"""Offline tests: signing, order rounding, strategy invariants, backtest smoke test."""
import numpy as np
import pandas as pd
import pytest

import backtest
from bot import strategy
from bot.config import StrategyParams
from bot.execution import Executor, PairRule, floor_to, fmt
from bot.roostoo_client import RoostooClient


def synthetic(n=1100, coins=("BTC", "ETH", "SOL", "XRP", "DOGE", "ADA"), seed=0, drift=None):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=n, freq="h")
    data = {}
    for i, c in enumerate(coins):
        mu = drift[i] if drift else rng.normal(0, 0.0004)
        r = rng.normal(mu, 0.01, n)
        data[c] = 100 * np.exp(np.cumsum(r))
    return pd.DataFrame(data, index=idx)


def test_signature_matches_roostoo_doc_example():
    c = RoostooClient("USEAPIKEYASMYID", "S1XP1e3UZj6A7H5fATj0jNhqPxxdSJYdInClVN65XAbvqqMKjVHjA7PZj4W12oep")
    params = {"pair": "BNB/USD", "quantity": 2000, "side": "BUY", "timestamp": 1580774512000, "type": "MARKET"}
    total, sig = c.sign(params)
    assert total == "pair=BNB/USD&quantity=2000&side=BUY&timestamp=1580774512000&type=MARKET"
    assert sig == "20b7fd5550b67b3bf0c1684ed0f04885261db8fdabd38611e9e6af23c19b7fff"


def test_rounding():
    assert floor_to(1.23456, 3) == 1.234
    assert fmt(5.0, 0) == "5"
    assert fmt(0.1, 4) == "0.1000"


def test_weights_long_only_capped_no_leverage():
    p = StrategyParams()
    closes = synthetic(drift=[0.001, 0.0008, 0.0012, 0.0006, 0.0009, 0.0007])
    w, info = strategy.decide(closes, set(), p, drawdown=0.0)
    assert (w >= 0).all()
    assert w.sum() <= p.gross_cap + 1e-9
    assert all(w[a] <= (p.max_weight_major if a in ("BTC", "ETH") else p.max_weight) + 1e-9 for a in w.index)
    assert len(w) <= p.top_k + 1
    assert info["regime"] == "risk_on"


def test_risk_off_without_shorts_goes_to_cash():
    from dataclasses import replace
    p = replace(StrategyParams(), allow_shorts=False)
    closes = synthetic(drift=[-0.001] * 6)
    w, info = strategy.decide(closes, {"ETH"}, p)
    assert w.empty or (w == 0).all()
    assert info["regime"] == "risk_off"


def test_risk_off_with_shorts_goes_short_no_leverage():
    p = StrategyParams()
    closes = synthetic(drift=[-0.001, -0.0012, -0.0008, -0.0011, -0.0009, -0.001])
    w, info = strategy.decide(closes, set(), p)
    assert info["regime"] == "risk_off"
    assert len(w) and (w < 0).all()
    assert w.abs().sum() <= p.gross_cap + 1e-9
    assert (w.abs() <= p.short_max_weight + 1e-9).all()
    assert len(w) <= p.short_top_k + 1


def test_risk_on_has_no_shorts():
    p = StrategyParams()
    closes = synthetic(drift=[0.001, 0.0008, 0.0012, 0.0006, 0.0009, 0.0007])
    w, info = strategy.decide(closes, set(), p)
    assert (w >= 0).all()


def test_drawdown_multiplier_has_floor():
    p = StrategyParams()
    assert strategy.drawdown_multiplier(0.0, p) == 1.0
    assert strategy.drawdown_multiplier(p.max_drawdown / 2, p) == pytest.approx(0.5)
    assert strategy.drawdown_multiplier(1.0, p) == p.dd_floor_mult


def test_rolling_drawdown_recovers_after_window():
    hist = [1.10] + [1.0] * 10
    assert strategy.rolling_drawdown(hist, 24) == pytest.approx(1 - 1.0 / 1.10)
    assert strategy.rolling_drawdown(hist, 5) == 0.0  # old peak has rolled out of the window


def test_regime_hysteresis_does_not_flip_near_ema():
    p = StrategyParams()
    mult, name, st = strategy.regime_state(102.0, 100.0, 0.3, {}, p)
    assert name == "risk_on"
    # price dips slightly below EMA and score slightly negative: still risk_on (inside buffers)
    mult, name, st = strategy.regime_state(99.5, 100.0, -0.05, st, p)
    assert name == "risk_on"
    # clear break on both: risk_off
    mult, name, st = strategy.regime_state(97.0, 100.0, -0.3, st, p)
    assert name == "risk_off" and mult == 0.0
    # tiny recovery is not enough to turn back on
    mult, name, st = strategy.regime_state(100.5, 100.0, 0.05, st, p)
    assert name == "risk_off"


class FakeClient:
    def __init__(self, allow_shorts=True):
        self.orders = []
        self.allow_shorts = allow_shorts

    def place_order(self, pair, side, qty, order_type, price):
        self.orders.append((pair, side, qty, order_type, price))
        return {"Success": True, "OrderDetail": {"Status": "FILLED" if order_type == "MARKET" else "PENDING"}}

    def short_open(self, pair, collateral):
        if not self.allow_shorts:
            return {"Success": False, "ErrMsg": "this competition does not allow short positions"}
        self.orders.append((pair, "SHORT_OPEN", collateral, "MARKET", None))
        return {"Success": True, "Status": "OPEN", "Collateral": float(collateral)}

    def short_close(self, pair, close_qty=None, close_pct=None):
        self.orders.append((pair, "SHORT_CLOSE", close_qty, "MARKET", None))
        return {"Success": True, "ReturnAmount": 1000.0, "FullyClosed": close_qty is None}


def _executor(tmp_path, allow_shorts=True):
    from bot import logger
    from bot.config import BotConfig
    logger.setup(str(tmp_path))
    cfg = BotConfig()
    cfg.dry_run = False
    fc = FakeClient(allow_shorts)
    ex = Executor(fc, cfg, {"BTC/USD": PairRule(2, 5, 1.0), "ETH/USD": PairRule(2, 4, 1.0)})
    book = {"BTC": {"bid": 99.9, "ask": 100.1, "last": 100.0}, "ETH": {"bid": 9.99, "ask": 10.01, "last": 10.0}}
    return ex, fc, book


def test_executor_opens_short_sized_by_collateral(tmp_path):
    ex, fc, book = _executor(tmp_path)
    ex.rebalance(pd.Series({"BTC": -0.20}), {}, 100_000, 100_000, book, {})
    assert fc.orders == [("BTC/USD", "SHORT_OPEN", "20000.00", "MARKET", None)]


def test_executor_flips_long_to_short_selling_first(tmp_path):
    ex, fc, book = _executor(tmp_path)
    ex.rebalance(pd.Series({"ETH": -0.10}), {"ETH": 1000.0}, 90_000, 100_000, book, {})
    assert fc.orders[0][:2] == ("ETH/USD", "SELL") and fc.orders[0][3] == "MARKET"
    assert fc.orders[1][:2] == ("ETH/USD", "SHORT_OPEN")


def test_executor_closes_and_reduces_shorts(tmp_path):
    ex, fc, book = _executor(tmp_path)
    shorts = {"BTC": {"qty": 200.0, "collateral": 20_000, "value": 20_000}}
    ex.rebalance(pd.Series(dtype=float), {}, 80_000, 100_000, book, {}, shorts=shorts)
    assert fc.orders == [("BTC/USD", "SHORT_CLOSE", None, "MARKET", None)]           # close all
    fc.orders.clear()
    ex.rebalance(pd.Series({"BTC": -0.10}), {}, 80_000, 100_000, book, {}, shorts=shorts)
    assert fc.orders == [("BTC/USD", "SHORT_CLOSE", "100.00000", "MARKET", None)]    # halve it


def test_executor_disables_shorts_when_exchange_refuses(tmp_path):
    ex, fc, book = _executor(tmp_path, allow_shorts=False)
    ex.rebalance(pd.Series({"BTC": -0.20}), {}, 100_000, 100_000, book, {})
    assert ex.shorts_disabled


def test_executor_band_sells_first_and_fallback(tmp_path):
    from bot import logger
    logger.setup(str(tmp_path))
    from bot.config import BotConfig
    cfg = BotConfig()
    cfg.dry_run = False
    fc = FakeClient()
    ex = Executor(fc, cfg, {"BTC/USD": PairRule(2, 5, 1.0), "ETH/USD": PairRule(2, 4, 1.0)})
    book = {"BTC": {"bid": 99.9, "ask": 100.1, "last": 100.0}, "ETH": {"bid": 9.99, "ask": 10.01, "last": 10.0}}
    attempts = {}
    # hold 500 ETH ($5k), want 0 ETH and 30% BTC of $100k
    ex.rebalance(pd.Series({"BTC": 0.30}), {"ETH": 500.0}, 95_000, 100_000, book, attempts)
    assert fc.orders[0][1] == "SELL" and fc.orders[1][1] == "BUY"
    assert fc.orders[1][3] == "LIMIT"
    # after max attempts the buy falls back to market
    fc.orders.clear()
    attempts["BTC"] = cfg.max_limit_attempts
    ex.rebalance(pd.Series({"BTC": 0.30}), {}, 95_000, 100_000, book, attempts)
    assert fc.orders[0][3] == "MARKET"
    # tiny change inside band -> no order
    fc.orders.clear()
    ex.rebalance(pd.Series({"BTC": 0.30}), {"BTC": 295.0}, 70_000, 100_000, book, {})
    assert fc.orders == []


def test_backtest_runs_and_reports_metrics():
    closes = synthetic(n=1800, drift=[0.0004, 0.0003, 0.0006, -0.0002, 0.0001, 0.0])
    eq, m = backtest.run(closes, StrategyParams())
    assert len(eq) > 500
    assert np.isfinite(m["composite"])
    assert eq.min() > 0


def test_backtest_shorts_profit_in_steady_downtrend():
    from dataclasses import replace
    closes = synthetic(n=2200, drift=[-0.0006] * 6, seed=3)
    _, with_shorts = backtest.run(closes, StrategyParams())
    _, long_only = backtest.run(closes, replace(StrategyParams(), allow_shorts=False))
    assert with_shorts["short_time"] > 0.2
    assert with_shorts["ret"] > long_only["ret"]
