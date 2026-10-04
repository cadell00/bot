"""Offline tests: signing, order rounding, strategy invariants, backtest smoke test."""
import numpy as np
import pandas as pd
import pytest

import backtest
from bot import strategy
from bot.config import StrategyParams
from bot.execution import Executor, PairRule, floor_to, fmt
from bot.roostoo_client import RoostooClient


def synthetic(n=900, coins=("BTC", "ETH", "SOL", "XRP", "DOGE", "ADA"), seed=0, drift=None):
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


def test_risk_off_goes_to_cash():
    p = StrategyParams()
    closes = synthetic(drift=[-0.001] * 6)
    w, info = strategy.decide(closes, {"ETH"}, p)
    assert w.empty or w.sum() == 0
    assert info["regime"] == "risk_off"


def test_drawdown_multiplier():
    p = StrategyParams()
    assert strategy.drawdown_multiplier(0.0, p) == 1.0
    assert strategy.drawdown_multiplier(p.max_drawdown / 2, p) == pytest.approx(0.5)
    assert strategy.drawdown_multiplier(1.0, p) == 0.0


class FakeClient:
    def __init__(self):
        self.orders = []

    def place_order(self, pair, side, qty, order_type, price):
        self.orders.append((pair, side, qty, order_type, price))
        return {"Success": True, "OrderDetail": {"Status": "FILLED" if order_type == "MARKET" else "PENDING"}}


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
    closes = synthetic(n=1200, drift=[0.0004, 0.0003, 0.0006, -0.0002, 0.0001, 0.0])
    eq, m = backtest.run(closes, StrategyParams())
    assert len(eq) > 500
    assert np.isfinite(m["composite"])
    assert eq.min() > 0
