"""Turns target weights into Roostoo orders.

Cost-aware by design (0.1% taker vs 0.05% maker):
  * changes smaller than the rebalance band are not traded (edge < cost);
  * routine rebalances are passive LIMIT orders at the touch (maker fee);
  * a limit that has not filled after N re-quotes falls back to MARKET;
  * risk exits (stops, crash guard, full exits flagged urgent) go MARKET immediately.
Sells are sent before buys so freed cash can fund new positions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from . import logger
from .config import BotConfig

log = logger.get("exec")


@dataclass
class PairRule:
    price_precision: int
    amount_precision: int
    min_notional: float


def parse_rules(exchange_info: dict) -> dict[str, PairRule]:
    rules = {}
    for pair, d in exchange_info.get("TradePairs", {}).items():
        if not d.get("CanTrade", True):
            continue
        rules[pair] = PairRule(int(d.get("PricePrecision", 2)), int(d.get("AmountPrecision", 2)),
                               float(d.get("MiniOrder", 1.0)))
    return rules


def floor_to(x: float, decimals: int) -> float:
    f = 10 ** decimals
    return math.floor(x * f + 1e-9) / f


def fmt(x: float, decimals: int) -> str:
    return f"{x:.{decimals}f}" if decimals > 0 else str(int(x))


class Executor:
    def __init__(self, client, cfg: BotConfig, rules: dict[str, PairRule]):
        self.client = client
        self.cfg = cfg
        self.rules = rules

    def pair(self, coin: str) -> str:
        return f"{coin}/{self.cfg.quote}"

    def cancel_all(self) -> None:
        if self.cfg.dry_run:
            return
        try:
            pc = self.client.pending_count()
            if pc.get("TotalPending", 0) > 0:
                res = self.client.cancel_order()
                logger.event("trades", action="cancel_all", response=res)
        except Exception as exc:
            log.warning("cancel_all failed: %s", exc)

    def _send(self, coin: str, side: str, qty: float, order_type: str, price: float | None,
              reason: str) -> dict | None:
        pair = self.pair(coin)
        rule = self.rules[pair]
        qty = floor_to(qty, rule.amount_precision)
        ref_px = price or 0.0
        if qty <= 0 or qty * ref_px < max(rule.min_notional, 1.0):
            return None
        q_str = fmt(qty, rule.amount_precision)
        p_str = fmt(round(price, rule.price_precision), rule.price_precision) if order_type == "LIMIT" else None
        rec = dict(action="order", pair=pair, side=side, type=order_type, quantity=q_str, price=p_str,
                   notional=round(qty * ref_px, 2), reason=reason, dry_run=self.cfg.dry_run)
        if self.cfg.dry_run:
            logger.event("trades", **rec)
            log.info("[DRY] %s %s %s %s @ %s (%s)", side, q_str, pair, order_type, p_str, reason)
            return {"Success": True, "dry_run": True}
        try:
            res = self.client.place_order(pair, side, q_str, order_type, p_str)
        except Exception as exc:
            res = {"Success": False, "ErrMsg": str(exc)}
        rec["response"] = res
        logger.event("trades", **rec)
        if res.get("Success"):
            od = res.get("OrderDetail", {})
            log.info("%s %s %s %s -> %s @ %s (%s)", side, q_str, pair, order_type, od.get("Status"),
                     od.get("FilledAverPrice") or p_str, reason)
        else:
            log.warning("order rejected %s %s %s: %s", side, q_str, pair, res.get("ErrMsg"))
        return res

    def rebalance(self, targets: pd.Series, holdings: dict[str, float], usd_free: float, equity: float,
                  book: dict[str, dict], attempts: dict[str, int], urgent: set | None = None,
                  reason: str = "rebalance") -> list:
        """targets: coin -> weight of equity. holdings: coin -> free quantity.
        book: coin -> {'bid','ask','last'}. attempts is mutated (limit re-quote counter)."""
        urgent = urgent or set()
        band = self.cfg.strategy.rebalance_band
        sells, buys = [], []
        coins = set(targets.index) | {c for c, q in holdings.items() if q > 0}
        for coin in coins:
            if self.pair(coin) not in self.rules or coin not in book:
                continue
            px = book[coin]["last"]
            cur_val = holdings.get(coin, 0.0) * px
            tgt_w = float(targets.get(coin, 0.0))
            tgt_val = tgt_w * equity
            diff = tgt_val - cur_val
            full_exit = tgt_w <= 0 and cur_val >= max(self.rules[self.pair(coin)].min_notional, 1.0) * 2
            if not full_exit and (abs(diff) / max(equity, 1) < band or abs(diff) < self.cfg.min_order_usd):
                attempts.pop(coin, None)
                continue
            if full_exit and cur_val < self.cfg.min_order_usd and coin not in urgent:
                continue  # dust, not worth a fee
            (sells if diff < 0 else buys).append((coin, diff, full_exit))

        results = []
        for coin, diff, full_exit in sells:
            b = book[coin]
            qty = holdings.get(coin, 0.0) if full_exit else min(-diff / b["last"], holdings.get(coin, 0.0))
            market = coin in urgent or attempts.get(coin, 0) >= self.cfg.max_limit_attempts
            if market:
                res = self._send(coin, "SELL", qty, "MARKET", b["bid"], reason + (":urgent" if coin in urgent else ":fallback"))
            else:
                res = self._send(coin, "SELL", qty, "LIMIT", b["ask"], reason)
                attempts[coin] = attempts.get(coin, 0) + 1
            if res and res.get("Success") and market:
                usd_free += qty * b["bid"] * (1 - self.cfg.strategy.taker_fee)
                attempts.pop(coin, None)
            results.append((coin, "SELL", res))

        budget = usd_free * 0.995
        for coin, diff, _ in sorted(buys, key=lambda x: -x[1]):
            b = book[coin]
            spend = min(diff, budget)
            if spend < self.cfg.min_order_usd:
                continue
            market = attempts.get(coin, 0) >= self.cfg.max_limit_attempts
            if market:
                qty = spend / (b["ask"] * (1 + self.cfg.strategy.taker_fee))
                res = self._send(coin, "BUY", qty, "MARKET", b["ask"], reason + ":fallback")
                attempts.pop(coin, None)
            else:
                qty = spend / (b["bid"] * (1 + self.cfg.strategy.maker_fee))
                res = self._send(coin, "BUY", qty, "LIMIT", b["bid"], reason)
                attempts[coin] = attempts.get(coin, 0) + 1
            if res and res.get("Success"):
                budget -= spend
            results.append((coin, "BUY", res))
        return results
