"""Turns target weights into Roostoo orders.

Cost-aware by design (0.1% taker vs 0.05% maker):
  * changes smaller than the rebalance band are not traded (edge < cost);
  * routine rebalances are passive LIMIT orders at the touch (maker fee);
  * a limit that has not filled after N re-quotes falls back to MARKET;
  * risk exits (stops, crash guard, full exits flagged urgent) go MARKET immediately.
Sells are sent before buys so freed cash can fund new positions.

Shorts use Roostoo's /v6 endpoints: opened at market with USD collateral equal to the
target notional (1x, no leverage), reduced/closed with reduce-only short_close. The short
fee is 0.1% either way, so there is no limit-order variant. If the exchange answers that
shorts are not allowed, the executor sets `shorts_disabled` and the bot goes long/cash.
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
        self.shorts_disabled = False

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

    # ------------------------------------------------------------------ shorts
    def _short(self, action: str, coin: str, reason: str, collateral: float = 0.0,
               qty: float | None = None, ref_px: float = 0.0) -> dict | None:
        """action: 'open' (sized by USD collateral) or 'close' (qty=None closes all)."""
        pair = self.pair(coin)
        rule = self.rules[pair]
        rec = dict(action=f"short_{action}", pair=pair, reason=reason, dry_run=self.cfg.dry_run)
        if action == "open":
            if collateral < max(self.cfg.min_order_usd, 1.0):
                return None
            rec["collateral"] = c_str = f"{floor_to(collateral, 2):.2f}"
        else:
            q_str = None
            if qty is not None:
                qty = floor_to(qty, rule.amount_precision)
                if qty <= 0 or qty * ref_px < 1.0:
                    return None
                q_str = fmt(qty, rule.amount_precision)
            rec["close_qty"] = q_str or "ALL"
        if self.cfg.dry_run:
            logger.event("trades", **rec)
            log.info("[DRY] SHORT_%s %s %s (%s)", action.upper(), pair, rec.get("collateral") or rec.get("close_qty"), reason)
            return {"Success": True, "dry_run": True}
        try:
            res = (self.client.short_open(pair, c_str) if action == "open"
                   else self.client.short_close(pair, close_qty=q_str))
        except Exception as exc:
            res = {"Success": False, "ErrMsg": str(exc)}
        rec["response"] = res
        logger.event("trades", **rec)
        if res.get("Success"):
            log.info("SHORT_%s %s %s (%s) -> %s", action.upper(), pair, rec.get("collateral") or rec.get("close_qty"),
                     reason, {k: res.get(k) for k in ("EntryPrice", "ShortQty", "ClosePrice", "RealizedPNL") if k in res})
        else:
            err = str(res.get("ErrMsg", ""))
            log.warning("short %s rejected %s: %s", action, pair, err)
            if "not allow short" in err.lower() or "permission" in err.lower():
                self.shorts_disabled = True
        return res

    # --------------------------------------------------------------- rebalance
    def rebalance(self, targets: pd.Series, holdings: dict[str, float], usd_free: float, equity: float,
                  book: dict[str, dict], attempts: dict[str, int], urgent: set | None = None,
                  reason: str = "rebalance", shorts: dict[str, dict] | None = None) -> list:
        """targets: coin -> SIGNED weight of equity (negative = short).
        holdings: coin -> free spot quantity. shorts: coin -> {'qty', 'collateral'}.
        book: coin -> {'bid','ask','last'}. attempts is mutated (limit re-quote counter)."""
        urgent = urgent or set()
        shorts = shorts or {}
        band = self.cfg.strategy.rebalance_band
        fee = self.cfg.strategy.taker_fee
        reduce_long, add_long, reduce_short, add_short = [], [], [], []
        coins = set(targets.index) | {c for c, q in holdings.items() if q > 0} | set(shorts)
        for coin in coins:
            if self.pair(coin) not in self.rules or coin not in book:
                continue
            px = book[coin]["last"]
            tgt = float(targets.get(coin, 0.0))
            long_val = holdings.get(coin, 0.0) * px
            sh = shorts.get(coin)
            short_val = sh["qty"] * px if sh else 0.0
            min_val = max(self.rules[self.pair(coin)].min_notional, 1.0) * 2

            # ---- short side: close / reduce / grow
            if sh and tgt >= 0:
                reduce_short.append((coin, None))                      # close the whole short
            elif tgt < 0:
                tgt_val = -tgt * equity
                diff = tgt_val - short_val
                if abs(diff) / max(equity, 1) >= band and abs(diff) >= self.cfg.min_order_usd:
                    if diff > 0:
                        add_short.append((coin, diff))
                    elif sh:
                        reduce_short.append((coin, sh["qty"] * min(1.0, -diff / short_val)))

            # ---- long side
            if tgt <= 0:
                if long_val >= min_val and (long_val >= self.cfg.min_order_usd or coin in urgent):
                    # flips (long -> short) and urgent exits go at market
                    reduce_long.append((coin, holdings[coin], True, tgt < 0 or coin in urgent))
                elif coin in attempts and long_val < min_val:
                    attempts.pop(coin, None)
                continue
            diff = tgt * equity - long_val
            if abs(diff) / max(equity, 1) < band or abs(diff) < self.cfg.min_order_usd:
                attempts.pop(coin, None)
                continue
            if diff < 0:
                reduce_long.append((coin, min(-diff / px, holdings.get(coin, 0.0)), False, coin in urgent))
            else:
                add_long.append((coin, diff))

        results = []
        # 1) reductions first: they free cash
        for coin, qty in reduce_short:
            res = self._short("close", coin, reason + (":urgent" if coin in urgent else ""), qty=qty,
                              ref_px=book[coin]["last"])
            if res and res.get("Success"):
                usd_free += float(res.get("ReturnAmount", 0.0))
            results.append((coin, "SHORT_CLOSE", res))
        for coin, qty, full_exit, force_market in reduce_long:
            b = book[coin]
            market = force_market or attempts.get(coin, 0) >= self.cfg.max_limit_attempts
            if market:
                res = self._send(coin, "SELL", qty, "MARKET", b["bid"], reason + (":urgent" if coin in urgent else ":market"))
                if res and res.get("Success"):
                    usd_free += qty * b["bid"] * (1 - fee)
                    attempts.pop(coin, None)
            else:
                res = self._send(coin, "SELL", qty, "LIMIT", b["ask"], reason)
                attempts[coin] = attempts.get(coin, 0) + 1
            results.append((coin, "SELL", res))

        # 2) additions, largest first, within free cash
        budget = usd_free * 0.995
        adds = [("long", c, d) for c, d in add_long] + [("short", c, d) for c, d in add_short]
        for kind, coin, diff in sorted(adds, key=lambda x: -x[2]):
            b = book[coin]
            spend = min(diff, budget / (1 + fee))
            if spend < self.cfg.min_order_usd:
                continue
            if kind == "short":
                if self.shorts_disabled:
                    continue
                res = self._short("open", coin, reason, collateral=spend)
            elif attempts.get(coin, 0) >= self.cfg.max_limit_attempts:
                res = self._send(coin, "BUY", spend / (b["ask"] * (1 + fee)), "MARKET", b["ask"], reason + ":fallback")
                attempts.pop(coin, None)
            else:
                res = self._send(coin, "BUY", spend / (b["bid"] * (1 + self.cfg.strategy.maker_fee)), "LIMIT",
                                 b["bid"], reason)
                attempts[coin] = attempts.get(coin, 0) + 1
            if res and res.get("Success"):
                budget -= spend * (1 + fee)
            results.append((coin, "SHORT_OPEN" if kind == "short" else "BUY", res))
        return results
