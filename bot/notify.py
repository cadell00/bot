"""Telegram notifications (optional; enabled when TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are set).

Sends: startup, position opened / closed / resized (detected from the account itself, so
limit orders that fill later are reported too), stops and guards, errors (rate-limited),
shutdown and crashes, and a daily summary.

Messages go out on a background thread with a short timeout, and every failure is
swallowed: a Telegram problem must never affect trading.
"""
from __future__ import annotations

import threading
import time

import requests

from . import logger

log = logger.get("notify")

MIN_NOTIONAL_CHANGE = 25.0       # ignore position changes smaller than this (USD)
ERROR_REPEAT_SECONDS = 1800      # same error is reported at most every 30 minutes


class Telegram:
    def __init__(self, token: str, chat_id: str, prefix: str = ""):
        self.token = token
        self.chat_id = chat_id
        self.prefix = prefix
        self.enabled = bool(token and chat_id)
        self._last_error: dict[str, float] = {}

    def _post(self, text: str) -> None:
        try:
            requests.post(f"https://api.telegram.org/bot{self.token}/sendMessage",
                          data={"chat_id": self.chat_id, "text": text[:4000],
                                "disable_web_page_preview": "true"}, timeout=8)
        except Exception as exc:
            log.debug("telegram send failed: %s", exc)

    def send(self, text: str, sync: bool = False) -> None:
        if not self.enabled:
            return
        text = f"{self.prefix}{text}"
        if sync:
            self._post(text)
        else:
            threading.Thread(target=self._post, args=(text,), daemon=True).start()

    def error(self, key: str, text: str) -> None:
        """Report an error, but the same `key` at most once per ERROR_REPEAT_SECONDS."""
        now = time.time()
        if now - self._last_error.get(key, 0.0) < ERROR_REPEAT_SECONDS:
            return
        self._last_error[key] = now
        self.send(f"⚠️ Error: {text}")


def _fmt_px(px: float) -> str:
    return f"{px:,.2f}" if px >= 1 else f"{px:.6g}"


def position_changes(prev: dict[str, float], cur: dict[str, float], prices: dict[str, float],
                     entry_px: dict[str, float]) -> list[str]:
    """Compare signed positions (coin -> qty; negative = short) between two snapshots and
    describe fills: opened, closed (with approximate P&L), flipped or resized."""
    msgs = []
    for coin in sorted(set(prev) | set(cur)):
        old, new = prev.get(coin, 0.0), cur.get(coin, 0.0)
        px = prices.get(coin)
        if not px or abs(new - old) * px < MIN_NOTIONAL_CHANGE:
            continue
        old_side = "LONG" if old > 0 else "SHORT"
        new_side = "LONG" if new > 0 else "SHORT"
        closed = abs(old) * px >= MIN_NOTIONAL_CHANGE and (abs(new) * px < MIN_NOTIONAL_CHANGE or old * new < 0)
        opened = abs(new) * px >= MIN_NOTIONAL_CHANGE and (abs(old) * px < MIN_NOTIONAL_CHANGE or old * new < 0)
        if closed:
            line = f"🔴 Closed {old_side} {coin}: {abs(old):g} @ ~{_fmt_px(px)}"
            entry = entry_px.get(coin)
            if entry:
                pnl = (px / entry - 1) * (1 if old > 0 else -1)
                line += f"  (P&L ≈ {pnl * 100:+.2f}%, ${pnl * abs(old) * entry:+,.0f})"
            msgs.append(line)
        if opened:
            msgs.append(f"🟢 Opened {new_side} {coin}: {abs(new):g} @ ~{_fmt_px(px)}  (${abs(new) * px:,.0f})")
        if not closed and not opened:
            verb = "Added to" if abs(new) > abs(old) else "Reduced"
            msgs.append(f"🔁 {verb} {new_side} {coin}: {abs(old):g} → {abs(new):g} @ ~{_fmt_px(px)}  "
                        f"(${abs(new) * px:,.0f})")
    return msgs