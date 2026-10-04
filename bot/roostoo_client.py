"""Roostoo mock-exchange REST client.

Implements the signing scheme from https://github.com/roostoo/Roostoo-API-Documents:
HMAC-SHA256 over the key-sorted `k=v&k=v` parameter string, sent in the
MSG-SIGNATURE header with RST-API-KEY. Adds client-side throttling, retries with
backoff, server-clock offset correction, and logs every call to logs/api.jsonl.
"""
from __future__ import annotations

import hashlib
import hmac
import time
from typing import Any

import requests

from . import logger

log = logger.get("client")


class RoostooError(RuntimeError):
    pass


class RoostooFatalError(RoostooError):
    """4xx response: retrying will not help (wrong key, wrong host, bad request)."""


class RoostooClient:
    def __init__(self, api_key: str, secret_key: str, base_url: str = "https://mock-api.roostoo.com",
                 min_interval: float = 0.35, timeout: float = 10.0, max_retries: int = 3):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url.rstrip("/")
        self.min_interval = min_interval
        self.timeout = timeout
        self.max_retries = max_retries
        self._last_call = 0.0
        self._clock_offset_ms = 0
        self.session = requests.Session()

    # ------------------------------------------------------------------ utils
    def _timestamp(self) -> str:
        return str(int(time.time() * 1000) + self._clock_offset_ms)

    def sign(self, params: dict) -> tuple[str, str]:
        total_params = "&".join(f"{k}={params[k]}" for k in sorted(params))
        sig = hmac.new(self.secret_key.encode(), total_params.encode(), hashlib.sha256).hexdigest()
        return total_params, sig

    def _throttle(self) -> None:
        wait = self.min_interval - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.time()

    def sync_clock(self) -> None:
        try:
            server = self.server_time()["ServerTime"]
            self._clock_offset_ms = int(server - time.time() * 1000)
            log.info("clock offset vs server: %d ms", self._clock_offset_ms)
        except Exception as exc:  # non-fatal
            log.warning("clock sync failed: %s", exc)

    def _request(self, method: str, path: str, params: dict | None = None,
                 signed: bool = False, timestamped: bool = False) -> dict:
        params = dict(params or {})
        url = f"{self.base_url}{path}"
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            p = dict(params)
            headers: dict[str, str] = {}
            if signed or timestamped:
                p["timestamp"] = self._timestamp()
            body = None
            if signed:
                body, sig = self.sign(p)
                headers["RST-API-KEY"] = self.api_key
                headers["MSG-SIGNATURE"] = sig
            self._throttle()
            try:
                if method == "GET":
                    resp = self.session.get(url, params=p, headers=headers, timeout=self.timeout)
                else:
                    headers["Content-Type"] = "application/x-www-form-urlencoded"
                    if body is None:
                        body, _ = self.sign(p)
                    resp = self.session.post(url, data=body, headers=headers, timeout=self.timeout)
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise RoostooError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                if resp.status_code >= 400:
                    # client errors (bad key, wrong host, bad params) won't fix themselves: fail fast
                    # and keep the server's own message for the logs
                    logger.event("api", method=method, path=path, params=params, success=False,
                                 err=f"HTTP {resp.status_code}: {resp.text[:300]}")
                    raise RoostooFatalError(f"{method} {path} -> HTTP {resp.status_code}: {resp.text[:300]}")
                data = resp.json()
                ok = data.get("Success", True) if isinstance(data, dict) else True
                logger.event("api", method=method, path=path,
                             params={k: v for k, v in p.items() if k != "timestamp"},
                             success=ok, err=(data.get("ErrMsg") if isinstance(data, dict) else None))
                return data
            except RoostooFatalError:
                raise
            except Exception as exc:
                last_exc = exc
                logger.event("api", method=method, path=path, params=params, success=False,
                             err=str(exc), attempt=attempt)
                if "timestamp" in str(exc).lower():
                    self.sync_clock()
                time.sleep(min(2 ** attempt, 15))
        raise RoostooError(f"{method} {path} failed after {self.max_retries} attempts: {last_exc}")

    # --------------------------------------------------------------- public
    def server_time(self) -> dict:
        return self._request("GET", "/v3/serverTime")

    def exchange_info(self) -> dict:
        return self._request("GET", "/v3/exchangeInfo")

    def ticker(self, pair: str | None = None) -> dict:
        params = {"pair": pair} if pair else {}
        return self._request("GET", "/v3/ticker", params, timestamped=True)

    # --------------------------------------------------------------- signed
    def balance(self) -> dict:
        return self._request("GET", "/v3/balance", signed=True)

    def pending_count(self) -> dict:
        return self._request("GET", "/v3/pending_count", signed=True)

    def place_order(self, pair: str, side: str, quantity: str, order_type: str = "MARKET",
                    price: str | None = None) -> dict:
        params: dict[str, Any] = {"pair": pair, "side": side.upper(), "type": order_type.upper(),
                                  "quantity": quantity}
        if order_type.upper() == "LIMIT":
            if price is None:
                raise ValueError("LIMIT order requires price")
            params["price"] = price
        return self._request("POST", "/v3/place_order", params, signed=True)

    def query_order(self, order_id: str | None = None, pair: str | None = None,
                    pending_only: bool | None = None, limit: int | None = None) -> dict:
        params: dict[str, Any] = {}
        if order_id:
            params["order_id"] = str(order_id)
        else:
            if pair:
                params["pair"] = pair
            if pending_only is not None:
                params["pending_only"] = "TRUE" if pending_only else "FALSE"
            if limit:
                params["limit"] = str(limit)
        return self._request("POST", "/v3/query_order", params, signed=True)

    def cancel_order(self, order_id: str | None = None, pair: str | None = None) -> dict:
        params: dict[str, Any] = {}
        if order_id:
            params["order_id"] = str(order_id)
        elif pair:
            params["pair"] = pair
        return self._request("POST", "/v3/cancel_order", params, signed=True)

    # ---------------------------------------------------------------- shorts (/v6)
    def short_open(self, pair: str, collateral: str) -> dict:
        """Market short sized by USD collateral (qty = collateral / entry price, 1x)."""
        return self._request("POST", "/v6/short_open", {"pair": pair, "collateral": collateral}, signed=True)

    def short_close(self, pair: str, close_qty: str | None = None, close_pct: str | None = None) -> dict:
        """Reduce-only close at MinAsk. No size = close the whole position."""
        params: dict[str, Any] = {"pair": pair}
        if close_qty is not None:
            params["close_qty"] = close_qty
        elif close_pct is not None:
            params["close_pct"] = close_pct
        return self._request("POST", "/v6/short_close", params, signed=True)

    def short_positions(self) -> dict:
        return self._request("GET", "/v6/short_positions", signed=True)