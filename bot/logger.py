"""Structured, append-only logs. These files are the audit trail that proves the
bot traded autonomously and consistently with its declared strategy:

  logs/bot.log          human-readable runtime log
  logs/api.jsonl        every Roostoo API call (endpoint, params, success, error)
  logs/decisions.jsonl  every rebalance decision: signals, regime, targets, reasons
  logs/trades.jsonl     every order placed and its exchange response
  logs/equity.csv       portfolio value snapshot every loop
"""
from __future__ import annotations

import json
import logging
import os
import time
from logging.handlers import RotatingFileHandler

_LOG_DIR = "logs"


def setup(log_dir: str = "logs") -> logging.Logger:
    global _LOG_DIR
    _LOG_DIR = log_dir
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("bot")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    fh = RotatingFileHandler(os.path.join(log_dir, "bot.log"), maxBytes=20_000_000, backupCount=10)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def get(name: str = "bot") -> logging.Logger:
    return logging.getLogger(name if name.startswith("bot") else f"bot.{name}")


def event(stream: str, **fields) -> None:
    """Append one JSON record to logs/<stream>.jsonl."""
    os.makedirs(_LOG_DIR, exist_ok=True)
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "ts_ms": int(time.time() * 1000)}
    rec.update(fields)
    with open(os.path.join(_LOG_DIR, f"{stream}.jsonl"), "a") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")


def equity_row(equity: float, cash: float, drawdown: float, exposure: float) -> None:
    os.makedirs(_LOG_DIR, exist_ok=True)
    path = os.path.join(_LOG_DIR, "equity.csv")
    new = not os.path.exists(path)
    with open(path, "a") as fh:
        if new:
            fh.write("ts,equity,cash,drawdown,exposure\n")
        fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())},{equity:.2f},{cash:.2f},{drawdown:.5f},{exposure:.4f}\n")
