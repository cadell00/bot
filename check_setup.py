"""Read-only pre-flight check run on the EC2 box before starting the bot.
Places NO orders: verifies keys/signing (balance), Roostoo connectivity, and that
free candle sources are reachable from this server.

    .venv/bin/python scripts/check_setup.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot import logger  # noqa: E402
from bot.config import BotConfig  # noqa: E402
from bot.data import fetch_binance, fetch_okx  # noqa: E402
from bot.roostoo_client import RoostooClient  # noqa: E402

cfg = BotConfig()
logger.setup(cfg.log_dir)
c = RoostooClient(cfg.api_key, cfg.secret_key, cfg.base_url)
print("server time :", c.server_time())
info = c.exchange_info()
print("pairs       :", len(info.get("TradePairs", {})), "| initial wallet:", info.get("InitialWallet"))
print("BTC ticker  :", c.ticker("BTC/USD").get("Data"))
bal = c.balance()
print("balance ok  :", bal.get("Success"), bal.get("ErrMsg") or "", {k: v for k, v in (bal.get("Wallet") or {}).items() if v.get("Free") or v.get("Lock")})
b = fetch_binance("BTC", 50)
print("binance 1h  :", len(b), "bars" + ("" if len(b) else "  (unreachable — will try OKX)"))
if not len(b):
    print("okx 1h      :", len(fetch_okx("BTC", 50)), "bars")
