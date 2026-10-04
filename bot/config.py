"""Central configuration. Everything tunable lives here and can be overridden
with environment variables (see .env.example) so that no strategy change ever
requires editing code on the server — every change goes through git."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default):
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    if isinstance(default, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    if isinstance(default, tuple):
        return tuple(type(default[0])(x) for x in raw.split(","))
    return raw


def _load_dotenv(path: str = ".env") -> None:
    """Tiny .env loader (avoids an extra dependency)."""
    if not os.path.exists(path):
        return
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

# Liquid crypto majors that also trade on Binance/OKX (needed for candle history).
DEFAULT_UNIVERSE = (
    "BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX", "SUI",
    "TRX", "LTC", "NEAR", "APT", "DOT", "UNI", "AAVE", "TON", "HBAR", "XLM",
    "ENA", "TAO", "ZEC", "PEPE", "WIF", "ARB", "FET", "ONDO", "CRV", "PENDLE",
)
MAJORS = ("BTC", "ETH")


@dataclass
class StrategyParams:
    # --- signal: volatility-normalised multi-horizon time-series momentum ---
    lookbacks: tuple = field(default_factory=lambda: _env("LOOKBACKS", (24, 72, 168)))
    vol_window: int = _env("VOL_WINDOW", 168)          # hours used for realised vol
    regime_ema: int = _env("REGIME_EMA", 200)          # BTC trend filter (hours)
    asset_ema: int = _env("ASSET_EMA", 100)            # per-asset trend filter (hours)
    # --- selection ---
    top_k: int = _env("TOP_K", 4)
    hold_buffer: int = _env("HOLD_BUFFER", 2)           # held assets survive down to rank top_k+buffer
    entry_threshold: float = _env("ENTRY_THRESHOLD", 0.25)
    exit_threshold: float = _env("EXIT_THRESHOLD", 0.05)
    min_hold_hours: int = _env("MIN_HOLD_HOURS", 24)    # no signal-driven exit before this (stops still apply)
    # --- sizing ---
    max_weight: float = _env("MAX_WEIGHT", 0.30)
    max_weight_major: float = _env("MAX_WEIGHT_MAJOR", 0.45)
    target_daily_vol: float = _env("TARGET_DAILY_VOL", 0.03)   # 3%/day portfolio vol target
    gross_cap: float = _env("GROSS_CAP", 0.95)                 # never > 95% invested (no leverage)
    full_conviction_score: float = _env("FULL_CONVICTION", 0.6)
    min_conviction: float = _env("MIN_CONVICTION", 0.35)
    neutral_regime_mult: float = _env("NEUTRAL_REGIME_MULT", 0.5)
    regime_score_smooth: int = _env("REGIME_SCORE_SMOOTH", 12)     # EMA span on BTC score (hours)
    regime_price_buffer: float = _env("REGIME_PRICE_BUFFER", 0.01)  # 1% hysteresis around BTC EMA
    regime_score_buffer: float = _env("REGIME_SCORE_BUFFER", 0.10)  # hysteresis around score 0
    # --- risk ---
    max_drawdown: float = _env("MAX_DRAWDOWN", 0.08)   # exposure -> floor as rolling DD approaches this
    dd_floor_mult: float = _env("DD_FLOOR_MULT", 0.25)  # never fully locked out of a recovery
    dd_lookback_hours: int = _env("DD_LOOKBACK_HOURS", 336)  # 14 days = competition length
    stop_vol_mult: float = _env("STOP_VOL_MULT", 2.5)  # trailing stop = k * daily vol
    stop_min_pct: float = _env("STOP_MIN_PCT", 0.06)
    stop_max_pct: float = _env("STOP_MAX_PCT", 0.15)
    stop_cooldown_hours: int = _env("STOP_COOLDOWN_HOURS", 12)
    crash_btc_1h: float = _env("CRASH_BTC_1H", -0.04)  # BTC -4% in 1h -> flatten
    crash_cooldown_hours: int = _env("CRASH_COOLDOWN_HOURS", 6)
    # --- costs / trading ---
    rebalance_band: float = _env("REBALANCE_BAND", 0.05)  # ignore weight changes < 5% of equity
    rebalance_every_hours: int = _env("REBALANCE_EVERY_HOURS", 8)  # UTC 00/08/16; stops run every minute
    taker_fee: float = 0.001
    maker_fee: float = 0.0005


@dataclass
class BotConfig:
    api_key: str = _env("ROOSTOO_API_KEY", "")
    secret_key: str = _env("ROOSTOO_SECRET_KEY", "")
    base_url: str = _env("ROOSTOO_BASE_URL", "https://mock-api.roostoo.com")
    universe: tuple = field(default_factory=lambda: _env("UNIVERSE", DEFAULT_UNIVERSE))
    quote: str = "USD"
    loop_seconds: int = _env("LOOP_SECONDS", 60)            # price/risk check cadence
    rebalance_minute: int = _env("REBALANCE_MINUTE", 2)     # rebalance at HH:02 (after hourly close)
    retry_minutes: int = _env("RETRY_MINUTES", 10)          # re-quote unfilled limit orders
    max_limit_attempts: int = _env("MAX_LIMIT_ATTEMPTS", 3) # then fall back to market
    min_order_usd: float = _env("MIN_ORDER_USD", 50.0)
    history_bars: int = _env("HISTORY_BARS", 500)
    min_request_interval: float = _env("MIN_REQUEST_INTERVAL", 0.35)  # seconds between API calls
    dry_run: bool = _env("DRY_RUN", False)
    log_dir: str = _env("LOG_DIR", "logs")
    data_dir: str = _env("DATA_DIR", "data")
    strategy: StrategyParams = field(default_factory=StrategyParams)
