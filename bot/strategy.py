"""Regime-gated, volatility-managed cross-asset momentum.

Pure functions only (no I/O), shared verbatim by the live bot and the backtester
so that what we test is exactly what trades.

Pipeline, evaluated once per closed hourly bar:

1. Signal  - for each asset and horizon h in {24h, 72h, 168h}:
             z_h = log(P_t / P_{t-h}) / (sigma_1h * sqrt(h))       (vol-normalised TSMOM)
             score = mean_h tanh(z_h)      in [-1, 1]
             An asset below its own 100h EMA has its score capped at 0.
             (Moskowitz, Ooi & Pedersen 2012; Liu & Tsyvinski 2021)
2. Regime  - BTC drives crypto beta. Risk-on (x1) when BTC's smoothed score > 0 AND BTC >
             200h EMA, neutral (x0.5) when exactly one holds, risk-off (cash) otherwise.
             Both conditions use hysteresis buffers so the regime does not flip hourly.
3. Select  - top-K positive-score assets. Hysteresis: a held asset stays while its
             score > exit threshold and rank < K + buffer; a new one needs score >
             entry threshold and rank < K. Cuts churn, and therefore fees.
4. Size    - weights proportional to score / vol (risk parity tilted to conviction),
             per-asset caps, then scaled so ex-ante portfolio vol <= target
             (Moreira & Muir 2017, volatility-managed portfolios), then by conviction,
             regime and the drawdown multiplier (drawdown measured from a rolling 7-day
             peak, with a floor). Gross exposure <= 95%: no leverage.
5. Cadence - positions are retargeted every 4 hours; stops and the crash guard run in
             between. Fees, not signal, were the binding constraint in backtests.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MAJORS, StrategyParams


def score_frame(closes: pd.DataFrame, p: StrategyParams) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (score, hourly_vol) frames aligned with `closes`."""
    logp = np.log(closes)
    r1 = logp.diff()
    vol = r1.rolling(p.vol_window, min_periods=max(24, p.vol_window // 2)).std()
    parts = []
    for h in p.lookbacks:
        z = (logp - logp.shift(h)) / (vol * np.sqrt(h))
        parts.append(np.tanh(z))
    score = sum(parts) / len(parts)
    ema = closes.ewm(span=p.asset_ema, adjust=False, min_periods=p.asset_ema // 2).mean()
    score = score.where(closes >= ema, np.minimum(score, 0.0))
    return score, vol


def regime_state(btc_px: float, btc_ema: float, btc_score_smooth: float, prev: dict | None,
                 p: StrategyParams) -> tuple[float, str, dict]:
    """BTC regime with hysteresis. Each condition needs a clear move past its line to
    switch on and a clear move back to switch off, so the regime (and therefore
    portfolio exposure) does not flip every hour when BTC hovers near its trend."""
    if any(np.isnan(x) for x in (btc_px, btc_ema, btc_score_smooth)):
        return 0.0, "unknown", dict(prev or {})
    prev = prev or {}
    pb, sb = p.regime_price_buffer, p.regime_score_buffer
    above = btc_px > btc_ema * (1 - pb) if prev.get("above") else btc_px > btc_ema * (1 + pb)
    positive = btc_score_smooth > -sb if prev.get("positive") else btc_score_smooth > sb
    new = {"above": bool(above), "positive": bool(positive)}
    if above and positive:
        return 1.0, "risk_on", new
    if above or positive:
        return p.neutral_regime_mult, "neutral", new
    return 0.0, "risk_off", new


def rolling_drawdown(equity_hist, lookback: int) -> float:
    """Drawdown from the peak of the last `lookback` equity samples (hourly). A rolling
    peak lets exposure recover after a loss instead of staying locked out for good."""
    tail = list(equity_hist)[-lookback:]
    if not tail:
        return 0.0
    peak = max(tail)
    return max(0.0, 1.0 - tail[-1] / peak) if peak > 0 else 0.0


def drawdown_multiplier(drawdown: float, p: StrategyParams) -> float:
    """Linear de-risking with a floor (Grossman-Zhou style)."""
    if p.max_drawdown <= 0:
        return 1.0
    return float(np.clip(1.0 - drawdown / p.max_drawdown, p.dd_floor_mult, 1.0))


def _cap(w: pd.Series, caps: pd.Series) -> pd.Series:
    w = w.copy()
    for _ in range(20):
        over = w > caps + 1e-12
        if not over.any():
            break
        excess = (w[over] - caps[over]).sum()
        w[over] = caps[over]
        room = (~over) & (w < caps)
        if not room.any() or w[room].sum() <= 0:
            break
        w[room] += excess * w[room] / w[room].sum()
    return w


def select_assets(scores: pd.Series, held: set, p: StrategyParams, locked: set | None = None) -> list:
    """`locked` = held positions younger than min_hold_hours: kept regardless of score
    (stops, crash guard and risk-off still close them). Prevents enter/exit whipsaw,
    which was the largest source of fees in backtests."""
    locked = locked or set()
    ranked = scores.dropna().sort_values(ascending=False)
    chosen = []
    for rank, (asset, s) in enumerate(ranked.items()):
        if asset in held and asset in locked:
            chosen.append(asset)
        elif asset in held and s > p.exit_threshold and rank < p.top_k + p.hold_buffer:
            chosen.append(asset)
        elif s > p.entry_threshold and rank < p.top_k:
            chosen.append(asset)
    return chosen[: p.top_k + 1]


def target_weights(scores: pd.Series, vols: pd.Series, recent_returns: pd.DataFrame,
                   held: set, p: StrategyParams, regime_mult: float = 1.0,
                   dd_mult: float = 1.0, excluded: set | None = None,
                   locked: set | None = None) -> tuple[pd.Series, dict]:
    """Compute long-only target weights (fraction of equity) for one rebalance."""
    info: dict = {}
    locked = locked or set()
    scores = scores.drop(labels=[a for a in (excluded or set()) if a in scores.index])
    chosen = select_assets(scores, held, p, locked)
    info["chosen"] = chosen
    if not chosen or regime_mult <= 0 or dd_mult <= 0:
        return pd.Series(dtype=float), info

    s = scores[chosen].clip(lower=1e-6)
    for a in chosen:  # a locked position keeps at least an entry-level weight
        if a in locked:
            s[a] = max(s[a], p.entry_threshold)
    v = vols[chosen].replace(0, np.nan).fillna(vols[chosen].median()).clip(lower=1e-4)
    raw = s / v
    w = raw / raw.sum()
    caps = pd.Series([p.max_weight_major if a in MAJORS else p.max_weight for a in chosen], index=chosen)
    w = _cap(w, caps)

    # ex-ante portfolio vol (daily) from recent hourly returns
    rr = recent_returns[chosen].dropna(how="all").fillna(0.0)
    if len(rr) >= 24:
        cov = rr.cov().values * 24.0
        port_vol = float(np.sqrt(max(w.values @ cov @ w.values, 1e-12)))
    else:
        port_vol = float((w * v * np.sqrt(24)).sum())
    vol_scale = min(1.0, p.target_daily_vol / port_vol) if port_vol > 0 else 1.0

    conviction = float(np.clip(s.mean() / p.full_conviction_score, p.min_conviction, 1.0))
    scale = vol_scale * conviction * regime_mult * dd_mult
    w = w * scale
    if w.sum() > p.gross_cap:
        w *= p.gross_cap / w.sum()

    info.update(port_vol_daily=round(port_vol, 5), vol_scale=round(vol_scale, 3),
                conviction=round(conviction, 3), regime_mult=regime_mult, dd_mult=round(dd_mult, 3),
                gross=round(float(w.sum()), 4))
    return w, info


def trailing_stop_pct(daily_vol: float, p: StrategyParams) -> float:
    return float(np.clip(p.stop_vol_mult * daily_vol, p.stop_min_pct, p.stop_max_pct))


def btc_regime_inputs(closes: pd.DataFrame, score: pd.DataFrame, p: StrategyParams) -> pd.DataFrame:
    """Per-bar BTC price, trend EMA and smoothed score (vectorised; used live and in backtests)."""
    if "BTC" not in closes:
        return pd.DataFrame(index=closes.index, columns=["px", "ema", "score"], dtype=float)
    btc = closes["BTC"]
    return pd.DataFrame({
        "px": btc,
        "ema": btc.ewm(span=p.regime_ema, adjust=False, min_periods=p.regime_ema // 2).mean(),
        "score": score["BTC"].ewm(span=p.regime_score_smooth, adjust=False).mean(),
    })


def decide(closes: pd.DataFrame, held: set, p: StrategyParams, drawdown: float = 0.0,
           excluded: set | None = None, prev_regime: dict | None = None,
           locked: set | None = None) -> tuple[pd.Series, dict]:
    """Full decision for the latest closed bar — the single entry point the live bot uses."""
    score, vol = score_frame(closes, p)
    last_scores = score.iloc[-1]
    last_vol = vol.iloc[-1]
    btc_score = float(last_scores.get("BTC", np.nan))
    ri = btc_regime_inputs(closes, score, p).iloc[-1]
    regime, regime_name, regime_st = regime_state(float(ri.px), float(ri.ema), float(ri.score), prev_regime, p)
    dd_mult = drawdown_multiplier(drawdown, p)
    recent = np.log(closes).diff().tail(p.vol_window)
    w, info = target_weights(last_scores, last_vol, recent, held, p, regime, dd_mult, excluded, locked)
    info["daily_vol"] = {k: round(float(x) * np.sqrt(24), 5) for k, x in last_vol.dropna().items()}
    info["regime_state"] = regime_st
    info.update(regime=regime_name, btc_score=round(btc_score, 3) if not np.isnan(btc_score) else None,
                scores={k: round(float(x), 3) for k, x in last_scores.dropna().sort_values(ascending=False).head(10).items()},
                bar=str(closes.index[-1]))
    return w, info
