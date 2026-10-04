"""Regime-gated, volatility-managed cross-asset momentum.

Pure functions only (no I/O), shared verbatim by the live bot and the backtester
so that what we test is exactly what trades.

Pipeline, evaluated once per closed hourly bar:

1. Signal  - for each asset and horizon h in {24h, 72h, 168h}:
             z_h = log(P_t / P_{t-h}) / (sigma_1h * sqrt(h))       (vol-normalised TSMOM)
             score = mean_h tanh(z_h)      in [-1, 1]
             An asset below its own 100h EMA has its score capped at 0.
             (Moskowitz, Ooi & Pedersen 2012; Liu & Tsyvinski 2021)
2. Regime  - BTC drives crypto beta. Risk-on (x1) when BTC score > 0 AND BTC > 200h EMA,
             neutral (x0.5) when exactly one holds, risk-off (cash) otherwise.
3. Select  - top-K positive-score assets. Hysteresis: a held asset stays while its
             score > exit threshold and rank < K + buffer; a new one needs score >
             entry threshold and rank < K. Cuts churn, and therefore fees.
4. Size    - weights proportional to score / vol (risk parity tilted to conviction),
             per-asset caps, then scaled so ex-ante portfolio vol <= target
             (Moreira & Muir 2017, volatility-managed portfolios), then by conviction,
             regime and the drawdown multiplier. Gross exposure <= 95%: no leverage.
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


def regime_multiplier(btc_closes: pd.Series, btc_score: float, p: StrategyParams) -> tuple[float, str]:
    if btc_closes is None or len(btc_closes) < p.regime_ema // 2 or np.isnan(btc_score):
        return 0.0, "unknown"
    ema = btc_closes.ewm(span=p.regime_ema, adjust=False).mean().iloc[-1]
    above = btc_closes.iloc[-1] > ema
    positive = btc_score > 0
    if above and positive:
        return 1.0, "risk_on"
    if above or positive:
        return p.neutral_regime_mult, "neutral"
    return 0.0, "risk_off"


def drawdown_multiplier(drawdown: float, p: StrategyParams) -> float:
    """Linear de-risking: full size at 0 DD, floor at max_drawdown (Grossman-Zhou style)."""
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


def select_assets(scores: pd.Series, held: set, p: StrategyParams) -> list:
    ranked = scores.dropna().sort_values(ascending=False)
    chosen = []
    for rank, (asset, s) in enumerate(ranked.items()):
        if asset in held and s > p.exit_threshold and rank < p.top_k + p.hold_buffer:
            chosen.append(asset)
        elif s > p.entry_threshold and rank < p.top_k:
            chosen.append(asset)
    return chosen[: p.top_k + 1]


def target_weights(scores: pd.Series, vols: pd.Series, recent_returns: pd.DataFrame,
                   held: set, p: StrategyParams, regime_mult: float = 1.0,
                   dd_mult: float = 1.0, excluded: set | None = None) -> tuple[pd.Series, dict]:
    """Compute long-only target weights (fraction of equity) for one rebalance."""
    info: dict = {}
    scores = scores.drop(labels=[a for a in (excluded or set()) if a in scores.index])
    chosen = select_assets(scores, held, p)
    info["chosen"] = chosen
    if not chosen or regime_mult <= 0 or dd_mult <= 0:
        return pd.Series(dtype=float), info

    s = scores[chosen].clip(lower=1e-6)
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


def decide(closes: pd.DataFrame, held: set, p: StrategyParams, drawdown: float = 0.0,
           excluded: set | None = None) -> tuple[pd.Series, dict]:
    """Full decision for the latest closed bar — the single entry point the live bot uses."""
    score, vol = score_frame(closes, p)
    last_scores = score.iloc[-1]
    last_vol = vol.iloc[-1]
    btc_score = float(last_scores.get("BTC", np.nan))
    regime, regime_name = regime_multiplier(closes.get("BTC"), btc_score, p)
    dd_mult = drawdown_multiplier(drawdown, p)
    recent = np.log(closes).diff().tail(p.vol_window)
    w, info = target_weights(last_scores, last_vol, recent, held, p, regime, dd_mult, excluded)
    info["daily_vol"] = {k: round(float(x) * np.sqrt(24), 5) for k, x in last_vol.dropna().items()}
    info.update(regime=regime_name, btc_score=round(btc_score, 3) if not np.isnan(btc_score) else None,
                scores={k: round(float(x), 3) for k, x in last_scores.dropna().sort_values(ascending=False).head(10).items()},
                bar=str(closes.index[-1]))
    return w, info
