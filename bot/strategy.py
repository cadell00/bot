"""Regime-switching, volatility-managed long/short momentum.

Pure functions only (no I/O), shared verbatim by the live bot and the backtester
so that what we test is exactly what trades. Target weights are SIGNED fractions
of equity: positive = long spot, negative = short (Roostoo /v6 short, 1x collateral).

Pipeline, evaluated on each rebalance bar:

1. Signal  - for each asset and horizon h in `lookbacks`:
             z_h = log(P_t / P_{t-h}) / (sigma_1h * sqrt(h))       (vol-normalised TSMOM)
             score = mean_h tanh(z_h)      in [-1, 1]
             Trend filter: above its own EMA an asset can only score >= 0, below it <= 0.
             (Moskowitz, Ooi & Pedersen 2012; Liu & Tsyvinski 2021)
2. Regime  - BTC drives crypto beta. Risk-on when BTC's smoothed score > 0 AND BTC is above
             its regime EMA; risk-off when neither holds; neutral otherwise. Hysteresis
             buffers stop the regime flipping every hour.
               risk_on  -> long book x1,  no shorts
               neutral  -> long book x neutral_regime_mult, short book x short_neutral_mult
               risk_off -> no longs,      short book x1
3. Select  - longs: top-K by score; shorts: bottom-K by score (most negative). Rank and
             threshold hysteresis plus a minimum hold keep turnover (fees) down.
4. Size    - per book: weight ~ |score| / vol, per-asset caps, scaled to an ex-ante daily
             vol target (Moreira & Muir 2017), then by conviction, regime and a drawdown
             multiplier (Grossman & Zhou 1993). Gross |exposure| <= gross_cap: no leverage.
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
    score = score.where(closes >= ema, np.minimum(score, 0.0))   # below EMA: no long signal
    score = score.where(closes <= ema, np.maximum(score, 0.0))   # above EMA: no short signal
    return score, vol


def regime_state(btc_px: float, btc_ema: float, btc_score_smooth: float, prev: dict | None,
                 p: StrategyParams) -> tuple[float, str, dict]:
    """BTC regime with hysteresis. Returns (long multiplier, regime name, new state)."""
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


def short_multiplier(regime_name: str, p: StrategyParams) -> float:
    if not p.allow_shorts:
        return 0.0
    return {"risk_off": 1.0, "neutral": p.short_neutral_mult}.get(regime_name, 0.0)


def rolling_drawdown(equity_hist, lookback: int) -> float:
    """Drawdown from the peak of the last `lookback` equity samples (hourly)."""
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


def select_assets(strength: pd.Series, held: set, top_k: int, entry: float, exit_: float,
                  hold_buffer: int, locked: set | None = None) -> list:
    """`strength` is the score in the book's direction (score for longs, -score for shorts).
    `locked` = held positions younger than min_hold_hours: kept regardless of score (stops,
    crash/squeeze guards and regime changes still close them)."""
    locked = locked or set()
    ranked = strength.dropna().sort_values(ascending=False)
    chosen = []
    for rank, (asset, s) in enumerate(ranked.items()):
        if asset in held and asset in locked:
            chosen.append(asset)
        elif asset in held and s > exit_ and rank < top_k + hold_buffer:
            chosen.append(asset)
        elif s > entry and rank < top_k:
            chosen.append(asset)
    return chosen[: top_k + 1]


def _size_book(strength: pd.Series, vols: pd.Series, recent_returns: pd.DataFrame, chosen: list,
               locked: set, entry: float, caps: pd.Series, mult: float, p: StrategyParams) -> tuple[pd.Series, dict]:
    s = strength[chosen].clip(lower=1e-6)
    for a in chosen:  # a locked position keeps at least an entry-level weight
        if a in locked:
            s[a] = max(s[a], entry)
    v = vols[chosen].replace(0, np.nan).fillna(vols[chosen].median()).clip(lower=1e-4)
    raw = s / v
    w = _cap(raw / raw.sum(), caps)
    rr = recent_returns[chosen].dropna(how="all").fillna(0.0)
    if len(rr) >= 24:
        cov = rr.cov().values * 24.0
        port_vol = float(np.sqrt(max(w.values @ cov @ w.values, 1e-12)))
    else:
        port_vol = float((w * v * np.sqrt(24)).sum())
    vol_scale = min(1.0, p.target_daily_vol / port_vol) if port_vol > 0 else 1.0
    conviction = float(np.clip(s.mean() / p.full_conviction_score, p.min_conviction, 1.0))
    w = w * vol_scale * conviction * mult
    return w, dict(port_vol_daily=round(port_vol, 5), vol_scale=round(vol_scale, 3), conviction=round(conviction, 3))


def target_weights(scores: pd.Series, vols: pd.Series, recent_returns: pd.DataFrame,
                   held: set, p: StrategyParams, regime_mult: float = 1.0,
                   dd_mult: float = 1.0, excluded: set | None = None,
                   locked: set | None = None, short_mult: float = 0.0,
                   held_short: set | None = None) -> tuple[pd.Series, dict]:
    """Signed target weights (fraction of equity) for one rebalance.
    held = coins held long, held_short = coins held short."""
    info: dict = {}
    locked = locked or set()
    held_short = held_short or set()
    scores = scores.drop(labels=[a for a in (excluded or set()) if a in scores.index])
    out = pd.Series(dtype=float)

    long_chosen = select_assets(scores, held, p.top_k, p.entry_threshold, p.exit_threshold,
                                p.hold_buffer, locked)
    info["chosen"] = long_chosen
    if long_chosen and regime_mult > 0 and dd_mult > 0:
        caps = pd.Series([p.max_weight_major if a in MAJORS else p.max_weight for a in long_chosen],
                         index=long_chosen)
        w, li = _size_book(scores, vols, recent_returns, long_chosen, locked, p.entry_threshold,
                           caps, regime_mult * dd_mult, p)
        out = pd.concat([out, w])
        info.update(li)

    short_chosen = []
    if short_mult > 0 and dd_mult > 0:
        short_chosen = select_assets(-scores, held_short, p.short_top_k, p.short_entry_threshold,
                                     p.short_exit_threshold, p.hold_buffer, locked)
        short_chosen = [a for a in short_chosen if a not in out.index]
        if short_chosen:
            caps = pd.Series(p.short_max_weight, index=short_chosen)
            w, si = _size_book(-scores, vols, recent_returns, short_chosen, locked, p.short_entry_threshold,
                               caps, short_mult * dd_mult, p)
            out = pd.concat([out, -w])
            info["short"] = si
    info["short_chosen"] = short_chosen

    gross = float(out.abs().sum()) if len(out) else 0.0
    if gross > p.gross_cap:
        out *= p.gross_cap / gross
    info.update(regime_mult=regime_mult, short_mult=short_mult, dd_mult=round(dd_mult, 3),
                gross=round(float(out.abs().sum()) if len(out) else 0.0, 4),
                net=round(float(out.sum()) if len(out) else 0.0, 4))
    return out, info


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
           locked: set | None = None, held_short: set | None = None) -> tuple[pd.Series, dict]:
    """Full decision for the latest closed bar — the single entry point the live bot uses."""
    score, vol = score_frame(closes, p)
    last_scores = score.iloc[-1]
    last_vol = vol.iloc[-1]
    btc_score = float(last_scores.get("BTC", np.nan))
    ri = btc_regime_inputs(closes, score, p).iloc[-1]
    regime, regime_name, regime_st = regime_state(float(ri.px), float(ri.ema), float(ri.score), prev_regime, p)
    dd_mult = drawdown_multiplier(drawdown, p)
    recent = np.log(closes).diff().tail(p.vol_window)
    w, info = target_weights(last_scores, last_vol, recent, held, p, regime, dd_mult, excluded, locked,
                             short_multiplier(regime_name, p), held_short)
    info["daily_vol"] = {k: round(float(x) * np.sqrt(24), 5) for k, x in last_vol.dropna().items()}
    info["regime_state"] = regime_st
    ranked = last_scores.dropna().sort_values(ascending=False)
    info.update(regime=regime_name, btc_score=round(btc_score, 3) if not np.isnan(btc_score) else None,
                top_scores={k: round(float(x), 3) for k, x in ranked.head(6).items()},
                bottom_scores={k: round(float(x), 3) for k, x in ranked.tail(6).items()},
                bar=str(closes.index[-1]))
    return w, info
