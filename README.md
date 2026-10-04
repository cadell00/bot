# Roostoo Regime-Switching Long/Short Momentum Bot

An autonomous, long-only crypto trading bot for the Roostoo mock exchange. It holds the strongest few crypto trends, sizes them by risk, and moves to cash when the market regime turns. It is built to score well on the competition's risk-adjusted metric: 0.4·Sortino + 0.3·Sharpe + 0.3·Calmar.

## Strategy in one paragraph

Every 8 hours (00:00, 08:00 and 16:00 UTC), after the candle closes, the bot scores about 30 liquid crypto assets on **volatility-normalised time-series momentum**. It then switches its book on the **BTC trend regime**, which uses hysteresis buffers so it doesn't flip every hour:

- **Risk-on:** it holds **long** spot positions in the strongest uptrends (top 4).
- **Risk-off:** it holds **short** positions in the weakest downtrends (bottom 3), using Roostoo's 1x-collateral short API.
- **Neutral:** it holds half-size longs and no shorts.

Each book is weighted by conviction divided by volatility and scaled to a **3%/day volatility target**. Exposure shrinks as **drawdown** grows, measured over 14 days with a floor. Rank and threshold hysteresis plus a **24-hour minimum hold** keep turnover down. Gross exposure is capped at 95%, so there's no leverage.

Every minute, independently of the rebalance schedule, it runs **volatility-scaled trailing stops** on both sides, a **crash guard** (BTC −4% in an hour closes all longs) and a **squeeze guard** (BTC +4% in an hour covers all shorts). Spot rebalances use maker limit orders. Short fees are 0.1% either way, so shorts are opened and closed at market. If the exchange ever rejects shorts as not allowed, the bot records that and continues long-or-cash only.

## Why this design

| Competition fact | Design response |
|---|---|
| Ranked first on return (top 20), then on Sortino/Sharpe/Calmar | Fully invested only in confirmed up-trends, otherwise cash. Cash has zero downside volatility, which helps Sortino and Calmar. |
| 0.1% taker / 0.05% maker fees | Hourly cadence, a rebalance band, hysteresis, and maker-first execution with a market fallback |
| No HFT / excessive requests | About 2 API calls per minute, with client-side throttling and backoff |
| 1x long and short, no leverage | Shorts are sized with collateral equal to their notional. Gross long plus short exposure is capped at 95%. |
| Log integrity and commit transparency | Every API call, decision and order goes to append-only JSONL logs. Parameters live in `bot/config.py`, so every change goes through git. |

### Research basis
- Moskowitz, Ooi & Pedersen (2012), *Time Series Momentum*, J. Financial Economics: past 1–12 month returns predict future returns across asset classes. We use vol-normalised returns, as they do.
- Liu & Tsyvinski (2021), *Risks and Returns of Cryptocurrency*, Review of Financial Studies; Liu, Tsyvinski & Wu (2022), *Common Risk Factors in Cryptocurrency*, J. Finance: strong time-series and cross-sectional momentum in crypto.
- Moreira & Muir (2017), *Volatility-Managed Portfolios*, J. Finance: scaling exposure inversely to volatility raises Sharpe ratios.
- Grossman & Zhou (1993), *Optimal Investment Strategies for Controlling Drawdowns*: exposure that shrinks as drawdown approaches a limit.

### Decision-science framing
Every trade must have positive expected value after costs. Weight changes smaller than the rebalance band aren't worth the fee, so they aren't traded. Position size follows risk rather than conviction alone. Downside is bounded by explicit, pre-committed rules (the drawdown multiplier, trailing stops and the crash guard) rather than by judgement made in the moment.

## Architecture

```
bot/
  config.py          all parameters (env-overridable)
  roostoo_client.py  signed REST client: HMAC-SHA256, throttling, retries, clock sync
  data.py            hourly candles: Binance -> OKX -> self-recorded Roostoo snapshots
  strategy.py        pure signal/regime/sizing functions (shared with the backtester)
  execution.py       target weights -> orders (band, maker-first, market fallback, precision)
  main.py            event loop: equity log, risk checks, hourly rebalance, re-quotes
  logger.py          audit logs
backtest.py          same strategy code, hourly simulation, competition metrics, sweep
tests/               offline tests (signing vs Roostoo doc vector, sizing invariants, executor)
deploy/              EC2 setup script + systemd unit (auto-restart)
scripts/check_setup.py  read-only pre-flight check (places no orders)
```

## Run it

```bash
git clone <repo> roostoo-bot && cd roostoo-bot
bash deploy/setup_ec2.sh          # venv, deps, systemd unit
nano .env                         # add ROOSTOO_API_KEY / ROOSTOO_SECRET_KEY
.venv/bin/python scripts/check_setup.py
.venv/bin/python -m pytest -q
.venv/bin/python backtest.py --days 180 --sweep --plot
.venv/bin/python -m bot.main --dry-run      # optional: watch decisions without trading
sudo systemctl start roostoo-bot
tail -f logs/bot.log
```

## Logs (audit trail)

| File | Contents |
|---|---|
| `logs/decisions.jsonl` | startup parameters, every rebalance (scores, regime, vol scale, drawdown multiplier, targets), stops, crash guard |
| `logs/trades.jsonl` | every order with reason (`rebalance`, `requote`, `risk_exit:urgent`, `…:fallback`) and the exchange response |
| `logs/api.jsonl` | every API call made by the bot |
| `logs/equity.csv` | equity, cash, drawdown and exposure every minute |

## Key parameters (`bot/config.py`)

| Param | Default | Meaning |
|---|---|---|
| `lookbacks` | 24, 72, 168 h | momentum horizons |
| `top_k` | 4 | max new long positions (held positions can stay up to rank 6) |
| `allow_shorts` / `short_top_k` | on / 3 | short book in risk-off regimes |
| `short_entry_threshold` / `short_max_weight` | 0.30 / 25% | how weak an asset must be to short, and the per-short size cap |
| `squeeze_btc_1h` | +4% | cover all shorts, 6 h cooldown |
| `rebalance_every_hours` | 8 | full retarget cadence (stops still run every minute) |
| `min_hold_hours` | 24 | no signal-driven exit before this; stops still apply |
| `target_daily_vol` | 3% | portfolio volatility target |
| `max_drawdown` / `dd_floor_mult` | 8% / 0.25 | exposure shrinks to 25% of normal at 8% drawdown (14-day window) |
| `regime_price_buffer` / `regime_score_buffer` | 1% / 0.10 | hysteresis on the BTC regime |
| `stop_vol_mult` | 2.5× daily vol (6–15%) | trailing stop width |
| `crash_btc_1h` | −4% | flatten everything, 6 h cooldown |
| `rebalance_band` | 5% of equity | minimum trade size worth paying fees for |

## Backtest findings that shaped v2

The first version made about +14% before fees over 6 months, with a third of BTC's drawdown. But 149x turnover (about 15% in fees) turned that into −1%. Two causes accounted for this:

- **Fee churn.** Hourly retargeting and a regime that flipped near the BTC trend line caused constant trading. The fix was a 4-hour cadence, regime hysteresis, a wider band and a minimum hold.
- **Drawdown lockout.** An all-time-peak drawdown rule with no floor kept size near zero through the August rally. The fix was a 25% floor and a 14-day window.

`backtest.py` now reports return before fees, fees paid, and turnover broken down by cause (entry, exit, resize, stop), so this kind of problem is visible straight away.

## Parameter choice (v3)

A 72-setting sweep over 6 months of hourly data was profitable in every configuration: +13% to +45%, with 8.5–15% max drawdown. Defaults were set to the best value of each parameter **by median composite across the whole grid**, not to the single best row, to limit overfitting: top_k 4, 3% daily volatility target, 8% drawdown limit, 8-hour rebalances, 24-hour minimum hold. That combination returned +40.2% with an 11.2% max drawdown in-sample.

These figures are in-sample, because the same period informed the v2 fixes. Check them out of sample with `python backtest.py --days 365 --until 2026-04-15`.

## Out-of-sample finding and the short book (v4)

On October 2025 to April 2026, which wasn't used in any design decision, the long-only version lost 17.2% (−11.4% before fees) while BTC fell 34%. Only 20% of 14-day windows were positive. Long-only momentum can only sit in cash during a bear market, and short-horizon signals kept buying rallies that failed. The competition allows shorts, so v4 adds a symmetric short book for risk-off regimes. `python backtest.py --days 365 --robust` judges every structural choice, including shorts on or off, on its worse half (bear or bull).
