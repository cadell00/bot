# Roostoo Regime-Gated Momentum Bot

An autonomous, long-only crypto trading bot for the Roostoo mock exchange. It holds the strongest few crypto trends, sizes them by risk, and moves to cash when the market regime turns. It is built to score well on the competition's risk-adjusted metric: 0.4·Sortino + 0.3·Sharpe + 0.3·Calmar.

## Strategy in one paragraph

Once an hour, after each candle closes, the bot scores about 30 liquid crypto assets on **volatility-normalised time-series momentum** over three horizons: 1, 3 and 7 days. It then takes the following steps:

1. It gates total exposure on the **BTC trend regime**.
2. It picks the top-K assets with **hysteresis**, so positions don't churn back and forth.
3. It weights the picks by conviction divided by volatility.
4. It scales the whole book to a **2%/day portfolio volatility target**.
5. It cuts exposure linearly as **drawdown** grows.

Independently, every minute it runs **volatility-scaled trailing stops** and a **BTC crash guard**. These are market-order exits that don't wait for the next hourly rebalance. Rebalances use passive **limit orders**, which pay the 0.05% maker fee, and changes below a 3%-of-equity band are skipped because the edge is smaller than the cost.

## Why this design

| Competition fact | Design response |
|---|---|
| Ranked first on return (top 20), then on Sortino/Sharpe/Calmar | Fully invested only in confirmed up-trends, otherwise cash. Cash has zero downside volatility, which helps Sortino and Calmar. |
| 0.1% taker / 0.05% maker fees | Hourly cadence, a rebalance band, hysteresis, and maker-first execution with a market fallback |
| No HFT / excessive requests | About 2 API calls per minute, with client-side throttling and backoff |
| Spot only, no leverage | Long-only. Gross exposure is capped at 95%. |
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
| `top_k` | 4 | max new positions (held positions can stay up to rank 6) |
| `target_daily_vol` | 2% | portfolio volatility target |
| `max_drawdown` | 12% | exposure reaches 0 at this drawdown |
| `stop_vol_mult` | 2.5× daily vol (6–15%) | trailing stop width |
| `crash_btc_1h` | −4% | flatten everything, 6 h cooldown |
| `rebalance_band` | 3% of equity | minimum trade size worth paying fees for |
