# VectorBT Tutorial — Companion Code

Runnable Python code for the Goosos tutorial
**[VectorBT Tutorial: Backtest a Strategy in 30 Minutes](https://goosos.com/vectorbt-tutorial)**.

## What's here

| File | Description |
|---|---|
| `ma_crossover.py` | Complete MA(20/50) crossover backtest on SPY — data download, signals, backtest, metrics, equity chart |
| `parameter_scan.py` | Grid-scan 9,801 fast/slow MA combinations + Sharpe heatmap |
| `equity.png` | Sample output chart (generated 2026-10-07) |
| `requirements.txt` | Python dependencies |

## Quickstart

```bash
pip install -r requirements.txt
python ma_crossover.py
```

Expected output (SPY daily, ~3 years):

```
Total return : 38.46%
Sharpe ratio : 1.29
Max drawdown : -11.31%
Win rate     : 83.33%  (6 trades)
```

> **Note:** the first run takes ~60s (Numba JIT compilation). Subsequent runs are fast.
> If `yfinance` can't reach Yahoo, the script falls back to synthetic data so it never crashes.

## Key lessons in the code

1. **No lookahead bias** — signals are shifted one bar via `.vbt.signals.fshift(1)`
2. **Realistic costs** — `fees=0.001` set explicitly (VectorBT defaults to zero!)
3. **Correct annualization** — `freq="1D"` passed (Sharpe is silently wrong without it)

## License

Code is MIT. VectorBT itself is Fair Code licensed — see the [tutorial](https://goosos.com/vectorbt-tutorial) for details.

---

Part of the [Goosos Quantitative Trading Lab](https://goosos.com) tutorial series.
