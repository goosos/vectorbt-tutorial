"""
VectorBT MA-crossover backtest — Goosos tutorial #1 companion code.

Strategy: go long SPY when the 20-day MA crosses ABOVE the 50-day MA,
exit when it crosses back BELOW. End-to-end, no lookahead bias.

Run:  python vbt_test.py
Needs: vectorbt, yfinance, pandas, numpy, matplotlib
"""
import matplotlib
matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import vectorbt as vbt

SYMBOL = "SPY"
FAST, SLOW = 20, 50
COMMISSION = 0.001  # 0.1% per trade


def load_data():
    """3 years of daily SPY closes. Falls back to synthetic data if offline."""
    try:
        import yfinance as yf
        df = yf.download(SYMBOL, period="3y", interval="1d",
                         auto_adjust=True, progress=False)
        if df is None or df.empty:
            raise RuntimeError("empty download")
        close = df["Close"].iloc[:, 0] if df["Close"].ndim > 1 else df["Close"]
        close = close.dropna()
        print(f"Data: {len(close)} daily bars of {SYMBOL} "
              f"({close.index[0].date()} -> {close.index[-1].date()})")
        return close
    except Exception as e:  # network blocked? use synthetic random-walk data
        print(f"yfinance failed ({e}); using SYNTHETIC data instead.")
        rng = np.random.default_rng(42)
        rets = rng.normal(0.0004, 0.012, 756)  # ~3y of trading days
        close = pd.Series(100 * np.exp(np.cumsum(rets)),
                          index=pd.date_range("2023-01-01", periods=756, freq="B"),
                          name="Close")
        print(f"Data: {len(close)} synthetic daily bars")
        return close


def main():
    close = load_data()

    # 1. Moving averages
    fast_ma = close.rolling(FAST).mean()
    slow_ma = close.rolling(SLOW).mean()

    # 2. Raw crossover signals, then SHIFT BY 1 BAR.
    #    Without the shift we'd trade on today's close using today's
    #    signal — classic lookahead bias. Shifting means: decide at
    #    today's close, trade at tomorrow's open (vectorbt default).
    entries = (fast_ma > slow_ma).shift(1).fillna(False).astype(bool)
    exits = (fast_ma < slow_ma).shift(1).fillna(False).astype(bool)

    # 3. Run the backtest
    pf = vbt.Portfolio.from_signals(
        close, entries, exits,
        fees=COMMISSION,      # 0.1% commission per side
        freq="1D",            # daily bars -> annualizes Sharpe correctly
    )

    # 4. Report
    print("\n==== Backtest results ====")
    print(f"Total return : {pf.total_return():.2%}")
    print(f"Sharpe ratio : {pf.sharpe_ratio():.2f}")
    print(f"Max drawdown : {pf.max_drawdown():.2%}")
    stats = pf.stats()
    trades = pf.trades.count()
    win_rate = pf.trades.win_rate()
    print(f"Win rate     : {win_rate:.2%}  ({trades} trades)")
    print(f"Start value  : 100.00 -> End value: {100 * (1 + pf.total_return()):.2f}")

    # 5. Equity curve + drawdown chart
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    pf.value().plot(ax=ax1, color="#22c55e", lw=1.5)
    ax1.set_title(f"MA({FAST}/{SLOW}) crossover on {SYMBOL} — equity curve")
    ax1.set_ylabel("Portfolio value ($)")
    ax1.grid(alpha=0.3)
    pf.drawdown().plot(ax=ax2, color="#ef4444", lw=1.2)
    ax2.set_ylabel("Drawdown")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    out = "/home/hatch/workspace/goosos-code/vectorbt-tutorial/equity.png"
    fig.savefig(out, dpi=120)
    print(f"\nChart saved to {out}")


if __name__ == "__main__":
    main()
