"""
VectorBT parameter scan — Goosos tutorial #1 companion code.

Scans 9,801 fast/slow MA combinations in a single vectorized pass,
prints the best Sharpe, and saves a heatmap.

Run:  python parameter_scan.py
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import vectorbt as vbt

from ma_crossover import load_data  # reuse the data loader

SYMBOL = "SPY"
COMMISSION = 0.001


def main():
    price = load_data()

    # 9,801 combinations: every (fast, slow) pair from 2..100
    windows = np.arange(2, 101)
    fast_ma, slow_ma = vbt.MA.run_combs(
        price, window=windows, r=2, short_names=["fast", "slow"])

    entries = fast_ma.ma_crossed_above(slow_ma).vbt.signals.fshift(1)
    exits = fast_ma.ma_crossed_below(slow_ma).vbt.signals.fshift(1)

    pf = vbt.Portfolio.from_signals(
        price, entries, exits,
        fees=COMMISSION,
        freq="1D",
    )

    sharpe = pf.sharpe_ratio()
    best = sharpe.idxmax()
    print(f"\nBest params : fast={best[1]}, slow={best[3]}")
    print(f"Best Sharpe : {sharpe.max():.2f}")
    print(f"Combos tested: {len(sharpe)}")

    # Heatmap of total return across the grid
    fig = pf.total_return().vbt.heatmap(
        x_level="fast_window", y_level="slow_window",
        trace_kwargs=dict(colorbar=dict(title="Total return")))
    fig.write_image(
        "/home/hatch/workspace/goosos-code/vectorbt-tutorial/heatmap.png",
        width=900, height=700)
    print("Heatmap saved to heatmap.png")

    print("\nReminder: a sharp lone peak = overfitting. "
          "Look for a broad plateau of good params.")


if __name__ == "__main__":
    main()
