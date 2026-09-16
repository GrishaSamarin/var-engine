"""
End-to-end pipeline: loads data, computes VaR under each model, backtests
each, and produces the comparison table and charts used in the README.
"""

from src.data import load_all


def main():
    cfg, prices, asset_rets, port_rets = load_all()
    print(f"loaded {len(port_rets)} days of portfolio returns")

    # TODO: point-in-time VaR across all models at each confidence level
    # TODO: risk attribution table (component VaR by position)
    # TODO: rolling VaR forecasts for each model
    # TODO: backtest each, build comparison table
    # TODO: plots


if __name__ == "__main__":
    main()