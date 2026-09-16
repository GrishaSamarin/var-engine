"""
End-to-end pipeline. Fill this in as modules come online.

Suggested order of build:
  1. data.py works           -> python -m src.data
  2. parametric_var          -> test_parametric_var_matches_hand_calculation passes
  3. parametric_var_portfolio + marginal_var
  4. historical_var + historical_es
  5. monte_carlo_var
  6. ewma_volatility -> parametric_var_ewma
  7. backtest.py, all of it
  8. plots.py
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
