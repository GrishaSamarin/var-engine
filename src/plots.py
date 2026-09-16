"""
Plots. Stubs — you write these.

Three charts carry the whole project. Don't make more than five.
"""


def plot_return_distribution(returns, var_levels: dict, save_path=None):
    """
    Histogram of portfolio returns with a fitted normal overlaid and vertical
    lines at each model's VaR estimate.

    The visual payoff: the empirical left tail sits well outside the normal
    curve, which is the argument for everything else in the project.
    """
    raise NotImplementedError


def plot_exception_timeline(var_forecasts, realised_returns, label="model", save_path=None):
    """
    Realised returns with the VaR band overlaid and exceptions marked.
    """
    import matplotlib.pyplot as plt

    exceptions = realised_returns < -var_forecasts
    breach_dates = realised_returns.index[exceptions]
    breach_values = realised_returns[exceptions]

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(realised_returns.index, realised_returns,
            color="#999999", linewidth=0.5, label="daily return")
    ax.plot(var_forecasts.index, -var_forecasts,
            color="#1f4e79", linewidth=1.2, label="99% VaR forecast")
    ax.scatter(breach_dates, breach_values,
               color="#c0392b", s=14, zorder=3, label=f"exceptions ({exceptions.sum()})")

    ax.axhline(0, color="black", linewidth=0.4)
    ax.set_title(f"{label} — {exceptions.sum()} exceptions vs {len(realised_returns)*0.01:.1f} expected")
    ax.set_ylabel("daily return")
    ax.legend(loc="lower left", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(-0.11, 0.09)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_rolling_var_comparison(var_series_dict, save_path=None):
    """
    All models' rolling VaR on one axis. Shows how differently they react to
    the same volatility shock — parametric-rolling lags, EWMA jumps.
    """
    raise NotImplementedError


def plot_component_var(tickers, component_vars, save_path=None):
    """
    Horizontal bar of component VaR vs portfolio weight. Positions that
    contribute more risk than capital are the story.
    """
    raise NotImplementedError
