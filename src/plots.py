"""
Plotting functions for the VaR project.

Three charts carry the analytical narrative: the fat-tail histogram, the
exception timeline, and the rolling model comparison. Component VaR is a
fourth, secondary view.
"""


def plot_return_distribution(returns, var_levels: dict, save_path=None):
    """
    Histogram of portfolio returns with a fitted normal distribution
    overlaid, and vertical lines marking each model's VaR estimate.

    Demonstrates the central empirical finding of the project: the observed
    left tail sits well outside what the normal distribution predicts,
    which motivates the move to EWMA, GARCH, and conditional EVT.

    Parameters
    ----------
    returns : pd.Series
        Portfolio return series.
    var_levels : dict
        Mapping of model name to VaR estimate (positive fraction of
        portfolio value), e.g. {"parametric": 0.023, "historical": 0.031}.
    save_path : str, optional
        If given, the figure is saved to this path.

    Returns
    -------
    matplotlib.figure.Figure

    Status
    ------
    Not yet implemented.
    """
    raise NotImplementedError("plot_return_distribution: planned, not yet implemented")


def plot_exception_timeline(var_forecasts, realised_returns, label="model", save_path=None):
    """
    Realised portfolio returns plotted against the VaR forecast band, with
    exceptions (realised loss exceeding forecast VaR) marked.

    Parameters
    ----------
    var_forecasts : pd.Series
        Daily VaR forecast, positive fraction of portfolio value, aligned
        to realised_returns.
    realised_returns : pd.Series
        Realised portfolio returns.
    label : str, default "model"
        Model name, used in the chart title.
    save_path : str, optional
        If given, the figure is saved to this path.

    Returns
    -------
    matplotlib.figure.Figure
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
    Rolling VaR forecasts from multiple models plotted on a single axis.

    Illustrates how differently each model responds to the same volatility
    shock — a fixed-window parametric estimate lags the shock, while EWMA
    and GARCH adjust more quickly.

    Parameters
    ----------
    var_series_dict : dict
        Mapping of model name to a pd.Series of VaR forecasts (positive
        fraction of portfolio value), all sharing a common index.
    save_path : str, optional
        If given, the figure is saved to this path.

    Returns
    -------
    matplotlib.figure.Figure

    Status
    ------
    Not yet implemented.
    """
    raise NotImplementedError("plot_rolling_var_comparison: planned, not yet implemented")


def plot_component_var(tickers, component_vars, save_path=None):
    """
    Horizontal bar chart of component VaR against portfolio weight per
    asset.

    Highlights positions whose contribution to portfolio risk is
    disproportionate to their capital allocation.

    Parameters
    ----------
    tickers : list[str]
        Asset identifiers, in the same order as component_vars.
    component_vars : np.ndarray
        Component VaR per asset (see marginal_var in the VaR module).
    save_path : str, optional
        If given, the figure is saved to this path.

    Returns
    -------
    matplotlib.figure.Figure

    Status
    ------
    Not yet implemented.
    """
    raise NotImplementedError("plot_component_var: planned, not yet implemented")