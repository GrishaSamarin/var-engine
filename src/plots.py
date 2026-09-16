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


def plot_exception_timeline(var_forecasts, realised_returns, save_path=None):
    """
    Realised returns as a line, the -VaR forecast as a band, exceptions marked.

    Clustering becomes visually obvious here in a way the Christoffersen
    statistic doesn't convey. Put this one in the README.
    """
    raise NotImplementedError


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
