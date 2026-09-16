"""
VaR backtesting.

Regulatory context: Basel requires banks to backtest their internal VaR
models daily against 250 days of P&L at 99% confidence, and the exception
count drives a capital multiplier. This module reproduces that machinery.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def count_exceptions(
    var_forecasts: pd.Series,
    realised_returns: pd.Series,
) -> pd.Series:
    """
    Boolean series marking days where the realised loss exceeded the VaR
    forecast.

    An exception on day t means realised_return[t] < -var_forecast[t]
    (VaR is stored as a positive number; returns are signed).

    Parameters
    ----------
    var_forecasts : pd.Series
        VaR forecast for each day. Must be computed using information
        available only through the prior day, to avoid look-ahead bias.
    realised_returns : pd.Series
        Realised returns, same index as var_forecasts.

    Returns
    -------
    pd.Series
        Boolean series indexed like realised_returns.
    """
    if not var_forecasts.index.equals(realised_returns.index):
        raise ValueError("forecast and return indices do not match")
    return realised_returns < -var_forecasts


def kupiec_pof_test(
    n_exceptions: int,
    n_observations: int,
    confidence: float = 0.99,
) -> dict:
    """
    Kupiec Proportion of Failures test: tests whether the exception rate
    matches the model's stated confidence level.

    H0: the true exception probability p equals (1 - confidence).

    Likelihood ratio statistic:

        p_hat = x / n
        LR_pof = -2 * ln[ ((1-p)^(n-x) * p^x) / ((1-p_hat)^(n-x) * p_hat^x) ]

    LR_pof is chi-squared distributed with 1 degree of freedom under H0.
    The null is rejected (model rejected) if LR_pof exceeds
    chi2.ppf(0.95, df=1) = 3.841.

    Parameters
    ----------
    n_exceptions : int
    n_observations : int
    confidence : float, default 0.99

    Returns
    -------
    dict
        n_exceptions, n_observations, expected_exceptions, exception_rate,
        lr_statistic, p_value, reject_null.

    Notes
    -----
    Zero exceptions makes the likelihood ratio's log term undefined and
    requires a special case; zero exceptions in 500 days at 99% confidence
    is itself a failure mode, indicating an overly conservative model.

    Status
    ------
    Not yet implemented.
    """
    raise NotImplementedError("kupiec_pof_test: planned, not yet implemented")


def christoffersen_independence_test(exceptions: pd.Series) -> dict:
    """
    Christoffersen independence test: tests whether exceptions are
    clustered in time rather than independently distributed.

    A model can have the correct exception count and still fail this test:
    if all exceptions land in the same week, the model is not reacting to
    changing volatility, which is the failure mode most damaging in
    practice, since losses arrive in clusters rather than evenly spaced.

    Transition counts from the exception indicator sequence:

        n00 = # of (no exception -> no exception)
        n01 = # of (no exception -> exception)
        n10 = # of (exception    -> no exception)
        n11 = # of (exception    -> exception)

        pi_01 = n01 / (n00 + n01)
        pi_11 = n11 / (n10 + n11)
        pi    = (n01 + n11) / (n00 + n01 + n10 + n11)

        LR_ind = -2 * ln[ (1-pi)^(n00+n10) * pi^(n01+n11) /
                          ((1-pi_01)^n00 * pi_01^n01 * (1-pi_11)^n10 * pi_11^n11) ]

    LR_ind is chi-squared distributed with 1 degree of freedom.

    Parameters
    ----------
    exceptions : pd.Series
        Boolean exception series, as returned by count_exceptions.

    Returns
    -------
    dict
        Transition counts, lr_statistic, p_value, reject_null. Also
        includes lr_cc = LR_pof + LR_ind, the joint conditional coverage
        test statistic (chi-squared with 2 df), which is the figure
        typically reported alongside the two individual tests.

    Status
    ------
    Not yet implemented.
    """
    raise NotImplementedError("christoffersen_independence_test: planned, not yet implemented")


def basel_traffic_light(n_exceptions: int, n_observations: int = 250) -> dict:
    """
    Basel traffic light zones for a 99% VaR model evaluated over 250
    trading days.

        Green  : 0-4  exceptions -> multiplier 3.00
        Yellow : 5-9  exceptions -> multiplier 3.40 to 3.85, rising with count
        Red    : 10+  exceptions -> multiplier 4.00, model rejected

    Yellow zone plus-factors added to the base 3.00 multiplier:
    5 -> 0.40, 6 -> 0.50, 7 -> 0.65, 8 -> 0.75, 9 -> 0.85.

    Parameters
    ----------
    n_exceptions : int
    n_observations : int, default 250
        The official Basel zones are defined for exactly 250 observation
        days; if n_observations differs, the thresholds should be scaled
        proportionally and that adjustment noted in the result.

    Returns
    -------
    dict
        zone, multiplier, and a one-line interpretation.

    Status
    ------
    Not yet implemented.
    """
    raise NotImplementedError("basel_traffic_light: planned, not yet implemented")


def run_backtest(
    var_forecasts: pd.Series,
    realised_returns: pd.Series,
    confidence: float = 0.99,
    label: str = "model",
) -> dict:
    """
    Run the full backtest battery on one model's VaR forecasts.

    Calls count_exceptions, kupiec_pof_test, christoffersen_independence_test,
    and basel_traffic_light, and assembles the results into a single dict.

    Parameters
    ----------
    var_forecasts : pd.Series
    realised_returns : pd.Series
    confidence : float, default 0.99
    label : str, default "model"

    Returns
    -------
    dict
        Combined results from all four tests, plus the dates of any
        exceptions (useful for identifying whether failures cluster
        around specific events, e.g. March 2020).

    Status
    ------
    Not yet implemented.
    """
    raise NotImplementedError("run_backtest: planned, not yet implemented")


def compare_models(results: list[dict]) -> pd.DataFrame:
    """
    Combine multiple run_backtest() outputs into a single comparison table.

    Parameters
    ----------
    results : list[dict]
        Outputs from run_backtest, one per model.

    Returns
    -------
    pd.DataFrame
        Columns: model, n_exceptions, expected, rate, kupiec_p,
        christoffersen_p, basel_zone, pass_fail.

    Status
    ------
    Not yet implemented.
    """
    raise NotImplementedError("compare_models: planned, not yet implemented")