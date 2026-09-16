"""
VaR backtesting: the part that makes this a risk project.

Anyone can compute a number. A risk analyst asks whether the number was any
good. That's what lives here.

Regulatory context: Basel requires banks to backtest their internal VaR models
daily against 250 days of P&L at 99% confidence, and the exception count drives
a capital multiplier. This module reproduces that machinery.

ALL STUBS — you write these.
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
    Boolean series: True where the realised loss exceeded the VaR forecast.

    An exception on day t means:  realised_return[t] < -var_forecast[t]
    (remember VaR is stored positive, returns are signed)

    CRITICAL: align these correctly. var_forecasts[t] must have been computed
    using data up to t-1 only. Shift if you need to, and write a test that
    catches look-ahead.

    Returns boolean Series indexed like realised_returns.
    """
    if not var_forecasts.index.equals(realised_returns.index):
        raise ValueError("forecast and return indices do not match")
    return realised_returns < - var_forecasts


def kupiec_pof_test(
    n_exceptions: int,
    n_observations: int,
    confidence: float = 0.99,
) -> dict:
    """
    Kupiec Proportion of Failures test — is the exception RATE correct?

    H0: the true exception probability p equals (1 - confidence).

    Likelihood ratio statistic:

        p_hat = x / n
        LR_pof = -2 * ln[ ((1-p)^(n-x) * p^x) / ((1-p_hat)^(n-x) * p_hat^x) ]

    LR_pof ~ chi-squared with 1 degree of freedom under H0.
    Reject (model is bad) if LR_pof > chi2.ppf(0.95, df=1) = 3.841

    Returns dict with: n_exceptions, n_observations, expected_exceptions,
    exception_rate, lr_statistic, p_value, reject_null (bool).

    EDGE CASE: x = 0 makes the log blow up. Handle it — zero exceptions in 500
    days at 99% is itself a failure (the model is too conservative and you're
    holding too much capital). Special-case the formula.
    """
    raise NotImplementedError


def christoffersen_independence_test(exceptions: pd.Series) -> dict:
    """
    Christoffersen independence test — are exceptions CLUSTERED?

    A model can have the right exception count and still be broken: if all 5
    exceptions land in the same week, the model isn't reacting to volatility.
    That's the failure mode that kills banks — losses arrive in clusters, not
    evenly spaced.

    Build the transition counts from the exception indicator sequence:
        n00 = # of (no exception -> no exception)
        n01 = # of (no exception -> exception)
        n10 = # of (exception   -> no exception)
        n11 = # of (exception   -> exception)

        pi_01 = n01 / (n00 + n01)
        pi_11 = n11 / (n10 + n11)
        pi    = (n01 + n11) / (n00 + n01 + n10 + n11)

        LR_ind = -2 * ln[ (1-pi)^(n00+n10) * pi^(n01+n11) /
                          ((1-pi_01)^n00 * pi_01^n01 * (1-pi_11)^n10 * pi_11^n11) ]

    ~ chi2 with 1 df.

    Returns dict with transition counts, lr_statistic, p_value, reject_null.

    Also compute LR_cc = LR_pof + LR_ind (joint conditional coverage test,
    chi2 with 2 df) — that's the one you actually quote.
    """
    raise NotImplementedError


def basel_traffic_light(n_exceptions: int, n_observations: int = 250) -> dict:
    """
    Basel traffic light zones for a 99% VaR model over 250 trading days.

        Green  : 0-4  exceptions   -> multiplier 3.00
        Yellow : 5-9  exceptions   -> multiplier 3.40 to 3.85 (rises with count)
        Red    : 10+  exceptions   -> multiplier 4.00, model rejected

    Yellow zone plus-factors: 5 -> 0.40, 6 -> 0.50, 7 -> 0.65, 8 -> 0.75,
    9 -> 0.85. Add to the base 3.00.

    Returns dict with zone, multiplier, and a one-line interpretation.

    Scale the thresholds if n_observations != 250, and say in the docstring
    that you did — the official zones are defined for exactly 250 days.

    This function is your interview soundbite. "My parametric model landed in
    the red zone over the 2020 window, historical simulation was yellow, and
    EWMA-parametric was green" is a real answer to "tell me about your project."
    """
    raise NotImplementedError


def run_backtest(
    var_forecasts: pd.Series,
    realised_returns: pd.Series,
    confidence: float = 0.99,
    label: str = "model",
) -> dict:
    """
    Run the full battery on one model's forecasts and return a tidy result dict.

    Should call: count_exceptions, kupiec_pof_test,
    christoffersen_independence_test, basel_traffic_light.

    Include the exception DATES in the output — being able to say "my
    exceptions cluster in March 2020 and October 2022" is worth more than the
    test statistic.
    """
    raise NotImplementedError


def compare_models(results: list[dict]) -> pd.DataFrame:
    """
    Stack multiple run_backtest() outputs into one comparison table.

    Columns: model, n_exceptions, expected, rate, kupiec_p, christoffersen_p,
    basel_zone, pass/fail.

    This table goes straight into your README. It is the single most
    informative artefact in the whole project.
    """
    raise NotImplementedError
