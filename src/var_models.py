"""
VaR and Expected Shortfall models.

THIS IS THE FILE YOU WRITE. Every function below is a stub with a full spec.
Implement them yourself — this is the code you will be asked about in an
interview, and you cannot defend code you did not write.

Sign convention used throughout this project:
    VaR is returned as a POSITIVE number representing a loss.
    A 1-day 99% VaR of 0.023 means "we expect to lose more than 2.3% on no
    more than 1 day in 100."
    Be consistent. Sign errors are the single most common bug in VaR code and
    an interviewer will probe for them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# 1. PARAMETRIC (variance-covariance)
# ---------------------------------------------------------------------------

def parametric_var(
    returns: pd.Series,
    confidence: float = 0.99,
    horizon: int = 1,
    include_mean: bool = False,
) -> float:
    """
    Variance-covariance VaR on a single return series.

        VaR = -(mu * h - z * sigma * sqrt(h))

    where z = stats.norm.ppf(confidence).

    Args:
        returns:    daily return series
        confidence: e.g. 0.99
        horizon:    horizon in days; scale with sqrt(h) for sigma, h for mu
        include_mean: if False, set mu = 0. For 1-day horizons the drift term
                    is ~2 orders of magnitude smaller than the vol term and is
                    conventionally dropped. Keep the switch so you can show
                    the difference.

    Returns:
        VaR as a positive decimal fraction of portfolio value.

    HINTS:
        - stats.norm.ppf(0.99) = 2.326
        - use ddof=1 on the std (sample, not population)
    """
    sigma = returns.std(ddof=1)
    z = stats.norm.ppf(confidence)
    mu = returns.mean() if include_mean else 0.0
    var = -(mu * horizon -z * sigma * np.sqrt(horizon))
    return var


def parametric_var_portfolio(
    asset_returns: pd.DataFrame,
    weights: np.ndarray,
    confidence: float = 0.99,
    horizon: int = 1,
) -> float:
    """
    Parametric VaR computed from the covariance matrix rather than from a
    pre-collapsed portfolio series.

        sigma_p = sqrt(w.T @ Sigma @ w)
        VaR     = z * sigma_p * sqrt(horizon)

    This SHOULD give (almost) the same answer as calling parametric_var() on
    the portfolio series. Verify that it does — if it doesn't, you have a
    weight-ordering bug or a log-vs-simple return inconsistency.

    HINTS:
        - Sigma = asset_returns.cov().to_numpy()
        - the quadratic form is w @ Sigma @ w  in numpy
        - assert weights are ordered to match asset_returns.columns
    """
    raise NotImplementedError


def marginal_var(
    asset_returns: pd.DataFrame,
    weights: np.ndarray,
    confidence: float = 0.99,
) -> np.ndarray:
    """
    Marginal VaR: d(VaR)/d(w_i) — how portfolio VaR changes per unit increase
    in each position's weight.

        MVaR_i = z * (Sigma @ w)_i / sigma_p

    Component VaR (= MVaR_i * w_i) sums exactly to total VaR. That additivity
    is the whole reason risk desks use it for limit allocation.

    Returns array of marginal VaRs ordered like asset_returns.columns.

    This function is what turns your project from "student calculates VaR"
    into "candidate understands risk attribution." Do not skip it.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 2. HISTORICAL SIMULATION
# ---------------------------------------------------------------------------

def historical_var(
    returns: pd.Series,
    confidence: float = 0.99,
    horizon: int = 1,
) -> float:
    """
    Historical simulation VaR: the empirical quantile of realised returns.

        VaR = -percentile(returns, (1 - confidence) * 100)

    No distributional assumption — the fat tails are whatever actually
    happened. The cost is that your tail estimate at 99% on a 250-day window
    rests on ~2.5 observations.

    HINTS:
        - np.percentile(returns, 1) for 99% confidence
        - scaling by sqrt(horizon) here is technically inconsistent (empirical
          returns aren't iid normal) — note that in your README rather than
          pretending it's fine
    """
    return -np.percentile(returns, (1 - confidence) * 100)


def historical_es(returns: pd.Series, confidence: float = 0.99) -> float:
    """
    Expected Shortfall (CVaR): mean loss GIVEN that the VaR threshold is
    breached.

        ES = -mean(returns[returns <= quantile])

    ES answers the question VaR cannot: how bad is it in the tail. Basel FRTB
    replaced VaR with ES at 97.5% for exactly this reason.

    ES must always be >= VaR. Assert that in your tests.
    """
    cutoff = np.percentile(returns, (1 - confidence) * 100)
    return -returns[returns <= cutoff].mean()


# ---------------------------------------------------------------------------
# 3. MONTE CARLO
# ---------------------------------------------------------------------------

def monte_carlo_var(
    asset_returns: pd.DataFrame,
    weights: np.ndarray,
    confidence: float = 0.99,
    n_sims: int = 50_000,
    horizon: int = 1,
    distribution: str = "normal",
    t_dof: int = 5,
    seed: int | None = 42,
) -> tuple[float, np.ndarray]:
    """
    Monte Carlo VaR with CORRELATED asset draws.

    This is the function that separates you from every other student project.
    Most people simulate each asset independently, which destroys the
    correlation structure and badly understates portfolio risk. Don't.

    Algorithm:
        1. Sigma = asset_returns.cov()
        2. L = np.linalg.cholesky(Sigma)          # lower triangular, L @ L.T = Sigma
        3. Z = standard normal draws, shape (n_assets, n_sims)
        4. correlated_returns = L @ Z             # now has covariance Sigma
        5. add the mean vector back if you're modelling drift
        6. portfolio_sims = weights @ correlated_returns
        7. VaR = -np.percentile(portfolio_sims, (1 - confidence) * 100)

    For distribution="t": draw from stats.t with t_dof degrees of freedom and
    STANDARDISE the draws (divide by sqrt(dof / (dof - 2))) before applying L,
    otherwise you inflate the variance and your VaR is wrong.

    Returns:
        (var, simulated_portfolio_returns) — return the sims too so you can
        plot the distribution and compute MC Expected Shortfall from it.

    HINTS:
        - np.random.default_rng(seed) for reproducibility
        - if cholesky raises LinAlgError, your covariance matrix isn't positive
          definite (usually means more assets than observations, or a duplicate
          column). Handle it and say something useful.
    """
    raise NotImplementedError

# ---------------------------------------------------------------------------
# 4. VOLATILITY MODELS
# ---------------------------------------------------------------------------

def ewma_volatility(returns: pd.Series, lam: float = 0.94) -> pd.Series:
    """
    Exponentially weighted volatility (RiskMetrics).

        sigma2_t = lam * sigma2_{t-1} + (1 - lam) * r_{t-1}^2

    lam = 0.94 is the RiskMetrics daily standard. Lower lambda = more
    reactive, noisier.

    Why this matters: an equally-weighted 250-day vol treats a return from
    250 days ago the same as yesterday's. After a vol regime shift, your VaR
    is stale for a year. EWMA fixes that, and it's two lines of code.

    Returns a series of conditional volatilities aligned to `returns`.

    HINT: pandas has .ewm(alpha=1-lam) but implement the recursion manually
    first so you understand it, then check the two agree.
    """
    variance = []
    var = returns.iloc[:30].var(ddof=1)
    for u in returns:
        variance.append(var)
        var = lam * var + (1 - lam ) * u ** 2
    return pd.Series(np.sqrt(variance),index = returns.index)




def garch_volatility(returns: pd.Series, horizon: int = 1) -> pd.Series:
    """
    GARCH(1,1) conditional volatility forecast.

        sigma2_t = omega + alpha * eps2_{t-1} + beta * sigma2_{t-1}

    Use the `arch` package:
        from arch import arch_model
        am = arch_model(returns * 100, vol="Garch", p=1, q=1, dist="t")
        res = am.fit(disp="off")

    Note the *100 — arch wants returns in percent or the optimiser struggles
    to converge. Remember to scale back.

    Check that alpha + beta < 1 (stationarity). If it's ~0.99 you have a
    near-integrated process; mention it rather than ignoring it.
    """
    from arch import arch_model
    am = arch_model(returns * 100, vol = "Garch", p = 1, q=1)
    res = am.fit(disp="off")
    vol = res.conditional_volatility / 100
    return pd.Series(vol.values, index = returns.index)



def parametric_var_ewma(
    returns: pd.Series,
    confidence: float = 0.99,
    lam: float = 0.94,
) -> pd.Series:
    """
    Rolling parametric VaR using EWMA vol instead of rolling-window vol.

    Returns a time series of VaR estimates, one per day. This is the input to
    your backtest — compare each day's VaR forecast to the NEXT day's realised
    return.

    Watch the off-by-one: VaR for day t must be computed using information
    available at t-1 only. Using day t's return to forecast day t's VaR is
    look-ahead bias and it will make your backtest look great and be worthless.
    """
    raise NotImplementedError

def garch_evt_var(
    returns: pd.Series,
    confidence: float = 0.99,
    threshold_pct: float = 0.95,
) -> float:

    z_t = returns / garch_volatility(returns)
    losses = -z_t
    u = losses.quantile(threshold_pct)
    exceedances  = losses[losses > u ] - u 
    from scipy.stats import genpareto
    shape, loc, scale = genpareto.fit(exceedances, floc = 0)
    n = len(losses)
    n_u = len(exceedances)
    p = 1 - confidence
    term = ((n / n_u) * p) ** (-shape)
    z_q = u + (scale / shape) * (term - 1)
    sigma_latest = (returns / z_t).iloc[-1]
    var = z_q * sigma_latest
    print("z_q:", round(z_q, 4))
    return var
   