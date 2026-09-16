"""
Value at Risk and Expected Shortfall models.

Sign convention: VaR and ES are returned as positive numbers representing a
loss. A 1-day 99% VaR of 0.023 means: on no more than 1 day in 100, the loss
is expected to exceed 2.3% of portfolio value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# 1. Parametric (variance-covariance)
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

    Parameters
    ----------
    returns : pd.Series
        Daily return series.
    confidence : float, default 0.99
        Confidence level (e.g. 0.99 for 99% VaR).
    horizon : int, default 1
        Horizon in days. Volatility scales with sqrt(h), drift with h.
    include_mean : bool, default False
        If False, the drift term is dropped, which is standard practice
        at 1-day horizons since it is roughly two orders of magnitude
        smaller than the volatility term.

    Returns
    -------
    float
        VaR as a positive fraction of portfolio value.
    """
    sigma = returns.std(ddof=1)
    z = stats.norm.ppf(confidence)
    mu = returns.mean() if include_mean else 0.0
    return -(mu * horizon - z * sigma * np.sqrt(horizon))


def parametric_var_portfolio(
    asset_returns: pd.DataFrame,
    weights: np.ndarray,
    confidence: float = 0.99,
    horizon: int = 1,
) -> float:
    """
    Parametric VaR computed from the asset covariance matrix and portfolio
    weights, rather than from a pre-collapsed portfolio return series.

        sigma_p = sqrt(w.T @ Sigma @ w)
        VaR     = z * sigma_p * sqrt(horizon)

    Parameters
    ----------
    asset_returns : pd.DataFrame
        Historical returns, one column per asset.
    weights : np.ndarray
        Portfolio weights, ordered to match asset_returns.columns.
    confidence : float, default 0.99
    horizon : int, default 1

    Returns
    -------
    float
        VaR as a positive fraction of portfolio value. Should agree with
        parametric_var() called on the corresponding portfolio series.
    """
    Sigma = asset_returns.cov().to_numpy()
    sigma_p = np.sqrt(weights @ Sigma @ weights)
    z = stats.norm.ppf(confidence)
    return z * sigma_p * np.sqrt(horizon)


def marginal_var(
    asset_returns: pd.DataFrame,
    weights: np.ndarray,
    confidence: float = 0.99,
) -> np.ndarray:
    """
    Marginal VaR: the sensitivity of portfolio VaR to each position's weight.

        MVaR_i = z * (Sigma @ w)_i / sigma_p

    Component VaR (MVaR_i * w_i) sums exactly to total portfolio VaR, which
    is why this decomposition is used for risk limit allocation.

    Parameters
    ----------
    asset_returns : pd.DataFrame
    weights : np.ndarray
    confidence : float, default 0.99

    Returns
    -------
    np.ndarray
        Marginal VaR per asset, ordered to match asset_returns.columns.
    """
    Sigma = asset_returns.cov().to_numpy()
    sigma_p = np.sqrt(weights @ Sigma @ weights)
    z = stats.norm.ppf(confidence)
    return z * (Sigma @ weights) / sigma_p


# ---------------------------------------------------------------------------
# 2. Historical simulation
# ---------------------------------------------------------------------------

def historical_var(
    returns: pd.Series,
    confidence: float = 0.99,
    horizon: int = 1,
) -> float:
    """
    Historical simulation VaR: the empirical quantile of realised returns.

        VaR = -percentile(returns, (1 - confidence) * 100)

    Makes no distributional assumption, so tail behaviour reflects whatever
    actually occurred in the sample. Note: scaling by sqrt(horizon) is not
    strictly valid here since historical returns are not i.i.d. normal;
    treat multi-day horizon estimates from this method as approximate.

    Returns
    -------
    float
        VaR as a positive fraction of portfolio value.
    """
    return -np.percentile(returns, (1 - confidence) * 100)


def historical_es(returns: pd.Series, confidence: float = 0.99) -> float:
    """
    Expected Shortfall (CVaR): the mean loss conditional on the VaR
    threshold being breached.

        ES = -mean(returns[returns <= quantile])

    ES captures the severity of tail losses, which VaR does not, and is
    the measure Basel FRTB adopted (at 97.5%) in place of VaR for this
    reason. ES >= VaR always holds at the same confidence level.

    Returns
    -------
    float
        Expected Shortfall as a positive fraction of portfolio value.
    """
    cutoff = np.percentile(returns, (1 - confidence) * 100)
    return -returns[returns <= cutoff].mean()


# ---------------------------------------------------------------------------
# 3. Monte Carlo
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
    Monte Carlo VaR using correlated asset return simulations.

    Correlation is preserved via Cholesky decomposition of the covariance
    matrix, rather than simulating each asset independently (which would
    understate portfolio risk by discarding the correlation structure).

    Algorithm
    ---------
    1. Sigma = asset_returns.cov()
    2. L = cholesky(Sigma), lower triangular, L @ L.T = Sigma
    3. Draw Z (n_assets x n_sims) from the chosen distribution
    4. correlated_returns = L @ Z
    5. portfolio_sims = weights @ correlated_returns
    6. VaR = -percentile(portfolio_sims, (1 - confidence) * 100)

    For distribution="t", draws are standardised (divided by
    sqrt(dof / (dof - 2))) before applying L, so the target covariance
    is preserved rather than inflated.

    Parameters
    ----------
    asset_returns : pd.DataFrame
    weights : np.ndarray
    confidence : float, default 0.99
    n_sims : int, default 50_000
    horizon : int, default 1
    distribution : {"normal", "t"}, default "normal"
    t_dof : int, default 5
        Degrees of freedom, used only when distribution="t".
    seed : int or None, default 42

    Returns
    -------
    tuple[float, np.ndarray]
        VaR, and the simulated portfolio return array (for plotting the
        distribution or computing Monte Carlo Expected Shortfall).

    Raises
    ------
    numpy.linalg.LinAlgError
        If the covariance matrix is not positive definite (commonly caused
        by more assets than observations, or duplicate columns).

    Status
    ------
    Not yet implemented. Planned: Cholesky decomposition of the asset
    covariance matrix to preserve cross-asset correlation in simulated
    draws, with optional Student's t innovations for fat-tailed returns.
    """
    raise NotImplementedError("monte_carlo_var: planned, not yet implemented")


# ---------------------------------------------------------------------------
# 4. Volatility models
# ---------------------------------------------------------------------------

def ewma_volatility(returns: pd.Series, lam: float = 0.94) -> pd.Series:
    """
    Exponentially weighted volatility (RiskMetrics methodology).

        sigma2_t = lam * sigma2_{t-1} + (1 - lam) * r_{t-1}^2

    lam = 0.94 is the RiskMetrics daily standard; a lower lambda weights
    recent observations more heavily and produces a noisier estimate.

    Unlike an equally-weighted rolling window, EWMA does not treat a return
    from 250 days ago the same as yesterday's, so it adapts faster after a
    volatility regime shift.

    Returns
    -------
    pd.Series
        Conditional volatility, aligned to `returns`.
    """
    variance = []
    var = returns.iloc[:30].var(ddof=1)
    for u in returns:
        variance.append(var)
        var = lam * var + (1 - lam) * u ** 2
    return pd.Series(np.sqrt(variance), index=returns.index)


def garch_volatility(returns: pd.Series, horizon: int = 1) -> pd.Series:
    """
    GARCH(1,1) conditional volatility forecast.

        sigma2_t = omega + alpha * eps2_{t-1} + beta * sigma2_{t-1}

    Fitted via the `arch` package. Returns are rescaled to percent before
    fitting, since the optimiser converges more reliably on that scale, and
    the result is rescaled back to decimal form.

    Parameters
    ----------
    returns : pd.Series
    horizon : int, default 1

    Returns
    -------
    pd.Series
        Conditional volatility, aligned to `returns`.

    Notes
    -----
    If alpha + beta is close to 1, the process is near-integrated
    (persistent volatility shocks); this is worth flagging rather than
    treating as a standard well-behaved fit.
    """
    from arch import arch_model
    am = arch_model(returns * 100, vol="Garch", p=1, q=1)
    res = am.fit(disp="off")
    vol = res.conditional_volatility / 100
    return pd.Series(vol.values, index=returns.index)


def parametric_var_ewma(
    returns: pd.Series,
    confidence: float = 0.99,
    lam: float = 0.94,
) -> pd.Series:
    """
    Rolling parametric VaR using EWMA volatility rather than a fixed
    rolling-window estimate.

    Returns a VaR forecast for each day, computed using information
    available only through the prior day (day t's VaR uses volatility
    estimated from returns through t-1). This alignment matters for
    backtesting: using day t's own return to produce day t's VaR forecast
    is look-ahead bias and will inflate apparent backtest accuracy.

    Parameters
    ----------
    returns : pd.Series
    confidence : float, default 0.99
    lam : float, default 0.94

    Returns
    -------
    pd.Series
        VaR forecast per day, aligned to `returns`.

    Status
    ------
    Not yet implemented. Planned: compute EWMA volatility, shift it forward
    one day to enforce the t-1 information cutoff, then scale by the normal
    quantile at the chosen confidence level.
    """
    raise NotImplementedError("parametric_var_ewma: planned, not yet implemented")


def garch_evt_var(
    returns: pd.Series,
    confidence: float = 0.99,
    threshold_pct: float = 0.95,
) -> float:
    """
    Conditional EVT VaR (McNeil-Frey approach).

    GARCH filtering removes volatility clustering from raw returns; a
    Generalized Pareto Distribution is then fit to the tail of the
    resulting standardized residuals via peaks-over-threshold, and the
    fitted tail quantile is rescaled by the current GARCH volatility
    forecast to produce VaR in real terms.

    Parameters
    ----------
    returns : pd.Series
    confidence : float, default 0.99
    threshold_pct : float, default 0.95
        Quantile used to define the GPD fitting threshold (e.g. 0.95 uses
        the worst 5% of standardized residuals as exceedances).

    Returns
    -------
    float
        VaR as a positive fraction of portfolio value.
    """
    from scipy.stats import genpareto

    z_t = returns / garch_volatility(returns)
    losses = -z_t
    u = losses.quantile(threshold_pct)
    exceedances = losses[losses > u] - u

    shape, loc, scale = genpareto.fit(exceedances, floc=0)
    n = len(losses)
    n_u = len(exceedances)
    p = 1 - confidence
    term = ((n / n_u) * p) ** (-shape)
    z_q = u + (scale / shape) * (term - 1)

    sigma_latest = (returns / z_t).iloc[-1]
    return z_q * sigma_latest