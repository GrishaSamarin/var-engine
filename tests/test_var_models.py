"""
Unit tests.

Almost no student project has these. Model validation teams at every Big Five
bank spend their days doing exactly this: checking that a model reproduces a
known answer. Having tests is a disproportionately strong signal.

Run with:  pytest -v

The first test is written for you as a template. Write the rest as you
implement each function.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import var_models as vm
from src import backtest as bt


# ---------------------------------------------------------------------------
# WRITTEN FOR YOU — template
# ---------------------------------------------------------------------------

def test_parametric_var_matches_hand_calculation():
    """
    Construct a series with known mean and std, then check the function
    reproduces the closed-form answer.

    Hand calculation:
        sigma = 0.02, mu dropped, confidence = 0.99
        z = 2.32635
        VaR = 2.32635 * 0.02 = 0.0465270
    """
    rng = np.random.default_rng(0)
    raw = rng.standard_normal(10_000)
    # force exact sample mean 0 and sample std 0.02
    raw = (raw - raw.mean()) / raw.std(ddof=1) * 0.02
    returns = pd.Series(raw)

    result = vm.parametric_var(returns, confidence=0.99, include_mean=False)
    expected = stats.norm.ppf(0.99) * 0.02

    assert result == pytest.approx(expected, rel=1e-6)
    assert result > 0, "VaR must be returned as a positive loss number"


# ---------------------------------------------------------------------------
# YOU WRITE THESE
# ---------------------------------------------------------------------------

def test_var_increases_with_confidence():
    """99% VaR must exceed 95% VaR. Catches sign and ppf errors."""
    pytest.skip("TODO")


def test_es_exceeds_var():
    """ES >= VaR always, by construction. If this fails your quantile is wrong."""
    pytest.skip("TODO")


def test_parametric_portfolio_matches_series_version():
    """
    parametric_var_portfolio(covariance route) should match
    parametric_var(collapsed series route) to within ~1e-6.

    This is the test that catches weight-ordering bugs. Write it early.
    """
    pytest.skip("TODO")


def test_monte_carlo_converges_to_parametric_under_normal():
    """
    With normal draws and enough simulations, MC VaR should converge to the
    parametric answer. Use a loose tolerance (rel=0.02) — it's a simulation.

    If these disagree by a lot, you probably forgot the Cholesky step or
    simulated assets independently.
    """
    pytest.skip("TODO")


def test_monte_carlo_reproducible_with_seed():
    """Same seed, same answer. Twice."""
    pytest.skip("TODO")


def test_cholesky_preserves_covariance():
    """
    Simulate a large sample, compute its covariance, and check it matches the
    input Sigma within tolerance. Directly validates the correlation structure.
    """
    pytest.skip("TODO")


def test_ewma_reacts_faster_than_rolling():
    """
    Build a series with a vol regime break (low vol, then high vol). EWMA vol
    should rise faster than a 250-day rolling std. Demonstrates the point of
    the model.
    """
    pytest.skip("TODO")


def test_kupiec_zero_exceptions_handled():
    """x=0 must not raise and must flag the model as too conservative."""
    pytest.skip("TODO")


def test_kupiec_rejects_bad_model():
    """25 exceptions in 250 days at 99% should reject decisively (p < 0.001)."""
    pytest.skip("TODO")


def test_kupiec_accepts_good_model():
    """2-3 exceptions in 250 days at 99% should not reject."""
    pytest.skip("TODO")


def test_basel_zones():
    """
    Boundary values: 4 -> green, 5 -> yellow, 9 -> yellow, 10 -> red.
    Off-by-one errors at zone boundaries are the classic bug here.
    """
    pytest.skip("TODO")


def test_no_lookahead_in_rolling_var():
    """
    The hardest and most important test. Construct returns where day N is a
    massive outlier. The VaR forecast FOR day N must be unaffected by it.

    If this test fails, your entire backtest is invalid.
    """
    pytest.skip("TODO")
