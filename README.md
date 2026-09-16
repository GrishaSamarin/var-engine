# Portfolio VaR Engine with Basel Backtesting

A multi-method Value at Risk engine with regulatory backtesting, built to
compare where VaR methodologies agree and — more usefully — where they don't.

> **Status:** in development. Sections marked TODO are placeholders to fill in
> as you build. Delete this note before pushing publicly.

---

## Why this exists

VaR is a single number that hides three separate modelling choices: the
distributional assumption, the estimation window, and the volatility model.
This project implements three methodologies over the same portfolio and
backtests each against realised P&L using the tests a bank's model validation
team would actually run.

## Methodology

**Portfolio.** TODO — describe the 6-asset portfolio, weights, and why (equity
/ rates / energy / defensive mix gives you meaningful correlation structure
rather than 6 tech names that all move together).

**Returns.** Log returns. Log returns are time-additive, which makes
square-root-of-time scaling internally consistent. The cost is that portfolio
log return is not the weighted sum of asset log returns; the approximation
error is negligible at daily frequency but is noted here because it is a real
assumption, not a free lunch.

**Models implemented**

| Model | Distributional assumption | Vol estimate |
|---|---|---|
| Parametric (variance-covariance) | Normal | Rolling 250d |
| Parametric + EWMA | Normal | EWMA, λ=0.94 |
| Parametric + GARCH(1,1) | Student-t | GARCH conditional |
| Historical simulation | None (empirical) | Implicit in window |
| Monte Carlo | Normal / Student-t | Sample covariance |

Monte Carlo draws are correlated via Cholesky decomposition of the covariance
matrix — assets are not simulated independently.

**Expected Shortfall** is reported alongside VaR at every confidence level.
VaR identifies where the tail begins; it says nothing about severity within it.
Basel FRTB replaced VaR with ES at 97.5% for exactly this reason.

**Risk attribution.** Marginal and component VaR are computed per position.
Component VaR sums exactly to total portfolio VaR, which is what makes it
usable for limit allocation.

## Backtesting

Each model's daily VaR forecast is compared against the next day's realised
return over an out-of-sample window. Forecasts use only information available
at t-1.

- **Kupiec POF test** — is the exception rate correct?
- **Christoffersen independence test** — are exceptions clustered? A model can
  have the right count and still be broken if all its failures land in one week.
- **Conditional coverage** — joint test, LR_cc = LR_pof + LR_ind.
- **Basel traffic light** — green / yellow / red zones and the resulting
  capital multiplier.

## Results

TODO — the comparison table goes here. This is the most important section of
the README; a reader should be able to see at a glance which models passed
and which failed.

| Model | Exceptions | Expected | Kupiec p | Christoffersen p | Basel zone |
|---|---|---|---|---|---|

TODO — 3-4 sentences on what you found. Candidate findings to look for:
- Parametric VaR fails badly in 2020 and 2022; normality is the culprit
- Historical simulation is slow to react to regime change — the window has to
  roll past the crisis before VaR rises
- EWMA/GARCH pass where constant-vol models fail, and by how much
- ES/VaR ratio is materially higher than the normal-distribution implied ratio

## Limitations

TODO — be honest here; it reads as maturity, not weakness. Candidates:
- Fixed weights assume daily rebalancing with no transaction costs
- Equities only, no derivatives — no non-linear payoffs, so no delta-gamma
- Sample covariance is noisy for large asset counts (no shrinkage estimator)
- Historical simulation at 99% over 250 days rests on ~2.5 observations
- Square-root-of-time scaling assumes iid returns, which returns are not

## Structure

```
var-engine/
├── config.yaml           portfolio, model, and backtest parameters
├── src/
│   ├── data.py           price download, cleaning, returns
│   ├── var_models.py     parametric / historical / Monte Carlo / vol models
│   ├── backtest.py       Kupiec, Christoffersen, Basel zones
│   └── plots.py          distribution, exception timeline, model comparison
├── tests/                pytest unit tests
└── run.py                end-to-end pipeline
```

## Running it

```bash
pip install -r requirements.txt
python -m src.data          # sanity check the data layer
pytest -v                   # run tests
python run.py               # full pipeline
```

## References

TODO — cite Jorion's *Value at Risk*, the Basel backtesting framework
document, and the original Kupiec (1995) and Christoffersen (1998) papers.
Citing primary sources on a student project is unusual and reads well.



Unconditional parametric VaR produced 55 exceptions against 27.6 expected over 2,764 days, and 30 of those fell in 2020 and 2022 alone. The model fails both unconditional and conditional coverage: it is too narrow on average, and it does not widen when volatility rises.