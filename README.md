# BSM + Monte Carlo Option Pricer

A professional Black-Scholes-Merton option pricing calculator with Monte Carlo simulation,
built with QuantLib and Streamlit.

## Features

- **BSM closed-form pricing** — exact analytical call/put prices using the Merton (1973)
  continuous dividend yield extension
- **QuantLib cross-validation** — independently prices using QuantLib's AnalyticEuropeanEngine
- **All six Greeks** — Delta (call+put), Gamma, Vega, Theta (call+put), Rho (call+put)
- **Monte Carlo simulation** — QuantLib GBM path generation with up to 500,000 paths
- **Variance reduction** — antithetic variates option halves variance at no cost
- **95% confidence intervals** on all Monte Carlo estimates
- **Convergence diagnostics** — BSM vs MC error with quality indicators
- **Payoff distribution charts** and terminal price histograms
- **Put-call parity check** — automatic verification

## Installation

```bash
# 1. Clone or download the files
# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run bsm_calculator.py
```

## Usage

1. Enter contract parameters in the sidebar:
   - Spot price (S), Strike price (K), Days to expiry
2. Enter market parameters:
   - Risk-free rate (r), Dividend yield (q), Volatility (σ)
3. Optionally configure Monte Carlo settings:
   - Number of paths (10k to 500k)
   - Time steps per path (50, 100, 252)
   - Antithetic variates on/off
4. Press **Calculate**

## Parameter Guide

| Symbol | Meaning | Example |
|--------|---------|---------|
| S | Current underlying price | 100.00 |
| K | Strike price | 105.00 |
| T | Days to expiry | 365 (= 1 year) |
| r | Risk-free rate (%) | 5.0 |
| q | Dividend yield (%) | 0.0 (no dividends) |
| σ | Annualised volatility (%) | 20.0 |

## Formulas

### Black-Scholes-Merton

```
d1 = [ln(S/K) + (r - q + σ²/2)T] / (σ√T)
d2 = d1 - σ√T

Call = S·e^(-qT)·N(d1) - K·e^(-rT)·N(d2)
Put  = K·e^(-rT)·N(-d2) - S·e^(-qT)·N(-d1)
```

### Monte Carlo (GBM)

```
S(t+dt) = S(t) · exp[(r - q - σ²/2)dt + σ√dt · Z]
where Z ~ N(0,1)

Call price = e^(-rT) · E[max(S(T) - K, 0)]
```

### Antithetic variates

For each random draw Z, also simulate with -Z.
This pairs each "high" path with a "low" path, cutting variance roughly in half.

## Dependencies

- `QuantLib` — professional-grade quantitative finance library
- `Streamlit` — interactive web UI framework
- `NumPy` / `SciPy` — numerical computation
- `Pandas` — data display
- `Matplotlib` — charting
