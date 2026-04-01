"""
Black-Scholes-Merton Option Calculator
with Monte Carlo Simulation (QuantLib + Streamlit)

Install dependencies:
    pip install streamlit QuantLib numpy pandas matplotlib scipy

Run:
    streamlit run bsm_calculator.py
"""

import streamlit as st
import QuantLib as ql
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="BSM + Monte Carlo Option Pricer",
    page_icon="📈",
    layout="wide",
)

st.markdown("""
<style>
    /* ── Global background ── */
    .stApp { background: #0f1117; }
    section[data-testid="stSidebar"] { background: #1a1d27 !important; }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    section[data-testid="stSidebar"] .stNumberInput input,
    section[data-testid="stSidebar"] .stSelectbox select {
        background: #252836 !important;
        border: 1px solid #3a3f55 !important;
        color: #e2e8f0 !important;
        border-radius: 6px;
    }
    section[data-testid="stSidebar"] hr { border-color: #2d3148 !important; }

    /* ── Main panel text ── */
    h1, h2, h3, h4, p, label, .stMarkdown { color: #e2e8f0 !important; }

    /* ── Metric cards — colour coded ── */
    div[data-testid="stMetric"] {
        border-radius: 10px !important;
        padding: 14px 16px !important;
        border: 1px solid rgba(255,255,255,0.07) !important;
    }
    div[data-testid="stMetric"] label {
        font-size: 11px !important;
        letter-spacing: .06em !important;
        text-transform: uppercase !important;
        color: #94a3b8 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 22px !important;
        font-weight: 700 !important;
        color: #f1f5f9 !important;
    }
    div[data-testid="stMetricDelta"] { font-size: 12px !important; }

    /* ── Section headers ── */
    h2 {
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 18px !important;
        font-weight: 700 !important;
        padding-bottom: 4px;
    }

    /* ── Dividers ── */
    hr { border-color: #2d3148 !important; }

    /* ── Expanders ── */
    details { background: #1a1d27 !important; border-radius: 8px !important;
              border: 1px solid #2d3148 !important; }
    details summary { color: #a5b4fc !important; }

    /* ── DataFrames ── */
    .stDataFrame { border-radius: 8px; overflow: hidden; }
    .stDataFrame thead th {
        background: #252836 !important;
        color: #a5b4fc !important;
        font-size: 12px !important;
    }
    .stDataFrame tbody tr:nth-child(even) { background: #1e2130 !important; }
    .stDataFrame tbody tr:nth-child(odd)  { background: #1a1d27 !important; }
    .stDataFrame tbody td { color: #e2e8f0 !important; font-size: 13px !important; }

    /* ── Spinner / status ── */
    .stSpinner > div { border-top-color: #6366f1 !important; }

    /* ── Success / info / warning banners ── */
    div[data-testid="stAlert"][data-baseweb="notification"] {
        border-radius: 8px !important;
    }

    /* ── Custom header banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 40%, #1e3a5f 100%);
        border-radius: 14px;
        padding: 24px 32px;
        margin-bottom: 24px;
        border: 1px solid rgba(99,102,241,0.3);
        text-align: center;
    }
    .hero-title {
        font-size: 22px;
        font-weight: 700;
        color: #f1f5f9 !important;
        -webkit-text-fill-color: #f1f5f9 !important;
        margin: 0 0 6px 0;
        text-align: center;
    }
    .hero-sub {
        font-size: 13px;
        color: #a5b4fc !important;
        margin: 0;
        text-align: center;
    }
    .hero-badges { margin-top: 14px; display: flex; gap: 8px; flex-wrap: wrap; justify-content: center; }
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: .04em;
    }
    .badge-blue   { background: #1e3a5f; color: #93c5fd; border: 1px solid #2563eb44; }
    .badge-purple { background: #2e1065; color: #c4b5fd; border: 1px solid #7c3aed44; }
    .badge-teal   { background: #042f2e; color: #5eead4; border: 1px solid #0d948844; }
    .badge-amber  { background: #431407; color: #fcd34d; border: 1px solid #d9770644; }

    /* ── Metric colour overrides per section ── */
    .call-metric  div[data-testid="stMetricValue"] { color: #34d399 !important; }
    .put-metric   div[data-testid="stMetricValue"] { color: #f87171 !important; }
    .mc-metric    div[data-testid="stMetricValue"] { color: #a78bfa !important; }
    .greek-metric div[data-testid="stMetricValue"] { color: #60a5fa !important; }

    /* ── Section label pills ── */
    .section-pill {
        display: inline-block;
        background: #312e81;
        color: #a5b4fc;
        font-size: 11px;
        font-weight: 700;
        padding: 3px 12px;
        border-radius: 20px;
        letter-spacing: .06em;
        text-transform: uppercase;
        margin-bottom: 12px;
        border: 1px solid rgba(99,102,241,0.3);
    }

    /* ── Primary button ── */
    .stButton > button {
        background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        height: 46px !important;
        letter-spacing: .03em;
        transition: opacity .15s;
    }
    .stButton > button:hover { opacity: .88 !important; }
    .stButton > button:active { transform: scale(.98) !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Helper: standard normal CDF / PDF
# ─────────────────────────────────────────────
def N(x):
    return norm.cdf(x)

def phi(x):
    return norm.pdf(x)


# ─────────────────────────────────────────────
# Analytical BSM (manual — always available)
# ─────────────────────────────────────────────
def bsm_analytical(S, K, T, r, q, sigma):
    """
    Black-Scholes-Merton with continuous dividend yield q.
    Returns call price, put price, and all Greeks.
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return None

    sqrt_T  = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)

    call = S * disc_q * N(d1) - K * disc_r * N(d2)
    put  = K * disc_r * N(-d2) - S * disc_q * N(-d1)

    # Greeks
    delta_call = disc_q * N(d1)
    delta_put  = disc_q * (N(d1) - 1)
    gamma      = disc_q * phi(d1) / (S * sigma * sqrt_T)
    vega       = S * disc_q * phi(d1) * sqrt_T / 100        # per 1% vol
    theta_call = (
        -S * disc_q * phi(d1) * sigma / (2 * sqrt_T)
        - r * K * disc_r * N(d2)
        + q * S * disc_q * N(d1)
    ) / 365                                                  # per calendar day
    theta_put  = (
        -S * disc_q * phi(d1) * sigma / (2 * sqrt_T)
        + r * K * disc_r * N(-d2)
        - q * S * disc_q * N(-d1)
    ) / 365
    rho_call   =  K * T * disc_r * N(d2) / 100              # per 1% rate
    rho_put    = -K * T * disc_r * N(-d2) / 100

    return {
        "call": call, "put": put,
        "d1": d1, "d2": d2,
        "delta_call": delta_call, "delta_put": delta_put,
        "gamma": gamma, "vega": vega,
        "theta_call": theta_call, "theta_put": theta_put,
        "rho_call": rho_call, "rho_put": rho_put,
    }


# ─────────────────────────────────────────────
# QuantLib BSM engine
# ─────────────────────────────────────────────
def bsm_quantlib(S, K, T, r, q, sigma, option_type="call"):
    """
    Price a European option with the QuantLib Black-Scholes-Merton engine.
    Returns the QuantLib NPV (price).
    """
    today      = ql.Date.todaysDate()
    expiry     = today + int(round(T * 365))
    calendar   = ql.NullCalendar()
    day_count  = ql.Actual365Fixed()
    convention = ql.Unadjusted

    ql.Settings.instance().evaluationDate = today

    payoff    = ql.PlainVanillaPayoff(
        ql.Option.Call if option_type == "call" else ql.Option.Put, K
    )
    exercise  = ql.EuropeanExercise(expiry)
    option    = ql.VanillaOption(payoff, exercise)

    spot_handle   = ql.QuoteHandle(ql.SimpleQuote(S))
    flat_ts       = ql.YieldTermStructureHandle(
        ql.FlatForward(today, r, day_count))
    div_ts        = ql.YieldTermStructureHandle(
        ql.FlatForward(today, q, day_count))
    flat_vol      = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(today, calendar, sigma, day_count))

    bsm_process   = ql.BlackScholesMertonProcess(
        spot_handle, div_ts, flat_ts, flat_vol)

    engine        = ql.AnalyticEuropeanEngine(bsm_process)
    option.setPricingEngine(engine)

    return option.NPV()


# ─────────────────────────────────────────────
# Monte Carlo simulation (QuantLib GBM paths)
# ─────────────────────────────────────────────
def monte_carlo_quantlib(S, K, T, r, q, sigma, N_paths, N_steps, antithetic, option_type="call"):
    """
    Price a European option via Monte Carlo using QuantLib path generation.
    Returns MC call/put price, standard error, and array of terminal prices.
    """
    today     = ql.Date.todaysDate()
    expiry    = today + int(round(T * 365))
    day_count = ql.Actual365Fixed()

    ql.Settings.instance().evaluationDate = today

    spot_handle = ql.QuoteHandle(ql.SimpleQuote(S))
    flat_ts     = ql.YieldTermStructureHandle(
        ql.FlatForward(today, r, day_count))
    div_ts      = ql.YieldTermStructureHandle(
        ql.FlatForward(today, q, day_count))
    flat_vol    = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(today, ql.NullCalendar(), sigma, day_count))

    bsm_process = ql.BlackScholesMertonProcess(
        spot_handle, div_ts, flat_ts, flat_vol)

    rng           = ql.GaussianRandomSequenceGenerator(
        ql.UniformRandomSequenceGenerator(
            N_steps,
            ql.UniformRandomGenerator(seed=42)
        )
    )
    seq_gen       = ql.GaussianPathGenerator(
        bsm_process, T, N_steps, rng, antithetic
    )

    disc     = np.exp(-r * T)
    payoffs  = []

    for _ in range(N_paths):
        path     = seq_gen.next().value()
        S_T      = path[N_steps]
        if option_type == "call":
            payoffs.append(max(S_T - K, 0.0))
        else:
            payoffs.append(max(K - S_T, 0.0))

    payoffs   = np.array(payoffs)
    price_mc  = disc * np.mean(payoffs)
    std_err   = disc * np.std(payoffs, ddof=1) / np.sqrt(N_paths)

    terminal  = np.array([
        seq_gen.next().value()[N_steps] for _ in range(min(5000, N_paths))
    ])

    return price_mc, std_err, payoffs, terminal


# ─────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────
def plot_payoff_hist(payoffs, K, option_type, bsm_price):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(6, 2.8), facecolor='#1a1d27')
    ax.set_facecolor('#252836')
    ax.hist(payoffs[payoffs > 0], bins=60, color="#378ADD", alpha=0.8,
            edgecolor="none", density=True)
    ax.axvline(np.mean(payoffs), color="#c92a2a", linewidth=1.4,
               linestyle="--", label=f"Mean payoff: ${np.mean(payoffs):.2f}")
    ax.set_xlabel("Payoff at expiry ($)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(f"MC {option_type.capitalize()} Payoff Distribution", fontsize=11)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


def plot_price_paths(terminal_prices, K, S, T):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(6, 2.8), facecolor='#1a1d27')
    ax.set_facecolor('#252836')
    ax.hist(terminal_prices, bins=60, color="#1D9E75", alpha=0.8,
            edgecolor="none", density=True)
    ax.axvline(S,  color="#185FA5", linewidth=1.2, linestyle="--", label=f"Spot S=${S:.2f}")
    ax.axvline(K,  color="#c92a2a", linewidth=1.2, linestyle="--", label=f"Strike K=${K:.2f}")
    ax.axvline(np.median(terminal_prices), color="#f08c00", linewidth=1.2,
               linestyle=":", label=f"Median=${np.median(terminal_prices):.2f}")
    ax.set_xlabel("Terminal price S(T)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Distribution of Simulated Terminal Prices", fontsize=11)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


def plot_sensitivity(S, K, T, r, q, sigma):
    """Sensitivity of call/put price to spot price around current level."""
    spots  = np.linspace(S * 0.5, S * 1.5, 100)
    calls  = [bsm_analytical(s, K, T, r, q, sigma)["call"] for s in spots]
    puts   = [bsm_analytical(s, K, T, r, q, sigma)["put"]  for s in spots]

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(6, 2.8), facecolor='#1a1d27')
    ax.set_facecolor('#252836')
    ax.plot(spots, calls, color="#185FA5", linewidth=2, label="Call")
    ax.plot(spots, puts,  color="#c92a2a", linewidth=2, label="Put")
    ax.axvline(S, color="#f08c00", linewidth=1, linestyle="--", label=f"Current S=${S:.2f}")
    ax.axvline(K, color="#6c757d", linewidth=1, linestyle=":",  label=f"Strike K=${K:.2f}")
    ax.set_xlabel("Spot price ($)", fontsize=10)
    ax.set_ylabel("Option price ($)", fontsize=10)
    ax.set_title("Option Price vs Spot (Payoff profile)", fontsize=11)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# UI — Title
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <div class="hero-title">📈 BSM + Monte Carlo Option Pricer by Doug Chingosho</div>
  <div class="hero-sub">Black-Scholes-Merton analytical pricing &nbsp;·&nbsp; QuantLib engine &nbsp;·&nbsp; Monte Carlo simulation</div>
  <div class="hero-badges">
    <span class="badge badge-blue">European Options</span>
    <span class="badge badge-purple">QuantLib Engine</span>
    <span class="badge badge-teal">Monte Carlo GBM</span>
    <span class="badge badge-amber">All 6 Greeks</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# UI — Sidebar inputs (the calculator)
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("**Contract parameters**")

    S     = st.number_input("Spot price (S)",     value=100.0,  min_value=0.01,  step=0.5,  format="%.2f")
    K     = st.number_input("Strike price (K)",   value=105.0,  min_value=0.01,  step=0.5,  format="%.2f")
    T_days= st.number_input("Days to expiry",     value=365,    min_value=1,     step=1)
    T     = T_days / 365.0

    st.divider()
    st.header("Market parameters")

    r     = st.number_input("Risk-free rate r (%)",   value=5.0,  min_value=0.0,  step=0.1,  format="%.2f") / 100
    q     = st.number_input("Dividend yield q (%)",   value=0.0,  min_value=0.0,  step=0.1,  format="%.2f") / 100
    sigma = st.number_input("Volatility σ (%)",       value=20.0, min_value=0.1,  step=0.5,  format="%.2f") / 100

    st.divider()
    st.header("Monte Carlo settings")

    run_mc     = st.checkbox("Run Monte Carlo simulation", value=True)
    N_paths    = st.selectbox("Number of paths",
                              [10_000, 50_000, 100_000, 500_000], index=1,
                              format_func=lambda x: f"{x:,}")
    N_steps    = st.selectbox("Time steps per path",
                              [50, 100, 252], index=1,
                              format_func=lambda x: f"{x} ({'daily' if x==252 else 'steps'})")
    antithetic = st.checkbox("Antithetic variates (variance reduction)", value=True)

    st.divider()
    calculate  = st.button("⚡  Calculate", use_container_width=True, type="primary")


# ─────────────────────────────────────────────
# Main panel — always show inputs summary
# ─────────────────────────────────────────────
st.markdown('<div class="section-pill">Current inputs</div>', unsafe_allow_html=True)
col_info1, col_info2, col_info3, col_info4 = st.columns(4)
col_info1.metric("Spot S",    f"${S:.2f}")
col_info2.metric("Strike K",  f"${K:.2f}")
col_info3.metric("Expiry",    f"{T_days}d  ({T:.4f}y)")
itm_label = "🟢 ITM" if S > K else ("🔴 OTM" if S < K else "🟡 ATM")
col_info4.metric("Moneyness", itm_label)

col_info5, col_info6, col_info7, col_info8 = st.columns(4)
col_info5.metric("Rate r",    f"{r*100:.2f}%")
col_info6.metric("Yield q",   f"{q*100:.2f}%")
col_info7.metric("Vol σ",     f"{sigma*100:.2f}%")
col_info8.metric("Paths",     f"{N_paths:,}" if run_mc else "—")
st.divider()

# ─────────────────────────────────────────────
# Compute on button press
# ─────────────────────────────────────────────
if calculate:
    # ── Validate ──────────────────────────────
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        st.error("All parameters must be positive numbers.")
        st.stop()

    # ── BSM Analytical ────────────────────────
    res = bsm_analytical(S, K, T, r, q, sigma)
    if res is None:
        st.error("Invalid parameters — check spot, strike, time, and volatility.")
        st.stop()

    # ── QuantLib cross-check ──────────────────
    try:
        ql_call = bsm_quantlib(S, K, T, r, q, sigma, "call")
        ql_put  = bsm_quantlib(S, K, T, r, q, sigma, "put")
        ql_ok   = True
    except Exception as e:
        ql_ok   = False
        ql_err  = str(e)

    # ─────────────────────────────────────────
    # Section 1 — BSM Prices
    # ─────────────────────────────────────────
    st.markdown('<div class="section-pill">1 — BSM Prices</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Call price (BSM)",         f"${res['call']:.4f}")
    c2.metric("Put price (BSM)",          f"${res['put']:.4f}")
    c3.metric("QuantLib call" if ql_ok else "QuantLib call", f"${ql_call:.4f}" if ql_ok else "N/A")
    c4.metric("QuantLib put"  if ql_ok else "QuantLib put",  f"${ql_put:.4f}"  if ql_ok else "N/A")

    if ql_ok:
        diff_c = abs(res["call"] - ql_call)
        diff_p = abs(res["put"]  - ql_put)
        if diff_c < 1e-6 and diff_p < 1e-6:
            st.success(f"✓ QuantLib cross-check passed — difference < $0.000001 (call: ${diff_c:.6f}, put: ${diff_p:.6f})")
        else:
            st.warning(f"⚠ Small numerical difference — call: ${diff_c:.6f}, put: ${diff_p:.6f}")

    # d1 / d2 details
    with st.expander("d1, d2 and put-call parity details"):
        parity = res["call"] - res["put"] - (S * np.exp(-q * T) - K * np.exp(-r * T))
        intrinsic_call = max(S - K, 0)
        intrinsic_put  = max(K - S, 0)
        dc1, dc2, dc3, dc4 = st.columns(4)
        dc1.metric("d1", f"{res['d1']:.4f}")
        dc2.metric("d2", f"{res['d2']:.4f}")
        dc3.metric("N(d1)", f"{norm.cdf(res['d1']):.4f}")
        dc4.metric("N(d2)", f"{norm.cdf(res['d2']):.4f}")
        dd1, dd2, dd3, dd4 = st.columns(4)
        dd1.metric("Intrinsic call", f"${intrinsic_call:.4f}")
        dd2.metric("Time value call", f"${res['call'] - intrinsic_call:.4f}")
        dd3.metric("Intrinsic put",  f"${intrinsic_put:.4f}")
        dd4.metric("Time value put", f"${res['put'] - intrinsic_put:.4f}")
        st.metric("Put-call parity error (should be ~0)",
                  f"${parity:.8f}",
                  delta="✓ Holds" if abs(parity) < 1e-6 else f"⚠ ${abs(parity):.6f}")

    # ─────────────────────────────────────────
    # Section 2 — Greeks
    # ─────────────────────────────────────────
    st.divider()
    st.markdown('<div class="section-pill">2 — Option Greeks</div>', unsafe_allow_html=True)

    g1, g2, g3, g4, g5, g6 = st.columns(6)
    g1.metric("Δ Delta (call)", f"{res['delta_call']:.4f}",
              help="Change in option price per $1 move in spot")
    g2.metric("Δ Delta (put)",  f"{res['delta_put']:.4f}")
    g3.metric("Γ Gamma",        f"{res['gamma']:.4f}",
              help="Change in delta per $1 move in spot")
    g4.metric("ν Vega (per 1%)", f"${res['vega']:.4f}",
              help="Change in price per 1% change in volatility")
    g5.metric("Θ Theta call/day", f"${res['theta_call']:.4f}",
              help="Time decay per calendar day")
    g6.metric("ρ Rho call (per 1%)", f"${res['rho_call']:.4f}",
              help="Change in price per 1% change in interest rate")

    g7, g8, _, _, _, _ = st.columns(6)
    g7.metric("Θ Theta put/day", f"${res['theta_put']:.4f}")
    g8.metric("ρ Rho put (per 1%)", f"${res['rho_put']:.4f}")

    # Sensitivity chart
    st.divider()
    st.markdown('<div class="section-pill">3 — Price Sensitivity</div>', unsafe_allow_html=True)
    st.pyplot(plot_sensitivity(S, K, T, r, q, sigma))

    # ─────────────────────────────────────────
    # Section 4 — Monte Carlo
    # ─────────────────────────────────────────
    if run_mc:
        st.divider()
        st.markdown('<div class="section-pill">4 — Monte Carlo Simulation</div>', unsafe_allow_html=True)

        with st.spinner(f"Simulating {N_paths:,} paths with {N_steps} steps each..."):
            mc_call, se_call, payoffs_c, terminal_c = monte_carlo_quantlib(
                S, K, T, r, q, sigma, N_paths, N_steps, antithetic, "call")
            mc_put,  se_put,  payoffs_p, terminal_p = monte_carlo_quantlib(
                S, K, T, r, q, sigma, N_paths, N_steps, antithetic, "put")

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("MC Call price",    f"${mc_call:.4f}",
                   delta=f"BSM diff: ${mc_call - res['call']:+.4f}")
        mc2.metric("MC Put price",     f"${mc_put:.4f}",
                   delta=f"BSM diff: ${mc_put - res['put']:+.4f}")
        mc3.metric("Call 95% CI",      f"±${1.96*se_call:.4f}")
        mc4.metric("Put 95% CI",       f"±${1.96*se_put:.4f}")

        mc5, mc6, mc7, mc8 = st.columns(4)
        err_call = abs(mc_call - res["call"])
        err_put  = abs(mc_put  - res["put"])
        mc5.metric("Call abs error",   f"${err_call:.4f}",
                   delta=f"{err_call/res['call']*100:.2f}% of BSM")
        mc6.metric("Put abs error",    f"${err_put:.4f}",
                   delta=f"{err_put/res['put']*100:.2f}% of BSM")
        mc7.metric("Paths simulated",  f"{N_paths:,}")
        mc8.metric("Variance method",  "Antithetic" if antithetic else "Standard")

        # Convergence quality indicator
        if err_call < 0.01:
            st.success(f"✓ Excellent convergence — MC call error ${err_call:.4f} < $0.01")
        elif err_call < 0.05:
            st.info(f"ℹ Good convergence — MC call error ${err_call:.4f}. Increase paths for better accuracy.")
        else:
            st.warning(f"⚠ Poor convergence — error ${err_call:.4f}. Try 100,000+ paths.")

        # Payoff distribution
        col_hist1, col_hist2 = st.columns(2)
        with col_hist1:
            st.pyplot(plot_payoff_hist(payoffs_c, K, "call", res["call"]))
        with col_hist2:
            st.pyplot(plot_price_paths(terminal_c, K, S, T))

        # Detailed MC statistics
        with st.expander("Detailed Monte Carlo statistics"):
            itm_frac = np.mean(payoffs_c > 0)
            df_stats = pd.DataFrame({
                "Statistic": [
                    "Paths simulated",
                    "Fraction ending ITM (call)",
                    "Mean payoff (undiscounted)",
                    "Median payoff (undiscounted)",
                    "Std dev of payoffs",
                    "Min payoff", "Max payoff",
                    "MC call price (discounted)",
                    "BSM call price",
                    "Absolute error",
                    "Relative error",
                    "Standard error (MC)",
                    "95% confidence interval",
                ],
                "Value": [
                    f"{N_paths:,}",
                    f"{itm_frac*100:.2f}%",
                    f"${np.mean(payoffs_c):.4f}",
                    f"${np.median(payoffs_c):.4f}",
                    f"${np.std(payoffs_c):.4f}",
                    f"${payoffs_c.min():.4f}",
                    f"${payoffs_c.max():.4f}",
                    f"${mc_call:.4f}",
                    f"${res['call']:.4f}",
                    f"${err_call:.4f}",
                    f"{err_call/res['call']*100:.3f}%",
                    f"${se_call:.4f}",
                    f"[${mc_call-1.96*se_call:.4f},  ${mc_call+1.96*se_call:.4f}]",
                ]
            })
            st.dataframe(df_stats, use_container_width=True, hide_index=True)

    # ─────────────────────────────────────────
    # Section 5 — Summary table
    # ─────────────────────────────────────────
    st.divider()
    st.markdown('<div class="section-pill">5 — Full Results Summary</div>', unsafe_allow_html=True)
    rows = [
        ("Spot price S",         f"${S:.4f}",            ""),
        ("Strike price K",       f"${K:.4f}",            ""),
        ("Time to expiry T",     f"{T:.6f} years",       f"({T_days} calendar days)"),
        ("Risk-free rate r",     f"{r*100:.4f}%",        ""),
        ("Dividend yield q",     f"{q*100:.4f}%",        ""),
        ("Volatility σ",         f"{sigma*100:.4f}%",    ""),
        ("d1",                   f"{res['d1']:.6f}",     ""),
        ("d2",                   f"{res['d2']:.6f}",     ""),
        ("BSM Call",             f"${res['call']:.6f}",  ""),
        ("BSM Put",              f"${res['put']:.6f}",   ""),
        ("Delta (call)",         f"{res['delta_call']:.6f}", "Δ"),
        ("Delta (put)",          f"{res['delta_put']:.6f}",  "Δ"),
        ("Gamma",                f"{res['gamma']:.6f}",  "Γ"),
        ("Vega (per 1% vol)",    f"${res['vega']:.6f}",  "ν"),
        ("Theta call (per day)", f"${res['theta_call']:.6f}", "Θ"),
        ("Theta put (per day)",  f"${res['theta_put']:.6f}",  "Θ"),
        ("Rho call (per 1% r)",  f"${res['rho_call']:.6f}",  "ρ"),
        ("Rho put (per 1% r)",   f"${res['rho_put']:.6f}",   "ρ"),
    ]
    if run_mc:
        rows += [
            ("MC Call price",        f"${mc_call:.6f}", f"±${1.96*se_call:.4f} (95% CI)"),
            ("MC Put price",         f"${mc_put:.6f}",  f"±${1.96*se_put:.4f} (95% CI)"),
            ("MC call error vs BSM", f"${err_call:.6f}", f"({err_call/res['call']*100:.3f}%)"),
        ]
    df_summary = pd.DataFrame(rows, columns=["Parameter", "Value", "Note"])
    st.dataframe(df_summary, use_container_width=True, hide_index=True)

else:
    st.info("👈  Enter your parameters in the sidebar and press **Calculate** to price the option.")
    st.markdown("""
    **How this calculator works:**

    - **BSM closed-form** — Computes the exact Black-Scholes-Merton price and all six Greeks analytically using the Merton (1973) continuous dividend yield extension.
    - **QuantLib cross-check** — Independently prices the option using QuantLib's `AnalyticEuropeanEngine` to verify the analytical solution.
    - **Monte Carlo** — Simulates thousands of GBM price paths using QuantLib's `GaussianPathGenerator`, discounts the average payoff, and reports a 95% confidence interval. Antithetic variates halve the variance at no cost.

    **Parameters:**

    | Symbol | Meaning |
    |--------|---------|
    | S | Current price of the underlying asset |
    | K | Strike price of the option |
    | T | Time to expiry in years (e.g. 0.5 = 6 months) |
    | r | Risk-free interest rate (annualised, continuous compounding) |
    | q | Continuous dividend yield (0 for non-dividend-paying stocks) |
    | σ | Annualised volatility of the underlying asset |
    """)
