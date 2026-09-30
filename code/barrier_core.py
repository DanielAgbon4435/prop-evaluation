"""
barrier_core.py
================
Analytic and Monte-Carlo machinery for modelling a FUTURES proprietary-trading
"evaluation" as a discrete double-barrier one-touch (digital / binary) option.

Why arithmetic (not geometric) Brownian motion?
------------------------------------------------
A futures evaluation is denominated in DOLLARS of account P&L, and a trader
trades a roughly fixed number of contracts, so each trade adds an (approximately)
fixed dollar amount to cumulative P&L.  The natural state variable is therefore
the cumulative profit-and-loss

    X_t  =  (account equity) - (starting balance),     X_0 = 0,

which we model as arithmetic Brownian motion with drift

    dX_t = mu dt + sigma dW_t,

where mu is the dollar drift per day (skill, net of fees) and sigma is the dollar
volatility per day (risk / position size).  Time is measured in trading days.

The canonical contract (the $100K evaluation on the supplied pricing card; the
same numbers as a widely-advertised $100K futures Combine):

    Profit target (upper barrier)   U =  +$6,000
    Max loss / "the One Rule"        D =  -$3,000   (trailing high-water-mark)
    Daily loss limit                 d =   $2,000   (0.5-0.66x the max loss)
    Consistency target               c =   50%      (best day <= 50% of profit)
    Time limit                       T =   30 days

The barrier reward-to-risk is U/|D| = 2:1 -- the target is TWICE as far as ruin.

A "pass" is {tau_U <= T and tau_U < tau_D}: cumulative P&L touches +6,000 before
touching the trailing -3,000 floor and before the 30-day clock expires.  This is
a discrete double-barrier ONE-TOUCH binary (American digital at U, knocked out
by D, European expiry at T).

The spectral / scale-function machinery below is generic arithmetic-BM
first-passage theory on a strip (D, U); the dollar contract just supplies the
numbers.
"""

from __future__ import annotations
import numpy as np


# --------------------------------------------------------------------------- #
#  Closed form 1: eventual (perpetual, T -> inf) hit probability
# --------------------------------------------------------------------------- #
def prob_hit_upper_eventual(mu, sigma, D, U, x0=0.0):
    """
    P(hit U before D | start x0) for arithmetic BM dX = mu dt + sigma dW on the
    line (no time limit) -- classic gambler's-ruin-with-drift / scale function.

        theta = 2 mu / sigma^2,   s(x) = exp(-theta x)
        P = (s(x0) - s(D)) / (s(U) - s(D))

    Driftless limit P = (x0 - D)/(U - D).  For the 2:1 contract (D=-3000, U=6000,
    x0=0) this is 3000/9000 = 1/3.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if abs(mu) < 1e-15:
        return (x0 - D) / (U - D)
    theta = 2.0 * mu / sigma ** 2
    s = lambda x: np.exp(-theta * x)
    return (s(x0) - s(D)) / (s(U) - s(D))


# --------------------------------------------------------------------------- #
#  Closed form 2: finite-T absorption via Dirichlet heat-kernel spectral series
# --------------------------------------------------------------------------- #
def _spectral_flux_integrals(mu, sigma, D, U, x0, T, n_terms=400):
    """
    Probability of absorption at the upper / lower boundary on or before T for
    arithmetic BM with drift mu, vol sigma, started at x0 in the strip (D, U).
    Obtained from the Dirichlet heat-kernel after a Girsanov/Doob h-transform.
    Returns (P_upper, P_lower).
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    L = U - D
    y0 = x0 - D                       # start measured from the lower boundary
    n = np.arange(1, n_terms + 1)
    lam = (n * np.pi / L) ** 2
    beta = mu ** 2 / (2.0 * sigma ** 2) + 0.5 * sigma ** 2 * lam
    sin0 = np.sin(n * np.pi * y0 / L)
    tfac = (1.0 - np.exp(-beta * T)) / beta
    pref = sigma ** 2 * np.pi / L ** 2

    P_up = pref * np.exp(mu / sigma ** 2 * (L - y0)) * np.sum(
        ((-1.0) ** (n + 1)) * n * sin0 * tfac)
    P_dn = pref * np.exp(mu / sigma ** 2 * (0.0 - y0)) * np.sum(
        n * sin0 * tfac)
    return float(P_up), float(P_dn)


def prob_pass_analytic(mu, sigma, D, U, T, x0=0.0, n_terms=400):
    """P(touch U before D, on or before T).  Clamped to [0, 1]."""
    P_up, _ = _spectral_flux_integrals(mu, sigma, D, U, x0, T, n_terms)
    if not np.isfinite(P_up):
        return np.nan
    return float(min(max(P_up, 0.0), 1.0))


def prob_knockout_analytic(mu, sigma, D, U, T, x0=0.0, n_terms=400):
    """P(touch D (bust) before U, on or before T)."""
    _, P_dn = _spectral_flux_integrals(mu, sigma, D, U, x0, T, n_terms)
    if not np.isfinite(P_dn):
        return np.nan
    return float(min(max(P_dn, 0.0), 1.0))


def prob_pass_safe(mu, sigma, D, U, T, x0=0.0, n_terms=600):
    """Analytic pass probability with an ABM Monte-Carlo fallback in the
    numerically unstable small-sigma / strong-drift corner."""
    val = prob_pass_analytic(mu, sigma, D, U, T, x0=x0, n_terms=n_terms)
    unstable = (sigma > 0) and (abs(mu) / sigma ** 2 * (U - D) > 25)
    if (not np.isfinite(val)) or unstable:
        return simulate_abm_paths(mu, sigma, D, U, T, steps_per_day=24,
                                  n_paths=60_000)["pass_rate"]
    return val


# --------------------------------------------------------------------------- #
#  Greeks (finite differences on the pass-probability surface in P&L space)
# --------------------------------------------------------------------------- #
def pass_prob_at(x0, mu, sigma, D, U, T, n_terms=400):
    """Pass probability as a function of current cumulative P&L x0 in (D, U)."""
    return prob_pass_analytic(mu, sigma, D, U, T, x0=x0, n_terms=n_terms)


def greek_theta(x0, mu, sigma, D, U, T, h=1e-3):
    """Theta = d(Pass)/d(elapsed) = -d(Pass)/dT  (per day)."""
    fp = pass_prob_at(x0, mu, sigma, D, U, T + h)
    fm = pass_prob_at(x0, mu, sigma, D, U, max(T - h, 1e-6))
    return -(fp - fm) / (2 * h)


def greek_vega(x0, mu, sigma, D, U, T, h=1.0):
    """Vega = d(Pass)/d(sigma)  (per $1 of daily dollar-vol)."""
    fp = pass_prob_at(x0, mu, sigma + h, D, U, T)
    fm = pass_prob_at(x0, mu, sigma - h, D, U, T)
    return (fp - fm) / (2 * h)


def greek_delta(x0, mu, sigma, D, U, T, h=25.0):
    """Delta = d(Pass)/dX  (per $1 of current P&L)."""
    fp = pass_prob_at(x0 + h, mu, sigma, D, U, T)
    fm = pass_prob_at(x0 - h, mu, sigma, D, U, T)
    return (fp - fm) / (2 * h)


def greek_gamma(x0, mu, sigma, D, U, T, h=60.0):
    """Gamma = d^2(Pass)/dX^2."""
    fp = pass_prob_at(x0 + h, mu, sigma, D, U, T)
    f0 = pass_prob_at(x0, mu, sigma, D, U, T)
    fm = pass_prob_at(x0 - h, mu, sigma, D, U, T)
    return (fp - 2 * f0 + fm) / h ** 2


# --------------------------------------------------------------------------- #
#  Trading style -> (mu, sigma) in DOLLARS  (arithmetic, no volatility drag)
# --------------------------------------------------------------------------- #
def style_to_abm(p, R, k, trades_per_day, cost_per_trade=0.0):
    """
    Convert a futures trading style to arithmetic-BM dollar drift / vol per day.

    Each trade risks a fixed dollar stop `k`:
        win  (prob p):   P&L += R*k - cost
        loss (prob 1-p): P&L += -k  - cost
    Per-trade moments:
        m1 = k*(p*(1+R) - 1) - cost
        v1 = k^2 * [ p*R^2 + (1-p) - (p*(1+R)-1)^2 ]
    With `trades_per_day` i.i.d. trades:
        mu    = trades_per_day * m1
        sigma = sqrt(trades_per_day * v1).
    """
    m1 = k * (p * (1 + R) - 1.0) - cost_per_trade
    struct = p * R ** 2 + (1 - p) - (p * (1 + R) - 1.0) ** 2
    v1 = k ** 2 * max(struct, 1e-18)
    mu = trades_per_day * m1
    sigma = np.sqrt(trades_per_day * v1)
    return mu, sigma


def per_trade_edge_dollars(p, R, k, cost_per_trade=0.0):
    """Expected dollar P&L per trade (arithmetic edge)."""
    return k * (p * (1 + R) - 1.0) - cost_per_trade


# --------------------------------------------------------------------------- #
#  Monte-Carlo: trade-level evaluation in P&L space with the full rule set
# --------------------------------------------------------------------------- #
def simulate_challenge_mc(
    p, R, k,
    U=6_000.0,            # profit target
    maxloss=3_000.0,      # max loss limit ("the One Rule"); D = -maxloss
    n_days=30,
    trades_per_day=4,
    cost_per_trade=0.0,
    daily_loss_limit=None,    # dollars, e.g. 2000 (0.66x) or 1500 (0.5x)
    trailing=False,           # trailing high-water-mark drawdown
    lock_at_breakeven=True,   # trailing floor locks at 0 P&L once peak>=maxloss
    consistency=None,         # e.g. 0.5: best single day <= 50% of total profit
    jump_prob=0.0,            # per-trade prob of an adverse gap
    jump_mult=0.0,            # extra loss on a jump, in units of k (e.g. 3 -> -3k)
    n_paths=100_000,
    rng=None,
):
    """
    Monte-Carlo the discrete futures evaluation at the trade level, in dollar
    P&L space.  Barriers monitored discretely at the close of every trade; the
    daily limit and consistency rule monitored at the close of every day.
    Returns pass / bust / timeout rates and diagnostics.
    """
    if rng is None:
        rng = np.random.default_rng(12345)
    D = -maxloss
    pnl = np.zeros(n_paths)
    peak = np.zeros(n_paths)
    max_day_gain = np.zeros(n_paths)
    alive = np.ones(n_paths, dtype=bool)
    passed = np.zeros(n_paths, dtype=bool)
    busted = np.zeros(n_paths, dtype=bool)
    win_pnl = R * k - cost_per_trade
    loss_pnl = -k - cost_per_trade

    for day in range(n_days):
        day_open = pnl.copy()
        for _ in range(trades_per_day):
            act = alive & ~passed & ~busted
            if not act.any():
                break
            wins = rng.random(n_paths) < p
            step = np.where(wins, win_pnl, loss_pnl)
            if jump_prob > 0.0 and jump_mult > 0.0:
                jumps = rng.random(n_paths) < jump_prob
                step = step - np.where(jumps, jump_mult * k, 0.0)
            pnl = np.where(act, pnl + step, pnl)
            peak = np.maximum(peak, pnl)

            if trailing:
                floor = peak - maxloss
                if lock_at_breakeven:
                    floor = np.minimum(floor, 0.0)
            else:
                floor = D
            hit_low = act & (pnl <= floor)
            if daily_loss_limit is not None:
                hit_low = hit_low | (act & (pnl <= day_open - daily_loss_limit))
            busted = busted | hit_low

        # end-of-day: update best-day gain and test the (consistency-gated) pass
        live = alive & ~passed & ~busted
        day_gain = pnl - day_open
        max_day_gain = np.where(live & (day_gain > max_day_gain),
                                day_gain, max_day_gain)
        if consistency is None:
            ok = live & (pnl >= U)
        else:
            ok = live & (pnl >= U) & (max_day_gain <= consistency * np.maximum(pnl, 1.0))
        passed = passed | ok
        alive = alive & ~busted & ~passed

    timeout = ~passed & ~busted
    return {
        "p": p, "R": R, "k": k,
        "pass_rate": float(passed.mean()),
        "bust_rate": float(busted.mean()),
        "timeout_rate": float(timeout.mean()),
        "n_paths": int(n_paths),
        "mean_final_pnl": float(pnl.mean()),
    }


def simulate_abm_paths(mu, sigma, D, U, T, steps_per_day=8, n_paths=100_000, rng=None):
    """Continuous-ish arithmetic-BM Monte-Carlo (fine Euler grid) to validate the
    analytic spectral formula.  Returns pass / bust / timeout rates."""
    if rng is None:
        rng = np.random.default_rng(7)
    n_steps = int(round(T * steps_per_day))
    dt = T / n_steps
    x = np.zeros(n_paths)
    passed = np.zeros(n_paths, dtype=bool)
    busted = np.zeros(n_paths, dtype=bool)
    sd = sigma * np.sqrt(dt)
    for _ in range(n_steps):
        live = ~passed & ~busted
        if not live.any():
            break
        x = x + np.where(live, mu * dt + sd * rng.standard_normal(n_paths), 0.0)
        busted = busted | (live & (x <= D))
        passed = passed | (live & ~busted & (x >= U))
    timeout = ~passed & ~busted
    return {"pass_rate": float(passed.mean()),
            "bust_rate": float(busted.mean()),
            "timeout_rate": float(timeout.mean())}


# --------------------------------------------------------------------------- #
#  Funded phase: the "prize" is itself a (trailing) double-barrier game
# --------------------------------------------------------------------------- #
def simulate_funded_extraction(
    p, R, k,
    maxloss=3_000.0,
    split=0.90,                 # futures firms pay a high split, but cap payouts
    payout_buffer=0.0,          # profit cushion to leave in before withdrawing
    payout_period=10,           # skim profit every N trading days
    payout_cap=None,            # max $ per payout period (e.g. 'DOUBLE' caps)
    trailing=True,
    lock_at_breakeven=True,
    daily_loss_limit=None,
    cost_per_trade=0.0,
    max_days=200,
    trades_per_day=4,
    n_paths=40_000,
    rng=None,
):
    """
    Expected cash a trader extracts from a FUNDED futures account, modelled as a
    repeated trailing double-barrier game.  Terminal death on the trailing floor;
    profit above the buffer is skimmed every `payout_period` days (trader keeps
    `split`, subject to an optional per-period `payout_cap`).  Returns mean cash.
    """
    if rng is None:
        rng = np.random.default_rng(424242)
    pnl = np.zeros(n_paths)
    peak = np.zeros(n_paths)
    cash = np.zeros(n_paths)
    dead = np.zeros(n_paths, dtype=bool)
    win_pnl = R * k - cost_per_trade
    loss_pnl = -k - cost_per_trade

    for day in range(max_days):
        day_open = pnl.copy()
        for _ in range(trades_per_day):
            live = ~dead
            if not live.any():
                break
            wins = rng.random(n_paths) < p
            step = np.where(wins, win_pnl, loss_pnl)
            pnl = np.where(live, pnl + step, pnl)
            peak = np.maximum(peak, pnl)
            if trailing:
                floor = peak - maxloss
                if lock_at_breakeven:
                    floor = np.minimum(floor, 0.0)
            else:
                floor = -maxloss
            hit_low = live & (pnl <= floor)
            if daily_loss_limit is not None:
                hit_low = hit_low | (live & (pnl <= day_open - daily_loss_limit))
            dead = dead | hit_low

        if (day + 1) % payout_period == 0:
            avail = np.where(~dead, pnl - payout_buffer, 0.0)
            avail = np.maximum(avail, 0.0)
            if payout_cap is not None:
                avail = np.minimum(avail, payout_cap)
            cash = cash + split * avail
            pnl = pnl - avail                 # withdrawn profit leaves the account
    return float(cash.mean())
