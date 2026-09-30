"""
run_simulation.py
=================
Generates every figure and headline number for the whitepaper

    "Two-to-One Against: A Double-Barrier Binary-Option Analysis of
     Futures Proprietary-Trading Evaluations"

Canonical contract (the $100K futures evaluation on the supplied card):
    profit target  U = +$6,000      max loss D = -$3,000   (ratio 2:1)
    daily loss limit $2,000 (0.66x)  consistency 50%        T = 30 days

Run:  python3 run_simulation.py
"""
from __future__ import annotations
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from barrier_core import (
    prob_pass_analytic, prob_knockout_analytic, prob_hit_upper_eventual,
    prob_pass_safe, pass_prob_at,
    greek_theta, greek_vega, greek_delta, greek_gamma,
    style_to_abm, per_trade_edge_dollars,
    simulate_challenge_mc, simulate_abm_paths, simulate_funded_extraction,
)

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 150, "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
    "axes.spines.right": False, "figure.autolayout": True,
    "axes.titleweight": "bold",
})
NAVY, RED, GREEN, GOLD = "#1f3b73", "#c0392b", "#1e8449", "#b7791f"

# ---- canonical contract ----
U = 6_000.0
MAXLOSS = 3_000.0
D = -MAXLOSS
T = 30.0
TRADES_PER_DAY = 4
DAILY = 2_000.0          # 0.66x max loss
DAILY_LOW = 1_500.0      # 0.5x max loss
CONSISTENCY = 0.5
P_INF0 = (0 - D) / (U - D)   # eventual zero-drift pass = 1/3

FIG = "../figures"
RESULTS = {}


def save(fig, name):
    fig.savefig(f"{FIG}/{name}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{name}")


# --------------------------------------------------------------------------- #
def validation():
    print("[0] validation")
    rows = []
    for name, mu, sig in [("zero", 0.0, 700.0), ("pos", 60.0, 700.0),
                          ("neg", -60.0, 700.0), ("hi_vol", 0.0, 1400.0)]:
        pa = prob_pass_analytic(mu, sig, D, U, T)
        mc = simulate_abm_paths(mu, sig, D, U, T, steps_per_day=16, n_paths=200_000)
        rows.append({"case": name, "mu": mu, "sigma": sig,
                     "pass_analytic": pa, "pass_mc": mc["pass_rate"],
                     "bust_mc": mc["bust_rate"]})
    RESULTS["validation"] = rows


# --------------------------------------------------------------------------- #
#  1.  The geometry: pass probability vs the barrier reward:risk ratio
# --------------------------------------------------------------------------- #
def fig_geometry():
    print("[1] barrier-ratio geometry")
    ratios = np.linspace(0.5, 3.0, 120)
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    # eventual (zero drift): P = |D| / (U + |D|) = 1/(1+ratio)
    ax.plot(ratios, 1.0 / (1.0 + ratios), color=NAVY, lw=2.4,
            label="eventual (no time limit), zero edge")
    # finite T=30, zero drift, moderate sigma=700
    pv = [prob_pass_safe(0.0, 700.0, -MAXLOSS, MAXLOSS * r, T) for r in ratios]
    ax.plot(ratios, pv, color=RED, lw=2.4,
            label="30-day limit, zero edge, $\\sigma=\\$700$/day")
    for r, name, col in [(0.8, "old equity model\n(+8%/-10%)", GOLD),
                         (1.5, "1.5:1", GREEN), (2.0, "2:1 (this card)", "#111")]:
        pe = 1.0 / (1.0 + r)
        ax.axvline(r, color=col, ls=":", lw=1.4)
        ax.annotate(name, (r, 0.62), color=col, fontsize=8.5, ha="center")
    ax.set_xlabel("barrier reward:risk   $U/|D|$")
    ax.set_ylabel("probability of passing")
    ax.set_title("Geometry alone decides the game before skill enters:\n"
                 "at 2:1 a coin-flip trader tops out at 1/3 — and far less with a clock")
    ax.legend(frameon=False, fontsize=9)
    ax.set_ylim(0, 0.7)
    save(fig, "fig1_geometry.png")
    RESULTS["geometry"] = {
        "eventual_zero_edge_2to1": 1/3, "eventual_zero_edge_1p5to1": 1/2.5,
        "finite30_zero_edge_2to1_sig700": float(prob_pass_safe(0.0, 700.0, D, U, T)),
    }


# --------------------------------------------------------------------------- #
#  2.  VEGA: pass probability vs dollar volatility
# --------------------------------------------------------------------------- #
def fig_vega():
    print("[2] vega")
    sig = np.linspace(150, 2600, 240)
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    drifts = [(-60.0, RED, r"$\mu=-\$60$/day (negative edge)"),
              (0.0, NAVY, r"$\mu=0$ (zero edge)"),
              (60.0, GREEN, r"$\mu=+\$60$/day (positive edge)")]
    opt = {}
    for mu, col, lab in drifts:
        pv = np.array([prob_pass_safe(mu, s, D, U, T) for s in sig])
        ax.plot(sig, pv, color=col, lw=2.2, label=lab)
        pe = prob_hit_upper_eventual(mu, 700.0, D, U)  # asymptote ref (sigma-indep for mu=0)
        i = int(np.argmax(pv))
        if mu > 0:
            ax.plot(sig[i], pv[i], "o", color=col, ms=7)
            opt[mu] = {"sigma_star": float(sig[i]), "pass_star": float(pv[i])}
    ax.axhline(P_INF0, color=NAVY, ls="--", lw=1.0, alpha=0.7)
    ax.text(2300, P_INF0 + 0.01, "1/3 cap (zero edge, $T=\\infty$)",
            color=NAVY, fontsize=8.5, ha="right")
    ax.axvspan(350, 1000, color=GOLD, alpha=0.10)
    ax.text(675, 0.02, "risk the rules permit\n(daily limit / consistency / contracts)",
            color=GOLD, fontsize=8.5, ha="center")
    ax.set_xlabel(r"daily dollar-volatility  $\sigma$  (\$/day)")
    ax.set_ylabel("probability of passing  $P_{\\mathrm{pass}}$")
    ax.set_title("Vega is the edgeless trader's only lever — and it is capped\n"
                 "(zero/negative edge: pass rises with $\\sigma$ but never beats 1/3)")
    ax.legend(frameon=False, fontsize=9, loc="center right")
    save(fig, "fig2_vega.png")
    RESULTS["optimal_sigma_pos_edge"] = opt


# --------------------------------------------------------------------------- #
#  3.  THETA: time decay
# --------------------------------------------------------------------------- #
def fig_theta():
    print("[3] theta")
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.4, 4.9))
    tt = np.linspace(0.5, 30, 200)
    for sig, col, lab in [(500, RED, r"$\sigma=\$500$/day"),
                          (800, NAVY, r"$\sigma=\$800$/day"),
                          (1200, GREEN, r"$\sigma=\$1200$/day")]:
        pv = [prob_pass_safe(0.0, sig, D, U, t) for t in tt]
        axL.plot(tt, pv, color=col, lw=2.2, label=lab)
    axL.set_xlabel("time remaining  $T-t$  (days)")
    axL.set_ylabel("$P_{\\mathrm{pass}}$")
    axL.set_title("Pass probability collapses as the clock runs out\n(zero-edge trader at breakeven P&L)")
    axL.legend(frameon=False); axL.invert_xaxis()

    days = np.linspace(1, 29.5, 120)
    for sig, col, lab in [(500, RED, r"$\sigma=\$500$"),
                          (800, NAVY, r"$\sigma=\$800$"),
                          (1200, GREEN, r"$\sigma=\$1200$")]:
        th = [greek_theta(0.0, 0.0, sig, D, U, max(T - dd, 0.3)) for dd in days]
        axR.plot(days, th, color=col, lw=2.2, label=lab)
    axR.set_xlabel("day of evaluation  $t$")
    axR.set_ylabel(r"$\Theta=\partial P_{\mathrm{pass}}/\partial t$  (per day)")
    axR.set_title("Theta: P&L stuck at breakeven bleeds pass-probability")
    axR.axhline(0, color="k", lw=0.6); axR.legend(frameon=False)
    save(fig, "fig3_theta.png")
    RESULTS["theta_example"] = {
        "pass_T30_sig800": prob_pass_safe(0.0, 800.0, D, U, 30),
        "pass_T2_sig800": prob_pass_safe(0.0, 800.0, D, U, 2),
        "pass_T30_sig500": prob_pass_safe(0.0, 500.0, D, U, 30),
        "pass_T2_sig500": prob_pass_safe(0.0, 500.0, D, U, 2),
    }


# --------------------------------------------------------------------------- #
#  4.  DELTA & GAMMA vs current P&L
# --------------------------------------------------------------------------- #
def fig_greeks_in_pnl():
    print("[4] delta/gamma")
    X = np.linspace(D + 80, U - 80, 240)
    fig, (axD, axG) = plt.subplots(1, 2, figsize=(12.4, 4.9))
    for tt, col, lab in [(28, RED, "day 28 ($T-t=2$)"),
                         (15, NAVY, "day 15"),
                         (2, GREEN, "day 2 ($T-t=28$)")]:
        rem = T - tt
        d = [greek_delta(x, 0.0, 800.0, D, U, rem) for x in X]
        g = [greek_gamma(x, 0.0, 800.0, D, U, rem) for x in X]
        axD.plot(X, np.array(d) * 1000, color=col, lw=2.0, label=lab)
        axG.plot(X, np.array(g) * 1e6, color=col, lw=2.0, label=lab)
    for ax in (axD, axG):
        ax.axvline(D, color=RED, ls="--", lw=1, alpha=.6)
        ax.axvline(U, color=GREEN, ls="--", lw=1, alpha=.6)
        ax.axvline(0, color="k", lw=0.6, alpha=.4)
        ax.set_xlabel("current cumulative P&L  $X$  (\\$)")
    axD.set_ylabel(r"$\Delta$  (per \$1000 of P&L)")
    axD.set_title("Delta: marginal pass-probability per dollar of P&L")
    axG.set_ylabel(r"$\Gamma$  ($\times 10^{-6}$)")
    axG.set_title("Gamma: convexity drives forced position sizing")
    axD.legend(frameon=False, fontsize=9); axG.legend(frameon=False, fontsize=9)
    save(fig, "fig4_delta_gamma.png")


# --------------------------------------------------------------------------- #
#  5.  PROBABILITY MATRIX over (win-rate, R:R) -- full rule set
# --------------------------------------------------------------------------- #
def fig_prob_matrix():
    print("[5] probability matrix")
    ps = np.round(np.linspace(0.30, 0.75, 10), 3)
    Rs = np.round(np.linspace(0.4, 3.0, 10), 3)
    k = 350.0
    rng = np.random.default_rng(2025)
    M = np.zeros((len(ps), len(Rs)))
    for i, p in enumerate(ps):
        for j, Rr in enumerate(Rs):
            res = simulate_challenge_mc(
                p, Rr, k, U=U, maxloss=MAXLOSS, n_days=int(T),
                trades_per_day=TRADES_PER_DAY, daily_loss_limit=DAILY,
                trailing=True, consistency=CONSISTENCY, n_paths=30_000, rng=rng)
            M[i, j] = res["pass_rate"]
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    im = ax.imshow(M, origin="lower", aspect="auto", cmap="viridis",
                   extent=[Rs[0], Rs[-1], ps[0], ps[-1]], vmin=0, vmax=0.6)
    Rline = np.linspace(Rs[0], Rs[-1], 100)
    ax.plot(Rline, 1.0 / (1.0 + Rline), color="white", lw=2.2, ls="--",
            label="zero-edge frontier  $p=1/(1+R)$")
    cs = ax.contour(Rs, ps, M, levels=[0.1, 0.2, 0.35], colors="white",
                    linewidths=1.0, alpha=0.85)
    ax.clabel(cs, fmt="%.2f", fontsize=8)
    ax.set_xlabel("reward : risk  $R$")
    ax.set_ylabel("win rate  $p$")
    ax.set_title(f"Probability of passing the 2:1 evaluation\n"
                 f"(full rules: trailing DD, \\${DAILY:.0f} daily, {CONSISTENCY:.0%} consistency)")
    fig.colorbar(im, ax=ax, label="$P_{\\mathrm{pass}}$")
    ax.legend(frameon=True, loc="lower right", fontsize=9)
    save(fig, "fig5_prob_matrix.png")
    np.savetxt("../data/prob_matrix_pass.csv", M, delimiter=",",
               header="rows=p[0.30..0.75], cols=R[0.4..3.0]")

    arche = {}
    for label, p, Rr in [("scalper_neg_skew(p=0.667,R=0.5)", 2/3, 0.5),
                        ("symmetric(p=0.5,R=1)", 0.5, 1.0),
                        ("trend_pos_skew(p=0.333,R=2)", 1/3, 2.0)]:
        res = simulate_challenge_mc(
            p, Rr, k, U=U, maxloss=MAXLOSS, n_days=int(T),
            trades_per_day=TRADES_PER_DAY, daily_loss_limit=DAILY,
            trailing=True, consistency=CONSISTENCY, n_paths=100_000, rng=rng)
        mu, sg = style_to_abm(p, Rr, k, TRADES_PER_DAY)
        arche[label] = {"pass": res["pass_rate"], "bust": res["bust_rate"],
                        "timeout": res["timeout_rate"], "mu": mu, "sigma": sg}
    RESULTS["skew_archetypes"] = arche


# --------------------------------------------------------------------------- #
#  6.  RISK SWEEP + the rule set as a sigma cap
# --------------------------------------------------------------------------- #
def fig_rules():
    print("[6] risk sweep & rule effects")
    ks = np.linspace(80, 800, 26)
    rng = np.random.default_rng(7)
    p, Rr = 0.5, 1.0
    variants = [
        ("static max-loss only", dict(trailing=False, daily_loss_limit=None, consistency=None), NAVY),
        ("+ trailing drawdown", dict(trailing=True, daily_loss_limit=None, consistency=None), GREEN),
        ("+ \\$2,000 daily limit", dict(trailing=True, daily_loss_limit=DAILY, consistency=None), GOLD),
        ("+ 50% consistency", dict(trailing=True, daily_loss_limit=DAILY, consistency=CONSISTENCY), RED),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    summ = {}
    for lab, kw, col in variants:
        pr = []
        for k in ks:
            res = simulate_challenge_mc(p, Rr, k, U=U, maxloss=MAXLOSS,
                                        n_days=int(T), trades_per_day=TRADES_PER_DAY,
                                        n_paths=30_000, rng=rng, **kw)
            pr.append(res["pass_rate"])
        pr = np.array(pr)
        sig_axis = ks * np.sqrt(TRADES_PER_DAY)     # sigma ~ k*sqrt(n) at p=.5,R=1
        ax.plot(sig_axis, pr, color=col, lw=2.2, label=lab)
        i = int(np.argmax(pr))
        summ[lab.replace("\\", "")] = {"sigma_star": float(sig_axis[i]),
                                       "pass_star": float(pr[i]),
                                       "pass_at_sig700": float(pr[np.argmin(np.abs(sig_axis-700))])}
    ax.set_xlabel(r"daily dollar-volatility  $\sigma$  (\$/day)")
    ax.set_ylabel("$P_{\\mathrm{pass}}$")
    ax.set_title("Each rule peels back the variance pathway:\n"
                 "the consistency + daily rules cap $\\sigma$ where Theta wins")
    ax.legend(frameon=False, fontsize=9)
    save(fig, "fig6_rules_cap.png")
    RESULTS["rule_effect"] = summ


# --------------------------------------------------------------------------- #
#  7.  Jump-diffusion robustness
# --------------------------------------------------------------------------- #
def fig_jumps():
    print("[7] jumps")
    ks = np.linspace(80, 800, 22)
    rng = np.random.default_rng(33)
    p, Rr = 0.5, 1.0
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    cfgs = [("pure diffusion", 0.0, 0.0, NAVY),
            ("jumps: 2% prob, -3k gap", 0.02, 3.0, GOLD),
            ("jumps: 4% prob, -4k gap", 0.04, 4.0, RED)]
    out = {}
    for lab, jp, jm, col in cfgs:
        pr = []
        for k in ks:
            res = simulate_challenge_mc(p, Rr, k, U=U, maxloss=MAXLOSS,
                                        n_days=int(T), trades_per_day=TRADES_PER_DAY,
                                        trailing=True, daily_loss_limit=DAILY,
                                        consistency=CONSISTENCY, n_paths=30_000,
                                        rng=rng, jump_prob=jp, jump_mult=jm)
            pr.append(res["pass_rate"])
        pr = np.array(pr)
        ax.plot(ks * np.sqrt(TRADES_PER_DAY), pr, color=col, lw=2.2, label=lab)
        out[lab] = {"max_pass": float(pr.max())}
    ax.set_xlabel(r"daily dollar-volatility  $\sigma$  (\$/day)")
    ax.set_ylabel("$P_{\\mathrm{pass}}$")
    ax.set_title("Adverse gaps only lower the ceiling — result is robust")
    ax.legend(frameon=False, fontsize=9)
    save(fig, "fig7_jumps.png")
    RESULTS["jump_effect"] = out


# --------------------------------------------------------------------------- #
#  8.  Net EV vs genuine edge (evaluation + funded phase)
# --------------------------------------------------------------------------- #
def fig_net_ev():
    print("[8] net EV vs edge")
    FEE, SPLIT, k = 129.0, 0.90, 350.0
    Rr = 1.0
    ps = np.linspace(0.45, 0.62, 22)
    rng = np.random.default_rng(808)

    def journey(p):
        res = simulate_challenge_mc(p, Rr, k, U=U, maxloss=MAXLOSS, n_days=int(T),
                                    trades_per_day=TRADES_PER_DAY,
                                    daily_loss_limit=DAILY, trailing=True,
                                    consistency=CONSISTENCY, n_paths=20_000, rng=rng)
        prize = simulate_funded_extraction(
            p, Rr, k, maxloss=MAXLOSS, split=SPLIT, payout_period=10,
            payout_cap=2_500.0, trailing=True, daily_loss_limit=DAILY,
            trades_per_day=TRADES_PER_DAY, n_paths=12_000, max_days=160, rng=rng)
        return res["pass_rate"] * prize - FEE, res["pass_rate"], prize

    evs, passes, prizes = [], [], []
    for p in ps:
        e, pp, pr = journey(p); evs.append(e); passes.append(pp); prizes.append(pr)
    evs = np.array(evs)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.4, 4.9))
    axL.plot(ps * 100, evs, color=RED, lw=2.4)
    axL.axhline(0, color="k", lw=0.8); axL.axvline(50, color="gray", ls="--", lw=1, alpha=.6)
    sgn = np.sign(evs); cross = np.where(np.diff(sgn) > 0)[0]
    be = None
    if len(cross):
        i = cross[0]
        be = ps[i] + (ps[i+1]-ps[i]) * (0 - evs[i]) / (evs[i+1]-evs[i])
        axL.axvline(be*100, color=NAVY, ls=":", lw=1.4)
        axL.annotate(f"break-even\n$p\\approx{be*100:.0f}\\%$", (be*100, 0),
                     (be*100+1, max(evs)*0.4), color=NAVY, fontsize=9)
    axL.set_xlabel("genuine win rate  $p$  (%, R:R = 1)")
    axL.set_ylabel("net expected value of the evaluation (\\$)")
    axL.set_title("Now you need a real edge ABOVE a coin flip just to break even")
    axR.plot(ps * 100, passes, color=NAVY, lw=2.2, label="$P_{\\mathrm{pass}}$")
    axR2 = axR.twinx(); axR2.plot(ps * 100, prizes, color=GOLD, lw=2.2)
    axR.set_xlabel("genuine win rate  $p$  (%)"); axR.set_ylabel("$P_{\\mathrm{pass}}$", color=NAVY)
    axR2.set_ylabel("funded extraction $\\Pi$ (\\$)", color=GOLD)
    axR.axvline(50, color="gray", ls="--", lw=1, alpha=.6)
    axR.set_title("The two stages of the prize")
    save(fig, "fig8_net_ev.png")
    RESULTS["journey_ev"] = {"fee": FEE, "split": SPLIT, "k": k,
                             "breakeven_winrate_pct": (float(be*100) if be else None),
                             "ev_at_p50": float(evs[np.argmin(np.abs(ps-0.5))]),
                             "pass_at_p50": float(passes[np.argmin(np.abs(ps-0.5))])}


# --------------------------------------------------------------------------- #
#  9.  Sample P&L paths
# --------------------------------------------------------------------------- #
def fig_paths():
    print("[9] paths")
    rng = np.random.default_rng(11)
    p, Rr, k = 0.5, 1.0, 350.0
    n_show, n_steps = 60, int(T) * TRADES_PER_DAY
    win_pnl, loss_pnl = Rr * k, -k
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    t_axis = np.linspace(0, T, n_steps + 1)
    for _ in range(n_show):
        x = [0.0]; peak = 0.0; col = "#888888"
        for s in range(n_steps):
            x.append(x[-1] + (win_pnl if rng.random() < p else loss_pnl))
            peak = max(peak, x[-1]); floor = min(peak - MAXLOSS, 0.0)
            if x[-1] <= floor:
                col = RED; break
            if x[-1] >= U:
                col = GREEN; break
        ax.plot(t_axis[:len(x)], x, color=col, lw=0.8, alpha=0.6)
    ax.axhline(U, color=GREEN, lw=2, ls="--", label="profit target $+\\$6{,}000$")
    ax.axhline(D, color=RED, lw=2, ls="--", label="max loss $-\\$3{,}000$ (trailing)")
    ax.axhline(0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel("day"); ax.set_ylabel("cumulative P&L (\\$)")
    ax.set_title("60 simulated P&L paths in the 2:1 double-barrier strip\n"
                 "(zero-edge trader, \\$350 risk/trade)")
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    save(fig, "fig9_paths.png")


# --------------------------------------------------------------------------- #
#  10.  The volatility filter: pass-probability haircut by edge
# --------------------------------------------------------------------------- #
def fig_haircut():
    print("[10] rule haircut by edge")
    sig_grid = np.linspace(200, 2400, 200)
    regimes = [(-60.0, "negative\nedge"), (0.0, "zero\nedge"),
               (60.0, "mild +edge\n(+$60/d)"), (150.0, "strong +edge\n(+$150/d)")]
    sig_perm = 600.0     # permitted volatility
    labels, p_best, p_perm = [], [], []
    out = {}
    for mu, name in regimes:
        pv = np.array([prob_pass_safe(mu, s, D, U, T) for s in sig_grid])
        best = float(pv.max())
        perm = prob_pass_safe(mu, sig_perm, D, U, T)
        labels.append(name); p_best.append(best); p_perm.append(perm)
        out[name.replace(chr(10), " ")] = {
            "best_pass": best, "permitted_pass": float(perm),
            "haircut_pct": float((best - perm) / max(best, 1e-9) * 100)}
    x = np.arange(len(labels)); w = 0.34
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.bar(x - w/2, p_best, w, color=GREEN, label="best attainable (high $\\sigma$, unconstrained)")
    ax.bar(x + w/2, p_perm, w, color=RED, label=f"under the rules ($\\sigma=\\${sig_perm:.0f}$)")
    for xi, a, b in zip(x, p_best, p_perm):
        ax.text(xi - w/2, a + 0.006, f"{a:.2f}", ha="center", fontsize=8.5)
        ax.text(xi + w/2, b + 0.006, f"{b:.2f}", ha="center", fontsize=8.5)
        hc = (a - b) / max(a, 1e-9) * 100
        ax.text(xi, max(a, b) + 0.03, f"$-{hc:.0f}\\%$", ha="center",
                fontsize=8.5, color="#333", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("$P_{\\mathrm{pass}}$")
    ax.set_title("The $\\sigma$-cap is an edge filter: the relative haircut shrinks\n"
                 "as edge grows — it spares skill and crushes the edgeless")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_ylim(0, max(p_best) * 1.28)
    save(fig, "fig10_rule_haircut.png")
    RESULTS["rule_haircut"] = out


# --------------------------------------------------------------------------- #
def main():
    validation()
    fig_geometry()
    fig_vega()
    fig_theta()
    fig_greeks_in_pnl()
    fig_prob_matrix()
    fig_rules()
    fig_jumps()
    fig_net_ev()
    fig_paths()
    fig_haircut()

    RESULTS["contract"] = {"U": U, "D": D, "maxloss": MAXLOSS, "ratio": U/MAXLOSS,
                           "T": T, "daily_limit": DAILY, "consistency": CONSISTENCY,
                           "trades_per_day": TRADES_PER_DAY,
                           "eventual_zero_edge": P_INF0}
    with open("../data/results.json", "w") as fh:
        json.dump(RESULTS, fh, indent=2)
    print("\nwrote data/results.json")


if __name__ == "__main__":
    main()
