"""Cross-check the arithmetic-BM spectral formula against Monte-Carlo,
for the 2:1 futures contract (U=+6000, D=-3000, T=30 days)."""
import numpy as np
from barrier_core import (prob_pass_analytic, prob_knockout_analytic,
                          prob_hit_upper_eventual, simulate_abm_paths)

U, maxloss, T = 6_000.0, 3_000.0, 30.0
D = -maxloss
print(f"contract: D={D:.0f}  U={U:.0f}  ratio U/|D| = {U/maxloss:.1f}:1")
print(f"eventual zero-drift pass (gambler's ruin) = (0-D)/(U-D) = {(0-D)/(U-D):.4f}\n")

cases = [
    ("zero drift",      0.0,    700.0),
    ("positive drift", 60.0,    700.0),
    ("negative drift",-60.0,    700.0),
    ("low vol",         0.0,    350.0),
    ("high vol",        0.0,   1400.0),
]
print(f"{'case':<16}{'mu':>7}{'sigma':>8}{'pass(an)':>11}{'pass(MC)':>11}"
      f"{'bust(an)':>11}{'bust(MC)':>11}{'P_inf':>9}")
for name, mu, sig in cases:
    pa = prob_pass_analytic(mu, sig, D, U, T)
    ka = prob_knockout_analytic(mu, sig, D, U, T)
    mc = simulate_abm_paths(mu, sig, D, U, T, steps_per_day=16, n_paths=200_000)
    pinf = prob_hit_upper_eventual(mu, sig, D, U)
    print(f"{name:<16}{mu:>7.0f}{sig:>8.0f}{pa:>11.4f}{mc['pass_rate']:>11.4f}"
          f"{ka:>11.4f}{mc['bust_rate']:>11.4f}{pinf:>9.4f}")

print("\nT -> infinity check (zero drift, sigma=700):")
for Tt in [30, 200, 1000, 5000]:
    pu = prob_pass_analytic(0.0, 700.0, D, U, Tt)
    pd = prob_knockout_analytic(0.0, 700.0, D, U, Tt)
    print(f"  T={Tt:>5}:  pass={pu:.4f}  bust={pd:.4f}  sum={pu+pd:.4f}"
          f"  (P_inf={(0-D)/(U-D):.4f})")
