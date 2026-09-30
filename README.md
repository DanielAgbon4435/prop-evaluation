# A futures evaluation is a double-barrier binary

A futures prop-firm evaluation pays a fixed profit target and fails at a loss limit. On a 100,000-dollar card those barriers are +6,000 and −3,000, so the target is twice as far as ruin, and the clock is 30 days. Because the account is scored in dollars of P&L, the model is arithmetic Brownian motion, not geometric motion on a price.

With no drift and no time limit, gambler's ruin gives the pass probability as 3,000 / 9,000 = 1/3. The 30-day clock cuts the zero-drift case, at a daily volatility of 700 dollars, to about 0.12. A low-volatility profile that satisfies the consistency and daily-loss rules passes about 1.9% of the time in the simulation. The closed form is a spectral expansion for arithmetic Brownian motion on a strip. A Monte Carlo check sits next to it in `data/results.json`.

The paper is `whitepaper.md`. Figures are in `figures/`.

## Reproduce

```bash
cd code
python3 sanity_check.py
python3 run_simulation.py
```

Requires Python 3.9+, NumPy, SciPy, matplotlib, and pandas (`requirements.txt`). The full simulation takes about 15 seconds.
