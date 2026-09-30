---
title: "Two-to-One Against"
subtitle: "A Double-Barrier Binary-Option Analysis of Futures Proprietary-Trading Evaluations"
date: "June 2026"
---

# Two-to-One Against

### A Double-Barrier Binary-Option Analysis of Futures Proprietary-Trading Evaluations

---

## Abstract

Retail traders are sold a futures proprietary-trading-firm ("prop firm")
*evaluation* as a funding opportunity: pay a monthly fee, hit a profit target on
a simulated account without breaking a loss rule, and trade the firm's capital
for a high profit split. We argue this is the wrong accounting frame. The
correct frame is that the firm is **writing an exotic derivative and selling it
to an unsophisticated counterparty.**

We prove that a standard futures evaluation is mathematically identical to a
**discrete double-barrier one-touch (digital) option** written on the trader's
cumulative profit-and-loss. Because the contract is denominated in dollars and a
futures trader trades a roughly fixed number of contracts, the natural model is
**arithmetic Brownian motion on P&L**, not geometric motion on equity. We take
the exact contract from a live 100,000 pricing card — **profit target
6,000, max-loss "One Rule" 3,000, 2,000 daily loss limit, 50% consistency,
30-day cycle** — whose numbers coincide with the widely advertised 100K
futures Combine archetype.

The single most important number falls straight out of the geometry. The target
is **twice as far as ruin** (a 2:1 barrier reward-to-risk). By the
gambler's-ruin theorem, a genuinely edgeless trader's probability of ever hitting
the target before busting — *with no time limit at all* — is

$$
P_{\text{pass}}^{\infty}\big|_{\text{zero edge}} = \frac{|D|}{U+|D|} = \frac{3000}{9000} = \frac{1}{3}.
$$

The 30-day clock and the rules only subtract from this ceiling. We derive the
finite-horizon pass probability in closed form (a Dirichlet heat-kernel spectral
expansion), validate it against a 200,000-path Monte-Carlo, and compute the
Greeks. **Theta** is brutal: at realistic risk the pass probability decays from
0.12–0.16 over the month to **effectively zero** in the final two days.
**Vega** is positive but *capped*: an edgeless trader can raise the pass
probability by taking more volatility, yet can never beat the 1/3 ceiling — and
a *skilled* trader has a modest interior-optimal volatility that already sits
inside the permitted band. **Delta and Gamma** describe the forced
position-sizing dynamics near each barrier.

We then value the whole journey honestly — the funded account is itself a second
(trailing) barrier option with capped payouts. Unlike a trader-favourable
equity-percentage contract, the 2:1 futures geometry makes the evaluation
**negative expected value for any trader without a genuine, positive edge**: the
break-even win rate at 1:1 reward-to-risk is **≈ 50%** *before costs* and clearly
above it once fees and slippage are included. The "Responsible Trading" rule set
— trailing drawdown, daily loss limit, consistency — functions as a precise
**cap on volatility $\sigma$** that acts as an *edge filter*: the relative
pass-probability haircut it imposes falls monotonically with edge, from **−93%**
(negative edge) and **−80%** (zero edge) to just **−16%** for a strongly skilled
trader, who barely needs variance. It strips the edgeless majority — who could
*only* have passed via variance — while sparing genuine skill. With variance
removed, Theta finishes the job. The "responsible" low-volatility scalper the
rules nominally reward has, in our simulation, a **1.9%** pass rate.

All results are reproducible from the accompanying Python (`code/`); every quoted
number is emitted to `data/results.json`. We keep the analysis **general** to
futures evaluations; the supplied card is used only as a concrete, canonical
instance.

---

## 1. Introduction

### 1.1 The two ways to read a futures evaluation

A modern futures prop firm sells a product that looks like this:

> Pay 129/month for a 100,000 "buying-power" simulated account. Grow its P&L
> to **+6,000** (the **profit target**) and you are funded, trading our capital
> for a high split. But if your P&L ever falls to **−3,000** below your
> high-water mark (the trailing **max-loss limit** — "Don't Break the One
> Rule") — or you lose **2,000** in a single day — you fail. To get funded you
> must also satisfy a **50% consistency** rule (no single day may be more than
> half your total profit). The cycle resets monthly.

The retail reading is *"a cheap, fair route to capital; I just have to trade well
for a month."* The institutional reading is *"someone has written me a
path-dependent digital option, set the target twice as far as the knock-out, and
attached a list of rules that cap how fast I can move."* This paper formalises
the second reading and shows it is not a metaphor but an identity — and that, for
the futures 2:1 structure specifically, the mathematics is decisively against the
edgeless buyer.

### 1.2 Thesis

> **A futures prop-firm evaluation is a discrete double-barrier one-touch binary
> option on cumulative P&L. Its 2:1 barrier geometry caps an edgeless trader's
> lifetime pass probability at 1/3; the 30-day clock (Theta) and the trailing /
> daily / consistency rules (a cap on volatility $\sigma$) cut it to roughly one
> in ten; and the funded payoff is itself a capped second barrier option. The
> result is a structurally negative-expected-value purchase for anyone without a
> genuine statistical edge — and the rules are precisely the mechanism that
> filters such traders out.**

### 1.3 Contributions

1. **A formal dictionary** (Section 2) translating every line of the pricing card
  into exotic-option notation.
2. **The geometry result** (Section 3.3): the 2:1 reward-to-risk fixes the
  edgeless ceiling at 1/3 *before skill or time is considered*.
3. **Closed-form pass probabilities** (Section 3): a scale-function result for
  the perpetual contract and a Dirichlet heat-kernel spectral expansion for the
   30-day contract, validated to Monte-Carlo accuracy.
4. **The Greeks of an evaluation** (Section 4): Theta, the *capped* Vega, Delta,
  and Gamma of the pass-probability functional.
5. **A 100,000-path Monte-Carlo study** (Sections 5–6) mapping trading style
  (win rate, reward-to-risk, dollar risk, skew) to pass probability under the
   full rule set, with jump-diffusion robustness.
6. **An honest economic valuation** (Section 7): the funded account as a capped
  second barrier option, the break-even edge, and the volatility filter that is
   the real house edge.

### 1.4 A note on intellectual honesty (and what changed from the equity model)

An earlier version of this analysis modelled the underlying as an
equity-percentage process with a **+8% / −10%** target/loss (a 0.8:1 geometry).
That structure is mildly *trader-favourable* in the perpetual limit (eventual
pass ≈ 0.58), and we reported — against our own initial hypothesis — that priced
as a stand-alone instrument it was roughly fair. **The futures card is a
different instrument.** Its 2:1 dollar geometry inverts the asymmetry: the
perpetual edgeless ceiling is 1/3, not 0.58. We re-derived everything in P&L
space and report the consequences, which are far more adverse to the buyer and
make the predatory reading robust rather than borderline. We flag the model
switch explicitly so the two analyses are not confused.

---

## 2. The Contract and Its Exotic-Option Identity

### 2.1 The dictionary

Let the trader's cumulative account P&L at time $t$ be $X_t$ (dollars above the
starting balance), monitored in trading days $t \in [0,T]$.


| Pricing-card line   | Symbol | Card value            | Option-theoretic object                                   |
| ------------------- | ------ | --------------------- | --------------------------------------------------------- |
| 100K buying power   | $S_0$  | 100,000 notional      | Collateral / notional (P&L starts at 0)                   |
| Profit target       | $U$    | **+6,000**            | Upper **knock-in** barrier; one touch triggers the payoff |
| Max loss (One Rule) | $D$    | **−3,000**            | Lower **knock-out** barrier (trailing high-water-mark)    |
| Barrier reward:risk | $U/    | D                     | $                                                         |
| Time limit / cycle  | $T$    | 30 days               | Maturity                                                  |
| "Funded"            | —      | binary                | Fixed (digital) payoff $\Pi$ on knock-in                  |
| Daily loss limit    | $d$    | 2,000 (0.66×)         | Path constraint that **caps $\sigma$**                    |
| Consistency target  | $c$    | 50%                   | Best day ≤ 50% of profit — a further **$\sigma$ cap**     |
| Max contract limit  | —      | 10 mini / 100 micro   | Hard cap on per-trade dollar risk $k$ (hence $\sigma$)    |
| Profit split        | $s$    | ~90% (capped payouts) | Scaling of the funded-phase payoff                        |
| Monthly fee / reset | $F$    | 129                   | Option **premium** (recurring)                            |


### 2.2 The payoff, formally

Define the first-passage times of the P&L process

$$
\tau_U = \inft \ge 0 : X_t \ge U, \qquad
\tau_D = \inft \ge 0 : X_t \le D_t,
$$

where $D_t = \max(\text{peak}_t - 3000, 0)$ is the **trailing** max-loss floor
(it ratchets up with the high-water mark and, in the typical implementation,
locks at break-even). The trader **passes** iff

$$
\text{PASS} = \tau_U \le T \wedge \tau_U < \tau_D \cap \text{consistency satisfied},
$$

and the cash payoff is digital, $\text{Payoff} = \Pi\cdot\mathbf 1_{\text{PASS}} -
F$, with $\Pi$ the value of being funded (Section 7). The firm is **short** this
digital and **long** the recurring fee.

### 2.3 Taxonomy

The instrument is a **double-barrier one-touch binary**: an American digital /
one-touch at $U$ (you pass the instant P&L tags +6,000), a down-and-out
knock-out at the trailing $D_t$ (one touch and the contract is dead), and
European expiry at $T$. Barriers are monitored **discretely** (trade-by-trade),
hence *discrete* double barrier. The consistency rule is a path functional layered
on top. For "unlimited-time" account types set $T=\infty$; the trailing
drawdown and daily limit then become the binding constraints (Section 3.3).

---

## 3. The Mathematical Framework

### 3.1 P&L as arithmetic Brownian motion

A futures trader trades a roughly fixed number of contracts, so each trade adds an
approximately fixed dollar increment to cumulative P&L. The natural state
variable is therefore $X_t$ itself, modelled as **arithmetic** Brownian motion
with drift,

$$
dX_t = \mudt + \sigmadW_t, \qquad X_0 = 0,
$$

where $\mu$ is the **dollar drift per day** (skill, net of fees) and $\sigma$ is
the **dollar volatility per day** (risk / position size). This differs
deliberately from a geometric-Brownian equity model: there is **no volatility
drag** term, so a genuinely break-even trader has $\mu = 0$ exactly, and the
barriers are fixed dollar levels rather than percentages.

The barriers are $U = +6000$ and (static reference) $D = -3000$, a strip of width
$L = U - D = 9000$, with the trader starting a distance $b = 6000$ below the
target and $a = 3000$ above ruin. **The defining asymmetry is $b = 2a$:** the
target is twice as far as the knock-out. Everything follows from this.

### 3.2 Pass probability as a first-passage problem

$$
P_{\text{pass}}(\mu,\sigma,T) = \Pr\big(\tau_U \le T \wedge \tau_U < \tau_D \mid X_0 = 0\big).
$$

We give the perpetual answer first, then the finite-$T$ spectral series.

### 3.3 Perpetual contract: the geometry decides

For $T=\infty$ the scale function of arithmetic BM is $s(x)=e^{-\theta x}$ with
$\theta = 2\mu/\sigma^2$, and optional stopping gives

$$
\boxed{P_{\text{pass}}^{\infty} = \frac{1 - e^{-\theta D}}{e^{-\theta U} - e^{-\theta D}}}
\qquad\xrightarrow[\mu\to 0]{}\qquad \frac{0-D}{U-D} = \frac{|D|}{U+|D|}.
$$

For the **edgeless** trader ($\mu = 0$):

$$
P_{\text{pass}}^{\infty}\big|_{\text{zero edge}} = \frac{3000}{9000} = \frac{1}{3}.
$$

Pass probability versus barrier reward-to-risk

*Figure 1. Pass probability versus the barrier reward-to-risk $U/|D|$. Blue: the
perpetual (no-time-limit) edgeless probability $1/(1+U/|D|)$. Red: with the
30-day clock at moderate risk. The 2:1 card sits where a coin-flip trader tops
out at 1/3 with no clock — and ~0.12 with one. The old +8%/−10% equity model
(0.8:1) sat in trader-favourable territory.*

This is the headline the marketing hides: **before skill, time, or any rule, the
2:1 geometry caps an honest coin-flipper at one in three.** A 1.5:1 account
(common on smaller sizes) caps at $1/2.5 = 0.40$; the old 0.8:1 equity contract
sat at 0.56. Geometry alone moves the goalposts by tens of percentage points.

### 3.4 Finite horizon: the Dirichlet heat-kernel spectral expansion

Adding the clock, we solve the Fokker–Planck equation on the strip $(D,U)$ with
absorbing (Dirichlet) boundaries and integrate the probability flux through the
upper boundary over $[0,T]$. Shifting to $y = x - D \in (0,L)$ with start
$y_0 = -D = a$, removing drift with the Doob $h$-transform
$p = e^{\frac{\mu}{\sigma^2}(y-y_0) - \frac{\mu^2}{2\sigma^2}t}q$, and expanding
the resulting heat equation in $\sin(n\pi y/L)$ eigenfunctions yields the
**finite-horizon pass probability**

$$
\boxed{
P_{\text{pass}}(\mu,\sigma,T) =
\frac{\sigma^2 \pi}{L^2} e^{\frac{\mu}{\sigma^2}(L - y_0)}
\sum_{n=1}^{\infty} (-1)^{n+1} n \sin\frac{n\pi y_0}{L}
\frac{1 - e^{-\beta_n T}}{\beta_n}
},\qquad
\beta_n = \frac{\mu^2}{2\sigma^2} + \frac{\sigma^2}{2}\Big(\frac{n\pi}{L}\Big)^2 .
$$

The lower-boundary flux gives the **bust probability** $P_{\text{bust}}$; the
residual is the **time-out probability**. As $T\to\infty$ this collapses to the
scale-function result of Section 3.3, and $P_{\text{pass}}+P_{\text{bust}}\to 1$.
Both checks pass numerically. (Implementation: 400–600 terms; a Monte-Carlo
fallback covers the small-$\sigma$/large-$|\mu|$ corner. See
`code/barrier_core.py`.)

### 3.5 Validation against Monte-Carlo

Cross-checking the closed form against a 200,000-path arithmetic-BM Monte-Carlo
with fine (16 steps/day) monitoring:


| Case            | $\mu$ (/day) | $\sigma$ (/day) | $P_{\text{pass}}$ (analytic) | $P_{\text{pass}}$ (MC) | $P_{\text{bust}}$ (MC) |
| --------------- | ------------ | --------------- | ---------------------------- | ---------------------- | ---------------------- |
| Zero edge       | 0            | 700             | 0.1168                       | 0.1092                 | 0.4191                 |
| Positive edge   | +60          | 700             | 0.2263                       | 0.2135                 | 0.2733                 |
| Negative edge   | −60          | 700             | 0.0521                       | 0.0481                 | 0.5826                 |
| High volatility | 0            | 1400            | 0.3189                       | 0.3183                 | 0.6391                 |


The analytic (continuous-monitoring) value sits just above the discrete
Monte-Carlo, as theory predicts. Note the headline already: a zero-edge trader
risking 700/day passes only **≈ 11–12%** of the time in 30 days — about a third
of the already-modest 1/3 perpetual ceiling, the rest eaten by Theta.

---

## 4. The Greeks of an Evaluation

We treat the **pass probability** $P_{\text{pass}}$ as the object of interest
(the risk-neutral price of the digital is $e^{-rT}\PiP_{\text{pass}}^{\mathbb
Q}$, so probability Greeks and price Greeks differ only by constants). With
current P&L $X$ and time-to-maturity $\tau = T-t$:

$$
\Theta = \frac{\partial P_{\text{pass}}}{\partial t} = -\frac{\partial P_{\text{pass}}}{\partial \tau},\quad
\nu_{\text{Vega}} = \frac{\partial P_{\text{pass}}}{\partial \sigma},\quad
\Delta = \frac{\partial P_{\text{pass}}}{\partial X},\quad
\Gamma = \frac{\partial^2 P_{\text{pass}}}{\partial X^2},
$$

computed by central differences on the closed form.

### 4.1 Theta ($\Theta$): the clock is the firm's ally

Because the upper barrier is a knock-in, time spent not-yet-touching +6,000 is
time that cannot be recovered, and the distance to travel ($b = 6000$) is large.

Theta decay of the pass probability

*Figure 2. Left: pass probability versus time remaining for an edgeless trader at
break-even P&L, at three risk levels. Right: the Theta felt day-by-day by a
trader whose P&L is treading water.*


| Risk               | $P_{\text{pass}}$ with 30 days left | $P_{\text{pass}}$ with 2 days left |
| ------------------ | ----------------------------------- | ---------------------------------- |
| $\sigma = 800$/day | 0.164                               | **≈ 0.000**                        |
| $\sigma = 500$/day | 0.028                               | **≈ 0.000**                        |


By the final two days, a trader who has not already banked most of the +6,000
has a pass probability indistinguishable from zero: there is no longer enough
time for diffusion to span the gap to the target. Theta quietly converts the
majority of accounts — neither far ahead nor yet busted — to worthless.

### 4.2 Vega ($\nu_{\text{Vega}}$): a lever that is capped

Because Theta is lethal, the edgeless trader's only way to *reach* the target in
time is to widen the distribution — raise $\sigma$. But here the arithmetic
geometry bites in a way the equity model did not.

Pass probability versus dollar volatility

*Figure 3. Pass probability versus daily dollar-volatility. For zero/negative
edge it rises monotonically with $\sigma$ but is asymptotically **capped at 1/3**
(dashed). For positive edge it has an interior optimum (here $\sigma^\star \approx
1{,}410$, $P\approx 0.38$). Shaded: the volatility the rules actually permit.*

Three facts:

1. **Edgeless Vega is positive but bounded.** For $\mu \le 0$, $P_{\text{pass}}$
  is increasing in $\sigma$ — the trader is mathematically *incentivised to take
   more risk* — yet **no amount of volatility can beat the 1/3 ceiling**, and
   inside the permitted band the pass probability is only 0.05–0.17. The
   edgeless trader is structurally trapped below one in three.
2. **Skill creates a finite optimum.** For a positive-edge trader the curve
  *humps* and then declines toward 1/3: too much $\sigma$ washes out the drift
   advantage. The optimum is a *modest* risk level that sits near or inside the
   permitted band. As in the equity analysis, **the sign and shape of Vega is a
   litmus test for edge** — and the rules, by capping $\sigma$, implicitly test
   exactly that.
3. **The rules confiscate the only lever the edgeless trader has.** Capping
  $\sigma$ barely matters to the skilled trader (whose optimum is already low)
   and is devastating to the edgeless one (Section 7.3).

### 4.3 Delta ($\Delta$) and Gamma ($\Gamma$): forced position sizing

Delta and Gamma versus current P&L

*Figure 4. Delta (left) and Gamma (right) of the pass probability versus current
cumulative P&L, at three points in the cycle. Dashed lines mark the barriers and
break-even.*

- **Delta** — marginal pass-probability per dollar of P&L — is largest in the
interior and near the target late in the cycle, when little time remains and
each dollar moves you across the now-narrow remaining gap.
- **Gamma** is the destabiliser. Approaching the *target* with little time left,
Gamma is large and positive: small favourable moves sharply raise the pass
probability, rationally inviting the trader to *press* size to lock the touch.
Approaching the trailing *floor*, the convexity and the daily limit conspire to
knock the account out before any "martingale out of the hole" can pay. In
options language the trader is **short Gamma against the clock** — the position
demands its largest adjustments exactly when it is most fragile, which is what
the daily-loss and trailing rules exist to exploit.

---

## 5. From Trading Style to $(\mu, \sigma)$

We parametrise $(\mu,\sigma)$ by **win rate** $p$, **reward-to-risk** $R$, and
**dollar risk per trade** $k$ (the stop, in dollars), under fixed-size futures
trading. Each trade adds $+Rk$ with probability $p$ or $-k$ with probability
$1-p$. With $n$ trades/day ($n=4$):

$$
\mu = n\big[k(p(1+R)-1) - \text{cost}\big],\qquad
\sigma = k\sqrt{n\big[pR^2 + (1-p) - (p(1+R)-1)^2\big]} .
$$

Two structural facts:

- **Dollar risk $k$ controls $\sigma$ linearly.** The max-contract limit, the
daily loss limit, and the consistency rule are all, ultimately, **caps on $k$
and hence on $\sigma$.**
- **Skill $(p,R)$ controls $\mu$**, with the *zero-edge frontier* at
$p = 1/(1+R)$. Unlike the equity model there is no volatility drag, so a
break-even trader sits exactly at $\mu = 0$ — but trading costs (commissions,
exchange fees, slippage) push real traders to $\mu < 0$.

### 5.1 Skew profiles (on the zero-edge frontier)

- **Scalper / negative skew** ($p=0.667$, $R=0.5$): many small wins, rare large
losses; low $\sigma \approx 495$/day. This is the "responsible," consistent
style the rules nominally reward.
- **Symmetric** ($p=0.5$, $R=1$): $\sigma \approx 700$/day.
- **Trend / positive skew** ($p=0.333$, $R=2$): many small losses, rare large
wins; high $\sigma \approx 990$/day.

Because Vega is positive, the model predicts the high-$\sigma$ positive-skew style
passes most often — the Monte-Carlo confirms it (Section 6.2), with the
uncomfortable corollary that the *prudent* low-$\sigma$ scalper is the
least likely to pass.

---

## 6. Monte-Carlo Study

### 6.1 Engine and assumptions

A vectorised trade-level Monte-Carlo (`simulate_challenge_mc`) simulates P&L
trade-by-trade (4 trades/day × 30 days), monitors both barriers discretely,
applies the **trailing** high-water-mark drawdown (locking at break-even), the
**2,000 daily loss limit**, and the **50% consistency** rule (a pass requires
the best single day to be ≤ 50% of total profit), with optional adverse jumps.
Up to **100,000 paths** per configuration, fixed seeds.

Sample P&L paths in the 2:1 strip

*Figure 5. Sixty P&L paths for a zero-edge trader risking 350/trade. Green =
passed (+6,000), red = busted (trailing −3,000), grey = timed out. The
target sits twice as far from the start as the floor.*

### 6.2 The probability matrix and the skew result

Probability matrix over win rate and reward-to-risk

*Figure 6. Monte-Carlo pass probability over $(R, p)$ at 350 risk/trade under
the full rule set. Dashed white: the zero-edge frontier $p=1/(1+R)$; contours at
0.10, 0.20, 0.35.*

Pass probability is organised by **which side of the zero-edge frontier** the
trader is on — i.e. by genuine edge — but even well above the frontier it is held
down by the 2:1 geometry and the rules. Holding edge at zero and varying skew:


| Profile             | $(p,R)$       | $\sigma$ (/day) | Pass      | Bust  | Time-out |
| ------------------- | ------------- | --------------- | --------- | ----- | -------- |
| Scalper (neg. skew) | $(0.667,0.5)$ | 495             | **0.019** | 0.438 | 0.543    |
| Symmetric           | $(0.5,1.0)$   | 700             | 0.091     | 0.679 | 0.229    |
| Trend (pos. skew)   | $(0.333,2.0)$ | 990             | 0.165     | 0.786 | 0.049    |


The "responsible," consistent scalper — exactly the trader the rules claim to
cultivate — passes **1.9%** of the time. The high-variance trend trader passes
most (16.5%) but busts 79% of the time. **Variance is the only route to the
target, and the rules forbid variance.**

### 6.3 The rule set is a volatility cap

Adding the rules one at a time to the risk sweep:

Effect of the rule set

*Figure 7. Pass probability versus $\sigma$ under four rule regimes. Each added
rule peels back the high-$\sigma$ region where the pass probability would
otherwise climb toward the 1/3 ceiling.*


| Rule regime          | Peak $P_{\text{pass}}$ |
| -------------------- | ---------------------- |
| Static max-loss only | 0.314                  |
| + trailing drawdown  | 0.221                  |
| + 2,000 daily limit  | 0.173                  |
| + 50% consistency    | 0.170                  |


The static contract lets a high-$\sigma$ gambler approach the 1/3 ceiling (0.31);
the trailing drawdown, daily limit, and consistency rule together **cut the
attainable peak roughly in half** (to 0.17) and make the high-$\sigma$ region
unsurvivable. The rules do not change what the trader *should* do to pass (take
variance); they change whether it *works* (it no longer does).

### 6.4 Jump-diffusion robustness

Jump-diffusion robustness

*Figure 8. Adding Merton-style adverse gaps lowers the ceiling further; the
qualitative picture is unchanged.*


| Model                   | Peak $P_{\text{pass}}$ |
| ----------------------- | ---------------------- |
| Pure diffusion          | 0.175                  |
| Jumps: 2% prob, −3k gap | 0.103                  |
| Jumps: 4% prob, −4k gap | 0.063                  |


Adverse gaps make the trailing floor and daily limit easier to hit, so they only
worsen the buyer's odds. The double-barrier structure and the $\sigma$-cap
mechanism are robust to the diffusion assumption.

---

## 7. The Economics: Is It a Fair Bet?

### 7.1 The prize is a capped second barrier option

Passing delivers not cash but a **funded account**, which is *itself* a trailing
double-barrier game: keep P&L above a trailing −3,000 floor while skimming
profits, of which the trader keeps a high split $s\approx 90$ — but subject to
**payout caps** (the card's "DOUBLE Payout Caps") and minimum-period rules. We
model the funded phase honestly (`simulate_funded_extraction`): trailing floor
locking at break-even, profit skimmed every 10 days up to a per-period cap,
terminal death on the floor. The expected cash extracted, $\Pi(\mu,\sigma)$, is
the true prize — and it is **bounded**, because the caps truncate exactly the
high-variance upside a lucky funded trader would otherwise realise.

### 7.2 Net expected value and the break-even edge

With recurring fee $F = 129$,

$$
\mathrm{EV} = P_{\text{pass}}(\mu,\sigma,T)\cdot\Pi(\mu,\sigma) - F .
$$

Net EV versus genuine win rate

*Figure 9. Left: net expected value of the evaluation versus the trader's genuine
win rate (1:1 reward-to-risk, 350 risk/trade). Right: the two components —
challenge pass probability and capped funded extraction.*

The break-even win rate is **≈ 50.1%** — i.e. you need a **genuinely positive
edge** (above a coin flip) merely to break even, *before* counting trading costs.
At an honest coin flip ($p = 0.5$) the pass probability is **8.7%** and the
journey is **negative-EV** (≈ −21 per cycle on our generous, high-split,
no-cost funded model). This is the decisive contrast with the trader-favourable
+8%/−10% equity contract: **the 2:1 futures geometry makes the evaluation a
negative-expected-value purchase for every trader without a real, positive
edge** — and once commissions, exchange fees, and slippage push $\mu$ below zero,
the negativity deepens and the break-even edge rises further.

### 7.3 The real mechanism: the volatility filter

The $\sigma$-cap imposed by the trailing/daily/consistency rules has a wildly
asymmetric effect by edge. Comparing each trader's *best attainable* pass
probability (taking as much variance as they like) against the pass probability
the rules *permit* ($\sigma \approx 600$/day):

The volatility cap as an edge filter

*Figure 10. Best-attainable versus rule-permitted pass probability, by edge. The
relative haircut shrinks monotonically as edge grows.*


| Trader                          | Best attainable $P_{\text{pass}}$ | Under the rules ($\sigma{=}600$) | Haircut  |
| ------------------------------- | --------------------------------- | -------------------------------- | -------- |
| Negative edge (−60/day)         | 0.312                             | 0.022                            | **−93%** |
| Zero edge                       | 0.333                             | 0.067                            | **−80%** |
| Mild positive edge (+60/day)    | 0.378                             | 0.164                            | **−57%** |
| Strong positive edge (+150/day) | 0.504                             | 0.423                            | **−16%** |


This is the thesis quantified. **The volatility cap is the product.** It is not
risk management for the trader; it is an *edge filter* whose relative bite falls
monotonically with edge — from −93% for the negative-edge trader to just **−16%**
for a strongly skilled one, who barely needs variance because drift carries them.
The rule set removes the variance pathway to the target, which is the *only*
pathway available to the edgeless majority. With variance gone, Theta (Section
4.1) drives their pass probability toward zero as the month ends.

### 7.4 Where the house edge comes from

The futures evaluation's economics rest on three stacked layers, none of which is
a hidden fee markup:

1. **Geometry.** 2:1 caps the edgeless ceiling at 1/3 before anything else
  happens (Section 3.3).
2. **Theta.** The 30-day clock cuts the realistic pass probability to 1 in 10
  (Sections 3.5, 4.1).
3. **The $\sigma$-cap.** The "Responsible Trading" rules remove the variance
  pathway, filtering out precisely the edgeless majority (Section 7.3), while a
   capped, split-reduced funded payoff bounds the firm's liability on the rare
   winner (Section 7.1).

The marketing sells the **target** ("hit +6,000"); the rules enforce the
**unstated requirement** ("possess a genuine edge, and never use the variance
that could substitute for one"); the recurring fee is collected from everyone in
between. A secondary, well-documented channel — internalising (B-booking)
simulated order flow — means failed and timed-out accounts are pure fee revenue,
but our model needs no such assumption: the geometry, Theta, and the
$\sigma$-filter already explain the edge.

---

## 8. Discussion and Limitations

- **Fixed-size discipline flatters the trader.** Our trader never over-sizes
beyond a constant $k$, never revenge-trades, and has stationary $(p,R)$. Real
evaluation buyers do all three, inflating realised $\sigma$ into the bust
region and depressing $\mu$ — pushing outcomes further in the firm's favour
than shown. Our pass probabilities are *upper bounds* for the typical buyer.
- **Costs are excluded from the headline.** We set per-trade cost to zero in the
EV; real futures fees/slippage make $\mu < 0$ and raise the break-even edge.
- **Arithmetic BM is an idealisation.** Streaks, fat tails, and intraday
seasonality matter; the jump-diffusion check (Section 6.4) addresses tails and
only strengthens the conclusion.
- **The funded value $\Pi$ is model-dependent** and was chosen generously to the
trader (90% split, modest caps). Tighter caps, minimum-day rules, or
discretionary payout frictions lower $\Pi$ and push break-even higher.
- **General, not firm-specific.** The supplied card is a concrete instance; the
framework is parametric. Substitute any firm's $U$, $|D|$, daily limit,
consistency, $T$, split, and caps in `code/`.

None of these caveats touches the **structural identity** (Section 2), the
**2:1 geometry ceiling** (Section 3.3), or the **volatility-filter result**
(Section 7.3) — the load-bearing claims.

---

## 9. Conclusion

A futures proprietary-trading evaluation is, exactly, a **discrete double-barrier
one-touch binary option** on cumulative P&L: the target is an upper knock-in
digital, the max-loss "One Rule" is a trailing lower knock-out, and the monthly
cycle is maturity. We priced it in closed form, validated by Monte-Carlo, and
computed its Greeks.

The design is laid bare by three numbers. **Geometry:** the 2:1 reward-to-risk
caps an edgeless trader's lifetime pass probability at **1/3**. **Theta:** the
30-day clock cuts the realistic figure to **1 in 10**, and to essentially zero
in the final days. **The $\sigma$-cap:** the trailing drawdown, 2,000 daily
limit, and 50% consistency rule — the "Responsible Trading Advantage" — strip
**80–93%** of the pass probability from zero/negative-edge traders while costing
a skilled trader little, because variance is the edgeless trader's only path to a
target set twice as far as ruin.

Unlike a trader-favourable equity-percentage contract, the futures 2:1 structure
is **negative expected value for anyone without a genuine, positive edge** — the
break-even win rate is at the coin flip before costs and above it after. The
prudent, low-volatility "consistent" trader the rules claim to reward is, in our
simulation, the *least* likely to pass (1.9%).

**The consistency rule and daily drawdown limit are not there to protect the
trader's capital. They are there to cap the trader's $\sigma$, so that — against
a target set twice as far as the bust level — Theta kills the option before it
can ever touch the upper barrier.** That is the predation: not in the premium,
but in the geometry and the Greeks.

---

## Appendix A. Derivation of the finite-horizon pass probability

Solve $\partial_t p = -\mu\partial_y p + \tfrac12\sigma^2\partial_{yy}p$ on
$(0,L)$ with $p(0,t)=p(L,t)=0$, $p(y,0)=\delta(y-y_0)$. The $h$-transform
$p = e^{\frac{\mu}{\sigma^2}(y-y_0)-\frac{\mu^2}{2\sigma^2}t}q$ reduces this to the
heat equation $\partial_t q = \tfrac12\sigma^2\partial_{yy}q$ with the same
Dirichlet data, whose eigenexpansion is $q=\tfrac2L\sum_n \sin\frac{n\pi
y_0}{L}\sin\frac{n\pi y}{L}e^{-\frac12\sigma^2\lambda_n t}$, $\lambda_n =
(n\pi/L)^2$. The upper-boundary flux $J_U = -\tfrac12\sigma^2\partial_y
p|_{y=L}$ (using $\partial_y\sin\frac{n\pi y}{L}|*L = \frac{n\pi}{L}(-1)^n$) and
$\int_0^T e^{-\beta_n t}dt = (1-e^{-\beta_n T})/\beta_n$ give the boxed formula of
Section 3.4. The lower-boundary flux gives $P*{\text{bust}}$. $\square$

## Appendix B. Numerical methods and reproducibility

- **Closed form:** series truncated at $N=400$–$600$; convergence verified by the
$T\to\infty$ identity $P_{\text{pass}}+P_{\text{bust}}\to 1$ and the
scale-function limit $\to 1/3$.
- **Greeks:** central finite differences on the closed form.
- **Monte-Carlo:** vectorised NumPy, fixed seeds, up to 100k paths/config,
discrete (per-trade) barrier monitoring; trailing DD, daily limit, and
consistency enforced as in the contract.
- **Stability:** `prob_pass_safe` falls back to arithmetic-BM Monte-Carlo in the
rare unstable corner.

```bash
cd code
python3 sanity_check.py      # analytic vs Monte-Carlo validation
python3 run_simulation.py    # all figures + data/results.json
python3 build_html.py        # self-contained whitepaper.html
```

## Appendix C. Canonical contract parameters


| Symbol | Meaning                        | Value                 |
| ------ | ------------------------------ | --------------------- |
| $S_0$  | buying power / notional        | 100,000               |
| $U$    | profit target (upper barrier)  | +6,000                |
| $D$    | max loss "One Rule" (trailing) | −3,000                |
| $U/    | D                              | $                     |
| $d$    | daily loss limit               | 2,000 (0.66× $        |
| $c$    | consistency target             | 50%                   |
| $T$    | cycle length                   | 30 days               |
| $n$    | trades per day                 | 4                     |
| $s$    | profit split                   | ~90% (capped payouts) |
| $F$    | monthly fee / reset            | 129                   |


*All figures and numbers are generated by the accompanying code and are
reproducible from a clean checkout. Analysis is general to futures evaluations;
the card is a canonical instance.*