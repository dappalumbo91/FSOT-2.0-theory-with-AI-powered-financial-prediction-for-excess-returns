# FSOT Market Monitor — Methodology

**Version:** research snapshot 2026-07-22  
**Repository:** https://github.com/dappalumbo91/FSOT-2.0-theory-with-AI-powered-financial-prediction-for-excess-returns  
**Theory authority:** https://github.com/dappalumbo91/FSOT-2.1-Lean · pin `backend/app/fsot/PIN.json`  
**Free parameters:** **0**

This document specifies *what the system computes*, *why each step is allowed under FSOT*, and *how results are evaluated*. It is the scientific record for replication. Operational clone-and-run steps live in [`REPRODUCIBILITY.md`](./REPRODUCIBILITY.md).

---

## 1. Scope and claims

### 1.1 In scope

| Component | Claim type |
|-----------|------------|
| FSOT scalar \(S = K(T_1+T_2+T_3)\) | Closed-form from seeds + preregistered domain folds |
| Market mapping | Measured returns/volume/sentiment enter **observer path** only (e.g. via \(\mathrm{atan}\)), not free regression |
| Directional μ forecast | Causal function of past bars only |
| Monte Carlo multipath | Ensemble of futures under observer collapse TRUE/FALSE |
| Pattern memory | Online solidify/soften of discrete FSOT signatures |
| Buy/Hold/Sell (BHS) | Selective commits; HOLD default |
| Synthetic paper portfolio | Theoretical USD P&amp;L on **real** OHLCV |
| Forward journal | Score predictions after **future** bars exist |

### 1.2 Explicit non-claims

- **Not** financial advice; **not** a guarantee of profit.  
- **Not** always-in 70–80% accuracy on every bar (markets ≈ Hurst 0.5).  
- Target **70–80%** is for **selective commits** (when gates fire), not every calendar day.  
- Paper P&amp;L is **synthetic**; broker module is **dry-run** unless dual live flags are set.  
- Historical walk-forwards are **not** the same as live forward journal scores.

### 1.3 Forbidden practices (pin)

From `PIN.json`:

- Invented RSI/volume free coefficients  
- Per-ticker \(D_{\mathrm{eff}}\) fit  
- \(S \times\) multiplier \(+\) base hacks  

---

## 2. Mathematical foundations (FSOT 2.1)

### 2.1 Seeds (Layer 0)

All derived constants come from:

\[
\pi,\; e,\; \varphi = \frac{1+\sqrt{5}}{2},\; \gamma\ (\text{Euler–Mascheroni}),\; G\ (\text{Catalan})
\]

Implementation: `app/fsot/fast.py` (float64 hot path), `app/fsot/compute_authority.py` (mpmath pin).

### 2.2 Canonical scalar

\[
S = K \cdot (T_1 + T_2 + T_3)
\]

with \(K\) and terms from the archive engine under preregistered **domain folds** (not fitted per asset).

### 2.3 Representative derived constants (runtime float64)

| Symbol | Role | Approx. value |
|--------|------|----------------|
| \(K\) | Scalar prefactor | 0.420222 |
| \(C\) (consciousness) | Observer / quirk strength | 0.287600 |
| \(\gamma\) | Vol persistence β **and** Sharpe-class scale | 0.577216 |
| Poof | Decoherence residual floor | 0.153482 |
| \(\varphi\) | Fractal / EWMA / Fib ladder | 1.618034 |
| Kelly \(f^*\) | \(1/e\) position cap | 0.367879 |
| Solidify accuracy | \(0.5 + \mathrm{Poof}\) | 0.653482 |

Spine scalars (preregistered folds, `spine_scalars()`):

| Domain | \(D_{\mathrm{eff}}\) | hits | \(\delta\psi\) | \(S\) (approx.) |
|--------|----------------------|------|----------------|-----------------|
| Economics | 20 | 3 | 1.5 | 0.646005 |
| Finance_Markets | 19 | 2 | 0.75 | 0.570566 |
| Finance_Markets_Panel | 19 | 2 | 0.65 | 0.455662 |
| Econometrics | 19 | 2 | 0.7 | 0.509919 |
| Econophysics | (spine) | — | — | 0.646005 |

Domain factor for coupled telemetry: `DOMAIN_FACTOR_ECONOMICS = 0.0004` (pin).

### 2.4 Lean observer / consciousness

\[
\mathrm{quirk\_mod}(\delta\psi) = \exp(C \cdot P_{\mathrm{var}}) \cdot \cos(\delta\psi + P_{\mathrm{var}})
\quad (\text{observed}=\mathrm{True})
\]

Growth (engine):

\[
\mathrm{growth} = \exp\bigl(\alpha \cdot (1 - \mathrm{hits}/N) \cdot \gamma/\varphi\bigr)
\]

Vol persistence (econophysics):

\[
\sigma^2_{t+1} = \gamma\,\sigma^2_t + (1-\gamma)\,r_t^2
\]

---

## 3. Market observation mapping

**Code:** `app/fsot/intrinsic.py` → `market_observer_state`, `evaluate_market_bar`.

### 3.1 Inputs (measurement only)

- Daily OHLCV closes → returns \(r_t\)  
- Optional volume → relative intensity in \(P\)  
- Optional news `observer_mod` \(\in [-1,1]\) → sentiment  

### 3.2 Observer phase

Base fold \(\delta\psi_0\) from scale route (Fib window → Economics / Finance_Markets / …).

\[
\begin{aligned}
\mathrm{fluc} &= \frac{\mu_w}{\sigma_w \gamma}, \\
\mathrm{shock} &= \frac{r_t}{\sigma_t \gamma}, \\
\delta\psi &= \delta\psi_0 + \mathrm{atan}(\mathrm{fluc}) + \mathrm{atan}(\mathrm{sentiment}) + \mathrm{atan}(\mathrm{shock})
\end{aligned}
\]

Live scalar uses measured \(N\), hits, \(\rho\), amplitude, trend_bias with **seed-only** couplings (\(\mathrm{atan}\), \(\gamma\), \(\varphi\)).

### 3.3 Directional field → return units

\[
\begin{aligned}
d_{\mathrm{obs}} &= S_{\mathrm{live}} - S_{\mathrm{base}}, \\
\mathrm{field} &= K \cdot C \cdot \mathrm{growth} \cdot d_{\mathrm{obs}}, \\
\mu_{\mathrm{engine}} &= \tanh(\mathrm{field}/C) \cdot \sigma_{\mathrm{pred}} \cdot \gamma, \\
\mu &= (1 - 1/\varphi)\,\mu_{\mathrm{engine}} + (1/\varphi)\,\mu_{\varphi\text{-EWMA}}
\end{aligned}
\]

Poof extreme: if \(|r_t|\) is large vs \(\sigma_t\), μ is flipped/faded (decoherence).

Signal: \(\mathrm{sign}(\mu)\). Size: \(\min(1/e,\, |\mu|/\sigma \cdot 1/e)\).

### 3.4 Dual-scale (As Above So Below)

Windows Fib **8** (fast) and **55** (slow) plus mid window (default Fib **21**).

- \(\mu_{\mathrm{cross}}\) from \(S_{\mathrm{fast}} - S_{\mathrm{slow}}\)  
- Agreement: same sign of cross vs mid → boost by \((1+C)\); conflict → damp by Poof  
- Used as a **gate** in BHS (require `scale_agree > 0`)

---

## 4. Pattern memory (intelligence layer)

**Code:** `app/fsot/pattern_memory.py`

### 4.1 Signature

Discrete fingerprint of FSOT state (not free clustering), e.g.:

```text
D{D_eff}|mu{±1}|d{±1}|f{±1}|s{±1}|q{±1}|p{0|1}|a{scale_agree}|i{low|mid|high}
```

Intensity bins use seed \(\gamma\) and \(C\), not sample quantiles as free parameters.

### 4.2 Online learning

For each causal bar \(j\) with prediction direction \(d_{\mathrm{pred}}\) and realized direction \(d_{\mathrm{real}}\) (horizon-aligned for BHS):

- Hit indicator \(h \in \{0,1\}\) (or partial on FLAT)  
- \(\mathrm{acc}_\varphi \leftarrow \frac{1}{\varphi} h + (1-\frac{1}{\varphi})\,\mathrm{acc}_\varphi\)  

### 4.3 Solidify / soften (seeds only)

| Rule | Definition |
|------|------------|
| Solidify accuracy | \(\mathrm{acc}_\varphi \ge 0.5 + \mathrm{Poof}\) (≈ 0.653) |
| Min trials | Fib **8** base; strict/BHS: Fib **13** full or exceptional early solidify if \(\mathrm{acc}_\varphi \ge 0.5 + \varphi\cdot\mathrm{Poof}\) |
| Soften | \(\mathrm{acc}_\varphi\) falls below soften bar + consecutive soft fails |
| Strength | \(C \cdot (\mathrm{acc}_\varphi - 0.5) / (0.5 - \mathrm{Poof})\) clipped to \([0,1]\) |
| Preferred direction | Sign of realized hit mass (measurement) |

**Quality flag:** solid **and** raw accuracy **and** \(\mathrm{acc}_\varphi\) still clear solidify bar → eligible for commit bias.

Ledgers (optional persist):  
`D:\training data\FSOT-Market-History\patterns\{SYM}_mc_pattern_ledger.json`

---

## 5. Monte Carlo multipath (observer collapse)

**Code:** `app/fsot/monte_carlo.py`

### 5.1 Ontology

Many futures exist as fluid possibilities until observation couples.

Each simulated day \(t\):

1. Rebuild \(\delta\psi\) from path buffer (fluctuation + sentiment mean-reversion with \(\varphi\)).  
2. Collapse probability:

\[
p = \mathrm{logistic}\bigl(C\cdot\varphi\cdot|\delta\psi-\delta\psi_0| + C\cdot\mathrm{atan}(\mathrm{sentiment})\bigr)
\]

clipped to \([\mathrm{Poof},\, 1-\mathrm{Poof}]\).

3. **TRUE:** \(r = \mu_t + \sigma_t Z\) with \(Z\) from historical residual bootstrap.  
4. **FALSE:** \(r = -\mu_t\cdot\mathrm{Poof} + \sigma_t(1+\mathrm{Poof}) Z\) (chaotic branch).  
5. Update variance with \(\gamma\)-GARCH (vol persistence = \(\gamma\)).

Horizon snapped to Fib ladder: \(\{5,8,13,21,34,55\}\). Default horizon **21**, default paths **512**.

### 5.2 Dynamic / intelligent MC

1. `train_pattern_memory` on causal history.  
2. `run_fsot_monte_carlo(..., dynamic=True, memory=...)` biases:

   - μ scale and direction toward solidified preferred dir  
   - collapse probability by historical observed-branch success  
   - path weights; optional pattern-conditional residuals  

3. Ensemble outputs: \(P(\mathrm{up})\), quantiles, mode bin, fan chart, mean collapse-true fraction.  
4. Commit signal: LONG/SHORT only if observed-branch \(P(\mathrm{up})\) clears \(0.5\pm\mathrm{Poof}\); dynamic mode may FLAT if unsolidified.

### 5.3 Causal evaluation (`mc_walkforward_hit`)

At bar \(i\): grow memory on past only → run MC → compare direction to realized return over horizon. Metrics: directional accuracy, confident subset (\(|p-0.5|>\mathrm{Poof}\)), solid-only accuracy, early vs late lift.

---

## 6. Buy / Hold / Sell (BHS)

**Code:** `app/fsot/bhs_engine.py`, portfolio wrapper `app/backtest/paper_portfolio.py`

### 6.1 Philosophy

- **HOLD** is the default.  
- Target **70–80%** accuracy **conditional on commit**, not always-in.  
- Learn and trade on the **same Fib horizon** (default **H = 5** trading days).

### 6.2 Training filter

Only update patterns when:

- dual-scale agree, and  
- \(|r_{t:t+H}| \ge \mathrm{Poof}\cdot\sigma\cdot\sqrt{H}\) (material move; ignore noise days).

### 6.3 Commit gates (all must pass)

1. Pattern **solidified**  
2. `scale_agree > 0`  
3. Field \(|K C \,\mathrm{growth}\, d_{\mathrm{obs}}| \ge C^2\) **or** edge \(|\mu|/\sigma \ge \gamma\cdot\mathrm{Poof}\)  
4. Strength above \(C\cdot\mathrm{Poof}^2\) floor  
5. Mature decayed solids blocked if trials ≥ 13 and raw acc slipped  
6. φ-momentum of recent returns must not **strongly** oppose preferred direction  
7. Live \(\mathrm{sign}(\mu)\) must not conflict with preferred_dir  

Actions: **BUY** / **SELL** / **HOLD**. Long-only mode maps SELL → HOLD.

### 6.4 Position sizing (paper)

\[
\mathrm{size} = \min\bigl(1/e,\; (|\mu|/\sigma)\cdot(1/e)\bigr)
\]

boosted by \((1 + C\cdot\mathrm{strength})\) when solid. Non-overlapping holds of length \(H\) (no stacked leverage).

### 6.5 Metrics

| Metric | Definition |
|--------|------------|
| Commit directional accuracy | Among BUY/SELL only: correct sign of \(r_{t:t+H}\) |
| % HOLD | Fraction of decision dates with no position |
| Paper P&amp;L | Synthetic USD from starting capital |
| Progress to 70–80% | Mapped score of commit accuracy into target band |
| vs buy&amp;hold | Same capital, 100% long path (often higher in strong bulls; BHS is selective) |

---

## 7. Synthetic paper portfolio

**Code:** `app/backtest/paper_portfolio.py` · API `GET /api/paper/{symbol}`

| Mode | Behavior |
|------|----------|
| `bhs` | §6 multi-gate (recommended) |
| `bhs_long_only` | BHS without shorts |
| `solid_gated` | 1d solid pattern, legacy |
| `always_in` | Every non-flat μ |
| `buy_hold` | 100% long benchmark |

**Important:** P&amp;L is theoretical. Prices are real. No broker is contacted.

---

## 8. Forward prediction journal (true future test)

**Code:** `app/monitor/forward_journal.py` · API `/api/monitor/forward/*`

Historical walk-forwards can overstate skill via look-ahead of *which* sample was chosen. The forward journal:

1. **Record** at time \(T\): BHS action, price, horizon \(H\), pattern state.  
2. **Resolve** only when bar \(T+H\) exists in live/archive OHLCV.  
3. Score hit/miss on that realized return.

This is the protocol for claiming *out-of-sample future* skill after sufficient open→resolved mass accumulates.

---

## 9. Data sources and leakage policy

| Stream | Source | Leakage rule |
|--------|--------|----------------|
| OHLCV | Yahoo Finance / CoinGecko; optional local CSV archive | Only bars \(\le t\) for decision at \(t\) |
| Quotes | Same providers (cached TTL) | Display / latest price |
| News observer | Public RSS + lexicon | Same-day sentiment; no future headlines |
| Pattern ledgers | Local JSON | Written causally; loaded only as past state |

**Causal rule:** any quantity used at index \(i\) is a function of \(\{0,\ldots,i\}\) only (plus constants). Forward returns for *scoring* use \(i+H\) but never enter the decision at \(i\).

---

## 10. Broker interface (non-methodological trading)

**Code:** `app/broker/robinhood_crypto.py`

- Default: **dry_run = true**  
- Live requires all of: `FSOT_BROKER_LIVE=1`, `FSOT_BROKER_I_UNDERSTAND=YES`, `FSOT_BROKER_DRY_RUN=0`  
- Crypto API only; stocks not automated via this adapter  

Broker wiring is **infrastructure**, not part of the FSOT accuracy claim until live studies are designed separately.

---

## 11. Evaluation protocols

### 11.1 Unit / pin tests

```text
pytest tests/   # expect 17 passed (as of this snapshot)
```

Includes scalar pin, MC shape, pattern solidify, paper/BHS, broker dry-run, forward journal.

### 11.2 BHS target eval

```text
python scripts/eval_bhs_target.py
```

- Symbols: SPY, QQQ, AAPL, MSFT, NVDA, BTC, ETH, IWM (default)  
- Capital: $10,000  
- Horizon: 5  
- Aggregate mean commit accuracy with min 8 trades filter  

### 11.3 MC intelligence eval

```text
python scripts/eval_longterm_mc_intelligence.py --quick
python scripts/smoke_dynamic_mc.py
```

### 11.4 Reported research snapshot (illustrative)

| Aggregate | Approx. value |
|-----------|----------------|
| Mean BHS commit accuracy (min 8 trades) | ~63–64% |
| Best liquid (IWM) | ~80% |
| Near-target (MSFT) | ~68% |
| Mean paper P&amp;L ($10k, selective) | often slightly positive |
| Mean % HOLD | ~90%+ |
| Gap to 70% mean | ~6 percentage points |

Re-run scripts on your machine/date; numbers move with data end-date and symbol set. Publish outputs under `verification/` when claiming a new snapshot.

---

## 12. Implementation map

| Concern | Path |
|---------|------|
| Seeds / float engine | `backend/app/fsot/fast.py` |
| mpmath authority | `backend/app/fsot/compute_authority.py` |
| Folds / routes | `backend/app/fsot/routes.py` |
| Market μ | `backend/app/fsot/intrinsic.py` |
| Pattern memory | `backend/app/fsot/pattern_memory.py` |
| Monte Carlo | `backend/app/fsot/monte_carlo.py` |
| BHS | `backend/app/fsot/bhs_engine.py` |
| Paper $ | `backend/app/backtest/paper_portfolio.py` |
| Forward journal | `backend/app/monitor/forward_journal.py` |
| Predict API | `backend/app/api/predict.py` |
| Paper API | `backend/app/api/paper.py` |
| Pin metadata | `backend/app/fsot/PIN.json` |

Standalone MC package (optional side sandbox):  
`Desktop/FSOT-Monte-Carlo-Intelligence` — same math, no UI.

---

## 13. Citation / pin

When citing methodology:

1. Repository commit hash (e.g. from `git rev-parse HEAD`)  
2. `PIN.json` `pinned_at` and source path  
3. Exact script command + symbol list + date range of OHLCV  
4. Whether results are **historical walk-forward**, **BHS paper**, or **forward journal**

---

## 14. Disclaimer

Research software. Not an offer to sell securities. Not investment advice. Past synthetic performance is not indicative of future results. Cryptocurrency and equity markets involve substantial risk of loss.
