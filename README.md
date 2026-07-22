# FSOT Market Monitor

**Full-stack financial monitoring, intelligent Monte Carlo, and synthetic-$ paper trading** powered by **Fluid Spacetime Omni-Theory (FSOT 2.1)**.

> Successor app for the original finance experiments in this repository lineage:  
> [FSOT-2.0-theory-with-AI-powered-financial-prediction-for-excess-returns](https://github.com/dappalumbo91/FSOT-2.0-theory-with-AI-powered-financial-prediction-for-excess-returns)

Monitors **S&P 500 / indices, equities, and multi-crypto**, maps market features onto the canonical FSOT scalar engine (zero free parameters), runs **dynamic Monte Carlo with pattern solidification**, and simulates **theoretical P&amp;L in adjustable USD** on real market history before live capital.

| Layer | Stack |
|-------|--------|
| Backend | FastAPI · Python · yfinance · CoinGecko |
| FSOT math | Pinned from `I:\FSOT-Physical-Archive\02_FSOT-2.1-Lean-Full\vendor\fsot_compute.py` (or vendored pin) |
| Frontend | Next.js 14 · Tailwind · TradingView Lightweight Charts |
| Paper $ | Causal walk · Kelly `1/e` sizing · solid-gated commits |

---

## Theory pin

```
S = K · (T1 + T2 + T3)
```

- **Seeds:** π, e, φ, γ (Euler–Mascheroni), G (Catalan) — zero free parameters  
- **Domain:** Economics — `D_eff=20`, `hits=3`, `δψ=1.5`, `observed=True`  
- **Authority:** [FSOT-2.1-Lean](https://github.com/dappalumbo91/FSOT-2.1-Lean) + physical archive  
- **Pin check:** `python scripts/verify_fsot_pin.py` → all 35 domain scalars match mpmath (rel err ~1e-15)

### Live method — intrinsic, zero free parameters

| Item | Value |
|------|--------|
| Authority | `I:\FSOT-Physical-Archive\02_FSOT-2.1-Lean-Full\vendor\fsot_compute.py` |
| Method | `fsot_intrinsic_zero_free` |
| Free parameters | **0** |
| Economics fold | D_eff=20, hits=3, δψ=1.5, observed=True → S≈0.646 |
| Finance_Markets fold | D_eff=19, hits=2, δψ=0.75, observed=True → S≈0.571 |
| Structure | Seed votes (φ multi-scale, φ fractal, γ-strength) on Fib windows |
| Forbidden | Invented RSI/volume coefficients; per-ticker D_eff fit; S×multiplier+base |

```
S = K · (T1 + T2 + T3)   # preregistered fold only
score = mean(seed structure votes on measured returns)
signal = sign(score)
```

### Formal verification

Market app **pins** archive `fsot_compute.py` (Economics scalar matches mpmath). Full Lean/Coq/Isabelle/F*/Rust gauntlet stays in `FSOT-2.1-Lean` on the I: hub.

### Historical training data (game drive)

```
D:\training data\FSOT-Market-History\
  ohlcv\          # ~20y daily bars (2005→now; crypto from first Yahoo date)
  patterns\       # emergence series + strategy_ledger_v2.json
  verification\   # fsot_pin_verification.json
  manifests\      # download_manifest.json
  news\           # optional dumps
```

```powershell
cd backend
.\.venv\Scripts\python scripts\download_history.py
.\.venv\Scripts\python scripts\build_historical_patterns.py
.\.venv\Scripts\python scripts\verify_fsot_pin.py
```

### Free news / observer (no credentials)

RSS: Yahoo Finance, CNBC, MarketWatch, CoinDesk, Cointelegraph, Fed press, SEC EDGAR, Google News finance queries.  
Lexicon `observer_mod ∈ [-1,1]` couples into FSOT `δψ` (observer path).  
API: `GET /api/news`, `GET /api/news/observer?symbol=SPY`

### Monte Carlo multipath (intelligent / dynamic pattern collapse)

When single-path μ is near coin-flip, ensemble futures + **pattern recognition** carry structure:

1. **Causal train:** walk history; each bar → discrete FSOT signature (D_eff, μ/d_obs/fluc/sentiment signs, Poof, intensity bin).
2. Score signature vs realized forward return; φ-EWMA accuracy online.
3. **Solidify** when `acc_φ > 0.5 + Poof` and trials ≥ Fib(8)=8; **soften** on repeated misses (Poof decoherence). Ledgers persist under `D:\training data\FSOT-Market-History\patterns\*_mc_pattern_ledger.json`.
4. **Dynamic MC:** each day rebuild `δψ`; `p_collapse` from C×phase×sentiment; solidified anchors bias μ direction, collapse, and pattern-conditional bootstrap shocks.
5. **Collapse TRUE** → FSOT μ + γ-vol; **FALSE** → Poof chaos. Ensemble → P(up), quantiles, most-probable bin, fan chart.

```powershell
cd backend
.\.venv\Scripts\python scripts\smoke_dynamic_mc.py
# Long-term accuracy + intelligence score (writes D:\...\verification\mc_longterm_intelligence_eval.json)
.\.venv\Scripts\python scripts\eval_longterm_mc_intelligence.py --quick
# Full multi-symbol run (slower):
# .\.venv\Scripts\python scripts\eval_longterm_mc_intelligence.py
# API: GET /api/predict/SPY/montecarlo?horizon=21&n_paths=512&dynamic=true
```

**Intelligence gate:** only **solidified** signatures may commit LONG/SHORT; unsolidified states stay **FLAT** (fluid possibilities). That gate is the core you can later extract as a standalone pattern-intelligence system (`pattern_memory.py` + solidify/soften + commit gate).

---

## Quick start

### 1. Backend

```powershell
cd C:\Users\damia\Desktop\FSOT-Market-Monitor\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

- API docs: http://127.0.0.1:8000/docs  
- Health: http://127.0.0.1:8000/api/health  

### 2. Frontend

```powershell
cd C:\Users\damia\Desktop\FSOT-Market-Monitor\frontend
npm install
npm run dev
```

Open http://localhost:3000  

Optional: `NEXT_PUBLIC_API_URL=http://127.0.0.1:8000`

### 3. Tests

```powershell
cd backend
.\.venv\Scripts\python -m pytest tests/ -q
```

---

## API surface

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Liveness + FSOT pin + Economics S₀ |
| GET | `/api/watchlist` | Indices / stocks / crypto from config |
| GET | `/api/market/{symbol}/ohlcv` | Candles |
| GET | `/api/market/{symbol}/quote` | Last price snapshot |
| GET | `/api/predict/{symbol}` | FSOT forecast + telemetry (`include_monte_carlo=true` optional) |
| GET | `/api/predict/{symbol}/montecarlo` | FSOT Monte Carlo multipath + observer collapse |
| GET | `/api/predict/batch` | All watchlist signals |
| GET | `/api/backtest/{symbol}` | Walk-forward metrics |
| GET | `/api/paper/{symbol}` | **Synthetic USD** paper portfolio (`capital`, `mode`, `range`) |

### Synthetic dollar paper portfolio

Theoretical money on **real OHLCV** — adjust starting capital to see P&amp;L and drawdown before live markets:

| Query | Default | Meaning |
|-------|---------|---------|
| `capital` | `10000` | Starting synthetic USD (100 … 1e8) |
| `mode` | `solid_gated` | `always_in` · `solid_gated` · `long_only` · `buy_hold` |
| `range` | `2y` | History window |

```
GET /api/paper/SPY?capital=25000&mode=solid_gated&range=2y
```

Dashboard: **Synthetic $ Portfolio** panel — capital presets, mode toggle, equity curve, ending equity, total P&amp;L $, max DD $, vs buy&hold.

Modes:
- **solid_gated** — only trade when FSOT pattern memory has solidified (intelligence gate)
- **always_in** — every non-flat FSOT μ signal
- **long_only** — solid longs only
- **buy_hold** — benchmark

Sizing: Kelly \(f^*=1/e\) × edge \(|\mu|/\sigma\), strength boost when solid. Causal (no lookahead).

---

## Watchlist

Edit `config/watchlist.yaml` to add/remove assets. Defaults include:

- **Indices:** ^GSPC, SPY, ^IXIC, ^DJI  
- **Stocks:** AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, JPM, JNJ, XOM  
- **Crypto:** BTC, ETH, SOL, BNB, XRP, ADA, DOGE, AVAX, LINK, LTC  

---

## Reproducibility checklist

```powershell
# 1. Clone
git clone https://github.com/dappalumbo91/FSOT-2.0-theory-with-AI-powered-financial-prediction-for-excess-returns.git
cd FSOT-2.0-theory-with-AI-powered-financial-prediction-for-excess-returns

# 2. Backend
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows
pip install -r requirements.txt
python -m pytest tests/ -q
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# 3. Frontend (other terminal)
cd frontend
npm install
npm run dev

# 4. Optional: long-term MC intelligence eval
python scripts/eval_longterm_mc_intelligence.py --quick

# 5. Optional: paper $ smoke
# GET http://127.0.0.1:8000/api/paper/SPY?capital=10000&mode=solid_gated
```

Open http://localhost:3000 · API docs http://127.0.0.1:8000/docs

**Note:** Large 20y OHLCV history is optional (place under `D:\training data\FSOT-Market-History` or download via `scripts/download_history.py`). Live quotes use yfinance / CoinGecko.

## Lineage

- Physical archive: `I:\FSOT-Physical-Archive`  
- This repo: full-stack **FSOT Market Monitor** (2.1 intrinsic + dynamic MC + paper $)  
- Prior scripts retained under `backend/legacy_ref/` for comparison  
- Theory authority: [FSOT-2.1-Lean](https://github.com/dappalumbo91/FSOT-2.1-Lean)  
- Author: Damian Arthur Palumbo  

---

## Disclaimer

This is a **research / monitoring** application. It is **not** financial advice.  
Paper portfolio P&amp;L is **synthetic** (theoretical dollars on historical bars). Walk-forward and Monte Carlo metrics are honest causal estimates — they do not guarantee future performance. Crypto and equity APIs may rate-limit; the app caches OHLCV/quotes.
