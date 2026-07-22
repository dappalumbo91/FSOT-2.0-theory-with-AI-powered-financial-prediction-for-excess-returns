# FSOT Market Monitor

**Full-stack financial monitoring, intelligent Monte Carlo, Buy/Hold/Sell paper trading, and forward prediction journals** powered by **Fluid Spacetime Omni-Theory (FSOT 2.1)**.

| | |
|--|--|
| **Repo** | [github.com/dappalumbo91/FSOT-2.0-theory-with-AI-powered-financial-prediction-for-excess-returns](https://github.com/dappalumbo91/FSOT-2.0-theory-with-AI-powered-financial-prediction-for-excess-returns) |
| **Theory** | [FSOT-2.1-Lean](https://github.com/dappalumbo91/FSOT-2.1-Lean) |
| **Methodology** | [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) — full scientific method |
| **Reproducibility** | [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) — clone → metrics end-to-end |
| **Free parameters** | **0** (seeds π, e, φ, γ, Catalan + preregistered folds only) |
| **Author** | Damian Arthur Palumbo |

Monitors **indices, equities, and multi-crypto** on **real market data** (Yahoo / CoinGecko), with **synthetic USD** paper P&amp;L so you can evaluate the model **before** real capital. Robinhood crypto API is **wired in dry-run** only (live trading off by default).

| Layer | Stack |
|-------|--------|
| Backend | FastAPI · Python 3.11+ · yfinance · CoinGecko · httpx |
| FSOT math | `backend/app/fsot/` (pinned seeds + domain folds) |
| Frontend | Next.js 14 · React 18 · Tailwind · Lightweight Charts |
| Paper $ | BHS multi-gate · Kelly \(f^*=1/e\) · adjustable capital |
| Broker | Robinhood Crypto adapter · **dry-run default** |

---

## Current research metrics (honest)

These are **causal historical** evaluations (not live guarantees). Target band for selective commits: **70–80%**. Overall markets remain near Hurst ≈ 0.5; the edge is in **when to commit** (HOLD is the default).

### Buy / Hold / Sell (`mode=bhs`, $10k paper, ~2–10y bars, hold=5d)

| Symbol | Commit accuracy | Paper P&amp;L | Trades | % HOLD | Notes |
|--------|-----------------|--------------|--------|--------|--------|
| **IWM** | **~80%** | ~+$200 | ~10 | ~98% | In 70–80% band |
| **MSFT** | **~68%** | ~+$590 | ~38 | ~92% | Near 70% |
| NVDA | ~62% | mixed | ~21 | ~96% | High B&amp;H baseline |
| BTC | ~61% | ~+$640 | ~18 | ~96% | Crypto path |
| AAPL | ~60% | ~+$1.1k | ~78 | ~84% | More trades |
| SPY | ~60% | ~+$180 | ~57 | ~88% | Index |
| QQQ | ~54% | mixed | ~37 | ~92% | Weaker |
| **Mean (min 8 commits)** | **~63–64%** | **positive mean P&amp;L** | — | **~92%+ cash** | Gap to 70% ≈ **6 pts** |

```powershell
cd backend
.\.venv\Scripts\python scripts\eval_bhs_target.py
# → D:\training data\FSOT-Market-History\verification\bhs_target_eval.json
```

### Monte Carlo intelligence (pattern solidify + observer collapse)

| Metric | Typical range (research runs) |
|--------|-------------------------------|
| Solidify threshold | `0.5 + Poof` ≈ **0.653** |
| Consciousness \(C\) | ≈ **0.288** |
| Patterns solidified per multi-year train | often **10–20+** per liquid ticker |
| Gated / confident subset accuracy | can exceed raw always-in (~50%) |
| Always-in raw directional | ≈ **50%** (expected) |

```powershell
.\.venv\Scripts\python scripts\smoke_dynamic_mc.py
.\.venv\Scripts\python scripts\eval_longterm_mc_intelligence.py --quick
```

### Design target

| Goal | Status |
|------|--------|
| Zero free parameters | Met |
| Synthetic $ before live | Met |
| Commit accuracy 70–80% (selective) | **In progress** — IWM in band; mean ~64% |
| Forward (true future) journal | Wired — fill via UI/API over time |
| Live broker | Dry-run only until you opt in |

---

## Requirements

### System

- **Windows / macOS / Linux**
- **Python 3.11+**
- **Node.js 18+** (frontend)
- Optional: long history on `D:\training data\FSOT-Market-History` (or any path after download)

### Backend (`backend/requirements.txt`)

```
fastapi>=0.115.0
uvicorn[standard]>=0.32.0
pydantic>=2.9.0
pydantic-settings>=2.6.0
mpmath>=1.3.0
numpy>=1.26.0
pandas>=2.2.0
yfinance>=0.2.40
requests>=2.32.0
pyyaml>=6.0.2
httpx>=0.27.0
python-multipart>=0.0.12
pytest>=8.3.0
pytest-asyncio>=0.24.0
cryptography>=43.0.0
```

### Frontend (`frontend/package.json`)

- next ^14 · react ^18 · tailwindcss · lightweight-charts · typescript

---

## How to use

### 1. Clone & backend

```powershell
git clone https://github.com/dappalumbo91/FSOT-2.0-theory-with-AI-powered-financial-prediction-for-excess-returns.git
cd FSOT-2.0-theory-with-AI-powered-financial-prediction-for-excess-returns

cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Windows
# source .venv/bin/activate           # macOS/Linux
pip install -r requirements.txt
python -m pytest tests/ -q
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

- API docs: http://127.0.0.1:8000/docs  
- Health: http://127.0.0.1:8000/api/health  

Or: `.\start-backend.ps1` from repo root.

### 2. Frontend

```powershell
cd frontend
npm install
npm run dev
```

Open **http://localhost:3000**  
Optional: `NEXT_PUBLIC_API_URL=http://127.0.0.1:8000`

Or: `.\start-frontend.ps1`

### 3. Dashboard workflow

1. Pick **indices / stocks / crypto** in the left watchlist.  
2. Read **price chart**, **FSOT signal**, **Monte Carlo** (right).  
3. **Synthetic $** panel: set capital ($1k–$100k+), mode **BHS (rec.)**, compare P&amp;L vs buy&amp;hold.  
4. **Forward journal**: **Record** a forecast today → **Resolve** after the horizon (true future test).  
5. **Auto** poll: Off / 30s / **1m** (default) / 2m / 5m / 15m — pauses when the tab is hidden.  
6. Broker strip shows **dry_run=true** (no real money).

### 4. Useful API calls

```http
GET  /api/health
GET  /api/predict/BTC?range=1y
GET  /api/predict/BTC/montecarlo?horizon=21&n_paths=256&dynamic=true
GET  /api/paper/BTC?capital=10000&mode=bhs&range=2y
GET  /api/broker/status
POST /api/broker/preview
POST /api/monitor/forward/record?symbol=BTC&horizon=5
POST /api/monitor/forward/resolve
GET  /api/monitor/forward
```

### 5. Research scripts

```powershell
cd backend
.\.venv\Scripts\python scripts\verify_fsot_pin.py
.\.venv\Scripts\python scripts\smoke_intrinsic.py
.\.venv\Scripts\python scripts\smoke_dynamic_mc.py
.\.venv\Scripts\python scripts\eval_bhs_target.py
.\.venv\Scripts\python scripts\eval_longterm_mc_intelligence.py --quick
```

### 6. Optional history download

```powershell
.\.venv\Scripts\python scripts\download_history.py
```

Default archive layout:

```
D:\training data\FSOT-Market-History\
  ohlcv\          # daily bars
  patterns\       # MC pattern ledgers
  verification\   # eval JSON outputs
  monitor\        # forward journal
```

### 7. Robinhood crypto (later only)

```powershell
# copy backend/.env.example → backend/.env  (never commit secrets)
# FSOT_RH_API_KEY=...
# FSOT_RH_PRIVATE_KEY_PATH=...

# Live trading requires ALL THREE (default: dry-run forever):
# FSOT_BROKER_LIVE=1
# FSOT_BROKER_I_UNDERSTAND=YES
# FSOT_BROKER_DRY_RUN=0
```

Credentials: [robinhood.com/account/crypto](https://robinhood.com/account/crypto) (desktop). **Stocks are not automated via this API** — crypto only.

---

## Theory pin

```
S = K · (T1 + T2 + T3)
```

| Item | Value |
|------|--------|
| Seeds | π, e, φ, γ, Catalan |
| Economics | D_eff=20, hits=3, δψ=1.5 → S≈0.646 |
| Finance_Markets | D_eff=19, hits=2, δψ=0.75 → S≈0.571 |
| Method | `fsot_intrinsic_zero_free` / `fsot_bhs_buy_hold_sell` / `fsot_dynamic_monte_carlo_pattern_collapse` |
| Forbidden | Free LSQ, invented RSI coefficients, per-ticker D_eff fit |

### Monte Carlo + pattern intelligence

1. Causal train → discrete FSOT signatures  
2. φ-EWMA accuracy; solidify when `acc_φ ≥ 0.5 + Poof`  
3. Collapse TRUE → FSOT μ; FALSE → Poof chaos  
4. BHS: HOLD unless dual-scale agree + solid quality + edge/field  

### Data sources (what is real vs synthetic)

| Data | Source |
|------|--------|
| OHLCV / quotes | **Real** — Yahoo Finance, CoinGecko (optional local CSV archive) |
| News observer | **Real** RSS + lexicon sentiment |
| Paper P&amp;L $ | **Synthetic** dollars on real bars |
| Broker orders | **Dry-run** unless you enable live flags |

---

## API surface

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Liveness + FSOT pin + Economics S₀ |
| GET | `/api/watchlist` | Indices / stocks / crypto |
| GET | `/api/market/{symbol}/ohlcv` | Candles |
| GET | `/api/market/{symbol}/quote` | Last price |
| GET | `/api/predict/{symbol}` | FSOT forecast + telemetry |
| GET | `/api/predict/{symbol}/montecarlo` | Dynamic MC multipath |
| GET | `/api/predict/batch` | Watchlist signals |
| GET | `/api/backtest/{symbol}` | Walk-forward metrics |
| GET | `/api/paper/{symbol}` | Synthetic USD portfolio |
| GET | `/api/broker/status` | Robinhood wire (dry-run) |
| POST | `/api/broker/preview` | Dry-run order preview |
| GET | `/api/monitor/forward` | Forward journal summary |
| POST | `/api/monitor/forward/record` | Record forecast now |
| POST | `/api/monitor/forward/resolve` | Score when future exists |
| GET | `/api/news` | Headlines |
| GET | `/api/news/observer` | Observer mod |

---

## Watchlist

Edit `config/watchlist.yaml`. Defaults include major indices, mega-cap equities, and multi-crypto (BTC, ETH, SOL, …).

---

## Project layout

```
backend/
  app/fsot/          # intrinsic, monte_carlo, pattern_memory, bhs_engine
  app/broker/        # robinhood_crypto dry-run
  app/monitor/       # forward journal
  app/api/           # FastAPI routes
  scripts/           # smoke + eval
  tests/
frontend/            # Next.js dashboard
config/watchlist.yaml
```

Standalone MC-only sandbox (optional side project on Desktop):  
`C:\Users\damia\Desktop\FSOT-Monte-Carlo-Intelligence` — not required to run this repo.

---

## Lineage

- Physical archive: `I:\FSOT-Physical-Archive`  
- Prior experiments: `backend/legacy_ref/`  
- Theory: [FSOT-2.1-Lean](https://github.com/dappalumbo91/FSOT-2.1-Lean)  

---

## Disclaimer

**Research / monitoring only — not financial advice.**  
Synthetic paper P&amp;L and historical commit rates **do not** guarantee future results. Live trading is disabled by default. Cryptocurrency and equity markets involve substantial risk of loss. Crypto held at brokers is typically not SIPC/FDIC insured.
