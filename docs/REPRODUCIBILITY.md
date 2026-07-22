# FSOT Market Monitor — Complete Reproducibility Instructions

**Goal:** A third party can clone the repo, install dependencies, run tests, start the app, and recompute research metrics without undocumented steps.

**Companion:** scientific specification in [`METHODOLOGY.md`](./METHODOLOGY.md).

---

## 0. Quick checklist

| Step | Command / action | Pass criterion |
|------|------------------|----------------|
| 1 | Clone repo | Files present |
| 2 | Python 3.11+ venv + `pip install -r requirements.txt` | No install errors |
| 3 | `pytest tests/ -q` | **17 passed** (current suite) |
| 4 | `uvicorn app.main:app --host 127.0.0.1 --port 8000` | `/api/health` → `status: ok` |
| 5 | `npm install` + `npm run dev` in `frontend/` | http://localhost:3000 loads |
| 6 | Optional: history download + eval scripts | JSON under `verification/` |
| 7 | Record `git rev-parse HEAD` with any published metrics | Traceable snapshot |

---

## 1. Prerequisites

### 1.1 Required software

| Tool | Version | Notes |
|------|---------|--------|
| Git | 2.x | Clone / commit |
| Python | **3.11+** | 3.11/3.12 tested on Windows |
| Node.js | **18+** | Frontend (includes npm) |
| Network | Yes | yfinance / CoinGecko / RSS on first fetch |

### 1.2 Hardware / OS

- Windows 10/11, macOS, or Linux  
- ~2 GB free disk for code + venv + node_modules  
- Optional: multi-GB for 20y OHLCV archive  

### 1.3 Optional local paths (author machine)

These are **not required** for basic reproduction (live APIs work without them):

```
D:\training data\FSOT-Market-History\     # OHLCV + ledgers + verification dumps
I:\FSOT-Physical-Archive\                # mpmath authority source (vendored pin also in-repo)
```

If `D:\...` is missing, eval scripts that load CSV will fail for those symbols; use live `GET /api/paper/...` instead, or run `scripts/download_history.py`.

---

## 2. Obtain the source

```powershell
git clone https://github.com/dappalumbo91/FSOT-2.0-theory-with-AI-powered-financial-prediction-for-excess-returns.git
cd FSOT-2.0-theory-with-AI-powered-financial-prediction-for-excess-returns
git rev-parse HEAD
git log -1 --oneline
```

Record the commit hash next to any metrics you publish.

**Windows desktop layout (author):**

```
C:\Users\damia\Desktop\FSOT-Market-Monitor   # this repo working copy
```

---

## 3. Backend environment

```powershell
cd backend
python -m venv .venv

# Windows
.\.venv\Scripts\Activate.ps1

# macOS / Linux
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3.1 Locked dependency list (`requirements.txt`)

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

### 3.2 Environment variables (optional)

Copy template:

```powershell
copy .env.example .env
```

| Variable | Default | Purpose |
|----------|---------|---------|
| `FSOT_RH_API_KEY` | empty | Robinhood crypto (optional) |
| `FSOT_RH_PRIVATE_KEY_PATH` | empty | Signing key path |
| `FSOT_BROKER_DRY_RUN` | `1` | Keep dry-run |
| `FSOT_BROKER_LIVE` | unset | Must not enable for pure research |
| `FSOT_BROKER_I_UNDERSTAND` | unset | Second live gate |
| `FSOT_BROKER_MAX_ORDER_USD` | `100` | Cap for previews |

**Do not commit `.env`.** For reproducibility of *FSOT metrics*, broker keys are unnecessary.

### 3.3 Verify pin & tests

```powershell
.\.venv\Scripts\python scripts\verify_fsot_pin.py
.\.venv\Scripts\python -m pytest tests/ -q
```

**Expected:** pin script reports agreement; pytest shows **17 passed** (count may grow—document if suite changes).

Current test modules:

| File | Coverage |
|------|----------|
| `test_scalar_pin.py` | Seed/scalar pin |
| `test_mapper.py` | Mapping helpers |
| `test_monte_carlo.py` | MC + pattern solidify |
| `test_paper_portfolio.py` | Paper + BHS mode |
| `test_broker_dry_run.py` | Dry-run safety |
| `test_forward_journal.py` | Forward journal I/O |

### 3.4 Start API

```powershell
.\.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Or from repo root: `.\start-backend.ps1`

**Smoke:**

```powershell
# PowerShell
Invoke-RestMethod http://127.0.0.1:8000/api/health
# expect: status = ok, free_parameters = 0, economics_S ≈ 0.646
```

Interactive docs: http://127.0.0.1:8000/docs

---

## 4. Frontend environment

```powershell
cd frontend
npm install
npm run dev
```

Or: `.\start-frontend.ps1`

| Setting | Value |
|---------|--------|
| URL | http://localhost:3000 or http://127.0.0.1:3000 |
| API base | `NEXT_PUBLIC_API_URL` or default `http://127.0.0.1:8000` |

### 4.1 Frontend dependencies (major)

- next ^14.2  
- react / react-dom ^18.3  
- tailwindcss ^3.4  
- lightweight-charts ^4.2  
- typescript ^5.6  

### 4.2 Dashboard reproduction checklist

1. Health indicator green  
2. Watchlist batch loads  
3. Select **BTC** or **SPY** → chart + prediction  
4. Monte Carlo panel loads (may take a few seconds)  
5. Synthetic $ panel, mode **BHS**, capital 10000  
6. Forward journal: Record → (later) Resolve  
7. Auto refresh: default **1m**; set Off to freeze  

---

## 5. Reproducing research metrics

All commands from `backend/` with venv active.

### 5.1 BHS commit accuracy & paper P&amp;L

```powershell
.\.venv\Scripts\python scripts\eval_bhs_target.py
```

| Parameter | Default in script |
|-----------|-------------------|
| Capital | 10000 USD |
| Symbols | SPY, QQQ, AAPL, MSFT, NVDA, BTC, ETH, IWM |
| Hold horizon | 5 |
| Mode | BHS multi-gate |

**Output:** console table +  
`D:\training data\FSOT-Market-History\verification\bhs_target_eval.json`  
(if D: unavailable, script may need path edit—or copy logic to write under `backend/scripts/`).

**How to report:**

```
commit = <git hash>
date = <ISO date>
command = python scripts/eval_bhs_target.py
mean_commit_accuracy = ...
symbols_above_70 = ...
```

### 5.2 Dynamic Monte Carlo

```powershell
.\.venv\Scripts\python scripts\smoke_dynamic_mc.py
.\.venv\Scripts\python scripts\eval_longterm_mc_intelligence.py --quick
```

### 5.3 Intrinsic walk-forward (engine only)

```powershell
.\.venv\Scripts\python scripts\smoke_intrinsic.py
```

### 5.4 Paper portfolio via API (live data)

With API running:

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/api/paper/BTC?capital=10000&mode=bhs&range=2y"
```

Fields: `commit_directional_accuracy`, `total_pnl`, `pct_hold`, `progress_to_70_80`.

### 5.5 Monte Carlo via API

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/api/predict/BTC/montecarlo?horizon=21&n_paths=256&dynamic=true&range=2y"
```

### 5.6 Forward journal (true future protocol)

```powershell
# Record now
Invoke-RestMethod -Method POST "http://127.0.0.1:8000/api/monitor/forward/record?symbol=BTC&horizon=5"
# After horizon trading days have elapsed:
Invoke-RestMethod -Method POST "http://127.0.0.1:8000/api/monitor/forward/resolve"
Invoke-RestMethod "http://127.0.0.1:8000/api/monitor/forward"
```

Ledger: `...\monitor\forward_journal.json` or `backend/data/forward_journal.json`.

---

## 6. Optional historical archive

### 6.1 Download

```powershell
.\.venv\Scripts\python scripts\download_history.py
```

### 6.2 Expected layout

```
D:\training data\FSOT-Market-History\
  ohlcv\{SYMBOL}.csv
  patterns\{SYMBOL}_mc_pattern_ledger.json
  verification\
  monitor\
  manifests\
```

CSV columns (normalized): `time/open/high/low/close/volume` (case-insensitive on load).

### 6.3 Without archive

Market service falls back to **yfinance / CoinGecko** for OHLCV and quotes. Reproduction still works; long-horizon offline eval needs CSVs.

---

## 7. Determinism notes

| Source of variation | Mitigation |
|---------------------|------------|
| RNG in Monte Carlo | Pass `seed=` (API `seed` query; scripts use fixed seeds) |
| Live OHLCV revisions | Prefer archived CSV for bit-stable metrics |
| News RSS timing | Use `use_news=false` for deterministic observer=0 |
| Network failures | Retry; or load CSV-only path |
| Package minor versions | Pin stricter if bit-identical required |

**FSOT seeds and folds are deterministic.** Ensemble quantiles need a fixed `seed` and identical residual bootstrap history.

---

## 8. Full API surface (reproduction targets)

| Method | Path |
|--------|------|
| GET | `/api/health` |
| GET | `/api/watchlist` |
| GET | `/api/market/{symbol}/ohlcv?range=` |
| GET | `/api/market/{symbol}/quote` |
| GET | `/api/predict/{symbol}` |
| GET | `/api/predict/{symbol}/montecarlo` |
| GET | `/api/predict/batch` |
| GET | `/api/backtest/{symbol}` |
| GET | `/api/paper/{symbol}` |
| GET | `/api/broker/status` |
| POST | `/api/broker/preview` |
| GET | `/api/monitor/forward` |
| POST | `/api/monitor/forward/record` |
| POST | `/api/monitor/forward/resolve` |
| GET | `/api/news` |
| GET | `/api/news/observer` |

---

## 9. Directory map

```
FSOT-Market-Monitor/
  README.md
  docs/
    METHODOLOGY.md          ← scientific method
    REPRODUCIBILITY.md      ← this file
  config/watchlist.yaml
  start-backend.ps1
  start-frontend.ps1
  backend/
    requirements.txt
    .env.example
    app/
      main.py
      fsot/                 ← theory + MC + BHS
      backtest/
      broker/
      monitor/
      market/
      news/
      predict/
      api/
    scripts/
    tests/
  frontend/
    package.json
    src/
```

---

## 10. Standalone MC package (optional)

Intelligence-only sandbox (no UI):

```
C:\Users\damia\Desktop\FSOT-Monte-Carlo-Intelligence
```

```powershell
cd C:\Users\damia\Desktop\FSOT-Monte-Carlo-Intelligence
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python -m pytest tests/ -q
.\.venv\Scripts\python scripts\smoke_mc.py
```

Not required to reproduce the full Market Monitor.

---

## 11. Common failures

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: app` | Run from `backend/` or set `PYTHONPATH=.` |
| pytest not found | Activate venv; `pip install -r requirements.txt` |
| API offline on UI | Start uvicorn; check port 8000; CORS allows 3000 |
| Empty OHLCV | Network / rate limit; try another symbol; download history |
| `eval_bhs` missing CSV | Create archive or change load path in script |
| MC slow | Reduce `n_paths` (e.g. 64–128) for smoke |
| Live broker accidental | Ensure `FSOT_BROKER_DRY_RUN=1` and live flags unset |

---

## 12. Publishing a reproduction package

When sharing results:

1. `git rev-parse HEAD`  
2. `python -m pytest tests/ -q` log  
3. Exact eval command(s)  
4. Output JSON (BHS / MC eval)  
5. OHLCV end date / range  
6. Link to `docs/METHODOLOGY.md` § evaluation protocol  

---

## 13. License / ethics

- Research use; see repository license if present.  
- Not financial advice.  
- Do not enable live trading flags unless you accept full capital risk.  
- Respect data provider ToS (Yahoo, CoinGecko, RSS, Robinhood).  

---

## 14. One-shot Windows script (copy-paste)

```powershell
# From empty machine with git, python, node installed
git clone https://github.com/dappalumbo91/FSOT-2.0-theory-with-AI-powered-financial-prediction-for-excess-returns.git
cd FSOT-2.0-theory-with-AI-powered-financial-prediction-for-excess-returns

cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest tests/ -q
Start-Process powershell -ArgumentList "-NoExit","-Command","cd '$PWD'; .\.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000"

cd ..\frontend
npm install
Start-Process powershell -ArgumentList "-NoExit","-Command","cd '$PWD'; npm run dev"

Start-Process "http://127.0.0.1:8000/docs"
Start-Process "http://localhost:3000"
```

Then open docs + UI, run paper/BHS on BTC/SPY, and optionally `eval_bhs_target.py` with history installed.
