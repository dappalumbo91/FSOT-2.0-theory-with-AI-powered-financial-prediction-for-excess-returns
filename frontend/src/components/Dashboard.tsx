"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  api,
  type Backtest,
  type BatchItem,
  type Health,
  type MonteCarloResult,
  type OhlcvBar,
  type Prediction,
  type Quote,
} from "@/lib/api";
import ChartPanel from "./ChartPanel";
import SignalBadge from "./SignalBadge";
import FsotTelemetry from "./FsotTelemetry";
import WatchlistGrid from "./WatchlistGrid";
import BacktestCard from "./BacktestCard";
import MonteCarloPanel from "./MonteCarloPanel";
import PaperPortfolio from "./PaperPortfolio";
import ForwardMonitor from "./ForwardMonitor";

type Tab = "all" | "indices" | "stocks" | "crypto";

/** Common polling cadences for research dashboards (not HFT). */
const AUTO_OPTIONS: { sec: number; label: string; hint: string }[] = [
  { sec: 0, label: "Off", hint: "Manual only" },
  { sec: 30, label: "30s", hint: "Quotes / crypto feel" },
  { sec: 60, label: "1m", hint: "Typical monitor default" },
  { sec: 120, label: "2m", hint: "Light load" },
  { sec: 300, label: "5m", hint: "Casual / daily bars" },
  { sec: 900, label: "15m", hint: "Background watch" },
];

const LS_KEY = "fsot_auto_refresh_sec";

function readAutoSec(): number {
  if (typeof window === "undefined") return 60;
  const v = localStorage.getItem(LS_KEY);
  if (v == null) return 60; // industry-ish default for dashboards
  const n = Number(v);
  return AUTO_OPTIONS.some((o) => o.sec === n) ? n : 60;
}

export default function Dashboard() {
  const [symbol, setSymbol] = useState("SPY");
  const [tab, setTab] = useState<Tab>("all");
  const [range, setRange] = useState("1y");
  const [health, setHealth] = useState<Health | null>(null);
  const [batch, setBatch] = useState<BatchItem[]>([]);
  const [bars, setBars] = useState<OhlcvBar[]>([]);
  const [quote, setQuote] = useState<Quote | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [backtest, setBacktest] = useState<Backtest | null>(null);
  const [monteCarlo, setMonteCarlo] = useState<MonteCarloResult | null>(null);
  const [loadingMain, setLoadingMain] = useState(false);
  const [loadingBatch, setLoadingBatch] = useState(false);
  const [loadingBt, setLoadingBt] = useState(false);
  const [loadingMc, setLoadingMc] = useState(false);
  const [mcError, setMcError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [clock, setClock] = useState("");
  const [autoSec, setAutoSec] = useState(60);
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null);
  const [secondsLeft, setSecondsLeft] = useState<number | null>(null);
  const [refreshNonce, setRefreshNonce] = useState(0);
  const [tabVisible, setTabVisible] = useState(true);
  const heavyTick = useRef(0);
  const refreshing = useRef(false);

  useEffect(() => {
    setAutoSec(readAutoSec());
  }, []);

  useEffect(() => {
    const tick = () => setClock(new Date().toLocaleString());
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);

  // Pause polling when browser tab is hidden (saves API quota)
  useEffect(() => {
    const onVis = () => setTabVisible(document.visibilityState === "visible");
    onVis();
    document.addEventListener("visibilitychange", onVis);
    return () => document.removeEventListener("visibilitychange", onVis);
  }, []);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
  }, []);

  const loadBatch = useCallback(async (silent = false) => {
    if (!silent) setLoadingBatch(true);
    try {
      const section = tab === "all" ? undefined : tab;
      const res = await api.batch(section);
      setBatch(res.items);
    } catch (e) {
      console.error(e);
    } finally {
      if (!silent) setLoadingBatch(false);
    }
  }, [tab]);

  useEffect(() => {
    loadBatch(false);
  }, [loadBatch]);

  const loadSymbol = useCallback(
    async (sym: string, opts?: { silent?: boolean; heavy?: boolean }) => {
      const silent = opts?.silent ?? false;
      const heavy = opts?.heavy ?? true;
      if (!silent) setLoadingMain(true);
      setError(null);
      try {
        const [ohlcv, pred] = await Promise.all([
          api.ohlcv(sym, range),
          api.predict(sym, range),
        ]);
        setBars(ohlcv.bars);
        setQuote(pred.quote);
        setPrediction(pred.prediction);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        if (!silent) setLoadingMain(false);
      }

      if (!heavy) return;

      if (!silent) setLoadingBt(true);
      try {
        const bt = await api.backtest(sym, range === "1y" ? "2y" : range);
        setBacktest(bt.backtest);
      } catch {
        setBacktest(null);
      } finally {
        if (!silent) setLoadingBt(false);
      }

      if (!silent) setLoadingMc(true);
      setMcError(null);
      try {
        const mcRes = await api.monteCarlo(sym, {
          range: range === "3mo" || range === "6mo" ? "1y" : range === "1y" ? "2y" : range,
          horizon: 21,
          n_paths: 384,
        });
        setMonteCarlo(mcRes.monte_carlo);
      } catch (e) {
        setMonteCarlo(null);
        setMcError(e instanceof Error ? e.message : String(e));
      } finally {
        if (!silent) setLoadingMc(false);
      }
    },
    [range]
  );

  useEffect(() => {
    loadSymbol(symbol, { silent: false, heavy: true });
  }, [symbol, loadSymbol]);

  const runAutoRefresh = useCallback(async () => {
    if (refreshing.current || document.visibilityState !== "visible") return;
    refreshing.current = true;
    try {
      heavyTick.current += 1;
      // Every 3rd poll: full heavy (MC + backtest); else quotes + signals + batch
      const heavy = heavyTick.current % 3 === 0;
      await Promise.all([
        loadBatch(true),
        loadSymbol(symbol, { silent: true, heavy }),
        api.health().then(setHealth).catch(() => setHealth(null)),
      ]);
      if (heavy) setRefreshNonce((n) => n + 1);
      setLastRefresh(new Date());
      setSecondsLeft(autoSec > 0 ? autoSec : null);
    } finally {
      refreshing.current = false;
    }
  }, [autoSec, loadBatch, loadSymbol, symbol]);

  // Auto-refresh interval
  useEffect(() => {
    if (autoSec <= 0 || !tabVisible) {
      setSecondsLeft(null);
      return;
    }
    setSecondsLeft(autoSec);
    const countdown = setInterval(() => {
      setSecondsLeft((s) => {
        if (s == null) return autoSec;
        if (s <= 1) return autoSec;
        return s - 1;
      });
    }, 1000);
    const poll = setInterval(() => {
      void runAutoRefresh();
    }, autoSec * 1000);
    return () => {
      clearInterval(countdown);
      clearInterval(poll);
    };
  }, [autoSec, tabVisible, runAutoRefresh]);

  const setAuto = (sec: number) => {
    setAutoSec(sec);
    try {
      localStorage.setItem(LS_KEY, String(sec));
    } catch {
      /* ignore */
    }
    setSecondsLeft(sec > 0 ? sec : null);
  };

  const manualRefresh = () => {
    heavyTick.current = 0;
    void loadBatch(false);
    void loadSymbol(symbol, { silent: false, heavy: true });
    void api.health().then(setHealth).catch(() => setHealth(null));
    setRefreshNonce((n) => n + 1);
    setLastRefresh(new Date());
    if (autoSec > 0) setSecondsLeft(autoSec);
  };

  const filteredBatch = useMemo(() => {
    if (tab === "all") return batch;
    return batch.filter((b) => {
      const c = (b.class || "").toLowerCase();
      if (tab === "indices") return c === "index" || c === "indices";
      if (tab === "stocks") return c === "equity" || c === "stocks";
      if (tab === "crypto") return c === "crypto";
      return true;
    });
  }, [batch, tab]);

  const up = (quote?.change_pct ?? 0) >= 0;

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="sticky top-0 z-20 border-b border-border bg-void/90 backdrop-blur-md">
        <div className="max-w-[1600px] mx-auto px-4 py-3 flex flex-wrap items-center gap-4 justify-between">
          <div className="flex items-center gap-3">
            <div className="h-9 w-9 rounded-lg bg-gradient-to-br from-fsot to-long flex items-center justify-center font-bold text-void text-sm">
              FS
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight">FSOT Market Monitor</h1>
              <p className="text-[11px] text-muted">
                Fluid Spacetime Omni-Theory · Economics domain D_eff=20
              </p>
            </div>
          </div>

          <div className="flex items-center gap-4 text-sm">
            <div className="flex items-center gap-2 text-muted">
              <span
                className={`h-2 w-2 rounded-full live-dot ${
                  health?.status === "ok" ? "bg-long" : "bg-short"
                }`}
              />
              <span className="font-mono text-xs">
                {health?.status === "ok"
                  ? `API v${health.version} · S₀=${health.fsot.economics_S.toFixed(4)}`
                  : "API offline"}
              </span>
            </div>
            <span className="text-xs text-muted font-mono hidden sm:inline">{clock}</span>

            {/* Auto-refresh */}
            <div className="flex items-center gap-1.5">
              <span className="text-[10px] uppercase tracking-wider text-muted hidden md:inline">
                Auto
              </span>
              <div className="flex rounded-lg border border-border bg-panel2 p-0.5">
                {AUTO_OPTIONS.map((o) => (
                  <button
                    key={o.sec}
                    title={o.hint}
                    onClick={() => setAuto(o.sec)}
                    className={`px-1.5 sm:px-2 py-1 text-[10px] sm:text-xs rounded-md font-mono transition ${
                      autoSec === o.sec
                        ? "bg-fsot/20 text-fsot font-semibold"
                        : "text-muted hover:text-slate-200"
                    }`}
                  >
                    {o.label}
                  </button>
                ))}
              </div>
              {autoSec > 0 && tabVisible && secondsLeft != null && (
                <span className="text-[10px] font-mono text-muted tabular-nums w-8">
                  {secondsLeft}s
                </span>
              )}
              {autoSec > 0 && !tabVisible && (
                <span className="text-[10px] text-muted">paused</span>
              )}
            </div>

            <button
              onClick={manualRefresh}
              className="px-3 py-1.5 rounded-lg border border-border bg-panel2 hover:bg-panel text-xs font-medium transition"
            >
              Refresh
            </button>
          </div>
        </div>
        {(autoSec > 0 || lastRefresh) && (
          <div className="max-w-[1600px] mx-auto px-4 pb-2 text-[10px] text-muted font-mono flex flex-wrap gap-x-4 gap-y-0.5">
            <span>
              Polling:{" "}
              {autoSec <= 0
                ? "manual"
                : `${autoSec}s · quotes/signals every tick · MC/backtest every 3rd`}
            </span>
            {lastRefresh && (
              <span>Last update {lastRefresh.toLocaleTimeString()}</span>
            )}
            <span className="text-muted/70">
              Typical: 15–30s quotes · 1–5m dashboards · daily bar models at close
            </span>
          </div>
        )}
      </header>

      <main className="max-w-[1600px] mx-auto px-4 py-4 grid grid-cols-1 lg:grid-cols-12 gap-4">
        {/* Left watchlist */}
        <aside className="lg:col-span-3 space-y-3">
          <div className="flex gap-1 p-1 rounded-xl bg-panel border border-border">
            {(["all", "indices", "stocks", "crypto"] as Tab[]).map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={`flex-1 text-xs py-1.5 rounded-lg capitalize transition ${
                  tab === t ? "bg-fsot/20 text-fsot font-semibold" : "text-muted hover:text-slate-200"
                }`}
              >
                {t}
              </button>
            ))}
          </div>
          <WatchlistGrid
            items={filteredBatch}
            selected={symbol}
            onSelect={setSymbol}
            loading={loadingBatch}
          />
        </aside>

        {/* Center chart + quote */}
        <section className="lg:col-span-6 space-y-3">
          <div className="rounded-xl border border-border bg-panel p-4">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <div className="flex items-center gap-2">
                  <h2 className="text-2xl font-bold">{symbol}</h2>
                  {quote?.name && <span className="text-muted text-sm">{quote.name}</span>}
                </div>
                <div className="flex items-baseline gap-3 mt-1">
                  <span className="text-3xl font-semibold num font-mono">
                    {quote?.price != null ? formatPrice(quote.price) : "—"}
                  </span>
                  <span className={`num font-mono font-medium ${up ? "text-long" : "text-short"}`}>
                    {quote?.change_pct != null
                      ? `${up ? "+" : ""}${quote.change_pct.toFixed(2)}%`
                      : ""}
                  </span>
                </div>
              </div>
              <div className="flex flex-col items-end gap-2">
                {prediction && (
                  <SignalBadge signal={prediction.signal} confidence={prediction.confidence} size="lg" />
                )}
                <div className="flex gap-1">
                  {["3mo", "6mo", "1y", "2y"].map((r) => (
                    <button
                      key={r}
                      onClick={() => setRange(r)}
                      className={`px-2 py-1 text-xs rounded-md border transition ${
                        range === r
                          ? "border-fsot text-fsot bg-fsot/10"
                          : "border-border text-muted hover:text-slate-200"
                      }`}
                    >
                      {r}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {prediction && (
              <div className="mt-3 grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
                <MiniStat label="Pred return (1d)" value={pct(prediction.pred_return)} tone={prediction.pred_return} />
                <MiniStat label="Pred price" value={formatPrice(prediction.pred_price_1d)} />
                <MiniStat
                  label="5d horizon"
                  value={
                    prediction.horizons?.["5"]
                      ? pct(prediction.horizons["5"].pred_return)
                      : "—"
                  }
                />
                <MiniStat
                  label="20d horizon"
                  value={
                    prediction.horizons?.["20"]
                      ? pct(prediction.horizons["20"].pred_return)
                      : "—"
                  }
                />
              </div>
            )}
          </div>

          {error && (
            <div className="rounded-lg border border-short/40 bg-short/10 text-short text-sm px-4 py-2">
              {error}
            </div>
          )}

          {loadingMain && !bars.length ? (
            <div className="rounded-xl border border-border bg-panel h-[420px] animate-pulse" />
          ) : (
            <ChartPanel bars={bars} />
          )}

          <BacktestCard data={backtest} loading={loadingBt} symbol={symbol} />
          <PaperPortfolio
            symbol={symbol}
            range={range === "3mo" || range === "6mo" ? "1y" : range === "1y" ? "2y" : range}
            refreshKey={refreshNonce}
          />
          <ForwardMonitor symbol={symbol} refreshKey={refreshNonce} />
        </section>

        {/* Right FSOT */}
        <aside className="lg:col-span-3 space-y-3">
          <FsotTelemetry prediction={prediction} loading={loadingMain && !prediction} />
          <MonteCarloPanel mc={monteCarlo} loading={loadingMc} error={mcError} />
          <div className="rounded-xl border border-border bg-panel p-4 text-xs text-muted leading-relaxed">
            <h4 className="text-slate-200 font-semibold mb-1">About FSOT finance</h4>
            Markets map onto Economics domain folds (seeds π, e, φ, γ, Catalan). Monte
            Carlo explores multipath futures: each step collapses True (FSOT μ) or False
            (Poof chaos) from consciousness × observer phase × sentiment — zero free params.
            <div className="mt-2 text-[10px] text-muted/80">
              Math pin: FSOT-Physical-Archive vendor/fsot_compute.py · GitHub FSOT-2.1-Lean
            </div>
          </div>
        </aside>
      </main>
    </div>
  );
}

function formatPrice(p: number) {
  if (p >= 1000) return p.toLocaleString(undefined, { maximumFractionDigits: 2 });
  if (p >= 1) return p.toFixed(2);
  return p.toPrecision(4);
}

function pct(x: number) {
  const sign = x >= 0 ? "+" : "";
  return `${sign}${(x * 100).toFixed(3)}%`;
}

function MiniStat({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone?: number;
}) {
  const cls =
    tone === undefined ? "" : tone >= 0 ? "text-long" : "text-short";
  return (
    <div className="rounded-lg bg-panel2 border border-border/50 px-2 py-1.5">
      <div className="text-[10px] text-muted uppercase tracking-wider">{label}</div>
      <div className={`num font-mono font-semibold ${cls}`}>{value}</div>
    </div>
  );
}
