"use client";

import { useCallback, useEffect, useState } from "react";
import { api, type PaperPortfolioResult } from "@/lib/api";

type Props = {
  symbol: string;
  range?: string;
};

const PRESETS = [1_000, 5_000, 10_000, 25_000, 50_000, 100_000];

export default function PaperPortfolio({ symbol, range = "2y" }: Props) {
  const [capital, setCapital] = useState(10_000);
  const [mode, setMode] = useState<"solid_gated" | "always_in" | "long_only" | "buy_hold">(
    "solid_gated"
  );
  const [data, setData] = useState<PaperPortfolioResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.paper(symbol, { capital, mode, range });
      setData(res.paper);
    } catch (e) {
      setData(null);
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [symbol, capital, mode, range]);

  useEffect(() => {
    load();
  }, [load]);

  const curve = data?.equity_curve || [];
  const minE = curve.length ? Math.min(...curve.map((c) => c.equity)) : 0;
  const maxE = curve.length ? Math.max(...curve.map((c) => c.equity)) : 1;
  const span = Math.max(maxE - minE, 1);

  return (
    <div className="rounded-xl border border-border bg-panel overflow-hidden">
      <div className="px-4 py-3 border-b border-border flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="font-semibold text-fsot">Synthetic $ Portfolio</h3>
          <p className="text-[11px] text-muted">
            Paper trading on real {symbol} history · theoretical P&amp;L before live capital
          </p>
        </div>
        <button
          onClick={load}
          disabled={loading}
          className="px-2 py-1 text-xs rounded-md border border-border hover:bg-panel2 disabled:opacity-50"
        >
          {loading ? "Running…" : "Recalculate"}
        </button>
      </div>

      <div className="p-4 space-y-3">
        <div>
          <label className="text-[10px] uppercase tracking-wider text-muted">
            Starting capital (USD)
          </label>
          <div className="flex flex-wrap gap-1 mt-1">
            {PRESETS.map((p) => (
              <button
                key={p}
                onClick={() => setCapital(p)}
                className={`px-2 py-1 text-xs rounded-md border font-mono ${
                  capital === p
                    ? "border-fsot text-fsot bg-fsot/10"
                    : "border-border text-muted hover:text-slate-200"
                }`}
              >
                ${p.toLocaleString()}
              </button>
            ))}
          </div>
          <input
            type="number"
            min={100}
            step={500}
            value={capital}
            onChange={(e) => setCapital(Math.max(100, Number(e.target.value) || 100))}
            className="mt-2 w-full rounded-lg bg-panel2 border border-border px-3 py-1.5 text-sm font-mono"
          />
        </div>

        <div>
          <label className="text-[10px] uppercase tracking-wider text-muted">Mode</label>
          <div className="flex flex-wrap gap-1 mt-1">
            {(
              [
                ["solid_gated", "Solid gate"],
                ["always_in", "Always in"],
                ["long_only", "Long only"],
                ["buy_hold", "Buy & hold"],
              ] as const
            ).map(([m, label]) => (
              <button
                key={m}
                onClick={() => setMode(m)}
                className={`px-2 py-1 text-xs rounded-md border ${
                  mode === m
                    ? "border-fsot text-fsot bg-fsot/10"
                    : "border-border text-muted hover:text-slate-200"
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        {error && (
          <div className="text-xs text-short border border-short/30 rounded-lg px-3 py-2">{error}</div>
        )}

        {loading && !data && (
          <div className="h-24 rounded-lg bg-panel2 animate-pulse" />
        )}

        {data && !data.error && (
          <>
            <div className="grid grid-cols-2 gap-2">
              <Stat
                label="Ending equity"
                value={usd(data.capital_end)}
                accent
              />
              <Stat
                label="Total P&L"
                value={usd(data.total_pnl)}
                tone={data.total_pnl}
              />
              <Stat label="Return" value={pct(data.total_return)} tone={data.total_return} />
              <Stat
                label="Max DD $"
                value={usd(data.max_drawdown_dollars)}
                tone={-1}
              />
              <Stat label="Sharpe" value={data.sharpe.toFixed(2)} tone={data.sharpe} />
              <Stat
                label="vs Buy&Hold $"
                value={usd(data.vs_buy_hold_pnl)}
                tone={data.vs_buy_hold_pnl}
              />
              <Stat label="Trades" value={String(data.trades)} />
              <Stat
                label="Win rate"
                value={
                  data.win_rate != null ? `${(data.win_rate * 100).toFixed(1)}%` : "—"
                }
              />
            </div>

            {curve.length > 2 && (
              <div>
                <div className="text-[10px] uppercase tracking-wider text-muted mb-1">
                  Equity curve · ${data.capital_start.toLocaleString()} →{" "}
                  {usd(data.capital_end)}
                </div>
                <div className="flex items-end gap-px h-16 rounded-lg bg-void/40 border border-border/40 px-1 py-1">
                  {curve.map((c, i) => {
                    const h = ((c.equity - minE) / span) * 100;
                    const up = c.equity >= data.capital_start;
                    return (
                      <div
                        key={i}
                        className={`flex-1 rounded-sm min-w-0 ${up ? "bg-long/70" : "bg-short/70"}`}
                        style={{ height: `${Math.max(h, 2)}%` }}
                        title={`${c.t}: ${usd(c.equity)}`}
                      />
                    );
                  })}
                </div>
              </div>
            )}

            <p className="text-[10px] text-muted leading-relaxed">{data.note}</p>
          </>
        )}
      </div>
    </div>
  );
}

function Stat({
  label,
  value,
  tone,
  accent,
}: {
  label: string;
  value: string;
  tone?: number;
  accent?: boolean;
}) {
  const cls =
    tone === undefined
      ? accent
        ? "text-fsot"
        : "text-slate-100"
      : tone >= 0
        ? "text-long"
        : "text-short";
  return (
    <div className="rounded-lg bg-panel2 border border-border/50 px-2 py-1.5">
      <div className="text-[10px] text-muted uppercase tracking-wider">{label}</div>
      <div className={`num font-mono font-semibold text-sm ${cls}`}>{value}</div>
    </div>
  );
}

function usd(n: number) {
  const sign = n < 0 ? "-" : "";
  return `${sign}$${Math.abs(n).toLocaleString(undefined, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`;
}

function pct(x: number) {
  const sign = x >= 0 ? "+" : "";
  return `${sign}${(x * 100).toFixed(2)}%`;
}
