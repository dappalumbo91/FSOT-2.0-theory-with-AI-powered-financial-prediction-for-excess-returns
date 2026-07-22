"use client";

import type { Backtest } from "@/lib/api";

type Props = {
  data: Backtest | null;
  loading?: boolean;
  symbol?: string;
};

export default function BacktestCard({ data, loading, symbol }: Props) {
  if (loading) {
    return (
      <div className="rounded-xl border border-border bg-panel p-4 animate-pulse h-40" />
    );
  }

  if (!data) {
    return (
      <div className="rounded-xl border border-border bg-panel p-4 text-sm text-muted">
        Walk-forward backtest will appear here.
      </div>
    );
  }

  const rows: [string, string, string?][] = [
    ["Directional accuracy", `${(data.directional_accuracy * 100).toFixed(1)}%`],
    ["Strategy return", pct(data.strategy_return), tone(data.strategy_return)],
    ["Buy & hold", pct(data.buy_hold_return), tone(data.buy_hold_return)],
    ["Sharpe (ann.)", data.sharpe.toFixed(2), tone(data.sharpe)],
    ["Max drawdown", pct(data.max_drawdown), "text-short"],
    ["Hit rate long", `${(data.hit_rate_long * 100).toFixed(1)}%`],
    ["Hit rate short", `${(data.hit_rate_short * 100).toFixed(1)}%`],
    ["Bars", String(data.n_bars)],
  ];

  return (
    <div className="rounded-xl border border-border bg-panel overflow-hidden">
      <div className="px-4 py-3 border-b border-border">
        <h3 className="font-semibold">Walk-forward backtest{symbol ? ` · ${symbol}` : ""}</h3>
        <p className="text-[11px] text-muted mt-0.5">{data.note}</p>
      </div>
      <div className="p-4 grid grid-cols-2 gap-2">
        {rows.map(([label, value, cls]) => (
          <div key={label} className="rounded-lg bg-panel2 border border-border/50 px-3 py-2">
            <div className="text-[10px] uppercase tracking-wider text-muted">{label}</div>
            <div className={`num font-mono font-semibold ${cls || ""}`}>{value}</div>
          </div>
        ))}
      </div>
      <div className="px-4 pb-3 text-[10px] text-muted">
        Research tool only — not financial advice. Metrics are causal (no lookahead).
      </div>
    </div>
  );
}

function pct(x: number) {
  return `${(x * 100).toFixed(2)}%`;
}

function tone(x: number) {
  return x >= 0 ? "text-long" : "text-short";
}
