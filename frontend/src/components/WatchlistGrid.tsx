"use client";

import SignalBadge from "./SignalBadge";
import type { BatchItem } from "@/lib/api";

type Props = {
  items: BatchItem[];
  selected?: string;
  onSelect: (symbol: string) => void;
  loading?: boolean;
};

export default function WatchlistGrid({ items, selected, onSelect, loading }: Props) {
  if (loading && !items.length) {
    return (
      <div className="rounded-xl border border-border bg-panel p-4 text-sm text-muted">
        Loading watchlist signals…
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-border bg-panel overflow-hidden flex flex-col max-h-[calc(100vh-8rem)]">
      <div className="px-3 py-2 border-b border-border flex items-center justify-between sticky top-0 bg-panel z-10">
        <h3 className="text-sm font-semibold">Watchlist</h3>
        <span className="text-xs text-muted">{items.length} assets</span>
      </div>
      <div className="overflow-y-auto">
        <table className="w-full text-sm">
          <thead className="text-[10px] uppercase tracking-wider text-muted sticky top-0 bg-panel2">
            <tr>
              <th className="text-left px-3 py-2 font-medium">Symbol</th>
              <th className="text-right px-2 py-2 font-medium">Price</th>
              <th className="text-right px-2 py-2 font-medium">Δ%</th>
              <th className="text-center px-2 py-2 font-medium">Signal</th>
              <th className="text-right px-3 py-2 font-medium">S</th>
            </tr>
          </thead>
          <tbody>
            {items.map((it) => {
              const active = selected === it.symbol;
              const up = (it.change_pct ?? 0) >= 0;
              return (
                <tr
                  key={it.symbol}
                  onClick={() => onSelect(it.symbol)}
                  className={`cursor-pointer border-t border-border/50 transition-colors ${
                    active ? "bg-fsot/10" : "hover:bg-panel2/80"
                  }`}
                >
                  <td className="px-3 py-2">
                    <div className="font-semibold">{it.symbol}</div>
                    <div className="text-[10px] text-muted truncate max-w-[100px]">{it.name}</div>
                  </td>
                  <td className="px-2 py-2 text-right num font-mono">
                    {it.price != null ? formatPrice(it.price) : "—"}
                  </td>
                  <td
                    className={`px-2 py-2 text-right num font-mono ${
                      up ? "text-long" : "text-short"
                    }`}
                  >
                    {it.change_pct != null
                      ? `${up ? "+" : ""}${it.change_pct.toFixed(2)}%`
                      : "—"}
                  </td>
                  <td className="px-2 py-2 text-center">
                    {it.error ? (
                      <span className="text-[10px] text-warn">err</span>
                    ) : (
                      <SignalBadge signal={it.signal || "FLAT"} size="sm" />
                    )}
                  </td>
                  <td className="px-3 py-2 text-right num font-mono text-fsot text-xs">
                    {it.S != null ? it.S.toFixed(3) : "—"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function formatPrice(p: number) {
  if (p >= 1000) return p.toLocaleString(undefined, { maximumFractionDigits: 2 });
  if (p >= 1) return p.toFixed(2);
  return p.toFixed(4);
}
