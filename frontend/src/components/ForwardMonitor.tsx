"use client";

import { useCallback, useEffect, useState } from "react";
import { api, type ForwardSummary, type BrokerStatus } from "@/lib/api";

type Props = {
  symbol: string;
};

export default function ForwardMonitor({ symbol }: Props) {
  const [summary, setSummary] = useState<ForwardSummary | null>(null);
  const [broker, setBroker] = useState<BrokerStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [s, b] = await Promise.all([api.forwardSummary(), api.brokerStatus()]);
      setSummary(s);
      setBroker(b);
    } catch (e) {
      setMsg(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const record = async () => {
    setLoading(true);
    setMsg(null);
    try {
      const r = await api.forwardRecord(symbol, 5);
      setMsg(
        `Recorded ${r.entry?.action ?? "?"} @ ${r.entry?.price_at_prediction?.toFixed?.(2) ?? "—"} — resolve after ${r.entry?.horizon_days ?? 5}d`
      );
      await refresh();
    } catch (e) {
      setMsg(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  const resolve = async () => {
    setLoading(true);
    setMsg(null);
    try {
      const r = await api.forwardResolve(symbol);
      setMsg(`Resolved ${r.n_resolved} · still open ${r.n_still_open}`);
      await refresh();
    } catch (e) {
      setMsg(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  const acc = summary?.forward_commit_accuracy;

  return (
    <div className="rounded-xl border border-border bg-panel overflow-hidden">
      <div className="px-4 py-3 border-b border-border">
        <h3 className="font-semibold text-fsot">Forward prediction journal</h3>
        <p className="text-[11px] text-muted">
          Record signals now · score when real future bars arrive (not only history)
        </p>
      </div>
      <div className="p-4 space-y-3">
        <div className="grid grid-cols-2 gap-2 text-sm">
          <Stat label="Open forecasts" value={String(summary?.n_open ?? "—")} />
          <Stat label="Resolved" value={String(summary?.n_resolved ?? "—")} />
          <Stat
            label="Forward commit acc"
            value={acc != null ? `${(acc * 100).toFixed(1)}%` : "—"}
            accent
          />
          <Stat
            label="Directional commits"
            value={String(summary?.n_directional_commits ?? "—")}
          />
        </div>

        <div className="flex flex-wrap gap-2">
          <button
            onClick={record}
            disabled={loading}
            className="px-3 py-1.5 text-xs rounded-lg border border-fsot/40 text-fsot bg-fsot/10 hover:bg-fsot/20 disabled:opacity-50"
          >
            Record {symbol} now
          </button>
          <button
            onClick={resolve}
            disabled={loading}
            className="px-3 py-1.5 text-xs rounded-lg border border-border hover:bg-panel2 disabled:opacity-50"
          >
            Resolve due
          </button>
          <button
            onClick={refresh}
            className="px-3 py-1.5 text-xs rounded-lg border border-border hover:bg-panel2"
          >
            Refresh
          </button>
        </div>

        {msg && <p className="text-xs text-muted font-mono">{msg}</p>}

        {broker && (
          <div className="rounded-lg bg-panel2 border border-border/50 px-3 py-2 text-[11px] space-y-1">
            <div className="font-semibold text-slate-200">Robinhood crypto wire</div>
            <div className="text-muted">
              dry_run=<span className="text-long">{String(broker.dry_run)}</span>
              {" · "}
              live=<span className={broker.live_trading_enabled ? "text-short" : "text-muted"}>
                {String(broker.live_trading_enabled)}
              </span>
              {" · "}
              keys={broker.credentials_configured ? "set" : "not set"}
            </div>
            <div className="text-muted/80">
              Real orders blocked. Synthetic $ + forward journal only until you opt in later.
            </div>
          </div>
        )}

        {summary?.note && (
          <p className="text-[10px] text-muted leading-relaxed">{summary.note}</p>
        )}
      </div>
    </div>
  );
}

function Stat({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent?: boolean;
}) {
  return (
    <div className="rounded-lg bg-panel2 border border-border/50 px-2 py-1.5">
      <div className="text-[10px] text-muted uppercase tracking-wider">{label}</div>
      <div className={`num font-mono font-semibold text-sm ${accent ? "text-fsot" : ""}`}>
        {value}
      </div>
    </div>
  );
}
