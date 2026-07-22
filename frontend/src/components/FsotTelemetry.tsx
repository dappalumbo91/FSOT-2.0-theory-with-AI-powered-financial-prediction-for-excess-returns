"use client";

import type { Prediction } from "@/lib/api";

function fmt(n: number | string | undefined | null, d = 4) {
  if (n === undefined || n === null || n === "") return "—";
  if (typeof n === "string") return n;
  if (Number.isNaN(n)) return "—";
  return n.toFixed(d);
}

type Props = {
  prediction: Prediction | null;
  loading?: boolean;
};

export default function FsotTelemetry({ prediction: p, loading }: Props) {
  if (loading) {
    return (
      <div className="rounded-xl border border-border bg-panel p-4 animate-pulse">
        <div className="h-4 w-32 bg-panel2 rounded mb-4" />
        <div className="space-y-2">
          <div className="h-8 bg-panel2 rounded" />
          <div className="h-8 bg-panel2 rounded" />
          <div className="h-8 bg-panel2 rounded" />
        </div>
      </div>
    );
  }

  if (!p) {
    return (
      <div className="rounded-xl border border-border bg-panel p-4 text-muted text-sm">
        Select an asset to view FSOT telemetry.
      </div>
    );
  }

  const params = p.meta?.params || {};
  const features = p.meta?.features || {};

  return (
    <div className="rounded-xl border border-border bg-panel overflow-hidden">
      <div className="px-4 py-3 border-b border-border flex items-center justify-between">
        <div>
          <h3 className="font-semibold text-fsot">FSOT Scalar Engine</h3>
          <p className="text-xs text-muted font-mono">S = K · (T1 + T2 + T3)</p>
        </div>
        <span
          className={`text-xs px-2 py-1 rounded-full border ${
            p.regime === "emergence"
              ? "border-long/40 text-long bg-long/10"
              : "border-short/40 text-short bg-short/10"
          }`}
        >
          {p.regime}
        </span>
      </div>

      <div className="p-4 grid grid-cols-2 gap-3">
        <Metric label="S (scalar)" value={fmt(p.S, 6)} accent="fsot" large />
        <Metric label="dS emergence pulse" value={fmt(p.dS, 6)} />
        <Metric label="Entropy (dispersal)" value={fmt(p.entropy, 4)} />
        <Metric label="Emergence score" value={fmt(p.emergence_score ?? p.score, 4)} accent="fsot" />
        <Metric label="T1 observer base" value={fmt(p.T1, 5)} />
        <Metric label="T2 linear mod" value={fmt(p.T2, 5)} />
        <Metric label="T3 valve-acoustic" value={fmt(p.T3, 5)} />
        <Metric label="Observer mod (news)" value={fmt(p.observer_mod, 4)} />
        <Metric label="Economics S₀ (telemetry)" value={fmt(p.base_S, 5)} />
        <Metric label="Method" value={p.method || "v2"} />
      </div>
      {p.base_S_note && (
        <p className="px-4 pb-2 text-[10px] text-muted">{p.base_S_note}</p>
      )}

      <div className="px-4 pb-3">
        <h4 className="text-xs uppercase tracking-wider text-muted mb-2">Mapped params · {p.meta?.domain}</h4>
        <div className="grid grid-cols-3 gap-2 text-xs font-mono">
          {[
            ["D_eff", params.D_eff],
            ["N", params.N],
            ["P", params.P],
            ["hits", params.recent_hits],
            ["δψ", params.delta_psi],
            ["δθ", params.delta_theta],
            ["ρ", params.rho],
            ["amp", params.amplitude],
            ["trend", params.trend_bias],
          ].map(([k, v]) => (
            <div key={String(k)} className="rounded-lg bg-panel2 border border-border/60 px-2 py-1.5">
              <div className="text-muted">{k}</div>
              <div className="num text-slate-100">{typeof v === "number" ? (v as number).toFixed(3) : String(v ?? "—")}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="px-4 pb-4">
        <h4 className="text-xs uppercase tracking-wider text-muted mb-2">Market features</h4>
        <div className="grid grid-cols-3 gap-2 text-xs font-mono">
          {[
            ["RSI", features.rsi],
            ["Vol", features.realized_vol],
            ["ATR%", features.atr_pct],
            ["Up days", features.up_day_count],
            ["Rel vol", features.relative_volume],
            ["Trend", features.trend_slope],
          ].map(([k, v]) => (
            <div key={String(k)} className="rounded-lg bg-void/40 border border-border/40 px-2 py-1.5">
              <div className="text-muted">{k}</div>
              <div className="num">{typeof v === "number" ? (v as number).toFixed(4) : "—"}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function Metric({
  label,
  value,
  accent,
  large,
}: {
  label: string;
  value: string;
  accent?: string;
  large?: boolean;
}) {
  // value is pre-formatted
  return (
    <div className="rounded-lg bg-panel2 border border-border/60 px-3 py-2">
      <div className="text-[10px] uppercase tracking-wider text-muted">{label}</div>
      <div
        className={`num font-mono font-semibold ${large ? "text-xl" : "text-sm"} ${
          accent === "fsot" ? "text-fsot" : "text-slate-100"
        }`}
      >
        {value}
      </div>
    </div>
  );
}
