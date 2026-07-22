"use client";

import type { MonteCarloResult } from "@/lib/api";
import SignalBadge from "./SignalBadge";

function fmt(n: number | undefined | null, d = 4) {
  if (n === undefined || n === null || Number.isNaN(n)) return "—";
  return n.toFixed(d);
}

function pct(n: number | undefined | null, d = 2) {
  if (n === undefined || n === null || Number.isNaN(n)) return "—";
  return `${(n * 100).toFixed(d)}%`;
}

type Props = {
  mc: MonteCarloResult | null;
  loading?: boolean;
  error?: string | null;
};

export default function MonteCarloPanel({ mc, loading, error }: Props) {
  if (loading) {
    return (
      <div className="rounded-xl border border-border bg-panel p-4 animate-pulse">
        <div className="h-4 w-48 bg-panel2 rounded mb-4" />
        <div className="h-24 bg-panel2 rounded" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-xl border border-short/30 bg-panel p-4 text-sm text-short">
        Monte Carlo: {error}
      </div>
    );
  }

  if (!mc || mc.error) {
    return (
      <div className="rounded-xl border border-border bg-panel p-4 text-muted text-sm">
        Monte Carlo ensemble not loaded.
      </div>
    );
  }

  const e = mc.ensemble;
  const mode = e.most_probable_return_bin;
  const fan = mc.fan_chart || [];
  const maxP = Math.max(...fan.map((f) => f.p90), mc.price0 * 1.01);
  const minP = Math.min(...fan.map((f) => f.p10), mc.price0 * 0.99);

  return (
    <div className="rounded-xl border border-border bg-panel overflow-hidden">
      <div className="px-4 py-3 border-b border-border flex items-center justify-between gap-2 flex-wrap">
        <div>
          <h3 className="font-semibold text-fsot">
            {mc.dynamic ? "Intelligent FSOT Monte Carlo" : "FSOT Monte Carlo"}
          </h3>
          <p className="text-xs text-muted font-mono">
            Pattern collapse · {mc.n_paths} paths · {mc.horizon}d · free_params=0
          </p>
        </div>
        <div className="flex items-center gap-2">
          <SignalBadge signal={mc.signal} confidence={mc.confidence} />
        </div>
      </div>

      <div className="p-4 grid grid-cols-2 sm:grid-cols-4 gap-3">
        <Metric label="P(up) ensemble" value={pct(e.p_up)} />
        <Metric label="P(up) observed branch" value={pct(e.p_up_observed_branch)} accent />
        <Metric label="E[return]" value={pct(e.expected_return)} />
        <Metric label="Median return" value={pct(e.median_return)} />
        <Metric label="Collapse True frac" value={pct(e.mean_collapse_true_fraction)} />
        <Metric label="Observed-path frac" value={pct(e.observed_branch_path_fraction)} />
        <Metric
          label="Most probable bin"
          value={`${pct(mode.low)}…${pct(mode.high)}`}
        />
        <Metric label="p50 terminal" value={fmt(e.quantiles_price.p50, 2)} accent />
      </div>

      {/* Pattern recognition / solidification */}
      {mc.pattern && (
        <div className="px-4 pb-3">
          <h4 className="text-xs uppercase tracking-wider text-muted mb-2">
            Pattern recognition · solidify when acc_φ &gt; 0.5+Poof
          </h4>
          <div className="grid grid-cols-2 gap-2 text-xs font-mono mb-2">
            <div className="rounded-lg bg-panel2 border border-border/60 px-2 py-1.5 col-span-2">
              <div className="text-muted">Active signature</div>
              <div className="text-[10px] text-slate-200 break-all">{mc.pattern.key}</div>
            </div>
            <div className="rounded-lg bg-panel2 border border-border/60 px-2 py-1.5">
              <div className="text-muted">Solidified (gate)</div>
              <div className={mc.pattern.bias.solidified ? "text-long" : "text-muted"}>
                {mc.pattern.bias.solidified
                  ? "YES · commit"
                  : "no · FLAT (fluid)"}
              </div>
            </div>
            <div className="rounded-lg bg-panel2 border border-border/60 px-2 py-1.5">
              <div className="text-muted">Strength / acc_φ</div>
              <div>
                {fmt(mc.pattern.bias.strength, 3)} / {pct(mc.pattern.bias.acc_phi)}
              </div>
            </div>
            {mc.pattern.memory_summary && (
              <>
                <div className="rounded-lg bg-panel2 border border-border/60 px-2 py-1.5">
                  <div className="text-muted">Patterns / solid</div>
                  <div>
                    {mc.pattern.memory_summary.n_patterns} / {mc.pattern.memory_summary.n_solidified}
                  </div>
                </div>
                <div className="rounded-lg bg-panel2 border border-border/60 px-2 py-1.5">
                  <div className="text-muted">Train updates</div>
                  <div>{mc.pattern.memory_summary.n_updates}</div>
                </div>
              </>
            )}
          </div>
          {mc.training && (
            <div className="grid grid-cols-2 gap-2 text-xs font-mono">
              <div className="rounded-lg bg-void/40 border border-border/40 px-2 py-1.5">
                <div className="text-muted">Raw FSOT acc</div>
                <div>{pct(mc.training.raw_directional_accuracy)}</div>
              </div>
              <div className="rounded-lg bg-void/40 border border-border/40 px-2 py-1.5">
                <div className="text-muted">Anchored acc</div>
                <div className="text-fsot">{pct(mc.training.anchored_directional_accuracy)}</div>
              </div>
              {mc.training.refinement_lift != null && (
                <div className="rounded-lg bg-void/40 border border-border/40 px-2 py-1.5 col-span-2">
                  <div className="text-muted">Refinement lift (late − early)</div>
                  <div className={mc.training.refinement_lift >= 0 ? "text-long" : "text-short"}>
                    {pct(mc.training.refinement_lift)} · early {pct(mc.training.anchored_early)} → late{" "}
                    {pct(mc.training.anchored_late)}
                  </div>
                </div>
              )}
            </div>
          )}
          {mc.pattern.memory_summary?.top_patterns && mc.pattern.memory_summary.top_patterns.length > 0 && (
            <div className="mt-2 space-y-1 max-h-28 overflow-y-auto">
              {mc.pattern.memory_summary.top_patterns.slice(0, 5).map((p) => (
                <div
                  key={p.key}
                  className="text-[10px] font-mono flex justify-between gap-2 border-b border-border/30 py-0.5"
                >
                  <span className="truncate text-muted" title={p.key}>
                    {p.solidified ? "◆" : "◇"} {p.key.slice(0, 28)}
                  </span>
                  <span className="shrink-0 text-slate-300">
                    φ={pct(p.acc_phi, 0)} n={p.trials}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Simple fan sparkline via CSS bars */}
      {fan.length > 2 && (
        <div className="px-4 pb-3">
          <h4 className="text-xs uppercase tracking-wider text-muted mb-2">
            Path fan (p10 / p50 / p90)
          </h4>
          <div className="flex items-end gap-0.5 h-20">
            {fan.map((f) => {
              const lo = ((f.p10 - minP) / (maxP - minP || 1)) * 100;
              const mid = ((f.p50 - minP) / (maxP - minP || 1)) * 100;
              const hi = ((f.p90 - minP) / (maxP - minP || 1)) * 100;
              return (
                <div
                  key={f.step}
                  className="flex-1 relative h-full"
                  title={`t=${f.step} p50=${f.p50.toFixed(2)}`}
                >
                  <div
                    className="absolute left-0 right-0 bg-fsot/20 rounded-sm"
                    style={{ bottom: `${lo}%`, height: `${Math.max(hi - lo, 2)}%` }}
                  />
                  <div
                    className="absolute left-0 right-0 h-0.5 bg-fsot"
                    style={{ bottom: `${mid}%` }}
                  />
                </div>
              );
            })}
          </div>
          <div className="flex justify-between text-[10px] text-muted font-mono mt-1">
            <span>t=0 · {fmt(mc.price0, 2)}</span>
            <span>
              p10={fmt(e.quantiles_price.p10, 2)} · p90={fmt(e.quantiles_price.p90, 2)}
            </span>
          </div>
        </div>
      )}

      <div className="px-4 pb-4 grid grid-cols-2 gap-2 text-xs font-mono">
        <div className="rounded-lg bg-panel2 border border-border/60 px-2 py-1.5">
          <div className="text-muted">C (consciousness)</div>
          <div>{fmt(mc.state0?.consciousness_factor, 4)}</div>
        </div>
        <div className="rounded-lg bg-panel2 border border-border/60 px-2 py-1.5">
          <div className="text-muted">δψ · quirk_mod</div>
          <div>
            {fmt(mc.state0?.delta_psi, 3)} · {fmt(mc.state0?.quirk_mod, 4)}
          </div>
        </div>
        <div className="rounded-lg bg-panel2 border border-border/60 px-2 py-1.5">
          <div className="text-muted">Sentiment (observer)</div>
          <div>{fmt(mc.sentiment, 4)}</div>
        </div>
        <div className="rounded-lg bg-panel2 border border-border/60 px-2 py-1.5">
          <div className="text-muted">Route / D_eff</div>
          <div>
            {mc.state0?.route_name ?? "—"} / {fmt(mc.state0?.D_eff, 1)}
          </div>
        </div>
      </div>

      {mc.note && (
        <p className="px-4 pb-3 text-[10px] text-muted leading-relaxed">{mc.note}</p>
      )}
    </div>
  );
}

function Metric({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent?: boolean;
}) {
  return (
    <div className="rounded-lg bg-panel2 border border-border/60 px-3 py-2">
      <div className="text-[10px] uppercase tracking-wider text-muted">{label}</div>
      <div className={`num font-mono font-semibold text-sm ${accent ? "text-fsot" : "text-slate-100"}`}>
        {value}
      </div>
    </div>
  );
}
