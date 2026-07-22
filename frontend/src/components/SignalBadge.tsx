"use client";

type Props = {
  signal: string;
  confidence?: number;
  size?: "sm" | "md" | "lg";
};

export default function SignalBadge({ signal, confidence, size = "md" }: Props) {
  const s = (signal || "FLAT").toUpperCase();
  const colors =
    s === "LONG"
      ? "bg-long/15 text-long border-long/40"
      : s === "SHORT"
        ? "bg-short/15 text-short border-short/40"
        : "bg-slate-500/15 text-slate-300 border-slate-500/40";

  const pad = size === "lg" ? "px-4 py-2 text-base" : size === "sm" ? "px-2 py-0.5 text-xs" : "px-3 py-1 text-sm";

  return (
    <span
      className={`inline-flex items-center gap-2 rounded-full border font-semibold tracking-wide ${colors} ${pad}`}
    >
      <span
        className={`h-2 w-2 rounded-full ${
          s === "LONG" ? "bg-long" : s === "SHORT" ? "bg-short" : "bg-slate-400"
        }`}
      />
      {s}
      {typeof confidence === "number" && (
        <span className="opacity-70 font-mono text-[0.85em]">{(confidence * 100).toFixed(0)}%</span>
      )}
    </span>
  );
}
