"use client";

import { useEffect, useRef } from "react";
import {
  createChart,
  type IChartApi,
  type ISeriesApi,
  type CandlestickData,
  type LineData,
  type HistogramData,
  type Time,
  ColorType,
  CrosshairMode,
} from "lightweight-charts";
import type { OhlcvBar } from "@/lib/api";

type Props = {
  bars: OhlcvBar[];
  signalMarkers?: { time: number; signal: string }[];
  height?: number;
};

export default function ChartPanel({ bars, height = 420 }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const volRef = useRef<ISeriesApi<"Histogram"> | null>(null);
  const smaRef = useRef<ISeriesApi<"Line"> | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const chart = createChart(containerRef.current, {
      height,
      layout: {
        background: { type: ColorType.Solid, color: "#141a22" },
        textColor: "#8b9bb4",
        fontFamily: "Inter, system-ui, sans-serif",
      },
      grid: {
        vertLines: { color: "rgba(36, 48, 65, 0.6)" },
        horzLines: { color: "rgba(36, 48, 65, 0.6)" },
      },
      crosshair: { mode: CrosshairMode.Normal },
      rightPriceScale: { borderColor: "#243041" },
      timeScale: { borderColor: "#243041", timeVisible: true },
    });

    const candles = chart.addCandlestickSeries({
      upColor: "#10b981",
      downColor: "#f43f5e",
      borderUpColor: "#10b981",
      borderDownColor: "#f43f5e",
      wickUpColor: "#10b981",
      wickDownColor: "#f43f5e",
    });

    const volume = chart.addHistogramSeries({
      priceFormat: { type: "volume" },
      priceScaleId: "vol",
    });
    chart.priceScale("vol").applyOptions({
      scaleMargins: { top: 0.8, bottom: 0 },
    });

    const sma = chart.addLineSeries({
      color: "#22d3ee",
      lineWidth: 1,
      priceLineVisible: false,
      lastValueVisible: false,
    });

    chartRef.current = chart;
    candleRef.current = candles;
    volRef.current = volume;
    smaRef.current = sma;

    const ro = new ResizeObserver(() => {
      if (containerRef.current) {
        chart.applyOptions({ width: containerRef.current.clientWidth });
      }
    });
    ro.observe(containerRef.current);

    return () => {
      ro.disconnect();
      chart.remove();
      chartRef.current = null;
    };
  }, [height]);

  useEffect(() => {
    if (!candleRef.current || !volRef.current || !smaRef.current || !bars?.length) return;

    const candleData: CandlestickData[] = bars.map((b) => ({
      time: b.time as Time,
      open: b.open,
      high: b.high,
      low: b.low,
      close: b.close,
    }));

    const volData: HistogramData[] = bars.map((b) => ({
      time: b.time as Time,
      value: b.volume,
      color: b.close >= b.open ? "rgba(16,185,129,0.35)" : "rgba(244,63,94,0.35)",
    }));

    // 20-period SMA
    const smaData: LineData[] = [];
    for (let i = 0; i < bars.length; i++) {
      if (i < 19) continue;
      const slice = bars.slice(i - 19, i + 1);
      const avg = slice.reduce((s, x) => s + x.close, 0) / 20;
      smaData.push({ time: bars[i].time as Time, value: avg });
    }

    candleRef.current.setData(candleData);
    volRef.current.setData(volData);
    smaRef.current.setData(smaData);
    chartRef.current?.timeScale().fitContent();
  }, [bars]);

  return (
    <div className="rounded-xl border border-border bg-panel overflow-hidden">
      <div className="flex items-center justify-between px-4 py-2 border-b border-border">
        <span className="text-xs uppercase tracking-wider text-muted">Price · Volume · SMA20 (cyan)</span>
        <span className="text-xs text-fsot/80 font-mono">lightweight-charts</span>
      </div>
      <div ref={containerRef} className="w-full" style={{ height }} />
    </div>
  );
}
