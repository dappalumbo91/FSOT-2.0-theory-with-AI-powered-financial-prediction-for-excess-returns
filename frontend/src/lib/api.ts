const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  return res.json() as Promise<T>;
}

export type Watchlist = {
  indices: AssetMeta[];
  stocks: AssetMeta[];
  crypto: AssetMeta[];
  defaults: Record<string, unknown>;
};

export type AssetMeta = {
  symbol: string;
  name: string;
  class?: string;
  yahoo?: string;
  coingecko?: string;
};

export type OhlcvBar = {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
};

export type Quote = {
  symbol: string;
  name: string;
  class?: string;
  price: number;
  change_pct: number;
  volume: number;
  high: number;
  low: number;
  as_of: string;
};

export type Prediction = {
  S: number;
  T1: number;
  T2: number;
  T3: number;
  K?: number;
  dS?: number;
  entropy?: number;
  emergence_score?: number;
  base_S: number;
  base_S_note?: string;
  residual?: number;
  score: number;
  signal: "LONG" | "SHORT" | "FLAT";
  confidence: number;
  regime: string;
  observer_mod?: number;
  method?: string;
  pred_return: number;
  last_price: number;
  pred_price_1d: number;
  horizons: Record<
    string,
    { horizon_days: number; S: number; pred_return: number; score?: number; emergence_score?: number }
  >;
  meta: {
    domain: string;
    window: number;
    features: Record<string, number>;
    params: Record<string, number | boolean>;
  };
  series_tail: {
    time: number | null;
    close: number | null;
    S: number | null;
    dS?: number | null;
    entropy?: number | null;
    emergence_score?: number | null;
    pred_return: number;
  }[];
  forecast_path: { time: number | null; value: number; S: number | null }[];
};

export type PredictResponse = {
  symbol: string;
  name: string;
  class?: string;
  quote: Quote;
  prediction: Prediction;
};

export type MonteCarloPatternBias = {
  solidified: number;
  strength: number;
  dir: number;
  mu_scale: number;
  collapse_boost: number;
  path_weight: number;
  acc_phi: number;
  trials: number;
};

export type MonteCarloResult = {
  error: string | null;
  method: string;
  free_parameters: number;
  symbol?: string;
  price0: number;
  horizon: number;
  n_paths: number;
  sentiment: number;
  dynamic?: boolean;
  pattern?: {
    key: string;
    bias: MonteCarloPatternBias;
    memory_summary?: {
      n_patterns: number;
      n_solidified: number;
      n_updates: number;
      refinement?: {
        early_acc_phi: number;
        late_acc_phi: number;
        final_acc_phi: number;
        lift: number;
      } | null;
      top_patterns?: {
        key: string;
        solidified: boolean;
        strength: number;
        acc_phi: number;
        trials: number;
        preferred_dir: number;
      }[];
    } | null;
  };
  training?: {
    raw_directional_accuracy?: number | null;
    anchored_directional_accuracy?: number | null;
    anchored_early?: number;
    anchored_late?: number;
    refinement_lift?: number;
    new_solidifications?: number;
  };
  state0?: {
    mu: number;
    sig: number;
    S_live: number;
    S_base: number;
    d_observer: number;
    growth: number;
    quirk_mod: number;
    consciousness_factor: number;
    D_eff: number;
    route_name: string;
    delta_psi: number;
  };
  ensemble: {
    p_up: number;
    p_down: number;
    p_up_observed_branch: number;
    expected_return: number;
    expected_return_observed_branch: number;
    median_return: number;
    mean_terminal_price: number;
    median_terminal_price: number;
    quantiles_price: {
      p05: number;
      p10: number;
      p25: number;
      p50: number;
      p75: number;
      p90: number;
      p95: number;
    };
    most_probable_return_bin: {
      low: number;
      high: number;
      center: number;
      count: number;
    };
    mean_collapse_true_fraction: number;
    observed_branch_path_fraction: number;
  };
  signal: "LONG" | "SHORT" | "FLAT" | string;
  confidence: number;
  fan_chart: { step: number; p10: number; p50: number; p90: number }[];
  note?: string;
};

export type MonteCarloResponse = {
  symbol: string;
  name?: string;
  quote: Quote;
  observer_mod: number;
  monte_carlo: MonteCarloResult;
  walkforward?: {
    directional_accuracy?: number | null;
    directional_accuracy_confident?: number | null;
    n_eval?: number;
    n_confident?: number;
    error?: string | null;
  };
};

export type BatchItem = {
  symbol: string;
  name?: string;
  class?: string;
  price?: number;
  change_pct?: number;
  signal?: string;
  confidence?: number;
  S?: number;
  pred_return?: number;
  regime?: string;
  error?: string | null;
};

export type Backtest = {
  n_bars: number;
  directional_accuracy: number;
  sharpe: number;
  max_drawdown: number;
  buy_hold_return: number;
  strategy_return: number;
  hit_rate_long: number;
  hit_rate_short: number;
  note: string;
};

export type Health = {
  status: string;
  version: string;
  fsot: {
    formula: string;
    K: number;
    economics_S: number;
    domain: Record<string, unknown>;
  };
};

export const api = {
  health: () => get<Health>("/api/health"),
  watchlist: () => get<Watchlist>("/api/watchlist"),
  ohlcv: (symbol: string, range = "1y") =>
    get<{ bars: OhlcvBar[]; symbol: string; name?: string }>(
      `/api/market/${encodeURIComponent(symbol)}/ohlcv?range=${range}`
    ),
  quote: (symbol: string) => get<Quote>(`/api/market/${encodeURIComponent(symbol)}/quote`),
  predict: (symbol: string, range = "1y") =>
    get<PredictResponse>(`/api/predict/${encodeURIComponent(symbol)}?range=${range}`),
  monteCarlo: (
    symbol: string,
    opts?: { range?: string; horizon?: number; n_paths?: number; walkforward?: boolean }
  ) => {
    const range = opts?.range ?? "2y";
    const horizon = opts?.horizon ?? 21;
    const n_paths = opts?.n_paths ?? 512;
    const wf = opts?.walkforward ? "&walkforward=true" : "";
    return get<MonteCarloResponse>(
      `/api/predict/${encodeURIComponent(symbol)}/montecarlo?range=${range}&horizon=${horizon}&n_paths=${n_paths}${wf}`
    );
  },
  batch: (section?: string) =>
    get<{ count: number; items: BatchItem[] }>(
      section ? `/api/predict/batch?section=${section}` : "/api/predict/batch"
    ),
  backtest: (symbol: string, range = "2y") =>
    get<{ symbol: string; backtest: Backtest }>(
      `/api/backtest/${encodeURIComponent(symbol)}?range=${range}`
    ),
  news: (limit = 40) => get<{ count: number; items: NewsItem[] }>(`/api/news?limit=${limit}`),
  observer: (symbol?: string) =>
    get<ObserverPayload>(
      symbol ? `/api/news/observer?symbol=${encodeURIComponent(symbol)}` : "/api/news/observer"
    ),
  historyStatus: () => get<Record<string, unknown>>("/api/history/status"),
  paper: (
    symbol: string,
    opts?: { capital?: number; mode?: string; range?: string }
  ) => {
    const capital = opts?.capital ?? 10000;
    const mode = opts?.mode ?? "solid_gated";
    const range = opts?.range ?? "2y";
    return get<PaperResponse>(
      `/api/paper/${encodeURIComponent(symbol)}?capital=${capital}&mode=${mode}&range=${range}`
    );
  },
};

export type PaperPortfolioResult = {
  error: string | null;
  method: string;
  free_parameters: number;
  mode: string;
  capital_start: number;
  capital_end: number;
  total_pnl: number;
  total_return: number;
  max_drawdown: number;
  max_drawdown_dollars: number;
  sharpe: number;
  buy_hold_return: number;
  buy_hold_final: number;
  buy_hold_pnl: number;
  vs_buy_hold_pnl: number;
  trades: number;
  wins: number;
  losses: number;
  win_rate: number | null;
  n_long_bars: number;
  n_short_bars: number;
  n_flat_bars: number;
  pct_time_in_market: number;
  commit_directional_accuracy?: number | null;
  progress_to_70_80?: number | null;
  pct_hold?: number | null;
  hold_horizon?: number;
  n_buy?: number;
  n_sell?: number;
  n_hold?: number;
  equity_curve: { t: string; equity: number; position: number; action?: string }[];
  note?: string;
};

export type PaperResponse = {
  symbol: string;
  name?: string;
  range: string;
  paper: PaperPortfolioResult;
};

export type NewsItem = {
  source: string;
  title: string;
  link: string;
  published?: string;
  sentiment: number;
};

export type ObserverPayload = {
  observer_mod: number;
  n_headlines: number;
  mean_sentiment: number;
  bull_share: number;
  bear_share: number;
  top: NewsItem[];
  as_of: string;
};

export { API_BASE };
