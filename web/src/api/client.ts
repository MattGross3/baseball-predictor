import type { BacktestResult, Game, GameFeaturesResponse, GamePredictions, GameSlateSummary, ModelInfo, OddsRefreshResult, Prediction } from './types'

// In dev, Vite proxies /api/* to the FastAPI backend (see vite.config.ts) -
// same trick nginx.conf uses in production (see web/Dockerfile). Neither
// the browser nor this code ever needs CORS or a hardcoded backend host.
const BASE = '/api'

class ApiError extends Error {
  status: number

  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function get<T>(path: string, params?: Record<string, string | undefined>): Promise<T> {
  const url = new URL(BASE + path, window.location.origin)
  if (params) {
    for (const [key, value] of Object.entries(params)) {
      if (value !== undefined) url.searchParams.set(key, value)
    }
  }
  const res = await fetch(url.toString())
  if (!res.ok) {
    const body = await res.text().catch(() => '')
    throw new ApiError(res.status, body || res.statusText)
  }
  return res.json() as Promise<T>
}

async function post<T>(path: string): Promise<T> {
  const res = await fetch(new URL(BASE + path, window.location.origin).toString(), { method: 'POST' })
  if (!res.ok) {
    const body = await res.text().catch(() => '')
    throw new ApiError(res.status, body || res.statusText)
  }
  return res.json() as Promise<T>
}

export const api = {
  gamesToday: (date?: string) => get<Game[]>('/games/today', { date }),
  getGameSlateSummary: (date?: string) => get<GameSlateSummary[]>('/games/today/summary', { date }),
  getGame: (id: number) => get<Game>(`/games/${id}`),
  getGameFeatures: (id: number, refresh?: boolean) =>
    get<GameFeaturesResponse>(`/games/${id}/features`, refresh ? { refresh: 'true' } : undefined),
  getGamePredictions: (id: number) => get<GamePredictions>(`/games/${id}/predictions`),
  predictionHistory: (dateRange: string, targetType?: string) =>
    get<Prediction[]>('/predictions/history', { date_range: dateRange, target_type: targetType }),
  backtestResults: (model: string, dateRange: string, refresh?: boolean) =>
    get<BacktestResult>('/backtest/results', { model, date_range: dateRange, refresh: refresh ? 'true' : undefined }),
  backtestSeasons: () => get<number[]>('/backtest/seasons'),
  listModels: () => get<ModelInfo[]>('/models'),
  refreshOdds: () => post<OddsRefreshResult>('/odds/refresh'),
}

export { ApiError }
