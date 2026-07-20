// Mirrors api/schemas.py - keep in sync by hand (small enough surface
// that generating an OpenAPI client isn't worth the build-step complexity
// yet; revisit if the schema grows).

export interface Team {
  id: number
  name: string
  abbreviation: string
  league: string
  division: string
}

export interface Venue {
  id: number
  name: string
  city: string | null
  park_factor_runs: number
  park_factor_hr: number
  roof_type: string | null
}

export interface Game {
  id: number
  mlb_game_id: number
  date: string
  start_time: string | null
  status: 'scheduled' | 'live' | 'final' | 'postponed' | 'cancelled'
  home_team: Team
  away_team: Team
  venue: Venue | null
  home_score: number | null
  away_score: number | null
  is_doubleheader: boolean
  game_number_in_series: number
}

export type TargetType = 'moneyline' | 'total' | 'nrfi' | 'prop_hr' | 'prop_hits' | 'prop_strikeouts'

export interface Prediction {
  id: number
  game_id: number
  model_name: string
  model_version: string
  target_type: TargetType
  predicted_value: number | null
  predicted_probability: number | null
  created_at: string
}

export interface GamePredictions {
  game_id: number
  predictions: Prediction[]
  edge_vs_market: { model_probability_home: number; market_implied_probability_home: number; edge: number } | null
}

// features/build_feature_matrix.py's nested dict - only the shapes the UI
// actually reads are typed strictly; the rest flows through as unknown so
// a new feature field doesn't require a frontend change to compile.
export interface StarterFeatures {
  era_season: number | null
  fip_season: number | null
  siera_season: number | null
  era_last_3_starts: number | null
  k_pct_rolling: number | null
  bb_pct_rolling: number | null
  velo_trend_last_3: number | null
  days_rest: number | null
  pitch_count_last_start: number | null
  home_away_split_era: { home: number | null; away: number | null }
  vs_opponent_career_era: number | null
  handedness: string | null
}

export interface BullpenFeatures {
  bullpen_era_rolling_7d: number | null
  bullpen_era_rolling_14d: number | null
  innings_thrown_last_3_games: number | null
  closer_available: boolean | null
  bullpen_hand_distribution: { L: number | null; R: number | null }
}

export interface TeamFormFeatures {
  win_pct_season: number | null
  win_pct_last_10: number | null
  run_diff_season: number | null
  pythag_win_pct: number | null
  home_away_win_pct: { home: number | null; away: number | null }
  oaa_defense_rating: number | null
}

export interface LineupFeatures {
  lineup_wOBA_weighted_by_order: number | null
  platoon_advantage_count: number | null
  hot_streak_players: { player_id: number; z_score: number }[]
  lineup_confirmed: boolean
}

export interface ParkFeatures {
  park_factor_runs: number | null
  park_factor_hr: number | null
  temp_f: number | null
  wind_out_mph: number | null
  roof_closed: boolean | null
}

export interface UmpireFeatures {
  strike_zone_size_percentile: number | null
  over_under_lean: number | null
  k_rate_boost: number | null
}

export interface GameFeatures {
  game_id: number
  home_starter: StarterFeatures
  away_starter: StarterFeatures
  home_bullpen: BullpenFeatures
  away_bullpen: BullpenFeatures
  home_team: TeamFormFeatures
  away_team: TeamFormFeatures
  home_lineup: LineupFeatures
  away_lineup: LineupFeatures
  park_weather: ParkFeatures
  umpire: UmpireFeatures
}

export interface GameFeaturesResponse {
  game_id: number
  features: GameFeatures
}

export interface BacktestResult {
  model: string
  target_type: string
  date_range: string
  accuracy: number | null
  log_loss: number | null
  brier_score: number | null
  roi_flat_bet: number | null
  roi_kelly: number | null
  clv_avg: number | null
  n_bets: number
  mae?: number
  rmse?: number
}
