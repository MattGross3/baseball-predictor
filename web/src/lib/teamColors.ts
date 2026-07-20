// Approximate primary brand color per team, keyed by the abbreviation our
// DB stores (from the MLB Stats API - see database/models.py Team.abbreviation).
// Used only for the small badge circle on game rows; not a claim of exact
// official brand hex values.
export const TEAM_COLORS: Record<string, string> = {
  ATH: '#003831',
  ATL: '#13274F',
  AZ: '#A71930',
  BAL: '#DF4601',
  BOS: '#BD3039',
  CHC: '#0E3386',
  CIN: '#C6011F',
  CLE: '#00385D',
  COL: '#33006F',
  CWS: '#27251F',
  DET: '#0C2340',
  HOU: '#002D62',
  KC: '#004687',
  LAA: '#BA0021',
  LAD: '#005A9C',
  MIA: '#00A3E0',
  MIL: '#12284B',
  MIN: '#002B5C',
  NYM: '#002D72',
  NYY: '#003087',
  PHI: '#E81828',
  PIT: '#27251F',
  SD: '#2F241D',
  SEA: '#0C2C56',
  SF: '#FD5A1E',
  STL: '#C41E3A',
  TB: '#092C5C',
  TEX: '#003278',
  TOR: '#134A8E',
  WSH: '#AB0003',
}

export function teamColor(abbr: string): string {
  return TEAM_COLORS[abbr] ?? '#64748b'
}
