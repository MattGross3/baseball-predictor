/** YYYY-MM-DD in the browser's local timezone.
 *
 * `Date.toISOString()` converts to UTC first - in US evening hours that's
 * already tomorrow's UTC date, so "today" silently became "tomorrow" in
 * the date picker default. Games/predictions are keyed by the date MLB
 * itself assigns (also not UTC), so local-component formatting is the
 * correct match, not an approximation.
 */
export function localIsoDate(date: Date): string {
  const year = date.getFullYear()
  const month = String(date.getMonth() + 1).padStart(2, '0')
  const day = String(date.getDate()).padStart(2, '0')
  return `${year}-${month}-${day}`
}

export function localIsoDaysAgo(days: number): string {
  const d = new Date()
  d.setDate(d.getDate() - days)
  return localIsoDate(d)
}
