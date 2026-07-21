import { chromium } from 'playwright'

const base = 'http://localhost:5173'
const pages = [
  { path: '/', name: 'today-slate' },
  { path: '/previous-games', name: 'previous-games' },
  { path: '/backtest', name: 'backtest' },
  { path: '/compare', name: 'compare' },
  { path: '/roi', name: 'roi' },
  { path: '/models', name: 'models' },
]

const browser = await chromium.launch()
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } })

const errors = []
page.on('console', (msg) => {
  if (msg.type() === 'error') errors.push(`[console] ${msg.text()}`)
})
page.on('pageerror', (err) => errors.push(`[pageerror] ${err.message}`))

for (const p of pages) {
  errors.length = 0
  // /backtest and /compare both trigger a live backtest run server-side
  // (full feature rebuild per game) - the default 30s nav timeout is too
  // tight for that even with the umpire/pitcher Statcast season caches
  // warm, so give those two more room. /roi fires 4 of those calls
  // concurrently (moneyline/total/spread/nrfi) for a full season each,
  // uncached the first time this exact range is ever requested, so it
  // gets the longest allowance. The page itself shows a loading state
  // well within these timeouts; this is purely the test's own patience.
  const timeout = p.path === '/roi' ? 300000 : p.path === '/backtest' || p.path === '/compare' ? 60000 : 30000
  await page.goto(base + p.path, { waitUntil: 'networkidle', timeout })
  await page.waitForTimeout(1500)
  await page.screenshot({ path: `verify_${p.name}.png`, fullPage: true })
  console.log(`${p.path} -> errors: ${errors.length ? errors.join(' | ') : 'none'}`)
}

// Click through to a game detail page from the slate
await page.goto(base + '/', { waitUntil: 'networkidle' })
await page.waitForTimeout(1500)
const firstCard = page.locator('a[href^="/games/"]').first()
if (await firstCard.count()) {
  errors.length = 0
  await firstCard.click()
  await page.waitForURL(/\/games\/\d+/)
  // Predictions tab is already-computed DB data - should render fast now
  // that it no longer waits on the Feature breakdown tab's live fetch.
  await page.getByText('Predicted total').first().waitFor({ timeout: 10000 })
  await page.getByText('Predictions', { exact: true }).waitFor({ timeout: 10000 })
  await page.waitForTimeout(500)
  await page.screenshot({ path: 'verify_game-detail.png', fullPage: true })
  console.log(`game-detail -> errors: ${errors.length ? errors.join(' | ') : 'none'}`)

  // Feature breakdown is fetched lazily only once this tab is opened -
  // first click can be slow (live Statcast), so wait for real content.
  errors.length = 0
  await page.getByRole('button', { name: 'Feature breakdown' }).click()
  await page.getByText('Starter').first().waitFor({ timeout: 45000 })
  await page.waitForTimeout(500)
  await page.screenshot({ path: 'verify_game-detail-features.png', fullPage: true })
  console.log(`game-detail feature-breakdown -> errors: ${errors.length ? errors.join(' | ') : 'none'}`)
} else {
  console.log('game-detail -> SKIPPED: no game cards found on slate')
}

await browser.close()
