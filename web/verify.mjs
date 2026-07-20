import { chromium } from 'playwright'

const base = 'http://localhost:5173'
const pages = [
  { path: '/', name: 'today-slate' },
  { path: '/previous-games', name: 'previous-games' },
  { path: '/backtest', name: 'backtest' },
  { path: '/compare', name: 'compare' },
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
  await page.goto(base + p.path, { waitUntil: 'networkidle' })
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
  // /games/{id}/features runs live Statcast/umpire lookups server-side -
  // slow (tens of seconds), so wait for real content, not a fixed delay.
  await page.getByText('expected win %').first().waitFor({ timeout: 45000 })
  await page.getByText('Predictions', { exact: true }).waitFor({ timeout: 45000 })
  await page.waitForTimeout(500)
  await page.screenshot({ path: 'verify_game-detail.png', fullPage: true })
  console.log(`game-detail -> errors: ${errors.length ? errors.join(' | ') : 'none'}`)
} else {
  console.log('game-detail -> SKIPPED: no game cards found on slate')
}

await browser.close()
