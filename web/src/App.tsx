import { Route, Routes } from 'react-router-dom'
import { Layout } from './components/Layout'
import { Backtest } from './pages/Backtest'
import { GameDetail } from './pages/GameDetail'
import { ModelComparison } from './pages/ModelComparison'
import { PreviousGames } from './pages/PreviousGames'
import { TodaySlate } from './pages/TodaySlate'

function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<TodaySlate />} />
        <Route path="games/:gameId" element={<GameDetail />} />
        <Route path="previous-games" element={<PreviousGames />} />
        <Route path="backtest" element={<Backtest />} />
        <Route path="compare" element={<ModelComparison />} />
      </Route>
    </Routes>
  )
}

export default App
