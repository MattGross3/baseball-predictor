import type { Prediction, TargetType } from '../api/types'

// Mirrors models/predict.py's PREFERRED_MODEL_BY_TARGET - since a game can
// now have a stored prediction from *every* trained model family for the
// same target (needed for Model Comparison's blend), display code needs
// to deliberately pick the same "headline" model the backend prefers,
// not just grab whichever row happens to sort first.
const PREFERRED_MODEL_BY_TARGET: Partial<Record<TargetType, string[]>> = {
  moneyline: ['moneyline_xgboost', 'moneyline_logistic'],
  nrfi: ['nrfi_logistic', 'nrfi_xgboost'],
  total: ['totals_xgboost', 'totals_poisson'],
}

/** The one prediction Today's Slate / Game Detail should display for a target. */
export function preferredPrediction(predictions: Prediction[], target: TargetType): Prediction | undefined {
  const candidates = predictions.filter((p) => p.target_type === target)
  for (const name of PREFERRED_MODEL_BY_TARGET[target] ?? []) {
    const match = candidates.find((p) => p.model_name === name)
    if (match) return match
  }
  return candidates[0]
}
