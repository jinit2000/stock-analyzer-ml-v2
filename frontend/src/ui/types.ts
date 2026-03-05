export type Reason = {
  feature: string
  contribution: number
  direction: 'up' | 'down' | string
  text: string
}

export type HorizonPrediction = {
  horizon_days: number
  target_return: number
  probability: number
  label: string
  reasons: Reason[]
}

export type AnalyzeResponse = {
  ticker: string
  as_of_date: string
  short_term: HorizonPrediction | null
  swing: HorizonPrediction
}
