import React, { useEffect, useMemo, useRef, useState } from 'react'
import { apiBaseUrl } from './env'
import type { AnalyzeResponse, HorizonPrediction } from './types'
import { Badge, Card, Divider, SmallLabel } from './components'

function pct(x: number) {
  return `${Math.round(x * 100)}%`
}

function money(x: unknown) {
  const n = Number(x)
  if (!Number.isFinite(n)) return '—'
  return `$${n.toFixed(2)}`
}

function labelTone(label: string | null | undefined) {
  const l = (label ?? '').toLowerCase()
  if (l.includes('buy') || l.includes('bull')) return 'good'
  if (l.includes('hold') || l.includes('neutral')) return 'ok'
  if (l.includes('avoid') || l.includes('bear') || l.includes('sell')) return 'bad'
  return 'neutral'
}

type Quote = {
  ticker: string
  price: number
  prev_close: number
  change: number
  change_pct: number
  as_of?: string
  source?: string
  note?: string
}

const SUGGESTED_TICKERS = [
  'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NFLX', 'AMD', 'INTC',
  'SPY', 'QQQ', 'DIA', 'IWM',
  'JPM', 'BAC', 'GS', 'V', 'MA',
  'XOM', 'CVX',
  'KO', 'PEP', 'COST', 'WMT',
  'DIS', 'UBER', 'SHOP',
  'NKE', 'ADBE', 'CRM',
]

function secondsAgo(ts: number | null) {
  if (!ts) return null
  return Math.max(0, Math.floor((Date.now() - ts) / 1000))
}

function HorizonCard(props: { title: string; pred: HorizonPrediction; subtitle: string }) {
  const { pred } = props
  return (
    <Card title={props.title} className="h-full">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-3xl font-bold">{pct(pred.probability)}</div>
          <SmallLabel>
            Chance of reaching ~{Math.round(pred.target_return * 100)}% in {pred.horizon_days} trading days
          </SmallLabel>
        </div>
        <Badge tone={labelTone(pred.label) as any}>{pred.label}</Badge>
      </div>

      <div className="mt-3 text-sm text-slate-300">{props.subtitle}</div>

      <Divider />

      <div className="text-sm font-semibold text-slate-200">Top reasons (explainable ML)</div>
      <ul className="mt-2 space-y-2">
        {pred.reasons.slice(0, 6).map((r, idx) => (
          <li key={idx} className="rounded-xl border border-slate-800 bg-slate-950/30 p-3">
            <div className="flex items-center justify-between gap-2">
              <div className="text-xs font-semibold text-slate-200">{r.feature}</div>
              <div className="text-xs text-slate-400">
                {r.direction === 'up' ? '↑' : r.direction === 'down' ? '↓' : ''}{' '}
                {r.contribution.toFixed(3)}
              </div>
            </div>
            <div className="mt-1 text-sm text-slate-300">{r.text}</div>
          </li>
        ))}
      </ul>
    </Card>
  )
}

export default function App() {
  const base = useMemo(() => apiBaseUrl(), [])

  const [ticker, setTicker] = useState('AAPL')
  const [loading, setLoading] = useState(false)

  const [data, setData] = useState<AnalyzeResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  const [quote, setQuote] = useState<Quote | null>(null)
  const [quoteUpdatedAt, setQuoteUpdatedAt] = useState<number | null>(null)
  const lastQuoteSymbolRef = useRef<string>('')

  const [nowTick, setNowTick] = useState(0) // updates “Updated Xs ago”
  const age = secondsAgo(quoteUpdatedAt)

  // --- Autocomplete / suggestions ---
  const inputRef = useRef<HTMLInputElement | null>(null)
  const containerRef = useRef<HTMLDivElement | null>(null)
  const [openSuggest, setOpenSuggest] = useState(false)
  const [activeIdx, setActiveIdx] = useState(0)

  const normalizedInput = ticker.trim().toUpperCase()

  const suggestions = useMemo(() => {
    if (!normalizedInput) return SUGGESTED_TICKERS.slice(0, 10)
    const starts = SUGGESTED_TICKERS.filter((t) => t.startsWith(normalizedInput))
    const contains = SUGGESTED_TICKERS.filter((t) => !t.startsWith(normalizedInput) && t.includes(normalizedInput))
    const merged = [...starts, ...contains]
    return merged.slice(0, 10)
  }, [normalizedInput])

  function chooseTicker(t: string) {
    setTicker(t)
    setOpenSuggest(false)
    setActiveIdx(0)
    // keep focus on input for quick “Enter”
    requestAnimationFrame(() => inputRef.current?.focus())
  }

  // close suggestions when clicking outside
  useEffect(() => {
    function onDocClick(e: MouseEvent) {
      const el = containerRef.current
      if (!el) return
      if (!el.contains(e.target as Node)) setOpenSuggest(false)
    }
    document.addEventListener('mousedown', onDocClick)
    return () => document.removeEventListener('mousedown', onDocClick)
  }, [])

  // update “Updated Xs ago”
  useEffect(() => {
    const id = setInterval(() => setNowTick((x) => x + 1), 1000)
    return () => clearInterval(id)
  }, [])

  async function fetchQuote(symbol: string) {
    const sym = symbol.trim().toUpperCase()
    if (!sym) return
    lastQuoteSymbolRef.current = sym

    try {
      const controller = new AbortController()
      const timer = setTimeout(() => controller.abort(), 3000)

      const res = await fetch(`${base}/quote/${encodeURIComponent(sym)}`, { signal: controller.signal })
      clearTimeout(timer)

      if (lastQuoteSymbolRef.current !== sym) return

      if (res.ok) {
        const body = (await res.json()) as Quote
        setQuote(body)
        setQuoteUpdatedAt(Date.now())
      }
    } catch {
      // ignore quote errors
    }
  }

  // ✅ Live quote refresh every 15 seconds
  useEffect(() => {
    const sym = normalizedInput
    if (!sym) return
    fetchQuote(sym)
    const id = setInterval(() => fetchQuote(sym), 15000)
    return () => clearInterval(id)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [normalizedInput, base])

  async function analyze() {
    const t = normalizedInput
    if (!t) return

    setLoading(true)
    setError(null)
    setData(null)

    try {
      const res = await fetch(`${base}/analyze/${encodeURIComponent(t)}`)
      const body = await res.json().catch(() => ({}))
      if (!res.ok) throw new Error((body as any)?.detail ?? 'Request failed')
      setData(body as AnalyzeResponse)
    } catch (e: any) {
      setError(e?.message ?? String(e))
    } finally {
      setLoading(false)
    }

    // also refresh quote after analyze (non-blocking)
    fetchQuote(t)
  }

  function onInputKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (!openSuggest) {
      if (e.key === 'Enter') analyze()
      return
    }

    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setActiveIdx((i) => Math.min(i + 1, suggestions.length - 1))
      return
    }
    if (e.key === 'ArrowUp') {
      e.preventDefault()
      setActiveIdx((i) => Math.max(i - 1, 0))
      return
    }
    if (e.key === 'Enter') {
      e.preventDefault()
      const pick = suggestions[activeIdx] ?? normalizedInput
      if (pick) chooseTicker(pick)
      analyze()
      return
    }
    if (e.key === 'Escape') {
      setOpenSuggest(false)
    }
  }

  return (
    <div className="min-h-screen">
      <header className="mx-auto max-w-6xl px-6 pt-10 pb-6">
        <div className="grid gap-6 md:grid-cols-12 md:items-start">
          {/* Left: title */}
          <div className="md:col-span-7">
            <div className="text-xs font-semibold tracking-widest text-slate-400">STOCK ANALYZER ML v2</div>
            <h1 className="mt-2 text-3xl font-bold leading-tight md:text-4xl">
              Explainable stock signals for two time windows
            </h1>
            <p className="mt-2 max-w-2xl text-slate-300">
              Enter a ticker to see probability signals for{' '}
              <span className="font-semibold text-slate-100">Short-term</span> (about 2 weeks) and{' '}
              <span className="font-semibold text-slate-100">Swing</span> (about 3 months). Educational project — not financial advice.
            </p>
          </div>

          {/* Right: search + quote */}
          <div className="md:col-span-5" ref={containerRef}>
            <Card title="Analyze a ticker">
              <div className="relative">
                <div className="flex items-center gap-2">
                  <input
                    ref={inputRef}
                    value={ticker}
                    onChange={(e) => {
                      setTicker(e.target.value)
                      setOpenSuggest(true)
                      setActiveIdx(0)
                    }}
                    onFocus={() => setOpenSuggest(true)}
                    onKeyDown={onInputKeyDown}
                    placeholder="AAPL"
                    className="w-full rounded-xl border border-slate-800 bg-slate-950/40 px-3 py-2 text-slate-100 outline-none focus:ring-2 focus:ring-slate-600"
                  />
                  <button
                    onClick={analyze}
                    disabled={loading || !normalizedInput}
                    className="rounded-xl border border-slate-700 bg-slate-200/10 px-4 py-2 text-sm font-semibold hover:bg-slate-200/15 disabled:opacity-60"
                  >
                    {loading ? 'Analyzing…' : 'Analyze'}
                  </button>
                </div>

                {/* Suggestions dropdown */}
                {openSuggest && suggestions.length > 0 ? (
                  <div className="absolute z-20 mt-2 w-full overflow-hidden rounded-xl border border-slate-800 bg-slate-950/95 shadow-xl backdrop-blur">
                    <div className="px-3 py-2 text-[11px] text-slate-400">
                      Suggestions
                    </div>
                    <div className="max-h-72 overflow-auto">
                      {suggestions.map((s, idx) => (
                        <button
                          key={s}
                          onClick={() => chooseTicker(s)}
                          onMouseEnter={() => setActiveIdx(idx)}
                          className={`flex w-full items-center justify-between px-3 py-2 text-left text-sm ${
                            idx === activeIdx ? 'bg-slate-800/60' : 'hover:bg-slate-800/40'
                          }`}
                        >
                          <span className="font-semibold text-slate-100">{s}</span>
                          <span className="text-xs text-slate-400">press Enter</span>
                        </button>
                      ))}
                    </div>
                  </div>
                ) : null}
              </div>

              <div className="mt-2 text-xs text-slate-400">
                API: <span className="font-mono text-slate-300">{base}</span>
              </div>

              {/* Live quote */}
              <div className="mt-3 rounded-xl border border-slate-800 bg-slate-950/30 p-3">
                {quote ? (
                  <>
                    <div className="flex items-center justify-between gap-3">
                      <div className="text-sm font-semibold text-slate-100">{quote.ticker}</div>
                      <div className="text-sm text-slate-200">{money(quote.price)}</div>
                    </div>

                    <div className="mt-1 flex items-center justify-between gap-3 text-xs">
                      <div className={`font-semibold ${quote.change >= 0 ? 'text-emerald-300' : 'text-rose-300'}`}>
                        {quote.change >= 0 ? '+' : ''}
                        {Number(quote.change).toFixed(2)} ({quote.change_pct >= 0 ? '+' : ''}
                        {(Number(quote.change_pct) * 100).toFixed(2)}%)
                      </div>
                      <div className="text-slate-400">Prev close: {money(quote.prev_close)}</div>
                    </div>

                    <div className="mt-2 flex items-center justify-between gap-3 text-[11px] text-slate-500">
                      <div>{quote.note ?? 'Price may be delayed depending on market/data source.'}</div>
                      <div>{age != null ? `Updated ${age}s ago` : ''}</div>
                    </div>
                  </>
                ) : (
                  <div className="text-xs text-slate-500">
                    Type a ticker to load live price (auto-refreshes every 15s).
                  </div>
                )}
              </div>
            </Card>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-6 pb-14">
        {/* Helper cards */}
        <div className="grid gap-6 md:grid-cols-12">
          <div className="md:col-span-7">
            <Card title="Time window guide">
              <div className="space-y-3 text-sm text-slate-300">
                <p>
                  A <span className="font-semibold text-slate-100">time window</span> is how far ahead the model looks.
                  For example, “10 trading days” asks whether the stock reaches the target move within about two weeks.
                </p>
                <p>
                  <span className="font-semibold text-slate-100">Short-term</span> is optimized for quicker moves and can be more selective.
                  <span className="font-semibold text-slate-100"> Swing</span> focuses on multi-week trends.
                </p>
                <p className="text-xs text-slate-400">
                  “Top reasons” shows which features pushed the probability up or down for this prediction.
                </p>
              </div>
            </Card>
          </div>

          <div className="md:col-span-5">
            <Card title="Quick demo tickers">
              <div className="flex flex-wrap gap-2">
                {['AAPL', 'MSFT', 'TSLA', 'NVDA', 'AMZN', 'GOOGL'].map((t) => (
                  <button
                    key={t}
                    onClick={() => chooseTicker(t)}
                    className="rounded-full border border-slate-800 bg-slate-950/20 px-3 py-1 text-xs font-semibold text-slate-200 hover:bg-slate-950/40"
                  >
                    {t}
                  </button>
                ))}
              </div>
              <div className="mt-3 text-xs text-slate-400">Tip: click one, then press Analyze.</div>
            </Card>
          </div>
        </div>

        {/* Errors */}
        {error ? (
          <div className="mt-6">
            <Card title="Error">
              <div className="text-sm text-rose-200">{error}</div>
              <div className="mt-2 text-xs text-slate-400">
                If you deployed the UI, make sure the API URL is correct and CORS is enabled on the backend.
              </div>
            </Card>
          </div>
        ) : null}

        {/* Results */}
        {data ? (
          <div className="mt-6 grid gap-6 md:grid-cols-2">
            {data.short_term ? (
              <HorizonCard
                title="Short-term signal"
                pred={data.short_term}
                subtitle="Use this for faster moves over ~2 weeks (10 trading days). It is stricter and may be unavailable if the short model is not trained."
              />
            ) : (
              <Card title="Short-term signal">
                <div className="text-sm text-slate-300">
                  Short-term model is not available on this deployment (not trained or not shipped).
                </div>
              </Card>
            )}

            <HorizonCard
              title="Swing signal"
              pred={data.swing}
              subtitle="Use this for multi-week trend signals over ~3 months (60 trading days)."
            />
          </div>
        ) : (
          <div className="mt-6 text-sm text-slate-400">Run an analysis to see results.</div>
        )}

        <div className="mt-10 text-xs text-slate-500">Disclaimer: educational project. No financial advice.</div>
      </main>
    </div>
  )
}