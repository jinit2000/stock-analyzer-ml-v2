import React from 'react'

export function Card(props: { title?: string; children: React.ReactNode; className?: string }) {
  return (
    <div className={'rounded-2xl border border-slate-800 bg-slate-900/60 p-5 shadow-sm ' + (props.className ?? '')}>
      {props.title ? <div className="mb-3 text-sm font-semibold text-slate-200">{props.title}</div> : null}
      {props.children}
    </div>
  )
}

export function Badge(props: { children: React.ReactNode; tone?: 'good'|'ok'|'bad'|'neutral' }) {
  const tone = props.tone ?? 'neutral'
  const cls =
    tone === 'good' ? 'bg-emerald-500/15 text-emerald-200 border-emerald-700/40' :
    tone === 'bad' ? 'bg-rose-500/15 text-rose-200 border-rose-700/40' :
    tone === 'ok' ? 'bg-amber-500/15 text-amber-200 border-amber-700/40' :
    'bg-slate-500/15 text-slate-200 border-slate-700/40'
  return <span className={'inline-flex items-center rounded-full border px-2.5 py-1 text-xs font-medium ' + cls}>{props.children}</span>
}

export function Divider() {
  return <div className="my-4 h-px bg-slate-800" />
}

export function SmallLabel(props: { children: React.ReactNode }) {
  return <div className="text-xs text-slate-400">{props.children}</div>
}
