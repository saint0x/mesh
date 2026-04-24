// Shared chart color tokens — keep in sync with App.css CSS variables where possible.
// Using inline hex/rgba so charts are resilient to CSS variable availability in tests.

export const chartColors = {
  reserved: '#6aa9ff',
  settled: '#66f0c0',
  released: '#b58cff',
  outstanding: '#ffbd59',
  accent: '#66f0c0',
  warm: '#ffbd59',
  cool: '#6aa9ff',
  violet: '#b58cff',
  danger: '#ff6b6b',
  grid: 'rgba(255,255,255,0.06)',
  gridStrong: 'rgba(255,255,255,0.12)',
  text: '#8b94a6',
  textStrong: '#f3f6fb',
  textMuted: '#5a6275',
  panel: '#11151c',
  panelElevated: '#161b24',
} as const

export type ChartColorToken = keyof typeof chartColors

export function formatChartNumber(n: number): string {
  const abs = Math.abs(n)
  if (abs >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (abs >= 1_000) return `${(n / 1_000).toFixed(1)}k`
  return Math.round(n).toString()
}

export function formatChartDecimal(n: number, digits = 2): string {
  if (Math.abs(n) >= 1_000) return formatChartNumber(n)
  return n.toFixed(digits)
}
