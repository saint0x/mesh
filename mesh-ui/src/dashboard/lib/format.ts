export function formatInteger(value: number): string {
  return new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 }).format(value)
}

export function formatDecimal(value: number, maximumFractionDigits = 1): string {
  return new Intl.NumberFormat('en-US', { maximumFractionDigits }).format(value)
}

export function formatPercent(value: number): string {
  return `${formatDecimal(value, 1)}%`
}

export function formatLatency(value: number): string {
  return `${formatInteger(value)} ms`
}

export function formatBytes(value: number): string {
  if (value >= 1024 ** 3) return `${formatDecimal(value / 1024 ** 3, 2)} GB`
  if (value >= 1024 ** 2) return `${formatDecimal(value / 1024 ** 2, 2)} MB`
  if (value >= 1024) return `${formatDecimal(value / 1024, 2)} KB`
  return `${formatInteger(value)} B`
}

type ChartValue = number | string | undefined | readonly (number | string)[]

function normalizeChartValue(value: ChartValue): number | string | undefined {
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'undefined') return value
  return value[0]
}

export function formatChartNumber(value: ChartValue): string {
  const normalized = normalizeChartValue(value)
  if (typeof normalized === 'number') return formatInteger(normalized)
  return normalized ?? ''
}

export function formatChartPercent(value: ChartValue): string {
  const normalized = normalizeChartValue(value)
  if (typeof normalized === 'number') return formatPercent(normalized)
  return normalized ?? ''
}

export function formatChartDecimal(value: ChartValue): string {
  const normalized = normalizeChartValue(value)
  if (typeof normalized === 'number') return formatDecimal(normalized)
  return normalized ?? ''
}
