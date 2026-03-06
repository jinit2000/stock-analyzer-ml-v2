export function apiBaseUrl(): string {
  const v = import.meta.env.VITE_API_BASE_URL as string | undefined
  return (v && v.trim()) ? v.trim().replace(/\/$/, '') : 'https://stock-analyzer-ml-v2.onrender.com'
}
