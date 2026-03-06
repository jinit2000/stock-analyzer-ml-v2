// frontend/src/env.ts
export function apiBaseUrl(): string {
  const v = import.meta.env.VITE_API_BASE_URL as string | undefined
  return (v && v.trim()) ? v.trim().replace(/\/$/, "") : "http://127.0.0.1:8000"
}