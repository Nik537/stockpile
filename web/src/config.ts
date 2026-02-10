/**
 * Shared API configuration.
 *
 * In development: Vite proxy handles /api/* and /ws/* -> localhost:8000
 * In production: Set VITE_API_BASE to your backend URL (e.g. https://api.example.com)
 */
export const API_BASE = import.meta.env.VITE_API_BASE ?? ''

/** Build a WebSocket URL that respects API_BASE. */
export function getWsUrl(path: string): string {
  if (API_BASE) {
    const protocol = API_BASE.startsWith('https') ? 'wss:' : 'ws:'
    const host = API_BASE.replace(/^https?:\/\//, '')
    return `${protocol}//${host}${path}`
  }
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${protocol}//${window.location.host}${path}`
}
