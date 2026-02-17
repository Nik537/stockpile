/**
 * Detect if the app is running inside a Tauri v2 webview
 */
export function isTauri(): boolean {
  return typeof window !== 'undefined' && '__TAURI_INTERNALS__' in window;
}
