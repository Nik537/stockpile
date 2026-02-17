import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.tsx'
import './index.css'
import { isTauri } from './lib/tauri'
import { setBackendPort } from './config'

async function init() {
  if (isTauri()) {
    const { invoke } = await import('@tauri-apps/api/core')
    const { listen } = await import('@tauri-apps/api/event')

    // Try to get port immediately (backend may already be ready)
    try {
      const port = await invoke<number>('get_backend_port')
      if (port > 0) {
        setBackendPort(port)
      }
    } catch {
      // Backend not ready yet, wait for event
    }

    // Listen for backend-ready event
    listen<{ port: number }>('backend-ready', (event) => {
      setBackendPort(event.payload.port)
    })
  }

  ReactDOM.createRoot(document.getElementById('root')!).render(
    <React.StrictMode>
      <App />
    </React.StrictMode>,
  )
}

init()
