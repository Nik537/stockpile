import { create } from 'zustand'
import { BackgroundJob, BackgroundJobType, BackgroundJobWSMessage } from '../types'
import { getWsUrl } from '../config'

interface JobQueueState {
  jobs: BackgroundJob[]
  addJob: (job: BackgroundJob) => void
  updateJob: (id: string, updates: Partial<BackgroundJob>) => void
  removeJob: (id: string) => void
  clearCompleted: () => void
  connectWebSocket: (jobId: string, type: BackgroundJobType) => void
  disconnectWebSocket: (jobId: string) => void
}

// Module-level WebSocket map (not in Zustand state to avoid serialization issues)
const activeWebSockets = new Map<string, WebSocket>()

export const useJobQueueStore = create<JobQueueState>((set, get) => ({
  jobs: [],

  addJob: (job) =>
    set((state) => ({ jobs: [...state.jobs, job] })),

  updateJob: (id, updates) =>
    set((state) => ({
      jobs: state.jobs.map((j) => (j.id === id ? { ...j, ...updates } : j)),
    })),

  removeJob: (id) => {
    // Also disconnect WS if active
    const ws = activeWebSockets.get(id)
    if (ws) {
      ws.close()
      activeWebSockets.delete(id)
    }
    set((state) => ({ jobs: state.jobs.filter((j) => j.id !== id) }))
  },

  clearCompleted: () =>
    set((state) => ({
      jobs: state.jobs.filter((j) => j.status === 'processing'),
    })),

  connectWebSocket: (jobId, type) => {
    // Don't open duplicate connections
    if (activeWebSockets.has(jobId)) return

    // image-edit uses the same WS path as image
    const wsType = type === 'image-edit' ? 'image' : type
    const wsUrl = getWsUrl(`/ws/${wsType}/${jobId}`)

    const ws = new WebSocket(wsUrl)
    activeWebSockets.set(jobId, ws)

    ws.onmessage = (event) => {
      try {
        const msg: BackgroundJobWSMessage = JSON.parse(event.data)

        if (msg.type === 'complete') {
          get().updateJob(jobId, {
            status: 'completed',
            completedAt: new Date().toISOString(),
            result: msg.result,
          })
          get().disconnectWebSocket(jobId)
        } else if (msg.type === 'error') {
          get().updateJob(jobId, {
            status: 'failed',
            completedAt: new Date().toISOString(),
            error: msg.error || msg.message || 'Unknown error',
          })
          get().disconnectWebSocket(jobId)
        } else if (msg.type === 'status' && msg.status) {
          get().updateJob(jobId, { status: msg.status })
        }
      } catch {
        // Ignore non-JSON messages
      }
    }

    ws.onerror = () => {
      get().updateJob(jobId, {
        status: 'failed',
        completedAt: new Date().toISOString(),
        error: 'WebSocket connection error',
      })
      get().disconnectWebSocket(jobId)
    }

    ws.onclose = () => {
      activeWebSockets.delete(jobId)
    }
  },

  disconnectWebSocket: (jobId) => {
    const ws = activeWebSockets.get(jobId)
    if (ws) {
      ws.close()
      activeWebSockets.delete(jobId)
    }
  },
}))
