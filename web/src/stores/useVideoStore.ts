import { create } from 'zustand'
import { VideoJob, VideoProduceParams, VideoWSMessage } from '../types'
import { API_BASE, getWsUrl } from '../config'

const activeWebSockets = new Map<string, WebSocket>()

interface VideoState {
  jobs: VideoJob[]
  activeJobId: string | null
  submitting: boolean
  error: string | null
  fetchJobs: () => Promise<void>
  startProduction: (params: VideoProduceParams) => Promise<string | null>
  deleteJob: (jobId: string) => Promise<void>
  connectWebSocket: (jobId: string) => void
  disconnectWebSocket: (jobId: string) => void
}

export const useVideoStore = create<VideoState>((set, get) => ({
  jobs: [],
  activeJobId: null,
  submitting: false,
  error: null,

  fetchJobs: async () => {
    try {
      const response = await fetch(`${API_BASE}/api/video/jobs`)
      if (response.ok) {
        const data: VideoJob[] = await response.json()
        set({ jobs: data })
      }
    } catch (e) {
      console.error('Failed to fetch video jobs:', e)
    }
  },

  startProduction: async (params) => {
    set({ submitting: true, error: null })
    try {
      const response = await fetch(`${API_BASE}/api/video/produce`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      })

      if (!response.ok) {
        const errorData = await response.json()
        const detail = errorData.detail
        const msg =
          typeof detail === 'string'
            ? detail
            : Array.isArray(detail)
              ? detail.map((e: Record<string, unknown>) => e.msg ?? String(e)).join(', ')
              : 'Production failed'
        throw new Error(msg)
      }

      const data = await response.json()
      const jobId = data.job_id

      // Add the new job to local state
      const newJob: VideoJob = {
        id: jobId,
        topic: params.topic,
        style: params.style,
        status: 'queued',
        created_at: new Date().toISOString(),
        progress: { stage: 'queued', percent: 0, message: 'Queued' },
      }
      set((state) => ({ jobs: [newJob, ...state.jobs], activeJobId: jobId }))

      return jobId
    } catch (e) {
      set({ error: e instanceof Error ? e.message : 'Production failed' })
      return null
    } finally {
      set({ submitting: false })
    }
  },

  deleteJob: async (jobId) => {
    try {
      const response = await fetch(`${API_BASE}/api/video/jobs/${jobId}`, {
        method: 'DELETE',
      })
      if (response.ok) {
        get().disconnectWebSocket(jobId)
        set((state) => ({
          jobs: state.jobs.filter((j) => j.id !== jobId),
        }))
      }
    } catch (e) {
      console.error('Failed to delete video job:', e)
    }
  },

  connectWebSocket: (jobId) => {
    if (activeWebSockets.has(jobId)) return

    const wsUrl = getWsUrl(`/ws/video/${jobId}`)
    const ws = new WebSocket(wsUrl)
    activeWebSockets.set(jobId, ws)

    ws.onmessage = (event) => {
      try {
        const msg: VideoWSMessage = JSON.parse(event.data)

        if (msg.type === 'progress') {
          set((state) => ({
            jobs: state.jobs.map((j) =>
              j.id === jobId
                ? {
                    ...j,
                    status: 'processing',
                    progress: {
                      stage: msg.stage ?? j.progress.stage,
                      percent: msg.percent ?? j.progress.percent,
                      message: msg.message ?? j.progress.message,
                    },
                  }
                : j
            ),
          }))
        } else if (msg.type === 'script') {
          set((state) => ({
            jobs: state.jobs.map((j) =>
              j.id === jobId
                ? {
                    ...j,
                    script: {
                      title: msg.title ?? '',
                      hook_voiceover: msg.hook_voiceover ?? '',
                      scenes: msg.scenes ?? [],
                    },
                  }
                : j
            ),
          }))
        } else if (msg.type === 'complete') {
          set((state) => ({
            jobs: state.jobs.map((j) =>
              j.id === jobId
                ? {
                    ...j,
                    status: 'completed',
                    progress: { ...j.progress, percent: 100, message: 'Complete' },
                  }
                : j
            ),
          }))
          get().disconnectWebSocket(jobId)
        } else if (msg.type === 'cost_update') {
          if (msg.cost) {
            set((state) => ({
              jobs: state.jobs.map((j) =>
                j.id === jobId ? { ...j, cost: msg.cost } : j
              ),
            }))
          }
        } else if (msg.type === 'status') {
          // Initial status may include cost
          if (msg.cost) {
            set((state) => ({
              jobs: state.jobs.map((j) =>
                j.id === jobId ? { ...j, cost: msg.cost } : j
              ),
            }))
          }
        } else if (msg.type === 'error') {
          set((state) => ({
            jobs: state.jobs.map((j) =>
              j.id === jobId
                ? {
                    ...j,
                    status: 'failed',
                    error: msg.error ?? msg.message ?? 'Unknown error',
                  }
                : j
            ),
          }))
          get().disconnectWebSocket(jobId)
        }
      } catch {
        // Ignore non-JSON messages
      }
    }

    ws.onerror = () => {
      set((state) => ({
        jobs: state.jobs.map((j) =>
          j.id === jobId
            ? { ...j, status: 'failed', error: 'WebSocket connection error' }
            : j
        ),
      }))
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
