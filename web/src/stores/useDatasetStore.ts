import { create } from 'zustand'

export interface DatasetItem {
  index: number
  status: string
  startUrl?: string
  endUrl?: string
  imageUrl?: string
  caption: string
  error?: string
  cost: number
}

interface DatasetState {
  jobId: string | null
  status: string
  items: DatasetItem[]
  totalCount: number
  completedCount: number
  failedCount: number
  totalCost: number
  estimatedCost: number
  zipPath: string | null
  error: string | null
  logs: string[]
  setJob: (jobId: string, totalCount: number, estimatedCost: number) => void
  updateStatus: (status: string) => void
  addItem: (item: DatasetItem) => void
  updateItem: (index: number, updates: Partial<DatasetItem>) => void
  updateProgress: (completedCount: number, failedCount: number, totalCost: number) => void
  setComplete: (totalCost: number, zipPath: string) => void
  setError: (error: string) => void
  addLog: (message: string) => void
  reset: () => void
}

export const useDatasetStore = create<DatasetState>((set) => ({
  jobId: null,
  status: 'idle',
  items: [],
  totalCount: 0,
  completedCount: 0,
  failedCount: 0,
  totalCost: 0,
  estimatedCost: 0,
  zipPath: null,
  error: null,
  logs: [],
  setJob: (jobId, totalCount, estimatedCost) =>
    set({
      jobId,
      totalCount,
      estimatedCost,
      status: 'pending',
      items: [],
      completedCount: 0,
      failedCount: 0,
      totalCost: 0,
      zipPath: null,
      error: null,
      logs: [],
    }),
  updateStatus: (status) => set({ status }),
  addItem: (item) =>
    set((state) => ({
      items: [...state.items, item],
    })),
  updateItem: (index, updates) =>
    set((state) => ({
      items: state.items.map((item) =>
        item.index === index ? { ...item, ...updates } : item
      ),
    })),
  updateProgress: (completedCount, failedCount, totalCost) =>
    set({ completedCount, failedCount, totalCost }),
  setComplete: (totalCost, zipPath) =>
    set({ status: 'completed', totalCost, zipPath }),
  setError: (error) => set({ error, status: 'failed' }),
  addLog: (message) =>
    set((state) => ({
      logs: [...state.logs, `[${new Date().toLocaleTimeString()}] ${message}`],
    })),
  reset: () =>
    set({
      jobId: null,
      status: 'idle',
      items: [],
      totalCount: 0,
      completedCount: 0,
      failedCount: 0,
      totalCost: 0,
      estimatedCost: 0,
      zipPath: null,
      error: null,
      logs: [],
    }),
}))
