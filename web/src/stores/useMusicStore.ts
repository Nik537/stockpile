import { create } from 'zustand'
import { MusicGenStatus } from '../types'

interface MusicState {
  status: MusicGenStatus
  serviceStatus: { configured: boolean; available: boolean; error?: string } | null
  error: string | null
  setStatus: (status: MusicGenStatus) => void
  setServiceStatus: (status: any) => void
  setError: (error: string | null) => void
}

export const useMusicStore = create<MusicState>((set) => ({
  status: 'idle',
  serviceStatus: null,
  error: null,
  setStatus: (status) => set({ status }),
  setServiceStatus: (serviceStatus) => set({ serviceStatus }),
  setError: (error) => set({ error }),
}))
