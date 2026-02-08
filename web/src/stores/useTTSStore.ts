import { create } from 'zustand'

interface TTSState {
  serverStatus: { colab: any; runpod: any } | null
  mode: 'runpod' | 'colab'
  generating: boolean
  error: string | null
  setServerStatus: (status: any) => void
  setMode: (mode: 'runpod' | 'colab') => void
  setGenerating: (generating: boolean) => void
  setError: (error: string | null) => void
}

export const useTTSStore = create<TTSState>((set) => ({
  serverStatus: null,
  mode: 'runpod',
  generating: false,
  error: null,
  setServerStatus: (status) => set({ serverStatus: status }),
  setMode: (mode) => set({ mode }),
  setGenerating: (generating) => set({ generating }),
  setError: (error) => set({ error }),
}))
