import { create } from 'zustand'

type TTSMode = 'runpod' | 'qwen3' | 'chatterbox-ext' | 'colab'

interface TTSState {
  serverStatus: { colab: any; runpod: any; qwen3?: any; 'chatterbox-ext'?: any } | null
  mode: TTSMode
  generating: boolean
  error: string | null
  setServerStatus: (status: any) => void
  setMode: (mode: TTSMode) => void
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
