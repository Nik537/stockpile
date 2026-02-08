import { create } from 'zustand'
import { GeneratedImage, ImageGenModel } from '../types'

interface ImageState {
  generatedImages: GeneratedImage[]
  generating: boolean
  error: string | null
  selectedModel: ImageGenModel
  generate: (images: GeneratedImage[]) => void
  setGenerating: (generating: boolean) => void
  setError: (error: string | null) => void
  setSelectedModel: (model: ImageGenModel) => void
  clearImages: () => void
}

export const useImageStore = create<ImageState>((set) => ({
  generatedImages: [],
  generating: false,
  error: null,
  selectedModel: 'flux-klein',
  generate: (images) => set({ generatedImages: images, generating: false }),
  setGenerating: (generating) => set({ generating }),
  setError: (error) => set({ error, generating: false }),
  setSelectedModel: (model) => set({ selectedModel: model }),
  clearImages: () => set({ generatedImages: [] }),
}))
