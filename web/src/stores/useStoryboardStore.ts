import { create } from 'zustand'

interface StoryboardState {
  defaultAspectRatio: string
  defaultSceneCount: number
  defaultRefModel: string
  defaultSceneModel: string
  setDefaultAspectRatio: (ratio: string) => void
  setDefaultSceneCount: (count: number) => void
  setDefaultRefModel: (model: string) => void
  setDefaultSceneModel: (model: string) => void
}

export const useStoryboardStore = create<StoryboardState>((set) => ({
  defaultAspectRatio: '9:16',
  defaultSceneCount: 6,
  defaultRefModel: 'flux-dev',
  defaultSceneModel: 'flux-kontext',
  setDefaultAspectRatio: (ratio) => set({ defaultAspectRatio: ratio }),
  setDefaultSceneCount: (count) => set({ defaultSceneCount: count }),
  setDefaultRefModel: (model) => set({ defaultRefModel: model }),
  setDefaultSceneModel: (model) => set({ defaultSceneModel: model }),
}))
