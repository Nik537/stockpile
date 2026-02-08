import { create } from 'zustand'
import {
  OutlierVideo,
  OutlierSearchStatus,
  OutlierSortField,
  SortDirection,
  OutlierFilters,
} from '../types'

interface OutlierState {
  // Search state
  searchId: string | null
  status: OutlierSearchStatus | null
  channelsAnalyzed: number
  totalChannels: number
  videosScanned: number
  outliers: OutlierVideo[]
  error: string | null
  isSearching: boolean

  // Sort/filter state
  sortField: OutlierSortField
  sortDirection: SortDirection
  filters: OutlierFilters

  // Actions
  startSearch: (searchId: string) => void
  setStatus: (status: OutlierSearchStatus) => void
  addOutlier: (outlier: OutlierVideo) => void
  setOutliers: (outliers: OutlierVideo[]) => void
  updateProgress: (channels: number, total: number, videos: number) => void
  setError: (error: string | null) => void
  setIsSearching: (searching: boolean) => void
  setSortField: (field: OutlierSortField) => void
  setSortDirection: (direction: SortDirection) => void
  setFilters: (filters: Partial<OutlierFilters>) => void
  reset: () => void
}

const initialState = {
  searchId: null,
  status: null,
  channelsAnalyzed: 0,
  totalChannels: 0,
  videosScanned: 0,
  outliers: [],
  error: null,
  isSearching: false,
  sortField: 'composite_score' as OutlierSortField,
  sortDirection: 'desc' as SortDirection,
  filters: {
    minEngagementRate: null,
    minVelocity: null,
    tiers: { exceptional: true, strong: true, solid: true },
    redditOnly: false,
  },
}

export const useOutlierStore = create<OutlierState>((set) => ({
  ...initialState,

  startSearch: (searchId) =>
    set({ searchId, status: 'searching', isSearching: true, outliers: [], error: null }),

  setStatus: (status) => set({ status }),

  addOutlier: (outlier) =>
    set((state) => ({ outliers: [...state.outliers, outlier] })),

  setOutliers: (outliers) => set({ outliers }),

  updateProgress: (channels, total, videos) =>
    set({ channelsAnalyzed: channels, totalChannels: total, videosScanned: videos }),

  setError: (error) => set({ error, isSearching: false }),

  setIsSearching: (searching) => set({ isSearching: searching }),

  setSortField: (field) => set({ sortField: field }),

  setSortDirection: (direction) => set({ sortDirection: direction }),

  setFilters: (newFilters) =>
    set((state) => ({ filters: { ...state.filters, ...newFilters } })),

  reset: () => set(initialState),
}))
