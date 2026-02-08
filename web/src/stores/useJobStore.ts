import { create } from 'zustand'
import { Job } from '../types'

interface JobState {
  jobs: Job[]
  loading: boolean
  error: string | null
  fetchJobs: () => Promise<void>
  addJob: (job: Job) => void
  updateJob: (jobId: string, updates: Partial<Job>) => void
  deleteJob: (jobId: string) => void
}

export const useJobStore = create<JobState>((set) => ({
  jobs: [],
  loading: true,
  error: null,

  fetchJobs: async () => {
    try {
      const response = await fetch('/api/jobs')
      const data = await response.json()
      set({ jobs: data.jobs, loading: false, error: null })
    } catch (error) {
      console.error('Failed to fetch jobs:', error)
      set({ loading: false, error: 'Failed to fetch jobs' })
    }
  },

  addJob: (job: Job) => {
    set((state) => ({ jobs: [job, ...state.jobs] }))
  },

  updateJob: (jobId: string, updates: Partial<Job>) => {
    set((state) => ({
      jobs: state.jobs.map((job) =>
        job.id === jobId ? { ...job, ...updates } : job
      ),
    }))
  },

  deleteJob: (jobId: string) => {
    set((state) => ({
      jobs: state.jobs.filter((job) => job.id !== jobId),
    }))
  },
}))
