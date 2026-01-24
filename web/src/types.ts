/**
 * Type definitions for Stockpile Web UI
 */

export type JobStatus = 'queued' | 'processing' | 'completed' | 'failed'

export interface JobProgress {
  stage: string
  percent: number
  message: string
}

export interface Job {
  id: string
  video_filename: string
  status: JobStatus
  created_at: string
  updated_at: string
  progress: JobProgress
  error?: string
  output_dir?: string
}

export interface UserPreferences {
  style?: string
  avoid?: string
  time_of_day?: string
  preferred_sources?: string
}

export interface StatusUpdate {
  job_id: string
  status: JobStatus
  progress: JobProgress
  error?: string
}
