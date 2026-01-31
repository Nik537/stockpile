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

// Outlier Finder Types
export type OutlierTier = 'solid' | 'strong' | 'exceptional'

export interface OutlierVideo {
  video_id: string
  title: string
  url: string
  thumbnail_url: string
  view_count: number
  outlier_score: number
  channel_average_views: number
  channel_name: string
  upload_date: string
  outlier_tier: OutlierTier
}

export interface OutlierSearchParams {
  topic: string
  max_channels: number
  min_score: number
  days?: number | null
  include_shorts: boolean
  min_subs?: number | null
  max_subs?: number | null
}

export type OutlierSearchStatus = 'searching' | 'completed' | 'failed'

export interface OutlierSearch {
  id: string
  topic: string
  status: OutlierSearchStatus
  channels_analyzed: number
  total_channels: number
  videos_scanned: number
  outliers: OutlierVideo[]
  error?: string
}

// WebSocket message types for outlier finder
export type OutlierWSMessageType = 'status' | 'progress' | 'outlier' | 'complete' | 'error'

export interface OutlierWSMessage {
  type: OutlierWSMessageType
  // Progress message
  channels_analyzed?: number
  total_channels?: number
  videos_scanned?: number
  // Outlier message
  outlier?: OutlierVideo
  // Complete message
  total_outliers?: number
  // Error message
  message?: string
  // Status message (initial)
  status?: OutlierSearchStatus
  outliers?: OutlierVideo[]
  error?: string
}
