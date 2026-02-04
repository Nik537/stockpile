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
  // Core fields
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

  // Engagement metrics
  like_count?: number | null
  comment_count?: number | null
  engagement_rate?: number | null // (likes + comments) / views * 100

  // Velocity metrics
  days_since_upload?: number | null
  views_per_day?: number | null
  velocity_score?: number | null // views_per_day / channel_median_velocity

  // Composite scoring
  composite_score?: number | null
  statistical_score?: number | null // IQR-based score
  engagement_score?: number | null // Normalized engagement score

  // Reddit integration
  found_on_reddit?: boolean
  reddit_score?: number | null
  reddit_subreddit?: string | null

  // Momentum tracking
  momentum_score?: number | null
  is_trending?: boolean
}

// Sorting options for outlier results
export type OutlierSortField =
  | 'composite_score'
  | 'outlier_score'
  | 'engagement_rate'
  | 'velocity_score'
  | 'view_count'
  | 'upload_date'

export type SortDirection = 'asc' | 'desc'

// Filter options for outlier results
export interface OutlierFilters {
  minEngagementRate?: number | null
  minVelocity?: number | null
  tiers: {
    exceptional: boolean
    strong: boolean
    solid: boolean
  }
  redditOnly: boolean
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

// TTS (Text-to-Speech) Types
export type TTSStatus = 'idle' | 'connecting' | 'generating' | 'completed' | 'error'

export interface TTSServerStatus {
  connected: boolean
  server_url?: string
  error?: string
}

export interface TTSGenerationParams {
  text: string
  voice?: File | null
  exaggeration: number
  cfg_weight: number
  temperature: number
}
