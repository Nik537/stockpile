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

// Image Generation Types
export type ImageGenStatus = 'idle' | 'generating' | 'completed' | 'error'
export type ImageGenMode = 'generate' | 'edit'
export type ImageGenModel = 'flux-klein' | 'z-image' | 'runpod-flux-schnell' | 'runpod-flux-dev'

// Provider type for routing to correct API
export type ImageGenProvider = 'fal' | 'runpod'

export interface ImageGenServerStatus {
  configured: boolean
  available: boolean
  default_model?: string
  error?: string
}

export interface GeneratedImage {
  url: string
  width: number
  height: number
  content_type: string
  seed?: number | null
}

export interface ImageGenerationResult {
  images: GeneratedImage[]
  model: string
  prompt: string
  generation_time_ms: number
  cost_estimate: number
}

export interface ImageGenerationParams {
  prompt: string
  model: ImageGenModel
  width: number
  height: number
  num_images: number
  seed?: number | null
  guidance_scale: number
}

export interface ImageEditParams {
  prompt: string
  image_url: string
  model: ImageGenModel
  strength: number
  seed?: number | null
  guidance_scale: number
}

// Bulk Image Generation Types
export type BulkImageStatus =
  | 'pending'
  | 'generating_prompts'
  | 'generating_images'
  | 'completed'
  | 'failed'
  | 'cancelled'

export type BulkImageModel =
  | 'runpod-flux-schnell'
  | 'runpod-flux-dev'
  | 'flux-klein'
  | 'z-image'
  | 'runpod-qwen-image'
  | 'runpod-qwen-image-lora'
  | 'runpod-seedream-3'
  | 'runpod-seedream-4'
  | 'gemini-flash'
  | 'replicate-flux-klein'

export interface BulkImagePrompt {
  index: number
  prompt: string
  rendering_style: string  // cartoon, claymation, isometric-3d, watercolor, etc.
  mood: string
  composition: string  // centered-character, scene, product-hero, poster, etc.
  has_text_space: boolean  // Whether prompt includes space for text/slogans
}

export interface BulkImageResult {
  index: number
  prompt: BulkImagePrompt
  image_url: string | null
  width: number
  height: number
  generation_time_ms: number
  status: 'completed' | 'failed'
  error?: string
}

export interface BulkImageJob {
  job_id: string
  meta_prompt: string
  model: BulkImageModel
  width: number
  height: number
  status: BulkImageStatus
  total_count: number
  completed_count: number
  failed_count: number
  prompts: BulkImagePrompt[]
  results: BulkImageResult[]
  total_cost: number
  estimated_cost: number
  error?: string
  created_at: string
  completed_at?: string
}

export type BulkImageWSMessageType =
  | 'status'
  | 'image_complete'
  | 'image_failed'
  | 'complete'
  | 'error'

export interface BulkImageWSMessage {
  type: BulkImageWSMessageType
  // Status message
  status?: BulkImageStatus
  total_count?: number
  completed_count?: number
  failed_count?: number
  error?: string
  // Image complete/failed message
  index?: number
  image_url?: string | null
  prompt?: BulkImagePrompt
  // Complete message
  total_cost?: number
  results?: BulkImageResult[]
  // Error message
  message?: string
}

export interface BulkImagePromptsResponse {
  job_id: string
  prompts: BulkImagePrompt[]
  estimated_cost: number
  estimated_time_seconds: number
}
