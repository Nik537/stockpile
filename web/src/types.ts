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
  min_views: number
  exclude_indian: boolean
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
export type ImageGenMode = 'generate' | 'edit' | 'inpaint'
export type ImageGenModel = 'runware-flux-klein-4b' | 'runware-z-image' | 'runware-flux-klein-9b' | 'gemini-flash' | 'nano-banana-pro'

export interface ImageGenProviderStatus {
  configured: boolean
  available: boolean
  models?: string[]
  free_quota?: string
  error?: string
}

export interface ImageGenServerStatus {
  runware: ImageGenProviderStatus
  gemini: ImageGenProviderStatus
  runpod: ImageGenProviderStatus
  default_model?: string
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

export interface ImageInpaintParams {
  prompt: string
  image_url: string
  mask_image: string
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
  | 'runware-flux-klein-4b'
  | 'runware-flux-klein-9b'
  | 'runware-z-image'
  | 'gemini-flash'
  | 'nano-banana-pro'

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

// Voice Library Types
export interface Voice {
  id: string
  name: string
  is_preset: boolean
  is_favorite: boolean
  audio_path: string
  created_at: string
  duration_seconds: number
}

// Music Generation Types
export type MusicGenStatus = 'idle' | 'generating' | 'completed' | 'error'

export interface MusicServiceStatus {
  configured: boolean
  available: boolean
  error?: string
}

export interface MusicGenerationParams {
  genres: string
  output_seconds: number
  seed: number | null
  steps: number
  cfg: number
}

// Background Job Queue Types
export type BackgroundJobType = 'tts' | 'image' | 'image-edit' | 'music' | 'video'
export type BackgroundJobStatus = 'processing' | 'completed' | 'failed'

export interface BackgroundJob {
  id: string
  type: BackgroundJobType
  status: BackgroundJobStatus
  label: string
  createdAt: string
  completedAt?: string
  error?: string
  result?: any
}

export type BackgroundJobWSMessage = {
  type: 'status' | 'complete' | 'error'
  status?: BackgroundJobStatus
  error?: string
  message?: string
  result?: any
}

// Dataset Generator Types
export type DatasetGenMode = 'pair' | 'single' | 'reference' | 'layered'
export type DatasetGenStatus = 'pending' | 'generating_prompts' | 'generating_images' | 'captioning' | 'packaging' | 'completed' | 'failed' | 'cancelled'

// Video Agent Types
export type VideoProductionStage =
  | 'queued'
  | 'script_generation'
  | 'narration'
  | 'word_timestamps'
  | 'asset_acquisition'
  | 'subtitle_generation'
  | 'video_composition'
  | 'director_review'

export interface VideoJobProgress {
  stage: VideoProductionStage
  percent: number
  message: string
}

export interface VideoScriptScene {
  id: number
  voiceover: string
  visual_type: string
  visual_keywords: string[]
  duration_est: number
}

export interface VideoScript {
  title: string
  hook_voiceover: string
  scenes: VideoScriptScene[]
}

export type VideoJobStatus = 'queued' | 'processing' | 'completed' | 'failed'

export interface VideoJobCost {
  tts: number
  images: number
  music: number
  broll: number
  director: number
  total: number
}

export interface VideoJob {
  id: string
  topic: string
  style: string
  status: VideoJobStatus
  created_at: string
  progress: VideoJobProgress
  script?: VideoScript
  cost?: VideoJobCost
  error?: string
  output_path?: string
}

export interface VideoProduceParams {
  topic: string
  style: string
  target_duration: number
  subtitle_style: string
  voice_id?: string | null
}

export type VideoWSMessageType = 'status' | 'progress' | 'script' | 'complete' | 'error' | 'cost_update'

export interface VideoWSMessage {
  type: VideoWSMessageType
  status?: VideoJobStatus
  stage?: VideoProductionStage
  percent?: number
  message?: string
  title?: string
  hook_voiceover?: string
  scenes?: VideoScriptScene[]
  error?: string
  job_id?: string
  cost?: VideoJobCost
}
