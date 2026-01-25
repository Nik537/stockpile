export type WorkflowStep =
  | "script"
  | "title-thumbnail"
  | "broll"
  | "avatar-tts"
  | "editor"
  | "export";

export interface WorkflowStepInfo {
  id: WorkflowStep;
  name: string;
  description: string;
  icon: string;
  path: string;
}

export const WORKFLOW_STEPS: WorkflowStepInfo[] = [
  {
    id: "script",
    name: "Script Writer",
    description: "Write and refine your video script with AI assistance",
    icon: "FileText",
    path: "/workflow/script",
  },
  {
    id: "title-thumbnail",
    name: "Title & Thumbnail",
    description: "Generate compelling titles and thumbnails",
    icon: "Image",
    path: "/workflow/title-thumbnail",
  },
  {
    id: "broll",
    name: "B-Roll Media",
    description: "Find and curate photos and videos for your content",
    icon: "Film",
    path: "/workflow/broll",
  },
  {
    id: "avatar-tts",
    name: "Avatar & TTS",
    description: "Configure AI avatar and text-to-speech settings",
    icon: "User",
    path: "/workflow/avatar-tts",
  },
  {
    id: "editor",
    name: "Timeline Editor",
    description: "Arrange and edit your video timeline",
    icon: "Layers",
    path: "/workflow/editor",
  },
  {
    id: "export",
    name: "Export",
    description: "Render and export your final video",
    icon: "Download",
    path: "/workflow/export",
  },
];

export interface Project {
  id: string;
  name: string;
  currentStep: WorkflowStep;
  createdAt: Date;
  updatedAt: Date;
  script?: string;
  title?: string;
  thumbnailUrl?: string;
  brollItems?: BRollItem[];
  avatarConfig?: AvatarConfig;
  timelineData?: TimelineData;
}

export interface BRollItem {
  id: string;
  type: "photo" | "video";
  url: string;
  thumbnailUrl: string;
  duration?: number;
  startTime?: number;
  endTime?: number;
  source: string;
  title: string;
}

export interface AvatarConfig {
  avatarId: string;
  voiceId: string;
  voiceSettings: {
    speed: number;
    pitch: number;
    stability: number;
  };
}

export interface TimelineData {
  duration: number;
  tracks: TimelineTrack[];
}

export interface TimelineTrack {
  id: string;
  type: "video" | "audio" | "text" | "overlay";
  clips: TimelineClip[];
}

export interface TimelineClip {
  id: string;
  trackId: string;
  startTime: number;
  duration: number;
  mediaUrl?: string;
  text?: string;
  style?: Record<string, unknown>;
}

// Avatar & TTS Types
export interface Voice {
  id: string;
  name: string;
  language: string;
  gender: "male" | "female" | "neutral";
  previewUrl: string;
  style: "natural" | "professional" | "casual" | "dramatic";
}

export interface TTSConfig {
  voiceId: string;
  speed: number; // 0.5 - 2.0
  pitch: number; // -12 to 12
  emotion?: "neutral" | "happy" | "sad" | "excited" | "serious";
}

export interface GeneratedAudio {
  id: string;
  url: string;
  duration: number;
  waveformData: number[];
  config: TTSConfig;
}

export interface Avatar {
  id: string;
  name: string;
  thumbnailUrl: string;
  style: "realistic" | "animated" | "professional";
}

export interface AvatarVideo {
  id: string;
  avatarId: string;
  audioId: string;
  videoUrl: string;
  duration: number;
}

export interface ScriptSection {
  id: string;
  text: string;
  startTime: number;
  endTime: number;
  audio?: GeneratedAudio;
}

export interface TTSGenerationRequest {
  text: string;
  voiceId: string;
  config: TTSConfig;
}

export interface AvatarVideoRequest {
  avatarId: string;
  audioId: string;
  script: string;
}

// Title & Thumbnail Types
export type TitleStyle = 'hook' | 'curiosity' | 'howto' | 'listicle' | 'story';

export interface TitleSuggestion {
  id: string;
  title: string;
  score: number; // AI confidence score (0-100)
  style: TitleStyle;
}

export type ThumbnailStyle = 'cinematic' | 'minimal' | 'bold' | 'viral' | 'professional';
export type ColorScheme = 'dark' | 'light' | 'vibrant';

export interface ThumbnailConfig {
  style: ThumbnailStyle;
  text?: string;
  colorScheme: ColorScheme;
}

export interface GeneratedThumbnail {
  id: string;
  url: string;
  prompt: string;
  config: ThumbnailConfig;
}

export interface TitleGenerationRequest {
  topic: string;
  scriptSummary?: string;
  targetAudience?: string;
  count?: number; // Number of titles to generate (default: 8)
}

export interface TitleGenerationResponse {
  suggestions: TitleSuggestion[];
  generatedAt: string;
}

export interface ThumbnailGenerationRequest {
  title: string;
  topic: string;
  config: ThumbnailConfig;
  count?: number; // Number of variations to generate (default: 4)
}

export interface ThumbnailGenerationResponse {
  thumbnails: GeneratedThumbnail[];
  generatedAt: string;
}

// Script Writer Chat Types
export interface ScriptWriterSection {
  id: string;
  timestamp: string; // e.g., "0:00", "0:30"
  content: string;
  brollSuggestions?: string[];
}

export interface Script {
  id: string;
  title: string;
  sections: ScriptWriterSection[];
  totalDuration: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
}

export interface ScriptGenerationRequest {
  messages: ChatMessage[];
  currentScript?: Script;
}

export interface ScriptGenerationResponse {
  script: Script;
  message: string;
}

// ============================================
// Video Timeline Editor Types
// ============================================

/**
 * Represents a clip on the timeline
 */
export interface EditorTimelineClip {
  id: string;
  trackId: string;
  type: 'video' | 'audio' | 'broll' | 'avatar';
  sourceUrl: string;
  thumbnailUrl?: string;
  name: string;
  startTime: number; // position on timeline (seconds)
  duration: number;
  trimStart: number; // trim from source start
  trimEnd: number; // trim from source end
  speed: number;
  volume?: number;
  // Original source duration (before trim/speed)
  sourceDuration?: number;
}

/**
 * Represents a track in the timeline
 */
export interface EditorTimelineTrack {
  id: string;
  name: string;
  type: 'main' | 'broll' | 'audio' | 'avatar';
  clips: EditorTimelineClip[];
  muted: boolean;
  locked: boolean;
  height?: number;
  color?: string;
}

/**
 * Complete timeline state
 */
export interface EditorTimeline {
  id: string;
  tracks: EditorTimelineTrack[];
  duration: number;
  playheadPosition: number;
  zoom: number; // pixels per second
  scrollOffset: number;
}

/**
 * Export configuration
 */
export interface ExportConfig {
  resolution: '1080p' | '720p' | '4k';
  fps: 30 | 60;
  format: 'mp4' | 'webm' | 'mov';
  quality: 'high' | 'medium' | 'low';
}

/**
 * Timeline editor state
 */
export interface TimelineEditorState {
  timeline: EditorTimeline;
  selectedClipId: string | null;
  selectedTrackId: string | null;
  isPlaying: boolean;
  isDragging: boolean;
  isResizing: boolean;
  dragStartX: number;
  dragClipId: string | null;
  resizeHandle: 'left' | 'right' | null;
  snapEnabled: boolean;
  history: EditorTimeline[];
  historyIndex: number;
}

/**
 * B-roll suggestion from AI
 */
export interface BRollSuggestion {
  id: string;
  timestamp: number; // where to place on timeline
  duration: number; // suggested duration
  searchQuery: string;
  reason: string;
  mediaUrl?: string;
  thumbnailUrl?: string;
  accepted: boolean;
}

/**
 * Script segment for AI placement
 */
export interface ScriptSegment {
  id: string;
  text: string;
  startTime: number;
  endTime: number;
  keywords: string[];
}

/**
 * Timeline action for undo/redo
 */
export type TimelineAction =
  | { type: 'ADD_CLIP'; payload: { trackId: string; clip: EditorTimelineClip } }
  | { type: 'REMOVE_CLIP'; payload: { trackId: string; clipId: string } }
  | { type: 'UPDATE_CLIP'; payload: { trackId: string; clipId: string; updates: Partial<EditorTimelineClip> } }
  | { type: 'MOVE_CLIP'; payload: { fromTrackId: string; toTrackId: string; clipId: string; newStartTime: number } }
  | { type: 'SPLIT_CLIP'; payload: { trackId: string; clipId: string; splitTime: number } }
  | { type: 'SET_PLAYHEAD'; payload: { position: number } }
  | { type: 'SET_ZOOM'; payload: { zoom: number } }
  | { type: 'TOGGLE_MUTE'; payload: { trackId: string } }
  | { type: 'TOGGLE_LOCK'; payload: { trackId: string } }
  | { type: 'SET_CLIP_SPEED'; payload: { trackId: string; clipId: string; speed: number } }
  | { type: 'SET_CLIP_VOLUME'; payload: { trackId: string; clipId: string; volume: number } }
  | { type: 'UNDO' }
  | { type: 'REDO' };
