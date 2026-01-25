// B-Roll specific types
export type MediaSource = 'pexels' | 'pixabay' | 'unsplash' | 'youtube';
export type MediaType = 'photo' | 'video';

export interface BRollItem {
  id: string;
  type: MediaType;
  source: MediaSource;
  url: string;
  thumbnailUrl: string;
  title: string;
  duration?: number; // for videos, in seconds
  dimensions: { width: number; height: number };
  relevanceScore?: number;
  photographer?: string;
  videographer?: string;
  tags?: string[];
  previewUrl?: string; // for video preview on hover
  downloadUrl?: string; // direct download link
  license?: string;
}

export interface BRollSearch {
  query: string;
  sources: MediaSource[];
  type: MediaType | 'all';
  results: BRollItem[];
  totalResults?: number;
  page?: number;
  perPage?: number;
}

export interface BRollSearchFilters {
  type: MediaType | 'all';
  sources: MediaSource[];
  orientation?: 'landscape' | 'portrait' | 'square' | 'all';
  minDuration?: number; // for videos
  maxDuration?: number; // for videos
  sortBy?: 'relevance' | 'newest' | 'popular';
}

export interface DownloadQueueItem {
  id: string;
  item: BRollItem;
  status: 'pending' | 'downloading' | 'complete' | 'error';
  progress: number;
  error?: string;
  startedAt?: Date;
  completedAt?: Date;
  downloadedPath?: string;
}

export interface AISuggestion {
  id: string;
  timestamp: string; // e.g., "0m30s", "1m45s"
  searchTerm: string;
  description: string;
  scriptContext: string;
  confidence: number; // 0-1
}

export interface ScriptSection {
  id: string;
  text: string;
  startTime: number;
  endTime: number;
  suggestedBRoll?: AISuggestion;
}

// API Response Types
export interface BRollSearchResponse {
  results: BRollItem[];
  total: number;
  page: number;
  perPage: number;
  sources: {
    source: MediaSource;
    count: number;
    error?: string;
  }[];
}

export interface BRollDownloadResponse {
  id: string;
  status: 'queued' | 'downloading' | 'complete' | 'error';
  downloadUrl?: string;
  localPath?: string;
  error?: string;
}

export interface AIBRollSuggestionsResponse {
  suggestions: AISuggestion[];
  scriptSections: ScriptSection[];
}

// Search request type
export interface BRollSearchRequest {
  query: string;
  sources: MediaSource[];
  type: MediaType | 'all';
  page?: number;
  perPage?: number;
  filters?: Partial<BRollSearchFilters>;
}

// Download request type
export interface BRollDownloadRequest {
  itemId: string;
  item: BRollItem;
  projectId?: string;
}

// AI suggestion request
export interface AIBRollSuggestionRequest {
  script: string;
  projectId?: string;
}
