// Workflow State Types for Unified Orchestrator

import type {
  WorkflowStep,
  Script,
  GeneratedThumbnail,
  Voice,
  Avatar,
  TTSConfig,
  GeneratedAudio,
  EditorTimeline,
  ScriptWriterSection,
  ChatMessage,
} from "./workflow";
import type { BRollItem, AISuggestion } from "./broll";

/**
 * Ideas step data - title/thumbnail concepts from brainstorming
 */
export interface IdeasStepData {
  selectedTitle: string;
  thumbnailConcept: string;
  keywords: string[];
}

/**
 * Script step data - generated script with sections
 */
export interface ScriptStepData {
  script: Script | null;
  sections: ScriptWriterSection[];
  summary: string;
  messages: ChatMessage[];
}

/**
 * Title/Thumbnail step data - final title and generated thumbnails
 */
export interface TitleThumbnailStepData {
  finalTitle: string;
  selectedThumbnail: GeneratedThumbnail | null;
  allThumbnails: GeneratedThumbnail[];
}

/**
 * B-Roll step data - downloaded media items
 */
export interface BRollStepData {
  selectedMedia: BRollItem[];
  downloadedMedia: BRollItem[];
  aiSuggestions: AISuggestion[];
}

/**
 * Avatar/TTS step data - voice and avatar configuration
 */
export interface AvatarTTSStepData {
  selectedVoice: Voice | null;
  selectedAvatar: Avatar | null;
  ttsConfig: TTSConfig;
  generatedAudio: GeneratedAudio[];
  isAvatarEnabled: boolean;
}

/**
 * Editor step data - timeline state
 */
export interface EditorStepData {
  timeline: EditorTimeline | null;
  hasChanges: boolean;
}

/**
 * Export step data - final export configuration
 */
export interface ExportStepData {
  resolution: "4k" | "1080p" | "720p" | "480p";
  aspectRatio: "16:9" | "9:16" | "1:1";
  format: "mp4" | "webm" | "mov";
  quality: number;
  exportStatus: "idle" | "preparing" | "rendering" | "complete" | "error";
  progress: number;
  outputUrl?: string;
}

/**
 * Complete workflow data structure
 */
export interface WorkflowData {
  ideas: IdeasStepData;
  script: ScriptStepData;
  titleThumbnail: TitleThumbnailStepData;
  broll: BRollStepData;
  avatarTts: AvatarTTSStepData;
  editor: EditorStepData;
  export: ExportStepData;
}

/**
 * Workflow project metadata
 */
export interface WorkflowProject {
  id: string;
  name: string;
  createdAt: string;
  updatedAt: string;
  currentStep: WorkflowStep;
  completedSteps: WorkflowStep[];
}

/**
 * Complete workflow state
 */
export interface WorkflowState {
  // Project metadata
  project: WorkflowProject;

  // Step data
  data: WorkflowData;

  // Navigation
  currentStep: WorkflowStep;
  completedSteps: WorkflowStep[];

  // UI state
  isLoading: boolean;
  isSaving: boolean;
  error: string | null;

  // Validation
  stepValidation: Record<WorkflowStep, boolean>;
}

/**
 * Initial/default workflow data
 */
export const defaultWorkflowData: WorkflowData = {
  ideas: {
    selectedTitle: "",
    thumbnailConcept: "",
    keywords: [],
  },
  script: {
    script: null,
    sections: [],
    summary: "",
    messages: [],
  },
  titleThumbnail: {
    finalTitle: "",
    selectedThumbnail: null,
    allThumbnails: [],
  },
  broll: {
    selectedMedia: [],
    downloadedMedia: [],
    aiSuggestions: [],
  },
  avatarTts: {
    selectedVoice: null,
    selectedAvatar: null,
    ttsConfig: {
      voiceId: "",
      speed: 1.0,
      pitch: 0,
      emotion: "neutral",
    },
    generatedAudio: [],
    isAvatarEnabled: false,
  },
  editor: {
    timeline: null,
    hasChanges: false,
  },
  export: {
    resolution: "1080p",
    aspectRatio: "16:9",
    format: "mp4",
    quality: 80,
    exportStatus: "idle",
    progress: 0,
  },
};

/**
 * Initial step validation state
 */
export const defaultStepValidation: Record<WorkflowStep, boolean> = {
  script: false,
  "title-thumbnail": false,
  broll: false,
  "avatar-tts": false,
  editor: false,
  export: false,
};

/**
 * Workflow step order for navigation
 */
export const STEP_ORDER: WorkflowStep[] = [
  "script",
  "title-thumbnail",
  "broll",
  "avatar-tts",
  "editor",
  "export",
];

/**
 * Get the index of a step in the workflow
 */
export function getStepIndex(step: WorkflowStep): number {
  return STEP_ORDER.indexOf(step);
}

/**
 * Get the next step in the workflow
 */
export function getNextStep(currentStep: WorkflowStep): WorkflowStep | null {
  const currentIndex = getStepIndex(currentStep);
  if (currentIndex === -1 || currentIndex >= STEP_ORDER.length - 1) {
    return null;
  }
  return STEP_ORDER[currentIndex + 1];
}

/**
 * Get the previous step in the workflow
 */
export function getPreviousStep(currentStep: WorkflowStep): WorkflowStep | null {
  const currentIndex = getStepIndex(currentStep);
  if (currentIndex <= 0) {
    return null;
  }
  return STEP_ORDER[currentIndex - 1];
}

/**
 * Check if a step can be navigated to
 */
export function canNavigateToStep(
  targetStep: WorkflowStep,
  currentStep: WorkflowStep,
  completedSteps: WorkflowStep[]
): boolean {
  const targetIndex = getStepIndex(targetStep);
  const currentIndex = getStepIndex(currentStep);

  // Can always go backwards
  if (targetIndex < currentIndex) {
    return true;
  }

  // Can go to current step
  if (targetIndex === currentIndex) {
    return true;
  }

  // Can only go forward one step at a time, and current step must be complete
  if (targetIndex === currentIndex + 1 && completedSteps.includes(currentStep)) {
    return true;
  }

  // Can jump to any completed step
  if (completedSteps.includes(targetStep)) {
    return true;
  }

  return false;
}
