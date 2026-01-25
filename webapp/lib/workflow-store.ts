"use client";

import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
import type { WorkflowStep, Script, GeneratedThumbnail, Voice, Avatar, TTSConfig, GeneratedAudio, EditorTimeline, ChatMessage } from "@/types/workflow";
import type { BRollItem, AISuggestion } from "@/types/broll";
import {
  type WorkflowState,
  type WorkflowData,
  type WorkflowProject,
  type ScriptStepData,
  type TitleThumbnailStepData,
  type BRollStepData,
  type AvatarTTSStepData,
  type EditorStepData,
  type ExportStepData,
  defaultWorkflowData,
  defaultStepValidation,
  STEP_ORDER,
  getStepIndex,
  getNextStep,
  getPreviousStep,
  canNavigateToStep,
} from "@/types/workflow-state";
import { generateId } from "@/lib/utils";

// Storage key for localStorage
const STORAGE_KEY = "stockpile_workflow_state";

/**
 * Workflow store actions interface
 */
interface WorkflowActions {
  // Project management
  createNewProject: (name?: string) => void;
  loadProject: (projectId: string) => void;
  saveProject: () => void;
  resetProject: () => void;

  // Navigation
  setCurrentStep: (step: WorkflowStep) => void;
  goToNextStep: () => void;
  goToPreviousStep: () => void;
  markStepComplete: (step: WorkflowStep) => void;
  markStepIncomplete: (step: WorkflowStep) => void;

  // Script step actions
  setScript: (script: Script | null) => void;
  setScriptSummary: (summary: string) => void;
  setScriptMessages: (messages: ChatMessage[]) => void;

  // Title/Thumbnail step actions
  setFinalTitle: (title: string) => void;
  setSelectedThumbnail: (thumbnail: GeneratedThumbnail | null) => void;
  addThumbnail: (thumbnail: GeneratedThumbnail) => void;

  // B-Roll step actions
  addSelectedMedia: (item: BRollItem) => void;
  removeSelectedMedia: (itemId: string) => void;
  setSelectedMedia: (items: BRollItem[]) => void;
  setDownloadedMedia: (items: BRollItem[]) => void;
  setAiSuggestions: (suggestions: AISuggestion[]) => void;

  // Avatar/TTS step actions
  setSelectedVoice: (voice: Voice | null) => void;
  setSelectedAvatar: (avatar: Avatar | null) => void;
  setTtsConfig: (config: Partial<TTSConfig>) => void;
  addGeneratedAudio: (audio: GeneratedAudio) => void;
  setAvatarEnabled: (enabled: boolean) => void;

  // Editor step actions
  setTimeline: (timeline: EditorTimeline | null) => void;
  setEditorHasChanges: (hasChanges: boolean) => void;

  // Export step actions
  setExportConfig: (config: Partial<ExportStepData>) => void;
  setExportProgress: (progress: number) => void;
  setExportStatus: (status: ExportStepData["exportStatus"]) => void;

  // Validation
  validateStep: (step: WorkflowStep) => boolean;
  validateAllSteps: () => void;

  // UI state
  setLoading: (loading: boolean) => void;
  setSaving: (saving: boolean) => void;
  setError: (error: string | null) => void;
}

/**
 * Complete workflow store type
 */
type WorkflowStore = WorkflowState & WorkflowActions;

/**
 * Create a new project with default data
 */
function createNewProject(name?: string): WorkflowProject {
  const now = new Date().toISOString();
  return {
    id: generateId(),
    name: name || `Project ${new Date().toLocaleDateString()}`,
    createdAt: now,
    updatedAt: now,
    currentStep: "script",
    completedSteps: [],
  };
}

/**
 * Initial state for the workflow store
 */
const initialState: WorkflowState = {
  project: createNewProject(),
  data: defaultWorkflowData,
  currentStep: "script",
  completedSteps: [],
  isLoading: false,
  isSaving: false,
  error: null,
  stepValidation: defaultStepValidation,
};

/**
 * Zustand store for workflow state management
 */
export const useWorkflowStore = create<WorkflowStore>()(
  persist(
    (set, get) => ({
      // Initial state
      ...initialState,

      // Project management
      createNewProject: (name?: string) => {
        const newProject = createNewProject(name);
        set({
          project: newProject,
          data: defaultWorkflowData,
          currentStep: "script",
          completedSteps: [],
          stepValidation: defaultStepValidation,
          error: null,
        });
      },

      loadProject: (projectId: string) => {
        // In a real app, this would load from a backend
        // For now, we just use the persisted state
        const state = get();
        if (state.project.id !== projectId) {
          set({ error: "Project not found" });
        }
      },

      saveProject: () => {
        set((state) => ({
          isSaving: true,
          project: {
            ...state.project,
            updatedAt: new Date().toISOString(),
          },
        }));
        // Simulated save delay
        setTimeout(() => set({ isSaving: false }), 500);
      },

      resetProject: () => {
        set({
          ...initialState,
          project: createNewProject(),
        });
      },

      // Navigation
      setCurrentStep: (step: WorkflowStep) => {
        const state = get();
        if (canNavigateToStep(step, state.currentStep, state.completedSteps)) {
          set({
            currentStep: step,
            project: {
              ...state.project,
              currentStep: step,
              updatedAt: new Date().toISOString(),
            },
          });
        }
      },

      goToNextStep: () => {
        const state = get();
        const nextStep = getNextStep(state.currentStep);
        if (nextStep && state.stepValidation[state.currentStep]) {
          // Mark current step as complete
          const newCompletedSteps = state.completedSteps.includes(state.currentStep)
            ? state.completedSteps
            : [...state.completedSteps, state.currentStep];

          set({
            currentStep: nextStep,
            completedSteps: newCompletedSteps,
            project: {
              ...state.project,
              currentStep: nextStep,
              completedSteps: newCompletedSteps,
              updatedAt: new Date().toISOString(),
            },
          });
        }
      },

      goToPreviousStep: () => {
        const state = get();
        const prevStep = getPreviousStep(state.currentStep);
        if (prevStep) {
          set({
            currentStep: prevStep,
            project: {
              ...state.project,
              currentStep: prevStep,
              updatedAt: new Date().toISOString(),
            },
          });
        }
      },

      markStepComplete: (step: WorkflowStep) => {
        const state = get();
        if (!state.completedSteps.includes(step)) {
          const newCompletedSteps = [...state.completedSteps, step];
          set({
            completedSteps: newCompletedSteps,
            project: {
              ...state.project,
              completedSteps: newCompletedSteps,
              updatedAt: new Date().toISOString(),
            },
          });
        }
      },

      markStepIncomplete: (step: WorkflowStep) => {
        const state = get();
        if (state.completedSteps.includes(step)) {
          const newCompletedSteps = state.completedSteps.filter((s) => s !== step);
          set({
            completedSteps: newCompletedSteps,
            project: {
              ...state.project,
              completedSteps: newCompletedSteps,
              updatedAt: new Date().toISOString(),
            },
          });
        }
      },

      // Script step actions
      setScript: (script: Script | null) => {
        set((state) => ({
          data: {
            ...state.data,
            script: {
              ...state.data.script,
              script,
              sections: script?.sections || [],
            },
          },
        }));
        get().validateStep("script");
      },

      setScriptSummary: (summary: string) => {
        set((state) => ({
          data: {
            ...state.data,
            script: {
              ...state.data.script,
              summary,
            },
          },
        }));
      },

      setScriptMessages: (messages: ChatMessage[]) => {
        set((state) => ({
          data: {
            ...state.data,
            script: {
              ...state.data.script,
              messages,
            },
          },
        }));
      },

      // Title/Thumbnail step actions
      setFinalTitle: (title: string) => {
        set((state) => ({
          data: {
            ...state.data,
            titleThumbnail: {
              ...state.data.titleThumbnail,
              finalTitle: title,
            },
          },
        }));
        get().validateStep("title-thumbnail");
      },

      setSelectedThumbnail: (thumbnail: GeneratedThumbnail | null) => {
        set((state) => ({
          data: {
            ...state.data,
            titleThumbnail: {
              ...state.data.titleThumbnail,
              selectedThumbnail: thumbnail,
            },
          },
        }));
        get().validateStep("title-thumbnail");
      },

      addThumbnail: (thumbnail: GeneratedThumbnail) => {
        set((state) => ({
          data: {
            ...state.data,
            titleThumbnail: {
              ...state.data.titleThumbnail,
              allThumbnails: [...state.data.titleThumbnail.allThumbnails, thumbnail],
            },
          },
        }));
      },

      // B-Roll step actions
      addSelectedMedia: (item: BRollItem) => {
        set((state) => {
          const exists = state.data.broll.selectedMedia.some(
            (m) => m.id === item.id && m.source === item.source
          );
          if (exists) return state;
          return {
            data: {
              ...state.data,
              broll: {
                ...state.data.broll,
                selectedMedia: [...state.data.broll.selectedMedia, item],
              },
            },
          };
        });
        get().validateStep("broll");
      },

      removeSelectedMedia: (itemId: string) => {
        set((state) => ({
          data: {
            ...state.data,
            broll: {
              ...state.data.broll,
              selectedMedia: state.data.broll.selectedMedia.filter(
                (m) => m.id !== itemId
              ),
            },
          },
        }));
        get().validateStep("broll");
      },

      setSelectedMedia: (items: BRollItem[]) => {
        set((state) => ({
          data: {
            ...state.data,
            broll: {
              ...state.data.broll,
              selectedMedia: items,
            },
          },
        }));
        get().validateStep("broll");
      },

      setDownloadedMedia: (items: BRollItem[]) => {
        set((state) => ({
          data: {
            ...state.data,
            broll: {
              ...state.data.broll,
              downloadedMedia: items,
            },
          },
        }));
      },

      setAiSuggestions: (suggestions: AISuggestion[]) => {
        set((state) => ({
          data: {
            ...state.data,
            broll: {
              ...state.data.broll,
              aiSuggestions: suggestions,
            },
          },
        }));
      },

      // Avatar/TTS step actions
      setSelectedVoice: (voice: Voice | null) => {
        set((state) => ({
          data: {
            ...state.data,
            avatarTts: {
              ...state.data.avatarTts,
              selectedVoice: voice,
              ttsConfig: voice
                ? { ...state.data.avatarTts.ttsConfig, voiceId: voice.id }
                : state.data.avatarTts.ttsConfig,
            },
          },
        }));
        get().validateStep("avatar-tts");
      },

      setSelectedAvatar: (avatar: Avatar | null) => {
        set((state) => ({
          data: {
            ...state.data,
            avatarTts: {
              ...state.data.avatarTts,
              selectedAvatar: avatar,
            },
          },
        }));
      },

      setTtsConfig: (config: Partial<TTSConfig>) => {
        set((state) => ({
          data: {
            ...state.data,
            avatarTts: {
              ...state.data.avatarTts,
              ttsConfig: {
                ...state.data.avatarTts.ttsConfig,
                ...config,
              },
            },
          },
        }));
      },

      addGeneratedAudio: (audio: GeneratedAudio) => {
        set((state) => ({
          data: {
            ...state.data,
            avatarTts: {
              ...state.data.avatarTts,
              generatedAudio: [...state.data.avatarTts.generatedAudio, audio],
            },
          },
        }));
        get().validateStep("avatar-tts");
      },

      setAvatarEnabled: (enabled: boolean) => {
        set((state) => ({
          data: {
            ...state.data,
            avatarTts: {
              ...state.data.avatarTts,
              isAvatarEnabled: enabled,
            },
          },
        }));
      },

      // Editor step actions
      setTimeline: (timeline: EditorTimeline | null) => {
        set((state) => ({
          data: {
            ...state.data,
            editor: {
              ...state.data.editor,
              timeline,
              hasChanges: true,
            },
          },
        }));
        get().validateStep("editor");
      },

      setEditorHasChanges: (hasChanges: boolean) => {
        set((state) => ({
          data: {
            ...state.data,
            editor: {
              ...state.data.editor,
              hasChanges,
            },
          },
        }));
      },

      // Export step actions
      setExportConfig: (config: Partial<ExportStepData>) => {
        set((state) => ({
          data: {
            ...state.data,
            export: {
              ...state.data.export,
              ...config,
            },
          },
        }));
      },

      setExportProgress: (progress: number) => {
        set((state) => ({
          data: {
            ...state.data,
            export: {
              ...state.data.export,
              progress,
            },
          },
        }));
      },

      setExportStatus: (status: ExportStepData["exportStatus"]) => {
        set((state) => ({
          data: {
            ...state.data,
            export: {
              ...state.data.export,
              exportStatus: status,
            },
          },
        }));
        if (status === "complete") {
          get().markStepComplete("export");
        }
      },

      // Validation
      validateStep: (step: WorkflowStep) => {
        const state = get();
        let isValid = false;

        switch (step) {
          case "script":
            isValid = state.data.script.script !== null &&
              state.data.script.script.sections.length > 0;
            break;
          case "title-thumbnail":
            isValid = state.data.titleThumbnail.finalTitle.trim().length > 0 &&
              state.data.titleThumbnail.selectedThumbnail !== null;
            break;
          case "broll":
            // B-Roll is optional, so it's valid by default
            // But if user selected media, ensure they're downloaded
            isValid = true;
            break;
          case "avatar-tts":
            isValid = state.data.avatarTts.selectedVoice !== null &&
              state.data.avatarTts.generatedAudio.length > 0;
            break;
          case "editor":
            isValid = state.data.editor.timeline !== null &&
              state.data.editor.timeline.tracks.some((t) => t.clips.length > 0);
            break;
          case "export":
            isValid = state.data.export.exportStatus === "complete";
            break;
        }

        set((state) => ({
          stepValidation: {
            ...state.stepValidation,
            [step]: isValid,
          },
        }));

        return isValid;
      },

      validateAllSteps: () => {
        STEP_ORDER.forEach((step) => get().validateStep(step));
      },

      // UI state
      setLoading: (loading: boolean) => set({ isLoading: loading }),
      setSaving: (saving: boolean) => set({ isSaving: saving }),
      setError: (error: string | null) => set({ error }),
    }),
    {
      name: STORAGE_KEY,
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        project: state.project,
        data: state.data,
        currentStep: state.currentStep,
        completedSteps: state.completedSteps,
        stepValidation: state.stepValidation,
      }),
    }
  )
);

/**
 * Selector hooks for specific parts of the state
 */
export const useWorkflowProject = () => useWorkflowStore((state) => state.project);
export const useWorkflowData = () => useWorkflowStore((state) => state.data);
export const useCurrentStep = () => useWorkflowStore((state) => state.currentStep);
export const useCompletedSteps = () => useWorkflowStore((state) => state.completedSteps);
export const useStepValidation = () => useWorkflowStore((state) => state.stepValidation);

// Step-specific data selectors
export const useScriptData = () => useWorkflowStore((state) => state.data.script);
export const useTitleThumbnailData = () => useWorkflowStore((state) => state.data.titleThumbnail);
export const useBRollData = () => useWorkflowStore((state) => state.data.broll);
export const useAvatarTtsData = () => useWorkflowStore((state) => state.data.avatarTts);
export const useEditorData = () => useWorkflowStore((state) => state.data.editor);
export const useExportData = () => useWorkflowStore((state) => state.data.export);
