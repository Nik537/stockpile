"use client";

import { useCallback, useEffect, useMemo, useReducer, useRef } from "react";
import { generateId } from "@/lib/utils";
import type {
  EditorTimeline,
  EditorTimelineClip,
  EditorTimelineTrack,
  TimelineAction,
  BRollSuggestion,
} from "@/types/workflow";

// Constants
const SNAP_THRESHOLD = 0.5; // seconds
const DEFAULT_ZOOM = 50; // pixels per second
const MIN_ZOOM = 10;
const MAX_ZOOM = 200;
const MAX_HISTORY = 50;

// Initial timeline state
function createInitialTimeline(): EditorTimeline {
  return {
    id: generateId(),
    tracks: [
      {
        id: "main-track",
        name: "Main Video",
        type: "main",
        clips: [],
        muted: false,
        locked: false,
        height: 60,
        color: "bg-blue-500",
      },
      {
        id: "broll-track",
        name: "B-Roll",
        type: "broll",
        clips: [],
        muted: false,
        locked: false,
        height: 50,
        color: "bg-purple-500",
      },
      {
        id: "audio-track",
        name: "Audio",
        type: "audio",
        clips: [],
        muted: false,
        locked: false,
        height: 40,
        color: "bg-green-500",
      },
      {
        id: "avatar-track",
        name: "Avatar",
        type: "avatar",
        clips: [],
        muted: false,
        locked: false,
        height: 50,
        color: "bg-orange-500",
      },
    ],
    duration: 180, // 3 minutes default
    playheadPosition: 0,
    zoom: DEFAULT_ZOOM,
    scrollOffset: 0,
  };
}

// Editor state
interface EditorState {
  timeline: EditorTimeline;
  selectedClipId: string | null;
  selectedTrackId: string | null;
  isPlaying: boolean;
  snapEnabled: boolean;
  history: EditorTimeline[];
  historyIndex: number;
}

// Action types
type EditorAction =
  | TimelineAction
  | { type: "SET_SELECTED_CLIP"; payload: string | null }
  | { type: "SET_SELECTED_TRACK"; payload: string | null }
  | { type: "SET_PLAYING"; payload: boolean }
  | { type: "TOGGLE_SNAP" }
  | { type: "SET_SCROLL_OFFSET"; payload: number }
  | { type: "SET_DURATION"; payload: number }
  | { type: "LOAD_TIMELINE"; payload: EditorTimeline };

// Reducer
function editorReducer(state: EditorState, action: EditorAction): EditorState {
  switch (action.type) {
    case "ADD_CLIP": {
      const { trackId, clip } = action.payload;
      const newTimeline = {
        ...state.timeline,
        tracks: state.timeline.tracks.map((track) =>
          track.id === trackId
            ? { ...track, clips: [...track.clips, clip] }
            : track
        ),
      };
      // Calculate new duration
      const allClips = newTimeline.tracks.flatMap((t) => t.clips);
      const maxEnd = Math.max(
        newTimeline.duration,
        ...allClips.map((c) => c.startTime + c.duration)
      );
      newTimeline.duration = maxEnd;

      return {
        ...state,
        timeline: newTimeline,
        history: [
          ...state.history.slice(0, state.historyIndex + 1),
          newTimeline,
        ].slice(-MAX_HISTORY),
        historyIndex: Math.min(state.historyIndex + 1, MAX_HISTORY - 1),
      };
    }

    case "REMOVE_CLIP": {
      const { trackId, clipId } = action.payload;
      const newTimeline = {
        ...state.timeline,
        tracks: state.timeline.tracks.map((track) =>
          track.id === trackId
            ? { ...track, clips: track.clips.filter((c) => c.id !== clipId) }
            : track
        ),
      };
      return {
        ...state,
        timeline: newTimeline,
        selectedClipId:
          state.selectedClipId === clipId ? null : state.selectedClipId,
        history: [
          ...state.history.slice(0, state.historyIndex + 1),
          newTimeline,
        ].slice(-MAX_HISTORY),
        historyIndex: Math.min(state.historyIndex + 1, MAX_HISTORY - 1),
      };
    }

    case "UPDATE_CLIP": {
      const { trackId, clipId, updates } = action.payload;
      const newTimeline = {
        ...state.timeline,
        tracks: state.timeline.tracks.map((track) =>
          track.id === trackId
            ? {
                ...track,
                clips: track.clips.map((clip) =>
                  clip.id === clipId ? { ...clip, ...updates } : clip
                ),
              }
            : track
        ),
      };
      // Update duration if needed
      const allClips = newTimeline.tracks.flatMap((t) => t.clips);
      const maxEnd = Math.max(
        ...allClips.map((c) => c.startTime + c.duration),
        newTimeline.duration
      );
      newTimeline.duration = maxEnd;

      return {
        ...state,
        timeline: newTimeline,
        history: [
          ...state.history.slice(0, state.historyIndex + 1),
          newTimeline,
        ].slice(-MAX_HISTORY),
        historyIndex: Math.min(state.historyIndex + 1, MAX_HISTORY - 1),
      };
    }

    case "MOVE_CLIP": {
      const { fromTrackId, toTrackId, clipId, newStartTime } = action.payload;
      let clipToMove: EditorTimelineClip | null = null;

      // Find and remove from source track
      const tracksAfterRemove = state.timeline.tracks.map((track) => {
        if (track.id === fromTrackId) {
          const clip = track.clips.find((c) => c.id === clipId);
          if (clip) {
            clipToMove = { ...clip, startTime: newStartTime, trackId: toTrackId };
          }
          return { ...track, clips: track.clips.filter((c) => c.id !== clipId) };
        }
        return track;
      });

      // Add to destination track
      const newTimeline = {
        ...state.timeline,
        tracks: tracksAfterRemove.map((track) =>
          track.id === toTrackId && clipToMove
            ? { ...track, clips: [...track.clips, clipToMove] }
            : track
        ),
      };

      return {
        ...state,
        timeline: newTimeline,
        history: [
          ...state.history.slice(0, state.historyIndex + 1),
          newTimeline,
        ].slice(-MAX_HISTORY),
        historyIndex: Math.min(state.historyIndex + 1, MAX_HISTORY - 1),
      };
    }

    case "SPLIT_CLIP": {
      const { trackId, clipId, splitTime } = action.payload;
      const newTimeline = {
        ...state.timeline,
        tracks: state.timeline.tracks.map((track) => {
          if (track.id !== trackId) return track;

          const clip = track.clips.find((c) => c.id === clipId);
          if (!clip) return track;

          // Calculate split point relative to clip
          const clipProgress =
            (splitTime - clip.startTime) / clip.duration;
          if (clipProgress <= 0 || clipProgress >= 1) return track;

          const firstDuration = clip.duration * clipProgress;
          const secondDuration = clip.duration * (1 - clipProgress);

          const firstClip: EditorTimelineClip = {
            ...clip,
            duration: firstDuration,
            trimEnd: clip.trimEnd + (clip.sourceDuration || clip.duration) * (1 - clipProgress),
          };

          const secondClip: EditorTimelineClip = {
            ...clip,
            id: generateId(),
            startTime: splitTime,
            duration: secondDuration,
            trimStart: clip.trimStart + (clip.sourceDuration || clip.duration) * clipProgress,
          };

          return {
            ...track,
            clips: [
              ...track.clips.filter((c) => c.id !== clipId),
              firstClip,
              secondClip,
            ],
          };
        }),
      };

      return {
        ...state,
        timeline: newTimeline,
        history: [
          ...state.history.slice(0, state.historyIndex + 1),
          newTimeline,
        ].slice(-MAX_HISTORY),
        historyIndex: Math.min(state.historyIndex + 1, MAX_HISTORY - 1),
      };
    }

    case "SET_PLAYHEAD": {
      return {
        ...state,
        timeline: {
          ...state.timeline,
          playheadPosition: Math.max(
            0,
            Math.min(action.payload.position, state.timeline.duration)
          ),
        },
      };
    }

    case "SET_ZOOM": {
      return {
        ...state,
        timeline: {
          ...state.timeline,
          zoom: Math.max(MIN_ZOOM, Math.min(action.payload.zoom, MAX_ZOOM)),
        },
      };
    }

    case "TOGGLE_MUTE": {
      const newTimeline = {
        ...state.timeline,
        tracks: state.timeline.tracks.map((track) =>
          track.id === action.payload.trackId
            ? { ...track, muted: !track.muted }
            : track
        ),
      };
      return { ...state, timeline: newTimeline };
    }

    case "TOGGLE_LOCK": {
      const newTimeline = {
        ...state.timeline,
        tracks: state.timeline.tracks.map((track) =>
          track.id === action.payload.trackId
            ? { ...track, locked: !track.locked }
            : track
        ),
      };
      return { ...state, timeline: newTimeline };
    }

    case "SET_CLIP_SPEED": {
      const { trackId, clipId, speed } = action.payload;
      const clampedSpeed = Math.max(0.25, Math.min(4, speed));
      const newTimeline = {
        ...state.timeline,
        tracks: state.timeline.tracks.map((track) =>
          track.id === trackId
            ? {
                ...track,
                clips: track.clips.map((clip) => {
                  if (clip.id !== clipId) return clip;
                  // Adjust duration based on speed change
                  const originalDuration =
                    (clip.sourceDuration || clip.duration) / clip.speed;
                  return {
                    ...clip,
                    speed: clampedSpeed,
                    duration: originalDuration / clampedSpeed,
                  };
                }),
              }
            : track
        ),
      };
      return {
        ...state,
        timeline: newTimeline,
        history: [
          ...state.history.slice(0, state.historyIndex + 1),
          newTimeline,
        ].slice(-MAX_HISTORY),
        historyIndex: Math.min(state.historyIndex + 1, MAX_HISTORY - 1),
      };
    }

    case "SET_CLIP_VOLUME": {
      const { trackId, clipId, volume } = action.payload;
      const newTimeline = {
        ...state.timeline,
        tracks: state.timeline.tracks.map((track) =>
          track.id === trackId
            ? {
                ...track,
                clips: track.clips.map((clip) =>
                  clip.id === clipId
                    ? { ...clip, volume: Math.max(0, Math.min(100, volume)) }
                    : clip
                ),
              }
            : track
        ),
      };
      return {
        ...state,
        timeline: newTimeline,
        history: [
          ...state.history.slice(0, state.historyIndex + 1),
          newTimeline,
        ].slice(-MAX_HISTORY),
        historyIndex: Math.min(state.historyIndex + 1, MAX_HISTORY - 1),
      };
    }

    case "UNDO": {
      if (state.historyIndex <= 0) return state;
      return {
        ...state,
        timeline: state.history[state.historyIndex - 1],
        historyIndex: state.historyIndex - 1,
      };
    }

    case "REDO": {
      if (state.historyIndex >= state.history.length - 1) return state;
      return {
        ...state,
        timeline: state.history[state.historyIndex + 1],
        historyIndex: state.historyIndex + 1,
      };
    }

    case "SET_SELECTED_CLIP":
      return { ...state, selectedClipId: action.payload };

    case "SET_SELECTED_TRACK":
      return { ...state, selectedTrackId: action.payload };

    case "SET_PLAYING":
      return { ...state, isPlaying: action.payload };

    case "TOGGLE_SNAP":
      return { ...state, snapEnabled: !state.snapEnabled };

    case "SET_SCROLL_OFFSET":
      return {
        ...state,
        timeline: { ...state.timeline, scrollOffset: action.payload },
      };

    case "SET_DURATION":
      return {
        ...state,
        timeline: { ...state.timeline, duration: action.payload },
      };

    case "LOAD_TIMELINE":
      return {
        ...state,
        timeline: action.payload,
        history: [action.payload],
        historyIndex: 0,
      };

    default:
      return state;
  }
}

// Custom hook
export function useTimeline() {
  const initialTimeline = createInitialTimeline();
  const [state, dispatch] = useReducer(editorReducer, {
    timeline: initialTimeline,
    selectedClipId: null,
    selectedTrackId: null,
    isPlaying: false,
    snapEnabled: true,
    history: [initialTimeline],
    historyIndex: 0,
  });

  const playbackRef = useRef<number | null>(null);
  const lastTimeRef = useRef<number>(0);

  // Playback control
  useEffect(() => {
    if (state.isPlaying) {
      lastTimeRef.current = performance.now();
      const animate = (currentTime: number) => {
        const delta = (currentTime - lastTimeRef.current) / 1000;
        lastTimeRef.current = currentTime;

        const newPosition = state.timeline.playheadPosition + delta;
        if (newPosition >= state.timeline.duration) {
          dispatch({ type: "SET_PLAYING", payload: false });
          dispatch({ type: "SET_PLAYHEAD", payload: { position: 0 } });
        } else {
          dispatch({ type: "SET_PLAYHEAD", payload: { position: newPosition } });
          playbackRef.current = requestAnimationFrame(animate);
        }
      };
      playbackRef.current = requestAnimationFrame(animate);
    } else if (playbackRef.current) {
      cancelAnimationFrame(playbackRef.current);
      playbackRef.current = null;
    }

    return () => {
      if (playbackRef.current) {
        cancelAnimationFrame(playbackRef.current);
      }
    };
  }, [state.isPlaying, state.timeline.duration, state.timeline.playheadPosition]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't handle shortcuts when typing in inputs
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement
      ) {
        return;
      }

      // Space: play/pause
      if (e.code === "Space") {
        e.preventDefault();
        dispatch({ type: "SET_PLAYING", payload: !state.isPlaying });
      }

      // Arrow keys: nudge playhead
      if (e.code === "ArrowLeft") {
        e.preventDefault();
        const nudge = e.shiftKey ? 5 : 1;
        dispatch({
          type: "SET_PLAYHEAD",
          payload: { position: state.timeline.playheadPosition - nudge },
        });
      }
      if (e.code === "ArrowRight") {
        e.preventDefault();
        const nudge = e.shiftKey ? 5 : 1;
        dispatch({
          type: "SET_PLAYHEAD",
          payload: { position: state.timeline.playheadPosition + nudge },
        });
      }

      // Ctrl/Cmd + Z: Undo
      if ((e.ctrlKey || e.metaKey) && e.code === "KeyZ" && !e.shiftKey) {
        e.preventDefault();
        dispatch({ type: "UNDO" });
      }

      // Ctrl/Cmd + Shift + Z or Ctrl/Cmd + Y: Redo
      if (
        (e.ctrlKey || e.metaKey) &&
        (e.code === "KeyY" || (e.code === "KeyZ" && e.shiftKey))
      ) {
        e.preventDefault();
        dispatch({ type: "REDO" });
      }

      // Delete/Backspace: remove selected clip
      if (
        (e.code === "Delete" || e.code === "Backspace") &&
        state.selectedClipId
      ) {
        e.preventDefault();
        const track = state.timeline.tracks.find((t) =>
          t.clips.some((c) => c.id === state.selectedClipId)
        );
        if (track && !track.locked) {
          dispatch({
            type: "REMOVE_CLIP",
            payload: { trackId: track.id, clipId: state.selectedClipId },
          });
        }
      }

      // S: Split at playhead
      if (e.code === "KeyS" && !e.ctrlKey && !e.metaKey && state.selectedClipId) {
        e.preventDefault();
        const track = state.timeline.tracks.find((t) =>
          t.clips.some((c) => c.id === state.selectedClipId)
        );
        if (track && !track.locked) {
          dispatch({
            type: "SPLIT_CLIP",
            payload: {
              trackId: track.id,
              clipId: state.selectedClipId,
              splitTime: state.timeline.playheadPosition,
            },
          });
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [state.isPlaying, state.selectedClipId, state.timeline]);

  // Snap to grid/clips
  const snapToGrid = useCallback(
    (time: number): number => {
      if (!state.snapEnabled) return time;

      const snapPoints: number[] = [0, state.timeline.duration];

      // Add playhead position
      snapPoints.push(state.timeline.playheadPosition);

      // Add all clip edges
      state.timeline.tracks.forEach((track) => {
        track.clips.forEach((clip) => {
          snapPoints.push(clip.startTime);
          snapPoints.push(clip.startTime + clip.duration);
        });
      });

      // Find closest snap point
      let closestPoint = time;
      let closestDistance = SNAP_THRESHOLD;

      snapPoints.forEach((point) => {
        const distance = Math.abs(time - point);
        if (distance < closestDistance) {
          closestDistance = distance;
          closestPoint = point;
        }
      });

      return closestPoint;
    },
    [state.snapEnabled, state.timeline]
  );

  // Get selected clip
  const selectedClip = useMemo(() => {
    if (!state.selectedClipId) return null;
    for (const track of state.timeline.tracks) {
      const clip = track.clips.find((c) => c.id === state.selectedClipId);
      if (clip) return clip;
    }
    return null;
  }, [state.selectedClipId, state.timeline.tracks]);

  // Actions
  const addClip = useCallback(
    (trackId: string, clip: Omit<EditorTimelineClip, "id" | "trackId">) => {
      const fullClip: EditorTimelineClip = {
        ...clip,
        id: generateId(),
        trackId,
      };
      dispatch({ type: "ADD_CLIP", payload: { trackId, clip: fullClip } });
      return fullClip.id;
    },
    []
  );

  const removeClip = useCallback((trackId: string, clipId: string) => {
    dispatch({ type: "REMOVE_CLIP", payload: { trackId, clipId } });
  }, []);

  const updateClip = useCallback(
    (
      trackId: string,
      clipId: string,
      updates: Partial<EditorTimelineClip>
    ) => {
      dispatch({ type: "UPDATE_CLIP", payload: { trackId, clipId, updates } });
    },
    []
  );

  const moveClip = useCallback(
    (
      fromTrackId: string,
      toTrackId: string,
      clipId: string,
      newStartTime: number
    ) => {
      const snappedTime = snapToGrid(newStartTime);
      dispatch({
        type: "MOVE_CLIP",
        payload: { fromTrackId, toTrackId, clipId, newStartTime: snappedTime },
      });
    },
    [snapToGrid]
  );

  const splitClip = useCallback(
    (trackId: string, clipId: string, splitTime?: number) => {
      const time = splitTime ?? state.timeline.playheadPosition;
      dispatch({
        type: "SPLIT_CLIP",
        payload: { trackId, clipId, splitTime: time },
      });
    },
    [state.timeline.playheadPosition]
  );

  const setPlayhead = useCallback((position: number) => {
    dispatch({ type: "SET_PLAYHEAD", payload: { position } });
  }, []);

  const setZoom = useCallback((zoom: number) => {
    dispatch({ type: "SET_ZOOM", payload: { zoom } });
  }, []);

  const toggleMute = useCallback((trackId: string) => {
    dispatch({ type: "TOGGLE_MUTE", payload: { trackId } });
  }, []);

  const toggleLock = useCallback((trackId: string) => {
    dispatch({ type: "TOGGLE_LOCK", payload: { trackId } });
  }, []);

  const setClipSpeed = useCallback(
    (trackId: string, clipId: string, speed: number) => {
      dispatch({
        type: "SET_CLIP_SPEED",
        payload: { trackId, clipId, speed },
      });
    },
    []
  );

  const setClipVolume = useCallback(
    (trackId: string, clipId: string, volume: number) => {
      dispatch({
        type: "SET_CLIP_VOLUME",
        payload: { trackId, clipId, volume },
      });
    },
    []
  );

  const selectClip = useCallback((clipId: string | null) => {
    dispatch({ type: "SET_SELECTED_CLIP", payload: clipId });
  }, []);

  const selectTrack = useCallback((trackId: string | null) => {
    dispatch({ type: "SET_SELECTED_TRACK", payload: trackId });
  }, []);

  const togglePlay = useCallback(() => {
    dispatch({ type: "SET_PLAYING", payload: !state.isPlaying });
  }, [state.isPlaying]);

  const play = useCallback(() => {
    dispatch({ type: "SET_PLAYING", payload: true });
  }, []);

  const pause = useCallback(() => {
    dispatch({ type: "SET_PLAYING", payload: false });
  }, []);

  const toggleSnap = useCallback(() => {
    dispatch({ type: "TOGGLE_SNAP" });
  }, []);

  const undo = useCallback(() => {
    dispatch({ type: "UNDO" });
  }, []);

  const redo = useCallback(() => {
    dispatch({ type: "REDO" });
  }, []);

  const setScrollOffset = useCallback((offset: number) => {
    dispatch({ type: "SET_SCROLL_OFFSET", payload: offset });
  }, []);

  const setDuration = useCallback((duration: number) => {
    dispatch({ type: "SET_DURATION", payload: duration });
  }, []);

  const loadTimeline = useCallback((timeline: EditorTimeline) => {
    dispatch({ type: "LOAD_TIMELINE", payload: timeline });
  }, []);

  // Apply B-roll suggestions from AI
  const applyBRollSuggestions = useCallback(
    (suggestions: BRollSuggestion[]) => {
      const brollTrack = state.timeline.tracks.find((t) => t.type === "broll");
      if (!brollTrack) return;

      suggestions
        .filter((s) => s.accepted && s.mediaUrl)
        .forEach((suggestion) => {
          addClip(brollTrack.id, {
            type: "broll",
            sourceUrl: suggestion.mediaUrl!,
            thumbnailUrl: suggestion.thumbnailUrl,
            name: suggestion.searchQuery,
            startTime: suggestion.timestamp,
            duration: suggestion.duration,
            trimStart: 0,
            trimEnd: 0,
            speed: 1,
            volume: 100,
          });
        });
    },
    [state.timeline.tracks, addClip]
  );

  // Convert timeline position to pixels
  const timeToPixels = useCallback(
    (time: number): number => {
      return time * state.timeline.zoom;
    },
    [state.timeline.zoom]
  );

  // Convert pixels to timeline position
  const pixelsToTime = useCallback(
    (pixels: number): number => {
      return pixels / state.timeline.zoom;
    },
    [state.timeline.zoom]
  );

  return {
    // State
    timeline: state.timeline,
    selectedClipId: state.selectedClipId,
    selectedTrackId: state.selectedTrackId,
    selectedClip,
    isPlaying: state.isPlaying,
    snapEnabled: state.snapEnabled,
    canUndo: state.historyIndex > 0,
    canRedo: state.historyIndex < state.history.length - 1,

    // Clip actions
    addClip,
    removeClip,
    updateClip,
    moveClip,
    splitClip,
    setClipSpeed,
    setClipVolume,

    // Selection
    selectClip,
    selectTrack,

    // Playback
    togglePlay,
    play,
    pause,
    setPlayhead,

    // View
    setZoom,
    setScrollOffset,
    timeToPixels,
    pixelsToTime,
    snapToGrid,
    toggleSnap,

    // Track controls
    toggleMute,
    toggleLock,

    // History
    undo,
    redo,

    // AI
    applyBRollSuggestions,

    // Load/save
    setDuration,
    loadTimeline,
  };
}

export type TimelineHook = ReturnType<typeof useTimeline>;
