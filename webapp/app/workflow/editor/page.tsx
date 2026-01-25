"use client";

import { useState, useCallback, useMemo, useEffect } from "react";
import { Header } from "@/components/layout/header";
import { EnhancedStepIndicator } from "@/components/workflow/enhanced-step-indicator";
import { WorkflowNav } from "@/components/workflow/workflow-nav";
import { generateId } from "@/lib/utils";
import { useTimeline } from "@/hooks/use-timeline";
import { useWorkflowStore, useEditorData, useBRollData, useAvatarTtsData } from "@/lib/workflow-store";
import {
  Timeline,
  VideoPreview,
  ClipInspector,
  Toolbar,
  PlaybackControls,
  BRollSuggestions,
} from "@/components/workflow/editor";
import type { BRollSuggestion, EditorTimelineClip } from "@/types/workflow";

// Sample data for demonstration
const SAMPLE_CLIPS: Omit<EditorTimelineClip, "id" | "trackId">[] = [
  {
    type: "video",
    name: "Main Video Intro",
    sourceUrl: "",
    startTime: 0,
    duration: 30,
    trimStart: 0,
    trimEnd: 0,
    speed: 1,
    volume: 100,
  },
  {
    type: "video",
    name: "Main Content",
    sourceUrl: "",
    startTime: 30,
    duration: 90,
    trimStart: 0,
    trimEnd: 0,
    speed: 1,
    volume: 100,
  },
  {
    type: "video",
    name: "Outro",
    sourceUrl: "",
    startTime: 120,
    duration: 60,
    trimStart: 0,
    trimEnd: 0,
    speed: 1,
    volume: 100,
  },
];

const SAMPLE_AUDIO: Omit<EditorTimelineClip, "id" | "trackId">[] = [
  {
    type: "audio",
    name: "Background Music",
    sourceUrl: "",
    startTime: 0,
    duration: 180,
    trimStart: 0,
    trimEnd: 0,
    speed: 1,
    volume: 30,
  },
];

export default function EditorPage() {
  // Workflow store
  const editorData = useEditorData();
  const brollData = useBRollData();
  const avatarTtsData = useAvatarTtsData();
  const setTimeline = useWorkflowStore((state) => state.setTimeline);
  const markStepComplete = useWorkflowStore((state) => state.markStepComplete);

  const timelineHook = useTimeline();
  const {
    timeline,
    selectedClipId,
    selectedTrackId,
    selectedClip,
    isPlaying,
    snapEnabled,
    canUndo,
    canRedo,
    addClip,
    removeClip,
    updateClip,
    splitClip,
    selectClip,
    togglePlay,
    setPlayhead,
    setZoom,
    toggleMute,
    toggleLock,
    undo,
    redo,
    toggleSnap,
    applyBRollSuggestions,
  } = timelineHook;

  const [currentTool, setCurrentTool] = useState<"select" | "move">("select");
  const [volume, setVolume] = useState(100);
  const [isMuted, setIsMuted] = useState(false);
  const [showBRollPanel, setShowBRollPanel] = useState(true);
  const [brollSuggestions, setBrollSuggestions] = useState<BRollSuggestion[]>([]);
  const [isLoadingSuggestions, setIsLoadingSuggestions] = useState(false);
  const [hasLoadedSampleData, setHasLoadedSampleData] = useState(false);

  // Create B-roll clips from workflow store data
  const brollClips: Omit<EditorTimelineClip, "id" | "trackId">[] = useMemo(() => {
    return brollData.selectedMedia.map((item, index) => ({
      type: "broll" as const,
      name: item.title,
      sourceUrl: item.downloadUrl || "",
      thumbnailUrl: item.thumbnailUrl,
      startTime: 5 + index * 20,
      duration: item.duration || 8,
      trimStart: 0,
      trimEnd: 0,
      speed: 1,
    }));
  }, [brollData.selectedMedia]);

  // Load sample data on first render
  useEffect(() => {
    if (!hasLoadedSampleData) {
      // Add sample clips to tracks
      const mainTrack = timeline.tracks.find((t) => t.type === "main");
      const brollTrack = timeline.tracks.find((t) => t.type === "broll");
      const audioTrack = timeline.tracks.find((t) => t.type === "audio");

      if (mainTrack && mainTrack.clips.length === 0) {
        SAMPLE_CLIPS.forEach((clip) => addClip(mainTrack.id, clip));
      }
      if (brollTrack && brollTrack.clips.length === 0) {
        // Use B-roll from workflow store if available, otherwise use samples
        const clipsToAdd = brollClips.length > 0 ? brollClips : [
          {
            type: "broll" as const,
            name: "City skyline aerial",
            sourceUrl: "",
            thumbnailUrl: "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=200",
            startTime: 5,
            duration: 8,
            trimStart: 0,
            trimEnd: 0,
            speed: 1,
          },
          {
            type: "broll" as const,
            name: "Office workspace",
            sourceUrl: "",
            thumbnailUrl: "https://images.unsplash.com/photo-1497366216548-37526070297c?w=200",
            startTime: 45,
            duration: 6,
            trimStart: 0,
            trimEnd: 0,
            speed: 1,
          },
        ];
        clipsToAdd.forEach((clip) => addClip(brollTrack.id, clip));
      }
      if (audioTrack && audioTrack.clips.length === 0) {
        SAMPLE_AUDIO.forEach((clip) => addClip(audioTrack.id, clip));
      }

      setHasLoadedSampleData(true);
    }
  }, [hasLoadedSampleData, timeline.tracks, addClip, brollClips]);

  // Sync timeline to workflow store
  useEffect(() => {
    if (timeline.tracks.some(t => t.clips.length > 0)) {
      setTimeline(timeline);
      markStepComplete("editor");
    }
  }, [timeline, setTimeline, markStepComplete]);

  // Find the track for the selected clip
  const selectedTrack = useMemo(() => {
    if (!selectedClipId) return null;
    return timeline.tracks.find((t) => t.clips.some((c) => c.id === selectedClipId));
  }, [selectedClipId, timeline.tracks]);

  // Handle clip update from inspector
  const handleClipUpdate = useCallback(
    (updates: Partial<EditorTimelineClip>) => {
      if (selectedClipId && selectedTrack) {
        updateClip(selectedTrack.id, selectedClipId, updates);
      }
    },
    [selectedClipId, selectedTrack, updateClip]
  );

  // Handle clip deletion
  const handleDeleteClip = useCallback(() => {
    if (selectedClipId && selectedTrack) {
      removeClip(selectedTrack.id, selectedClipId);
    }
  }, [selectedClipId, selectedTrack, removeClip]);

  // Handle clip split
  const handleSplitClip = useCallback(() => {
    if (selectedClipId && selectedTrack) {
      splitClip(selectedTrack.id, selectedClipId);
    }
  }, [selectedClipId, selectedTrack, splitClip]);

  // Handle clip duplication
  const handleDuplicateClip = useCallback(() => {
    if (!selectedClip || !selectedTrack) return;

    const newClip: Omit<EditorTimelineClip, "id" | "trackId"> = {
      ...selectedClip,
      startTime: selectedClip.startTime + selectedClip.duration + 0.5,
    };
    addClip(selectedTrack.id, newClip);
  }, [selectedClip, selectedTrack, addClip]);

  // Generate AI B-roll suggestions (simulated)
  const handleGenerateSuggestions = useCallback(async () => {
    setIsLoadingSuggestions(true);

    // Simulate AI processing delay
    await new Promise((resolve) => setTimeout(resolve, 2000));

    // Generate sample suggestions based on timeline
    const suggestions: BRollSuggestion[] = [
      {
        id: generateId(),
        timestamp: 15,
        duration: 5,
        searchQuery: "technology innovation",
        reason:
          "This section discusses new technology - relevant B-roll will enhance viewer engagement",
        thumbnailUrl:
          "https://images.unsplash.com/photo-1518770660439-4636190af475?w=200",
        accepted: false,
      },
      {
        id: generateId(),
        timestamp: 60,
        duration: 6,
        searchQuery: "business meeting teamwork",
        reason:
          "Team collaboration is mentioned here - showing people working together reinforces the message",
        thumbnailUrl:
          "https://images.unsplash.com/photo-1552664730-d307ca884978?w=200",
        accepted: false,
      },
      {
        id: generateId(),
        timestamp: 100,
        duration: 4,
        searchQuery: "success celebration",
        reason:
          "Conclusion section talks about achievements - celebratory imagery creates positive association",
        thumbnailUrl:
          "https://images.unsplash.com/photo-1533750349088-cd871a92f312?w=200",
        accepted: false,
      },
    ];

    setBrollSuggestions(suggestions);
    setIsLoadingSuggestions(false);
  }, []);

  // Accept B-roll suggestion
  const handleAcceptSuggestion = useCallback(
    (suggestionId: string) => {
      setBrollSuggestions((prev) =>
        prev.map((s) =>
          s.id === suggestionId ? { ...s, accepted: true } : s
        )
      );

      const suggestion = brollSuggestions.find((s) => s.id === suggestionId);
      if (suggestion) {
        const brollTrack = timeline.tracks.find((t) => t.type === "broll");
        if (brollTrack) {
          addClip(brollTrack.id, {
            type: "broll",
            name: suggestion.searchQuery,
            sourceUrl: suggestion.mediaUrl || "",
            thumbnailUrl: suggestion.thumbnailUrl,
            startTime: suggestion.timestamp,
            duration: suggestion.duration,
            trimStart: 0,
            trimEnd: 0,
            speed: 1,
          });
        }
      }
    },
    [brollSuggestions, timeline.tracks, addClip]
  );

  // Reject B-roll suggestion
  const handleRejectSuggestion = useCallback((suggestionId: string) => {
    setBrollSuggestions((prev) => prev.filter((s) => s.id !== suggestionId));
  }, []);

  // Accept all pending suggestions
  const handleAcceptAll = useCallback(() => {
    brollSuggestions
      .filter((s) => !s.accepted)
      .forEach((suggestion) => {
        handleAcceptSuggestion(suggestion.id);
      });
  }, [brollSuggestions, handleAcceptSuggestion]);

  // Reject all pending suggestions
  const handleRejectAll = useCallback(() => {
    setBrollSuggestions((prev) => prev.filter((s) => s.accepted));
  }, []);

  return (
    <div className="flex flex-col h-screen">
      <Header
        title="Timeline Editor"
        subtitle="Arrange and edit your video timeline"
      />

      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Step Indicator */}
        <div className="p-4 pb-0">
          <EnhancedStepIndicator />
        </div>

        {/* Toolbar */}
        <Toolbar
          hasSelection={!!selectedClipId}
          isLocked={selectedTrack?.locked || false}
          snapEnabled={snapEnabled}
          canUndo={canUndo}
          canRedo={canRedo}
          onSplit={handleSplitClip}
          onDelete={handleDeleteClip}
          onDuplicate={handleDuplicateClip}
          onUndo={undo}
          onRedo={redo}
          onToggleSnap={toggleSnap}
          currentTool={currentTool}
          onToolChange={setCurrentTool}
        />

        {/* Main content area */}
        <div className="flex-1 grid grid-cols-[1fr_280px] gap-4 p-4 overflow-hidden">
          {/* Left side: Preview + Timeline */}
          <div className="flex flex-col gap-4 min-w-0 overflow-hidden">
            {/* Video Preview */}
            <div className="flex-shrink-0">
              <VideoPreview
                timeline={timeline}
                isPlaying={isPlaying}
                onTimeUpdate={setPlayhead}
              />
            </div>

            {/* Playback Controls */}
            <div className="flex-shrink-0">
              <PlaybackControls
                currentTime={timeline.playheadPosition}
                duration={timeline.duration}
                isPlaying={isPlaying}
                volume={volume}
                zoom={timeline.zoom}
                isMuted={isMuted}
                onTogglePlay={togglePlay}
                onSeek={setPlayhead}
                onSkipBack={(s) =>
                  setPlayhead(Math.max(0, timeline.playheadPosition - (s || 1)))
                }
                onSkipForward={(s) =>
                  setPlayhead(
                    Math.min(
                      timeline.duration,
                      timeline.playheadPosition + (s || 1)
                    )
                  )
                }
                onJumpToStart={() => setPlayhead(0)}
                onJumpToEnd={() => setPlayhead(timeline.duration)}
                onVolumeChange={setVolume}
                onZoomChange={setZoom}
                onToggleMute={() => setIsMuted(!isMuted)}
              />
            </div>

            {/* Timeline */}
            <div className="flex-1 min-h-[200px] overflow-hidden">
              <Timeline
                timeline={timeline}
                selectedClipId={selectedClipId}
                selectedTrackId={selectedTrackId}
                timelineHook={timelineHook}
              />
            </div>
          </div>

          {/* Right side: Inspector + B-roll Suggestions */}
          <div className="flex flex-col gap-4 overflow-hidden">
            {/* Clip Inspector */}
            <div className="flex-shrink-0 h-[280px] border border-border rounded-lg bg-card overflow-hidden">
              <ClipInspector
                clip={selectedClip}
                onUpdate={handleClipUpdate}
                onDelete={handleDeleteClip}
                onDuplicate={handleDuplicateClip}
                onSplit={handleSplitClip}
                isLocked={selectedTrack?.locked}
              />
            </div>

            {/* B-Roll Suggestions Panel */}
            <div className="flex-1 border border-border rounded-lg bg-card overflow-hidden">
              <BRollSuggestions
                suggestions={brollSuggestions}
                isLoading={isLoadingSuggestions}
                onAccept={handleAcceptSuggestion}
                onReject={handleRejectSuggestion}
                onAcceptAll={handleAcceptAll}
                onRejectAll={handleRejectAll}
                onGenerateSuggestions={handleGenerateSuggestions}
                onRefresh={handleGenerateSuggestions}
              />
            </div>
          </div>
        </div>

        {/* Navigation */}
        <div className="p-4 border-t border-border">
          <WorkflowNav />
        </div>
      </div>
    </div>
  );
}
