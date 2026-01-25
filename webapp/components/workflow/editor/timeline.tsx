"use client";

import { useCallback, useRef, useEffect, useState } from "react";
import { cn } from "@/lib/utils";
import { formatDuration } from "@/lib/utils";
import { TimelineTrack } from "./timeline-track";
import type { EditorTimeline, EditorTimelineClip } from "@/types/workflow";
import type { TimelineHook } from "@/hooks/use-timeline";

interface TimelineProps {
  timeline: EditorTimeline;
  selectedClipId: string | null;
  selectedTrackId: string | null;
  timelineHook: TimelineHook;
}

export function Timeline({
  timeline,
  selectedClipId,
  selectedTrackId,
  timelineHook,
}: TimelineProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [isDraggingPlayhead, setIsDraggingPlayhead] = useState(false);

  const {
    selectClip,
    selectTrack,
    moveClip,
    updateClip,
    toggleMute,
    toggleLock,
    addClip,
    setPlayhead,
    timeToPixels,
    pixelsToTime,
    setScrollOffset,
  } = timelineHook;

  const timelineWidth = timeToPixels(timeline.duration);
  const playheadPosition = timeToPixels(timeline.playheadPosition);

  // Handle ruler click to set playhead
  const handleRulerClick = useCallback(
    (e: React.MouseEvent) => {
      const rect = e.currentTarget.getBoundingClientRect();
      const relativeX = e.clientX - rect.left + (containerRef.current?.scrollLeft || 0);
      const time = pixelsToTime(relativeX);
      setPlayhead(Math.max(0, Math.min(time, timeline.duration)));
    },
    [pixelsToTime, setPlayhead, timeline.duration]
  );

  // Handle playhead drag
  const handlePlayheadMouseDown = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      setIsDraggingPlayhead(true);

      const handleMouseMove = (e: MouseEvent) => {
        const container = containerRef.current;
        if (!container) return;

        const rect = container.getBoundingClientRect();
        const relativeX = e.clientX - rect.left + container.scrollLeft - 144; // Account for track controls width
        const time = pixelsToTime(relativeX);
        setPlayhead(Math.max(0, Math.min(time, timeline.duration)));
      };

      const handleMouseUp = () => {
        setIsDraggingPlayhead(false);
        document.removeEventListener("mousemove", handleMouseMove);
        document.removeEventListener("mouseup", handleMouseUp);
      };

      document.addEventListener("mousemove", handleMouseMove);
      document.addEventListener("mouseup", handleMouseUp);
    },
    [pixelsToTime, setPlayhead, timeline.duration]
  );

  // Scroll handler to sync offset
  const handleScroll = useCallback(
    (e: React.UIEvent<HTMLDivElement>) => {
      setScrollOffset(e.currentTarget.scrollLeft);
    },
    [setScrollOffset]
  );

  // Generate time markers
  const generateTimeMarkers = useCallback(() => {
    const markers: { time: number; label: string; isMajor: boolean }[] = [];
    const step = timeline.zoom < 30 ? 30 : timeline.zoom < 60 ? 10 : 5;
    const subStep = step / 5;

    for (let t = 0; t <= timeline.duration; t += subStep) {
      const isMajor = t % step === 0;
      markers.push({
        time: t,
        label: isMajor ? formatDuration(t) : "",
        isMajor,
      });
    }

    return markers;
  }, [timeline.duration, timeline.zoom]);

  const timeMarkers = generateTimeMarkers();

  // Handle drop on track
  const handleDropClip = useCallback(
    (trackId: string, time: number, clipData: Partial<EditorTimelineClip>) => {
      addClip(trackId, {
        type: clipData.type || "broll",
        sourceUrl: clipData.sourceUrl || "",
        thumbnailUrl: clipData.thumbnailUrl,
        name: clipData.name || "New Clip",
        startTime: time,
        duration: clipData.duration || 5,
        trimStart: 0,
        trimEnd: 0,
        speed: 1,
        volume: 100,
      });
    },
    [addClip]
  );

  // Handle clip resize
  const handleResizeClip = useCallback(
    (trackId: string, clipId: string, updates: Partial<EditorTimelineClip>) => {
      updateClip(trackId, clipId, updates);
    },
    [updateClip]
  );

  // Handle clip move (find source track and delegate)
  const handleMoveClip = useCallback(
    (trackId: string, clipId: string, newStartTime: number) => {
      const track = timeline.tracks.find((t) => t.id === trackId);
      const clip = track?.clips.find((c) => c.id === clipId);
      if (!clip || !track) return;

      // For now, just update within same track
      moveClip(trackId, trackId, clipId, newStartTime);
    },
    [timeline.tracks, moveClip]
  );

  // Auto-scroll to keep playhead visible during playback
  useEffect(() => {
    if (!containerRef.current) return;
    const container = containerRef.current;
    const playheadX = playheadPosition + 144; // Account for track controls

    // Only auto-scroll if playhead is near edge
    const viewportWidth = container.clientWidth;
    const scrollLeft = container.scrollLeft;
    const margin = 100;

    if (playheadX > scrollLeft + viewportWidth - margin) {
      container.scrollLeft = playheadX - viewportWidth + margin * 2;
    } else if (playheadX < scrollLeft + margin + 144) {
      container.scrollLeft = Math.max(0, playheadX - margin - 144);
    }
  }, [playheadPosition]);

  return (
    <div className="flex flex-col h-full border border-border rounded-lg overflow-hidden bg-card">
      {/* Time ruler */}
      <div className="flex border-b border-border bg-muted/30">
        {/* Track controls header spacer */}
        <div className="w-36 shrink-0 border-r border-border" />

        {/* Time ruler */}
        <div
          className="flex-1 h-6 relative overflow-hidden cursor-pointer"
          onClick={handleRulerClick}
          style={{ scrollSnapType: "x mandatory" }}
        >
          <div
            className="absolute inset-0"
            style={{ width: `${timelineWidth}px`, left: -timeline.scrollOffset }}
          >
            {timeMarkers.map((marker, i) => (
              <div
                key={i}
                className="absolute top-0 flex flex-col items-center"
                style={{ left: `${timeToPixels(marker.time)}px` }}
              >
                <div
                  className={cn(
                    "w-px",
                    marker.isMajor
                      ? "h-3 bg-muted-foreground/60"
                      : "h-2 bg-muted-foreground/30"
                  )}
                />
                {marker.label && (
                  <span className="text-[10px] text-muted-foreground mt-0.5 -translate-x-1/2">
                    {marker.label}
                  </span>
                )}
              </div>
            ))}

            {/* Playhead marker on ruler */}
            <div
              className="absolute top-0 w-3 h-3 -translate-x-1/2 cursor-ew-resize z-20"
              style={{ left: `${playheadPosition}px` }}
              onMouseDown={handlePlayheadMouseDown}
            >
              <div className="w-0 h-0 border-l-[6px] border-l-transparent border-r-[6px] border-r-transparent border-t-[8px] border-t-primary" />
            </div>
          </div>
        </div>
      </div>

      {/* Tracks container */}
      <div
        ref={containerRef}
        className="flex-1 overflow-auto relative"
        onScroll={handleScroll}
      >
        <div style={{ width: `${timelineWidth + 144}px`, minWidth: "100%" }}>
          {timeline.tracks.map((track) => (
            <TimelineTrack
              key={track.id}
              track={track}
              zoom={timeline.zoom}
              selectedClipId={selectedClipId}
              isSelected={selectedTrackId === track.id}
              onSelectClip={selectClip}
              onSelectTrack={() => selectTrack(track.id)}
              onMoveClip={(clipId, newStartTime) =>
                handleMoveClip(track.id, clipId, newStartTime)
              }
              onResizeClip={(clipId, updates) =>
                handleResizeClip(track.id, clipId, updates)
              }
              onToggleMute={() => toggleMute(track.id)}
              onToggleLock={() => toggleLock(track.id)}
              onDropClip={handleDropClip}
              pixelsToTime={pixelsToTime}
              timeToPixels={timeToPixels}
              duration={timeline.duration}
            />
          ))}
        </div>

        {/* Playhead line */}
        <div
          className={cn(
            "absolute top-0 bottom-0 w-0.5 bg-primary pointer-events-none z-10",
            isDraggingPlayhead && "bg-primary/80"
          )}
          style={{ left: `${playheadPosition + 144}px` }}
        />
      </div>
    </div>
  );
}
