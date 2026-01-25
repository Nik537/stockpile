"use client";

import { useCallback, useRef } from "react";
import { cn } from "@/lib/utils";
import type { EditorTimelineTrack, EditorTimelineClip } from "@/types/workflow";
import { TimelineClip } from "./timeline-clip";
import {
  Volume2,
  VolumeX,
  Lock,
  Unlock,
  Film,
  Music,
  Image,
  User,
} from "lucide-react";

interface TimelineTrackProps {
  track: EditorTimelineTrack;
  zoom: number;
  selectedClipId: string | null;
  isSelected: boolean;
  onSelectClip: (clipId: string) => void;
  onSelectTrack: () => void;
  onMoveClip: (clipId: string, newStartTime: number) => void;
  onResizeClip: (
    clipId: string,
    updates: { startTime?: number; duration?: number; trimStart?: number; trimEnd?: number }
  ) => void;
  onToggleMute: () => void;
  onToggleLock: () => void;
  onDropClip?: (trackId: string, time: number, clip: Partial<EditorTimelineClip>) => void;
  pixelsToTime: (pixels: number) => number;
  timeToPixels: (time: number) => number;
  duration: number;
}

const trackTypeIcons = {
  main: Film,
  broll: Image,
  audio: Music,
  avatar: User,
};

const trackTypeColors: Record<string, string> = {
  main: "text-blue-500",
  broll: "text-purple-500",
  audio: "text-green-500",
  avatar: "text-orange-500",
};

export function TimelineTrack({
  track,
  zoom,
  selectedClipId,
  isSelected,
  onSelectClip,
  onSelectTrack,
  onMoveClip,
  onResizeClip,
  onToggleMute,
  onToggleLock,
  onDropClip,
  pixelsToTime,
  timeToPixels,
  duration,
}: TimelineTrackProps) {
  const trackRef = useRef<HTMLDivElement>(null);

  const Icon = trackTypeIcons[track.type] || Film;
  const iconColor = trackTypeColors[track.type] || trackTypeColors.main;

  const handleTrackClick = useCallback(
    (e: React.MouseEvent) => {
      // Only select track if clicking on empty space
      if (e.target === trackRef.current || e.target === trackRef.current?.children[0]) {
        onSelectTrack();
      }
    },
    [onSelectTrack]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      if (track.locked || !onDropClip) return;

      const rect = trackRef.current?.getBoundingClientRect();
      if (!rect) return;

      const relativeX = e.clientX - rect.left;
      const time = pixelsToTime(relativeX);

      // Try to get clip data from drag event
      const clipData = e.dataTransfer.getData("application/json");
      if (clipData) {
        try {
          const clip = JSON.parse(clipData);
          onDropClip(track.id, time, clip);
        } catch {
          // Invalid JSON, ignore
        }
      }
    },
    [track, pixelsToTime, onDropClip]
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
  }, []);

  const timelineWidth = timeToPixels(duration);

  return (
    <div
      className={cn(
        "flex items-stretch border-b border-border transition-colors",
        isSelected && "bg-accent/30",
        track.locked && "opacity-60"
      )}
      style={{ height: track.height || 50 }}
    >
      {/* Track controls */}
      <div className="w-36 flex items-center gap-2 px-3 border-r border-border bg-card/50 shrink-0">
        <Icon className={cn("h-4 w-4", iconColor)} />
        <span className="text-xs font-medium text-foreground truncate flex-1">
          {track.name}
        </span>

        {/* Mute button */}
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onToggleMute();
          }}
          className={cn(
            "p-1 rounded hover:bg-muted transition-colors",
            track.muted ? "text-destructive" : "text-muted-foreground"
          )}
          title={track.muted ? "Unmute" : "Mute"}
        >
          {track.muted ? (
            <VolumeX className="h-3.5 w-3.5" />
          ) : (
            <Volume2 className="h-3.5 w-3.5" />
          )}
        </button>

        {/* Lock button */}
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onToggleLock();
          }}
          className={cn(
            "p-1 rounded hover:bg-muted transition-colors",
            track.locked ? "text-yellow-500" : "text-muted-foreground"
          )}
          title={track.locked ? "Unlock" : "Lock"}
        >
          {track.locked ? (
            <Lock className="h-3.5 w-3.5" />
          ) : (
            <Unlock className="h-3.5 w-3.5" />
          )}
        </button>
      </div>

      {/* Track timeline area */}
      <div
        ref={trackRef}
        className="flex-1 relative bg-background/50 overflow-hidden"
        onClick={handleTrackClick}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
      >
        {/* Grid lines */}
        <div
          className="absolute inset-0 pointer-events-none"
          style={{ width: `${timelineWidth}px` }}
        >
          {Array.from({ length: Math.ceil(duration / 10) }).map((_, i) => (
            <div
              key={i}
              className="absolute top-0 bottom-0 w-px bg-border/30"
              style={{ left: `${timeToPixels(i * 10)}px` }}
            />
          ))}
        </div>

        {/* Clips */}
        {track.clips.map((clip) => (
          <TimelineClip
            key={clip.id}
            clip={clip}
            zoom={zoom}
            trackColor={track.color || "bg-gray-500"}
            isSelected={selectedClipId === clip.id}
            isLocked={track.locked}
            onSelect={onSelectClip}
            onMove={(clipId, newStartTime) => onMoveClip(clipId, newStartTime)}
            onResize={(clipId, updates) => onResizeClip(clipId, updates)}
            pixelsToTime={pixelsToTime}
          />
        ))}

        {/* Empty track message */}
        {track.clips.length === 0 && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <span className="text-xs text-muted-foreground/50">
              Drag clips here
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
