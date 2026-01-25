"use client";

import { useCallback, useRef, useState } from "react";
import { cn } from "@/lib/utils";
import type { EditorTimelineClip } from "@/types/workflow";
import { Film, Music, User, Image } from "lucide-react";

interface TimelineClipProps {
  clip: EditorTimelineClip;
  zoom: number;
  trackColor: string;
  isSelected: boolean;
  isLocked: boolean;
  onSelect: (clipId: string) => void;
  onMove: (clipId: string, newStartTime: number) => void;
  onResize: (
    clipId: string,
    updates: { startTime?: number; duration?: number; trimStart?: number; trimEnd?: number }
  ) => void;
  pixelsToTime: (pixels: number) => number;
}

const clipTypeIcons = {
  video: Film,
  audio: Music,
  broll: Image,
  avatar: User,
};

const clipTypeColors: Record<string, string> = {
  video: "from-blue-500/90 to-blue-600/90",
  audio: "from-green-500/90 to-green-600/90",
  broll: "from-purple-500/90 to-purple-600/90",
  avatar: "from-orange-500/90 to-orange-600/90",
};

export function TimelineClip({
  clip,
  zoom,
  isSelected,
  isLocked,
  onSelect,
  onMove,
  onResize,
  pixelsToTime,
}: TimelineClipProps) {
  const clipRef = useRef<HTMLDivElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isResizing, setIsResizing] = useState<"left" | "right" | null>(null);
  const dragStartRef = useRef({ x: 0, startTime: 0, duration: 0, trimStart: 0 });

  const width = clip.duration * zoom;
  const left = clip.startTime * zoom;

  const Icon = clipTypeIcons[clip.type] || Film;
  const colorClass = clipTypeColors[clip.type] || clipTypeColors.video;

  // Handle drag start
  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (isLocked) return;
      e.stopPropagation();

      onSelect(clip.id);

      // Check if clicking on resize handles
      const rect = clipRef.current?.getBoundingClientRect();
      if (rect) {
        const relativeX = e.clientX - rect.left;
        if (relativeX < 8) {
          // Left resize handle
          setIsResizing("left");
          dragStartRef.current = {
            x: e.clientX,
            startTime: clip.startTime,
            duration: clip.duration,
            trimStart: clip.trimStart,
          };
        } else if (relativeX > rect.width - 8) {
          // Right resize handle
          setIsResizing("right");
          dragStartRef.current = {
            x: e.clientX,
            startTime: clip.startTime,
            duration: clip.duration,
            trimStart: clip.trimStart,
          };
        } else {
          // Drag the whole clip
          setIsDragging(true);
          dragStartRef.current = {
            x: e.clientX,
            startTime: clip.startTime,
            duration: clip.duration,
            trimStart: clip.trimStart,
          };
        }
      }
    },
    [clip, isLocked, onSelect]
  );

  // Handle drag
  const handleMouseMove = useCallback(
    (e: MouseEvent) => {
      if (isLocked) return;

      const deltaX = e.clientX - dragStartRef.current.x;
      const deltaTime = pixelsToTime(deltaX);

      if (isDragging) {
        const newStartTime = Math.max(0, dragStartRef.current.startTime + deltaTime);
        onMove(clip.id, newStartTime);
      } else if (isResizing === "left") {
        // Resize from left (adjust start time and duration)
        const newStartTime = Math.max(0, dragStartRef.current.startTime + deltaTime);
        const startDelta = newStartTime - dragStartRef.current.startTime;
        const newDuration = dragStartRef.current.duration - startDelta;

        if (newDuration > 0.1) {
          onResize(clip.id, {
            startTime: newStartTime,
            duration: newDuration,
            trimStart: dragStartRef.current.trimStart + startDelta,
          });
        }
      } else if (isResizing === "right") {
        // Resize from right (only adjust duration)
        const newDuration = Math.max(0.1, dragStartRef.current.duration + deltaTime);
        onResize(clip.id, { duration: newDuration });
      }
    },
    [clip.id, isDragging, isResizing, isLocked, onMove, onResize, pixelsToTime]
  );

  // Handle drag end
  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
    setIsResizing(null);
  }, []);

  // Attach global mouse events when dragging
  const handleDragStart = useCallback(
    (e: React.MouseEvent) => {
      handleMouseDown(e);

      const handleGlobalMove = (e: MouseEvent) => handleMouseMove(e);
      const handleGlobalUp = () => {
        handleMouseUp();
        document.removeEventListener("mousemove", handleGlobalMove);
        document.removeEventListener("mouseup", handleGlobalUp);
      };

      document.addEventListener("mousemove", handleGlobalMove);
      document.addEventListener("mouseup", handleGlobalUp);
    },
    [handleMouseDown, handleMouseMove, handleMouseUp]
  );

  return (
    <div
      ref={clipRef}
      className={cn(
        "absolute top-1 bottom-1 rounded-md cursor-pointer transition-all",
        "bg-gradient-to-b shadow-sm overflow-hidden group",
        colorClass,
        isSelected && "ring-2 ring-white ring-offset-1 ring-offset-transparent",
        isDragging && "opacity-80 cursor-grabbing",
        isResizing && "cursor-ew-resize",
        isLocked && "opacity-50 cursor-not-allowed"
      )}
      style={{
        left: `${left}px`,
        width: `${Math.max(width, 20)}px`,
      }}
      onMouseDown={handleDragStart}
      onClick={(e) => {
        e.stopPropagation();
        onSelect(clip.id);
      }}
    >
      {/* Left resize handle */}
      <div
        className={cn(
          "absolute left-0 top-0 bottom-0 w-2 cursor-ew-resize",
          "hover:bg-white/30 transition-colors",
          !isLocked && "group-hover:bg-white/20"
        )}
      />

      {/* Clip content */}
      <div className="flex items-center gap-1 px-2 h-full min-w-0">
        <Icon className="h-3 w-3 flex-shrink-0 text-white/80" />
        {width > 60 && (
          <span className="text-xs text-white/90 truncate font-medium">
            {clip.name}
          </span>
        )}
        {width > 100 && clip.speed !== 1 && (
          <span className="text-[10px] text-white/70 ml-auto flex-shrink-0">
            {clip.speed}x
          </span>
        )}
      </div>

      {/* Thumbnail preview (for video/broll clips) */}
      {clip.thumbnailUrl && width > 80 && (
        <div
          className="absolute inset-0 opacity-20 bg-cover bg-center pointer-events-none"
          style={{ backgroundImage: `url(${clip.thumbnailUrl})` }}
        />
      )}

      {/* Waveform visualization (placeholder for audio) */}
      {clip.type === "audio" && width > 50 && (
        <div className="absolute inset-x-2 bottom-1 h-2 flex items-end gap-px">
          {Array.from({ length: Math.floor(width / 4) }).map((_, i) => (
            <div
              key={i}
              className="flex-1 bg-white/40 rounded-t"
              style={{ height: `${20 + Math.random() * 80}%` }}
            />
          ))}
        </div>
      )}

      {/* Right resize handle */}
      <div
        className={cn(
          "absolute right-0 top-0 bottom-0 w-2 cursor-ew-resize",
          "hover:bg-white/30 transition-colors",
          !isLocked && "group-hover:bg-white/20"
        )}
      />

      {/* Speed indicator bar */}
      {clip.speed !== 1 && (
        <div
          className={cn(
            "absolute bottom-0 left-0 right-0 h-0.5",
            clip.speed > 1 ? "bg-yellow-400" : "bg-cyan-400"
          )}
        />
      )}
    </div>
  );
}
