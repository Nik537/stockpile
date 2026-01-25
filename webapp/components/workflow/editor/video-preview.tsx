"use client";

import { useRef, useEffect, useState, useCallback } from "react";
import { cn } from "@/lib/utils";
import { formatDuration } from "@/lib/utils";
import type { EditorTimeline, EditorTimelineClip } from "@/types/workflow";
import { Film, Maximize, Minimize, Volume2, VolumeX } from "lucide-react";

interface VideoPreviewProps {
  timeline: EditorTimeline;
  isPlaying: boolean;
  onTimeUpdate?: (time: number) => void;
}

export function VideoPreview({
  timeline,
  isPlaying,
  onTimeUpdate,
}: VideoPreviewProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [currentClip, setCurrentClip] = useState<EditorTimelineClip | null>(null);

  // Find the active clip at the current playhead position
  useEffect(() => {
    const playheadTime = timeline.playheadPosition;

    // Find active clips from main track or broll track (broll overlays main)
    let activeClip: EditorTimelineClip | null = null;

    // Check broll track first (overlay priority)
    const brollTrack = timeline.tracks.find((t) => t.type === "broll");
    if (brollTrack && !brollTrack.muted) {
      activeClip =
        brollTrack.clips.find(
          (clip) =>
            playheadTime >= clip.startTime &&
            playheadTime < clip.startTime + clip.duration
        ) || null;
    }

    // Fall back to main track
    if (!activeClip) {
      const mainTrack = timeline.tracks.find((t) => t.type === "main");
      if (mainTrack && !mainTrack.muted) {
        activeClip =
          mainTrack.clips.find(
            (clip) =>
              playheadTime >= clip.startTime &&
              playheadTime < clip.startTime + clip.duration
          ) || null;
      }
    }

    setCurrentClip(activeClip);
  }, [timeline.playheadPosition, timeline.tracks]);

  // Sync video playback with playhead
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !currentClip) return;

    // Calculate the time within the source video
    const clipProgress = timeline.playheadPosition - currentClip.startTime;
    const sourceTime = currentClip.trimStart + clipProgress * currentClip.speed;

    // Update video if time is significantly different
    if (Math.abs(video.currentTime - sourceTime) > 0.1) {
      video.currentTime = sourceTime;
    }

    // Control playback
    if (isPlaying && video.paused) {
      video.playbackRate = currentClip.speed;
      video.play().catch(() => {
        // Autoplay blocked, continue silently
      });
    } else if (!isPlaying && !video.paused) {
      video.pause();
    }
  }, [isPlaying, currentClip, timeline.playheadPosition]);

  // Handle fullscreen toggle
  const toggleFullscreen = useCallback(() => {
    if (!containerRef.current) return;

    if (!document.fullscreenElement) {
      containerRef.current.requestFullscreen().catch(() => {
        // Fullscreen not supported
      });
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  }, []);

  // Listen for fullscreen changes
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };

    document.addEventListener("fullscreenchange", handleFullscreenChange);
    return () =>
      document.removeEventListener("fullscreenchange", handleFullscreenChange);
  }, []);

  // Toggle mute
  const toggleMute = useCallback(() => {
    setIsMuted(!isMuted);
    if (videoRef.current) {
      videoRef.current.muted = !isMuted;
    }
  }, [isMuted]);

  return (
    <div
      ref={containerRef}
      className={cn(
        "relative bg-black rounded-lg overflow-hidden",
        isFullscreen ? "fixed inset-0 z-50" : "aspect-video"
      )}
    >
      {/* Video element */}
      {currentClip?.sourceUrl ? (
        <video
          ref={videoRef}
          src={currentClip.sourceUrl}
          className="w-full h-full object-contain"
          muted={isMuted}
          playsInline
          preload="auto"
          onTimeUpdate={(e) => {
            if (onTimeUpdate && currentClip) {
              const videoTime = e.currentTarget.currentTime;
              const clipTime =
                currentClip.startTime +
                (videoTime - currentClip.trimStart) / currentClip.speed;
              onTimeUpdate(clipTime);
            }
          }}
        />
      ) : (
        // Placeholder when no clip is active
        <div className="w-full h-full flex items-center justify-center">
          <div className="text-center text-muted-foreground">
            <Film className="h-16 w-16 mx-auto mb-2 opacity-50" />
            <p className="text-sm">No clip at current position</p>
            <p className="text-xs opacity-50 mt-1">
              Add clips to the timeline to preview
            </p>
          </div>
        </div>
      )}

      {/* Overlay controls */}
      <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/60 to-transparent p-4">
        <div className="flex items-center justify-between">
          {/* Time display */}
          <div className="flex items-center gap-2">
            <span className="text-white font-mono text-sm">
              {formatDuration(timeline.playheadPosition)}
            </span>
            <span className="text-white/50">/</span>
            <span className="text-white/70 font-mono text-sm">
              {formatDuration(timeline.duration)}
            </span>
          </div>

          {/* Controls */}
          <div className="flex items-center gap-2">
            {/* Current clip info */}
            {currentClip && (
              <span className="text-xs text-white/70 mr-2">
                {currentClip.name}
                {currentClip.speed !== 1 && ` (${currentClip.speed}x)`}
              </span>
            )}

            {/* Mute button */}
            <button
              type="button"
              onClick={toggleMute}
              className="p-2 rounded-full bg-white/10 hover:bg-white/20 transition-colors"
              title={isMuted ? "Unmute" : "Mute"}
            >
              {isMuted ? (
                <VolumeX className="h-4 w-4 text-white" />
              ) : (
                <Volume2 className="h-4 w-4 text-white" />
              )}
            </button>

            {/* Fullscreen button */}
            <button
              type="button"
              onClick={toggleFullscreen}
              className="p-2 rounded-full bg-white/10 hover:bg-white/20 transition-colors"
              title={isFullscreen ? "Exit fullscreen" : "Fullscreen"}
            >
              {isFullscreen ? (
                <Minimize className="h-4 w-4 text-white" />
              ) : (
                <Maximize className="h-4 w-4 text-white" />
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Playhead indicator */}
      <div className="absolute top-2 left-2">
        <div
          className={cn(
            "px-2 py-0.5 rounded text-xs font-medium",
            isPlaying ? "bg-red-500 text-white" : "bg-white/10 text-white/70"
          )}
        >
          {isPlaying ? "PLAYING" : "PAUSED"}
        </div>
      </div>
    </div>
  );
}
