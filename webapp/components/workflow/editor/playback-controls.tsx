"use client";

import { cn } from "@/lib/utils";
import { formatDuration } from "@/lib/utils";
import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  Volume2,
  VolumeX,
  ZoomIn,
  ZoomOut,
  Rewind,
  FastForward,
  RotateCcw,
} from "lucide-react";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";

interface PlaybackControlsProps {
  currentTime: number;
  duration: number;
  isPlaying: boolean;
  volume: number;
  zoom: number;
  onTogglePlay: () => void;
  onSeek: (time: number) => void;
  onSkipBack: (seconds?: number) => void;
  onSkipForward: (seconds?: number) => void;
  onJumpToStart: () => void;
  onJumpToEnd: () => void;
  onVolumeChange: (volume: number) => void;
  onZoomChange: (zoom: number) => void;
  isMuted?: boolean;
  onToggleMute?: () => void;
}

export function PlaybackControls({
  currentTime,
  duration,
  isPlaying,
  volume,
  zoom,
  onTogglePlay,
  onSeek,
  onSkipBack,
  onSkipForward,
  onJumpToStart,
  onJumpToEnd,
  onVolumeChange,
  onZoomChange,
  isMuted = false,
  onToggleMute,
}: PlaybackControlsProps) {
  return (
    <div className="flex items-center gap-4 p-4 border border-border rounded-lg bg-card">
      {/* Time display */}
      <div className="min-w-[140px] flex items-center gap-1">
        <span className="font-mono text-sm text-foreground tabular-nums">
          {formatDuration(currentTime)}
        </span>
        <span className="text-muted-foreground">/</span>
        <span className="font-mono text-sm text-muted-foreground tabular-nums">
          {formatDuration(duration)}
        </span>
      </div>

      {/* Playback controls */}
      <div className="flex items-center gap-1">
        {/* Jump to start */}
        <Button
          variant="ghost"
          size="sm"
          onClick={onJumpToStart}
          className="h-8 w-8 p-0"
          title="Jump to start"
        >
          <RotateCcw className="h-4 w-4" />
        </Button>

        {/* Skip back 10s */}
        <Button
          variant="ghost"
          size="sm"
          onClick={() => onSkipBack(10)}
          className="h-8 w-8 p-0"
          title="Skip back 10s"
        >
          <Rewind className="h-4 w-4" />
        </Button>

        {/* Skip back 1s */}
        <Button
          variant="ghost"
          size="sm"
          onClick={() => onSkipBack(1)}
          className="h-8 w-8 p-0"
          title="Skip back 1s (Left arrow)"
        >
          <SkipBack className="h-4 w-4" />
        </Button>

        {/* Play/Pause */}
        <Button
          onClick={onTogglePlay}
          className={cn(
            "h-10 w-10 rounded-full p-0 transition-colors",
            isPlaying
              ? "bg-destructive hover:bg-destructive/90"
              : "bg-primary hover:bg-primary/90"
          )}
          title={isPlaying ? "Pause (Space)" : "Play (Space)"}
        >
          {isPlaying ? (
            <Pause className="h-5 w-5" />
          ) : (
            <Play className="h-5 w-5 ml-0.5" />
          )}
        </Button>

        {/* Skip forward 1s */}
        <Button
          variant="ghost"
          size="sm"
          onClick={() => onSkipForward(1)}
          className="h-8 w-8 p-0"
          title="Skip forward 1s (Right arrow)"
        >
          <SkipForward className="h-4 w-4" />
        </Button>

        {/* Skip forward 10s */}
        <Button
          variant="ghost"
          size="sm"
          onClick={() => onSkipForward(10)}
          className="h-8 w-8 p-0"
          title="Skip forward 10s"
        >
          <FastForward className="h-4 w-4" />
        </Button>

        {/* Jump to end */}
        <Button
          variant="ghost"
          size="sm"
          onClick={onJumpToEnd}
          className="h-8 w-8 p-0"
          title="Jump to end"
        >
          <SkipForward className="h-4 w-4" />
        </Button>
      </div>

      {/* Timeline scrubber */}
      <div className="flex-1 px-2">
        <Slider
          value={[currentTime]}
          onValueChange={([value]) => onSeek(value)}
          min={0}
          max={duration}
          step={0.1}
          className="cursor-pointer"
        />
      </div>

      {/* Volume control */}
      <div className="flex items-center gap-2">
        <Button
          variant="ghost"
          size="sm"
          onClick={onToggleMute}
          className="h-8 w-8 p-0"
          title={isMuted ? "Unmute" : "Mute"}
        >
          {isMuted || volume === 0 ? (
            <VolumeX className="h-4 w-4 text-muted-foreground" />
          ) : (
            <Volume2 className="h-4 w-4 text-muted-foreground" />
          )}
        </Button>
        <div className="w-24">
          <Slider
            value={[isMuted ? 0 : volume]}
            onValueChange={([value]) => onVolumeChange(value)}
            min={0}
            max={100}
            step={1}
          />
        </div>
      </div>

      {/* Zoom control */}
      <div className="flex items-center gap-2 border-l border-border pl-4">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => onZoomChange(Math.max(10, zoom - 10))}
          className="h-8 w-8 p-0"
          title="Zoom out"
        >
          <ZoomOut className="h-4 w-4 text-muted-foreground" />
        </Button>
        <span className="text-xs text-muted-foreground min-w-[40px] text-center">
          {zoom}px/s
        </span>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => onZoomChange(Math.min(200, zoom + 10))}
          className="h-8 w-8 p-0"
          title="Zoom in"
        >
          <ZoomIn className="h-4 w-4 text-muted-foreground" />
        </Button>
      </div>
    </div>
  );
}
