"use client";

import { useCallback } from "react";
import { cn } from "@/lib/utils";
import { formatDuration } from "@/lib/utils";
import type { EditorTimelineClip } from "@/types/workflow";
import { Slider } from "@/components/ui/slider";
import { Input } from "@/components/ui/input";
import {
  Film,
  Music,
  Image,
  User,
  Clock,
  Gauge,
  Volume2,
  Scissors,
  Trash2,
  Copy,
} from "lucide-react";
import { Button } from "@/components/ui/button";

interface ClipInspectorProps {
  clip: EditorTimelineClip | null;
  onUpdate: (updates: Partial<EditorTimelineClip>) => void;
  onDelete: () => void;
  onDuplicate?: () => void;
  onSplit?: () => void;
  isLocked?: boolean;
}

const clipTypeIcons = {
  video: Film,
  audio: Music,
  broll: Image,
  avatar: User,
};

const clipTypeLabels: Record<string, string> = {
  video: "Video Clip",
  audio: "Audio Clip",
  broll: "B-Roll Clip",
  avatar: "Avatar Clip",
};

export function ClipInspector({
  clip,
  onUpdate,
  onDelete,
  onDuplicate,
  onSplit,
  isLocked = false,
}: ClipInspectorProps) {
  const handleSpeedChange = useCallback(
    (value: number[]) => {
      onUpdate({ speed: value[0] });
    },
    [onUpdate]
  );

  const handleVolumeChange = useCallback(
    (value: number[]) => {
      onUpdate({ volume: value[0] });
    },
    [onUpdate]
  );

  const handleStartTimeChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const value = parseFloat(e.target.value);
      if (!isNaN(value) && value >= 0) {
        onUpdate({ startTime: value });
      }
    },
    [onUpdate]
  );

  const handleDurationChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const value = parseFloat(e.target.value);
      if (!isNaN(value) && value > 0) {
        onUpdate({ duration: value });
      }
    },
    [onUpdate]
  );

  const handleNameChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      onUpdate({ name: e.target.value });
    },
    [onUpdate]
  );

  if (!clip) {
    return (
      <div className="h-full flex flex-col items-center justify-center p-4 text-center">
        <Scissors className="h-8 w-8 text-muted-foreground/40 mb-2" />
        <p className="text-sm text-muted-foreground">No clip selected</p>
        <p className="text-xs text-muted-foreground/60 mt-1">
          Click a clip on the timeline to edit its properties
        </p>
      </div>
    );
  }

  const Icon = clipTypeIcons[clip.type] || Film;
  const typeLabel = clipTypeLabels[clip.type] || "Clip";

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center gap-2 p-3 border-b border-border">
        <Icon className="h-4 w-4 text-primary" />
        <span className="text-sm font-medium text-foreground">{typeLabel}</span>
        {isLocked && (
          <span className="text-xs text-yellow-500 ml-auto">Locked</span>
        )}
      </div>

      {/* Properties */}
      <div className="flex-1 overflow-y-auto p-3 space-y-4">
        {/* Name */}
        <div className="space-y-1.5">
          <label className="text-xs font-medium text-muted-foreground">
            Name
          </label>
          <Input
            value={clip.name}
            onChange={handleNameChange}
            disabled={isLocked}
            className="h-8 text-sm"
          />
        </div>

        {/* Timing */}
        <div className="space-y-1.5">
          <label className="text-xs font-medium text-muted-foreground flex items-center gap-1">
            <Clock className="h-3 w-3" />
            Timing
          </label>
          <div className="grid grid-cols-2 gap-2">
            <div>
              <span className="text-[10px] text-muted-foreground">Start</span>
              <Input
                type="number"
                value={clip.startTime.toFixed(2)}
                onChange={handleStartTimeChange}
                disabled={isLocked}
                className="h-7 text-xs"
                step={0.1}
                min={0}
              />
            </div>
            <div>
              <span className="text-[10px] text-muted-foreground">Duration</span>
              <Input
                type="number"
                value={clip.duration.toFixed(2)}
                onChange={handleDurationChange}
                disabled={isLocked}
                className="h-7 text-xs"
                step={0.1}
                min={0.1}
              />
            </div>
          </div>
          <div className="text-[10px] text-muted-foreground/70">
            End: {formatDuration(clip.startTime + clip.duration)}
          </div>
        </div>

        {/* Trim info */}
        {(clip.trimStart > 0 || clip.trimEnd > 0) && (
          <div className="space-y-1.5">
            <label className="text-xs font-medium text-muted-foreground flex items-center gap-1">
              <Scissors className="h-3 w-3" />
              Trim
            </label>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="bg-muted/50 rounded p-2">
                <span className="text-muted-foreground">In:</span>{" "}
                <span className="text-foreground">
                  {formatDuration(clip.trimStart)}
                </span>
              </div>
              <div className="bg-muted/50 rounded p-2">
                <span className="text-muted-foreground">Out:</span>{" "}
                <span className="text-foreground">
                  {formatDuration(clip.trimEnd)}
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Speed */}
        <div className="space-y-2">
          <label className="text-xs font-medium text-muted-foreground flex items-center gap-1">
            <Gauge className="h-3 w-3" />
            Speed: {clip.speed.toFixed(2)}x
          </label>
          <Slider
            value={[clip.speed]}
            onValueChange={handleSpeedChange}
            min={0.25}
            max={4}
            step={0.25}
            disabled={isLocked}
            className="py-1"
          />
          <div className="flex justify-between text-[10px] text-muted-foreground/70">
            <span>0.25x</span>
            <span>1x</span>
            <span>4x</span>
          </div>
          {/* Speed presets */}
          <div className="flex gap-1">
            {[0.5, 1, 1.5, 2].map((speed) => (
              <button
                key={speed}
                type="button"
                onClick={() => onUpdate({ speed })}
                disabled={isLocked}
                className={cn(
                  "flex-1 py-1 text-[10px] rounded border transition-colors",
                  clip.speed === speed
                    ? "bg-primary text-primary-foreground border-primary"
                    : "border-border hover:bg-muted text-muted-foreground"
                )}
              >
                {speed}x
              </button>
            ))}
          </div>
        </div>

        {/* Volume (for audio/video clips) */}
        {(clip.type === "audio" || clip.type === "video") &&
          clip.volume !== undefined && (
            <div className="space-y-2">
              <label className="text-xs font-medium text-muted-foreground flex items-center gap-1">
                <Volume2 className="h-3 w-3" />
                Volume: {clip.volume}%
              </label>
              <Slider
                value={[clip.volume]}
                onValueChange={handleVolumeChange}
                min={0}
                max={100}
                step={1}
                disabled={isLocked}
                className="py-1"
              />
              <div className="flex justify-between text-[10px] text-muted-foreground/70">
                <span>0%</span>
                <span>50%</span>
                <span>100%</span>
              </div>
            </div>
          )}

        {/* Source info */}
        {clip.sourceUrl && (
          <div className="space-y-1.5">
            <label className="text-xs font-medium text-muted-foreground">
              Source
            </label>
            <div className="text-[10px] text-muted-foreground/70 break-all bg-muted/30 p-2 rounded">
              {clip.sourceUrl.length > 50
                ? clip.sourceUrl.substring(0, 50) + "..."
                : clip.sourceUrl}
            </div>
          </div>
        )}

        {/* Thumbnail preview */}
        {clip.thumbnailUrl && (
          <div className="space-y-1.5">
            <label className="text-xs font-medium text-muted-foreground">
              Preview
            </label>
            <div className="aspect-video bg-muted rounded overflow-hidden">
              <img
                src={clip.thumbnailUrl}
                alt={clip.name}
                className="w-full h-full object-cover"
              />
            </div>
          </div>
        )}
      </div>

      {/* Actions */}
      <div className="border-t border-border p-3 space-y-2">
        <div className="grid grid-cols-2 gap-2">
          {onSplit && (
            <Button
              variant="outline"
              size="sm"
              onClick={onSplit}
              disabled={isLocked}
              className="h-8 text-xs"
            >
              <Scissors className="h-3 w-3 mr-1" />
              Split
            </Button>
          )}
          {onDuplicate && (
            <Button
              variant="outline"
              size="sm"
              onClick={onDuplicate}
              disabled={isLocked}
              className="h-8 text-xs"
            >
              <Copy className="h-3 w-3 mr-1" />
              Duplicate
            </Button>
          )}
        </div>
        <Button
          variant="destructive"
          size="sm"
          onClick={onDelete}
          disabled={isLocked}
          className="w-full h-8 text-xs"
        >
          <Trash2 className="h-3 w-3 mr-1" />
          Delete Clip
        </Button>
      </div>
    </div>
  );
}
