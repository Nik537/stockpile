"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { Play, Pause, RotateCcw, Download, Volume2, VolumeX } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { formatDuration } from "@/lib/utils";
import type { GeneratedAudio } from "@/types/workflow";

interface VoicePreviewProps {
  audio: GeneratedAudio | null;
  isLoading?: boolean;
  onRegenerate?: () => void;
}

export function VoicePreview({
  audio,
  isLoading = false,
  onRegenerate,
}: VoicePreviewProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [volume, setVolume] = useState(1);
  const [isMuted, setIsMuted] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const animationRef = useRef<number | null>(null);

  const updateProgress = useCallback(() => {
    if (audioRef.current) {
      setCurrentTime(audioRef.current.currentTime);
      animationRef.current = requestAnimationFrame(updateProgress);
    }
  }, []);

  useEffect(() => {
    if (!audio?.url) return;

    const audioElement = new Audio(audio.url);
    audioRef.current = audioElement;

    audioElement.onended = () => {
      setIsPlaying(false);
      setCurrentTime(0);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };

    return () => {
      audioElement.pause();
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [audio?.url]);

  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.volume = isMuted ? 0 : volume;
    }
  }, [volume, isMuted]);

  const togglePlayPause = () => {
    if (!audioRef.current) return;

    if (isPlaying) {
      audioRef.current.pause();
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    } else {
      audioRef.current.play();
      animationRef.current = requestAnimationFrame(updateProgress);
    }
    setIsPlaying(!isPlaying);
  };

  const handleSeek = (value: number[]) => {
    if (!audioRef.current) return;
    const seekTime = (value[0] / 100) * (audio?.duration || 0);
    audioRef.current.currentTime = seekTime;
    setCurrentTime(seekTime);
  };

  const handleVolumeChange = (value: number[]) => {
    setVolume(value[0] / 100);
    setIsMuted(false);
  };

  const toggleMute = () => {
    setIsMuted(!isMuted);
  };

  const handleDownload = () => {
    if (!audio?.url) return;
    const link = document.createElement("a");
    link.href = audio.url;
    link.download = `audio-${audio.id}.mp3`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const progress = audio?.duration ? (currentTime / audio.duration) * 100 : 0;

  if (isLoading) {
    return (
      <div className="rounded-lg border border-border bg-card p-4">
        <div className="flex flex-col items-center justify-center py-8">
          <div className="h-12 w-12 animate-spin rounded-full border-4 border-primary border-t-transparent" />
          <p className="mt-4 text-sm text-muted-foreground">
            Generating audio...
          </p>
        </div>
      </div>
    );
  }

  if (!audio) {
    return (
      <div className="rounded-lg border border-dashed border-border bg-card/50 p-4">
        <div className="flex flex-col items-center justify-center py-8 text-center">
          <Volume2 className="h-12 w-12 text-muted-foreground/50" />
          <p className="mt-4 text-sm text-muted-foreground">
            No audio generated yet
          </p>
          <p className="mt-1 text-xs text-muted-foreground">
            Select a voice and enter text to generate audio
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      {/* Waveform Visualization */}
      <div className="mb-4 flex h-16 items-end justify-center gap-0.5 rounded-lg bg-muted/50 px-2">
        {audio.waveformData.map((value, index) => {
          const isActive = (index / audio.waveformData.length) * 100 <= progress;
          return (
            <div
              key={index}
              className={cn(
                "w-1 rounded-full transition-colors",
                isActive ? "bg-primary" : "bg-muted-foreground/30"
              )}
              style={{ height: `${Math.max(value * 100, 8)}%` }}
            />
          );
        })}
      </div>

      {/* Progress Bar */}
      <div className="mb-4">
        <Slider
          value={[progress]}
          onValueChange={handleSeek}
          max={100}
          step={0.1}
          className="w-full"
        />
        <div className="mt-1 flex justify-between text-xs text-muted-foreground">
          <span>{formatDuration(currentTime)}</span>
          <span>{formatDuration(audio.duration)}</span>
        </div>
      </div>

      {/* Controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {/* Play/Pause */}
          <Button
            onClick={togglePlayPause}
            variant="default"
            size="icon"
            className="h-10 w-10 rounded-full"
          >
            {isPlaying ? (
              <Pause className="h-5 w-5" />
            ) : (
              <Play className="h-5 w-5 ml-0.5" />
            )}
          </Button>

          {/* Volume */}
          <div className="flex items-center gap-2">
            <Button
              onClick={toggleMute}
              variant="ghost"
              size="icon"
              className="h-8 w-8"
            >
              {isMuted || volume === 0 ? (
                <VolumeX className="h-4 w-4" />
              ) : (
                <Volume2 className="h-4 w-4" />
              )}
            </Button>
            <Slider
              value={[isMuted ? 0 : volume * 100]}
              onValueChange={handleVolumeChange}
              max={100}
              className="w-20"
            />
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Regenerate */}
          {onRegenerate && (
            <Button
              onClick={onRegenerate}
              variant="outline"
              size="sm"
              className="gap-2"
            >
              <RotateCcw className="h-4 w-4" />
              Regenerate
            </Button>
          )}

          {/* Download */}
          <Button
            onClick={handleDownload}
            variant="outline"
            size="sm"
            className="gap-2"
          >
            <Download className="h-4 w-4" />
            Download
          </Button>
        </div>
      </div>

      {/* Audio Config Info */}
      <div className="mt-4 flex flex-wrap gap-2 border-t border-border pt-4">
        <span className="rounded-full bg-muted px-2 py-1 text-xs text-muted-foreground">
          Speed: {audio.config.speed}x
        </span>
        <span className="rounded-full bg-muted px-2 py-1 text-xs text-muted-foreground">
          Pitch: {audio.config.pitch > 0 ? "+" : ""}{audio.config.pitch}
        </span>
        {audio.config.emotion && (
          <span className="rounded-full bg-muted px-2 py-1 text-xs text-muted-foreground">
            Emotion: {audio.config.emotion}
          </span>
        )}
      </div>
    </div>
  );
}
