"use client";

import { useState } from "react";
import { Settings, Zap, Gauge, Music2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { Slider } from "@/components/ui/slider";
import type { TTSConfig } from "@/types/workflow";

interface TTSControlsProps {
  config: TTSConfig;
  onChange: (config: TTSConfig) => void;
}

type EmotionType = NonNullable<TTSConfig["emotion"]>;

const emotions: { value: EmotionType; label: string; icon: string }[] = [
  { value: "neutral", label: "Neutral", icon: "😐" },
  { value: "happy", label: "Happy", icon: "😊" },
  { value: "sad", label: "Sad", icon: "😢" },
  { value: "excited", label: "Excited", icon: "🤩" },
  { value: "serious", label: "Serious", icon: "😤" },
];

export function TTSControls({ config, onChange }: TTSControlsProps) {
  const [isAdvancedOpen, setIsAdvancedOpen] = useState(false);

  const handleSpeedChange = (value: number[]) => {
    onChange({ ...config, speed: value[0] });
  };

  const handlePitchChange = (value: number[]) => {
    onChange({ ...config, pitch: value[0] });
  };

  const handleEmotionChange = (emotion: EmotionType) => {
    onChange({ ...config, emotion });
  };

  const resetToDefaults = () => {
    onChange({
      ...config,
      speed: 1.0,
      pitch: 0,
      emotion: "neutral",
    });
  };

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="flex items-center gap-2 text-sm font-medium text-foreground">
          <Settings className="h-4 w-4 text-muted-foreground" />
          Voice Settings
        </h3>
        <button
          type="button"
          onClick={resetToDefaults}
          className="text-xs text-muted-foreground hover:text-foreground"
        >
          Reset to defaults
        </button>
      </div>

      {/* Speed Control */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Gauge className="h-4 w-4 text-blue-400" />
            <label className="text-sm font-medium text-foreground">
              Speed
            </label>
          </div>
          <span className="rounded bg-muted px-2 py-0.5 text-xs font-mono text-muted-foreground">
            {config.speed.toFixed(1)}x
          </span>
        </div>
        <div className="flex items-center gap-4">
          <span className="text-xs text-muted-foreground">0.5x</span>
          <Slider
            value={[config.speed]}
            onValueChange={handleSpeedChange}
            min={0.5}
            max={2.0}
            step={0.1}
            className="flex-1"
          />
          <span className="text-xs text-muted-foreground">2.0x</span>
        </div>
        <div className="flex justify-between">
          <button
            type="button"
            onClick={() => handleSpeedChange([0.75])}
            className={cn(
              "rounded px-2 py-0.5 text-xs transition-colors",
              config.speed === 0.75
                ? "bg-primary text-primary-foreground"
                : "bg-muted hover:bg-muted/80"
            )}
          >
            Slow
          </button>
          <button
            type="button"
            onClick={() => handleSpeedChange([1.0])}
            className={cn(
              "rounded px-2 py-0.5 text-xs transition-colors",
              config.speed === 1.0
                ? "bg-primary text-primary-foreground"
                : "bg-muted hover:bg-muted/80"
            )}
          >
            Normal
          </button>
          <button
            type="button"
            onClick={() => handleSpeedChange([1.25])}
            className={cn(
              "rounded px-2 py-0.5 text-xs transition-colors",
              config.speed === 1.25
                ? "bg-primary text-primary-foreground"
                : "bg-muted hover:bg-muted/80"
            )}
          >
            Fast
          </button>
          <button
            type="button"
            onClick={() => handleSpeedChange([1.5])}
            className={cn(
              "rounded px-2 py-0.5 text-xs transition-colors",
              config.speed === 1.5
                ? "bg-primary text-primary-foreground"
                : "bg-muted hover:bg-muted/80"
            )}
          >
            Very Fast
          </button>
        </div>
      </div>

      {/* Pitch Control */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Music2 className="h-4 w-4 text-purple-400" />
            <label className="text-sm font-medium text-foreground">
              Pitch
            </label>
          </div>
          <span className="rounded bg-muted px-2 py-0.5 text-xs font-mono text-muted-foreground">
            {config.pitch > 0 ? "+" : ""}{config.pitch}
          </span>
        </div>
        <div className="flex items-center gap-4">
          <span className="text-xs text-muted-foreground">-12</span>
          <Slider
            value={[config.pitch]}
            onValueChange={handlePitchChange}
            min={-12}
            max={12}
            step={1}
            className="flex-1"
          />
          <span className="text-xs text-muted-foreground">+12</span>
        </div>
        <div className="flex justify-center gap-2">
          <button
            type="button"
            onClick={() => handlePitchChange([-6])}
            className={cn(
              "rounded px-2 py-0.5 text-xs transition-colors",
              config.pitch === -6
                ? "bg-primary text-primary-foreground"
                : "bg-muted hover:bg-muted/80"
            )}
          >
            Lower
          </button>
          <button
            type="button"
            onClick={() => handlePitchChange([0])}
            className={cn(
              "rounded px-2 py-0.5 text-xs transition-colors",
              config.pitch === 0
                ? "bg-primary text-primary-foreground"
                : "bg-muted hover:bg-muted/80"
            )}
          >
            Normal
          </button>
          <button
            type="button"
            onClick={() => handlePitchChange([6])}
            className={cn(
              "rounded px-2 py-0.5 text-xs transition-colors",
              config.pitch === 6
                ? "bg-primary text-primary-foreground"
                : "bg-muted hover:bg-muted/80"
            )}
          >
            Higher
          </button>
        </div>
      </div>

      {/* Emotion Selector */}
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <Zap className="h-4 w-4 text-yellow-400" />
          <label className="text-sm font-medium text-foreground">
            Emotion
          </label>
        </div>
        <div className="grid grid-cols-5 gap-2">
          {emotions.map((emotion) => (
            <button
              key={emotion.value}
              type="button"
              onClick={() => handleEmotionChange(emotion.value)}
              className={cn(
                "flex flex-col items-center gap-1 rounded-lg border p-2 transition-all",
                config.emotion === emotion.value
                  ? "border-primary bg-primary/10"
                  : "border-border hover:border-primary/50 hover:bg-muted/50"
              )}
            >
              <span className="text-lg">{emotion.icon}</span>
              <span className="text-[10px] text-muted-foreground">
                {emotion.label}
              </span>
            </button>
          ))}
        </div>
      </div>

      {/* Advanced Settings Toggle */}
      <button
        type="button"
        onClick={() => setIsAdvancedOpen(!isAdvancedOpen)}
        className="flex w-full items-center justify-between rounded-lg border border-border p-3 text-sm hover:bg-muted/50"
      >
        <span className="text-muted-foreground">Advanced Settings</span>
        <span
          className={cn(
            "transition-transform",
            isAdvancedOpen && "rotate-180"
          )}
        >
          ^
        </span>
      </button>

      {/* Advanced Settings Panel */}
      {isAdvancedOpen && (
        <div className="space-y-4 rounded-lg border border-border bg-muted/30 p-4">
          <p className="text-xs text-muted-foreground">
            Advanced TTS parameters for fine-tuning voice output.
            These settings are specific to Fish-Speech and GPT-SoVITS engines.
          </p>

          {/* Placeholder for advanced settings */}
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-1">
              <label className="text-xs font-medium text-foreground">
                Temperature
              </label>
              <input
                type="number"
                min="0"
                max="1"
                step="0.1"
                defaultValue="0.7"
                className="w-full rounded border border-border bg-background px-2 py-1 text-sm"
              />
            </div>
            <div className="space-y-1">
              <label className="text-xs font-medium text-foreground">
                Top P
              </label>
              <input
                type="number"
                min="0"
                max="1"
                step="0.1"
                defaultValue="0.9"
                className="w-full rounded border border-border bg-background px-2 py-1 text-sm"
              />
            </div>
            <div className="space-y-1">
              <label className="text-xs font-medium text-foreground">
                Repetition Penalty
              </label>
              <input
                type="number"
                min="1"
                max="2"
                step="0.1"
                defaultValue="1.2"
                className="w-full rounded border border-border bg-background px-2 py-1 text-sm"
              />
            </div>
            <div className="space-y-1">
              <label className="text-xs font-medium text-foreground">
                Max New Tokens
              </label>
              <input
                type="number"
                min="100"
                max="4096"
                step="100"
                defaultValue="1024"
                className="w-full rounded border border-border bg-background px-2 py-1 text-sm"
              />
            </div>
          </div>
        </div>
      )}

      {/* Current Settings Summary */}
      <div className="rounded-lg bg-muted/50 p-3">
        <p className="text-xs text-muted-foreground">
          <span className="font-medium text-foreground">Current Settings: </span>
          {config.speed}x speed, {config.pitch > 0 ? "+" : ""}{config.pitch} pitch
          {config.emotion && config.emotion !== "neutral" && `, ${config.emotion} emotion`}
        </p>
      </div>
    </div>
  );
}
