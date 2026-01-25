"use client";

import { useState, useRef } from "react";
import { Volume2, Play, Pause, Check, Upload, Mic } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { Voice } from "@/types/workflow";

interface VoiceSelectorProps {
  voices: Voice[];
  selectedVoiceId: string | null;
  onVoiceSelect: (voiceId: string) => void;
  onUploadVoice?: (file: File) => void;
}

export function VoiceSelector({
  voices,
  selectedVoiceId,
  onVoiceSelect,
  onUploadVoice,
}: VoiceSelectorProps) {
  const [playingVoiceId, setPlayingVoiceId] = useState<string | null>(null);
  const [filterGender, setFilterGender] = useState<"all" | "male" | "female" | "neutral">("all");
  const [filterStyle, setFilterStyle] = useState<"all" | "natural" | "professional" | "casual" | "dramatic">("all");
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const filteredVoices = voices.filter((voice) => {
    if (filterGender !== "all" && voice.gender !== filterGender) return false;
    if (filterStyle !== "all" && voice.style !== filterStyle) return false;
    return true;
  });

  const handlePlayPreview = (voice: Voice) => {
    if (playingVoiceId === voice.id) {
      audioRef.current?.pause();
      setPlayingVoiceId(null);
      return;
    }

    if (audioRef.current) {
      audioRef.current.pause();
    }

    const audio = new Audio(voice.previewUrl);
    audioRef.current = audio;

    audio.onended = () => setPlayingVoiceId(null);
    audio.onerror = () => setPlayingVoiceId(null);

    audio.play();
    setPlayingVoiceId(voice.id);
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && onUploadVoice) {
      onUploadVoice(file);
    }
  };

  const getStyleColor = (style: Voice["style"]) => {
    switch (style) {
      case "natural":
        return "bg-green-500/20 text-green-400 border-green-500/30";
      case "professional":
        return "bg-blue-500/20 text-blue-400 border-blue-500/30";
      case "casual":
        return "bg-orange-500/20 text-orange-400 border-orange-500/30";
      case "dramatic":
        return "bg-purple-500/20 text-purple-400 border-purple-500/30";
      default:
        return "bg-muted text-muted-foreground";
    }
  };

  return (
    <div className="space-y-4">
      <Tabs defaultValue="library" className="w-full">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="library" className="flex items-center gap-2">
            <Volume2 className="h-4 w-4" />
            Voice Library
          </TabsTrigger>
          <TabsTrigger value="custom" className="flex items-center gap-2">
            <Upload className="h-4 w-4" />
            Custom Voice
          </TabsTrigger>
        </TabsList>

        <TabsContent value="library" className="space-y-4">
          {/* Filters */}
          <div className="flex flex-wrap gap-2">
            <div className="flex items-center gap-2">
              <span className="text-xs text-muted-foreground">Gender:</span>
              <div className="flex gap-1">
                {(["all", "male", "female", "neutral"] as const).map((gender) => (
                  <button
                    key={gender}
                    type="button"
                    onClick={() => setFilterGender(gender)}
                    className={cn(
                      "rounded-md px-2 py-1 text-xs transition-colors",
                      filterGender === gender
                        ? "bg-primary text-primary-foreground"
                        : "bg-muted hover:bg-muted/80"
                    )}
                  >
                    {gender === "all" ? "All" : gender.charAt(0).toUpperCase() + gender.slice(1)}
                  </button>
                ))}
              </div>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs text-muted-foreground">Style:</span>
              <div className="flex gap-1">
                {(["all", "natural", "professional", "casual", "dramatic"] as const).map((style) => (
                  <button
                    key={style}
                    type="button"
                    onClick={() => setFilterStyle(style)}
                    className={cn(
                      "rounded-md px-2 py-1 text-xs transition-colors",
                      filterStyle === style
                        ? "bg-primary text-primary-foreground"
                        : "bg-muted hover:bg-muted/80"
                    )}
                  >
                    {style === "all" ? "All" : style.charAt(0).toUpperCase() + style.slice(1)}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Voice Grid */}
          <div className="grid gap-3 sm:grid-cols-2">
            {filteredVoices.map((voice) => (
              <button
                key={voice.id}
                type="button"
                onClick={() => onVoiceSelect(voice.id)}
                className={cn(
                  "group relative flex items-center gap-3 rounded-lg border p-3 text-left transition-all",
                  selectedVoiceId === voice.id
                    ? "border-primary bg-primary/10"
                    : "border-border hover:border-primary/50 hover:bg-muted/50"
                )}
              >
                {/* Voice Icon */}
                <div
                  className={cn(
                    "flex h-10 w-10 items-center justify-center rounded-full",
                    voice.gender === "male"
                      ? "bg-blue-500/20"
                      : voice.gender === "female"
                      ? "bg-pink-500/20"
                      : "bg-purple-500/20"
                  )}
                >
                  <Mic
                    className={cn(
                      "h-5 w-5",
                      voice.gender === "male"
                        ? "text-blue-400"
                        : voice.gender === "female"
                        ? "text-pink-400"
                        : "text-purple-400"
                    )}
                  />
                </div>

                {/* Voice Info */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <p className="text-sm font-medium text-foreground truncate">
                      {voice.name}
                    </p>
                    {selectedVoiceId === voice.id && (
                      <Check className="h-4 w-4 text-primary" />
                    )}
                  </div>
                  <div className="flex items-center gap-2 mt-1">
                    <span className="text-xs text-muted-foreground">
                      {voice.language}
                    </span>
                    <span
                      className={cn(
                        "rounded-full border px-2 py-0.5 text-[10px] font-medium",
                        getStyleColor(voice.style)
                      )}
                    >
                      {voice.style}
                    </span>
                  </div>
                </div>

                {/* Play Button */}
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    handlePlayPreview(voice);
                  }}
                  className={cn(
                    "flex h-8 w-8 items-center justify-center rounded-full transition-colors",
                    playingVoiceId === voice.id
                      ? "bg-primary text-primary-foreground"
                      : "bg-muted text-muted-foreground hover:bg-muted/80"
                  )}
                >
                  {playingVoiceId === voice.id ? (
                    <Pause className="h-4 w-4" />
                  ) : (
                    <Play className="h-4 w-4" />
                  )}
                </button>
              </button>
            ))}
          </div>

          {filteredVoices.length === 0 && (
            <div className="flex flex-col items-center justify-center py-8 text-center">
              <Volume2 className="h-12 w-12 text-muted-foreground/50" />
              <p className="mt-2 text-sm text-muted-foreground">
                No voices match your filters
              </p>
            </div>
          )}
        </TabsContent>

        <TabsContent value="custom" className="space-y-4">
          <div className="rounded-lg border border-dashed border-border p-8">
            <div className="flex flex-col items-center justify-center text-center">
              <Upload className="h-12 w-12 text-muted-foreground/50" />
              <h3 className="mt-4 text-lg font-medium text-foreground">
                Upload Voice Sample
              </h3>
              <p className="mt-2 text-sm text-muted-foreground">
                Upload a voice sample for Fish-Speech or GPT-SoVITS voice cloning
              </p>
              <p className="mt-1 text-xs text-muted-foreground">
                Supported formats: WAV, MP3, M4A (10-30 seconds recommended)
              </p>
              <input
                ref={fileInputRef}
                type="file"
                accept="audio/*"
                onChange={handleFileUpload}
                className="hidden"
              />
              <Button
                onClick={() => fileInputRef.current?.click()}
                variant="outline"
                className="mt-4"
              >
                <Upload className="h-4 w-4" />
                Choose File
              </Button>
            </div>
          </div>

          <div className="rounded-lg border border-border bg-muted/50 p-4">
            <h4 className="text-sm font-medium text-foreground">
              Voice Cloning Tips
            </h4>
            <ul className="mt-2 space-y-1 text-xs text-muted-foreground">
              <li>- Use a clear recording without background noise</li>
              <li>- 10-30 seconds of continuous speech works best</li>
              <li>- Avoid recordings with music or sound effects</li>
              <li>- Higher quality audio produces better results</li>
            </ul>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
