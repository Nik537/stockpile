"use client";

import { useState, useEffect, useCallback } from "react";
import {
  User,
  Volume2,
  FileText,
  RefreshCw,
  Wand2,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import { Header } from "@/components/layout/header";
import { EnhancedStepIndicator } from "@/components/workflow/enhanced-step-indicator";
import { WorkflowNav } from "@/components/workflow/workflow-nav";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  VoiceSelector,
  VoicePreview,
  AvatarGenerator,
  TTSControls,
  AudioWaveform,
} from "@/components/workflow/avatar-tts";
import { useWorkflowStore, useAvatarTtsData, useScriptData } from "@/lib/workflow-store";
import type {
  Voice,
  Avatar,
  TTSConfig,
  GeneratedAudio,
  ScriptSection,
} from "@/types/workflow";
import { cn, formatDuration } from "@/lib/utils";

export default function AvatarTTSPage() {
  // Workflow store
  const avatarTtsData = useAvatarTtsData();
  const scriptData = useScriptData();
  const setSelectedVoice = useWorkflowStore((state) => state.setSelectedVoice);
  const setSelectedAvatar = useWorkflowStore((state) => state.setSelectedAvatar);
  const setTtsConfig = useWorkflowStore((state) => state.setTtsConfig);
  const addGeneratedAudio = useWorkflowStore((state) => state.addGeneratedAudio);
  const markStepComplete = useWorkflowStore((state) => state.markStepComplete);

  // Get script sections from previous step if available
  const scriptSectionsFromStore: ScriptSection[] = scriptData.script?.sections.map((s, i) => ({
    id: s.id,
    text: s.content,
    startTime: i * 8,
    endTime: (i + 1) * 8,
  })) || [];

  // Default script sections if none from store
  const defaultSections: ScriptSection[] = [
    {
      id: "s1",
      text: "Welcome to our channel! Today we're going to explore the fascinating world of artificial intelligence and how it's transforming the way we create content.",
      startTime: 0,
      endTime: 8,
    },
    {
      id: "s2",
      text: "First, let's talk about the basics. AI has come a long way in recent years, and it's now capable of generating incredibly realistic speech and visuals.",
      startTime: 8,
      endTime: 16,
    },
    {
      id: "s3",
      text: "With tools like Fish-Speech and GPT-SoVITS, you can clone voices and create natural-sounding narration without recording a single word yourself.",
      startTime: 16,
      endTime: 25,
    },
    {
      id: "s4",
      text: "But that's just the beginning. When you combine AI voice generation with avatar technology, the possibilities are endless.",
      startTime: 25,
      endTime: 32,
    },
    {
      id: "s5",
      text: "Thank you for watching! Don't forget to like and subscribe for more content on AI-powered video creation.",
      startTime: 32,
      endTime: 38,
    },
  ];

  // State
  const [voices, setVoices] = useState<Voice[]>([]);
  const [avatars, setAvatars] = useState<Avatar[]>([]);
  const [selectedVoiceId, setSelectedVoiceId] = useState<string | null>(
    avatarTtsData.selectedVoice?.id || null
  );
  const [selectedAvatarId, setSelectedAvatarId] = useState<string | null>(
    avatarTtsData.selectedAvatar?.id || null
  );
  const [isAvatarEnabled, setIsAvatarEnabled] = useState(false);
  const [ttsConfig, setLocalTTSConfig] = useState<TTSConfig>(
    avatarTtsData.ttsConfig || {
      voiceId: "",
      speed: 1.0,
      pitch: 0,
      emotion: "neutral",
    }
  );
  const [scriptSections, setScriptSections] = useState<ScriptSection[]>(
    scriptSectionsFromStore.length > 0 ? scriptSectionsFromStore : defaultSections
  );
  const [activeSectionId, setActiveSectionId] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedAudio, setGeneratedAudio] = useState<GeneratedAudio | null>(null);
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set());
  const [isLoading, setIsLoading] = useState(true);

  // Fetch voices and avatars on mount
  useEffect(() => {
    const fetchData = async () => {
      try {
        const [voicesRes, avatarsRes] = await Promise.all([
          fetch("/api/avatar-tts/voices"),
          fetch("/api/avatar-tts/avatar"),
        ]);

        const voicesData = await voicesRes.json();
        const avatarsData = await avatarsRes.json();

        setVoices(voicesData.voices || []);
        setAvatars(avatarsData.avatars || []);
      } catch (error) {
        console.error("Failed to fetch data:", error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
  }, []);

  // Update TTS config when voice changes
  useEffect(() => {
    if (selectedVoiceId) {
      setLocalTTSConfig((prev) => ({ ...prev, voiceId: selectedVoiceId }));
      const voice = voices.find((v) => v.id === selectedVoiceId);
      if (voice) {
        setSelectedVoice(voice);
      }
    }
  }, [selectedVoiceId, voices, setSelectedVoice]);

  // Sync TTS config to store
  useEffect(() => {
    setTtsConfig(ttsConfig);
  }, [ttsConfig, setTtsConfig]);

  // Sync avatar to store
  useEffect(() => {
    if (selectedAvatarId && isAvatarEnabled) {
      const avatar = avatars.find((a) => a.id === selectedAvatarId);
      if (avatar) {
        setSelectedAvatar(avatar);
      }
    } else {
      setSelectedAvatar(null);
    }
  }, [selectedAvatarId, isAvatarEnabled, avatars, setSelectedAvatar]);

  // Mark step complete when audio is generated
  useEffect(() => {
    const generatedCount = scriptSections.filter((s) => s.audio).length;
    if (generatedCount > 0 && selectedVoiceId) {
      markStepComplete("avatar-tts");
    }
  }, [scriptSections, selectedVoiceId, markStepComplete]);

  // Generate audio for a section
  const handleGenerateAudio = useCallback(
    async (section: ScriptSection) => {
      if (!selectedVoiceId) return;

      setIsGenerating(true);
      setActiveSectionId(section.id);

      try {
        const response = await fetch("/api/avatar-tts/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text: section.text,
            voiceId: selectedVoiceId,
            config: ttsConfig,
          }),
        });

        const data = await response.json();

        if (response.ok) {
          const audio = data.audio as GeneratedAudio;

          // Update section with generated audio
          setScriptSections((prev) =>
            prev.map((s) => (s.id === section.id ? { ...s, audio } : s))
          );

          setGeneratedAudio(audio);
          addGeneratedAudio(audio);
        } else {
          console.error("Generation failed:", data.error);
        }
      } catch (error) {
        console.error("Failed to generate audio:", error);
      } finally {
        setIsGenerating(false);
      }
    },
    [selectedVoiceId, ttsConfig, addGeneratedAudio]
  );

  // Generate all sections
  const handleGenerateAll = async () => {
    if (!selectedVoiceId) return;

    for (const section of scriptSections) {
      await handleGenerateAudio(section);
    }
  };

  // Toggle section expansion
  const toggleSection = (sectionId: string) => {
    setExpandedSections((prev) => {
      const next = new Set(prev);
      if (next.has(sectionId)) {
        next.delete(sectionId);
      } else {
        next.add(sectionId);
      }
      return next;
    });
  };

  // Handle voice upload
  const handleUploadVoice = async (file: File) => {
    const formData = new FormData();
    formData.append("audio", file);
    formData.append("name", file.name.replace(/\.[^/.]+$/, ""));

    try {
      const response = await fetch("/api/avatar-tts/voices", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      if (response.ok) {
        setVoices((prev) => [...prev, data.voice]);
        setSelectedVoiceId(data.voice.id);
      }
    } catch (error) {
      console.error("Failed to upload voice:", error);
    }
  };

  // Handle create avatar
  const handleCreateAvatar = async (
    name: string,
    style: Avatar["style"]
  ) => {
    const formData = new FormData();
    formData.append("name", name);
    formData.append("style", style);

    try {
      const response = await fetch("/api/avatar-tts/avatar", {
        method: "PUT",
        body: formData,
      });

      const data = await response.json();
      if (response.ok) {
        setAvatars((prev) => [...prev, data.avatar]);
        setSelectedAvatarId(data.avatar.id);
      }
    } catch (error) {
      console.error("Failed to create avatar:", error);
    }
  };

  // Calculate total duration
  const totalDuration = scriptSections.reduce((acc, section) => {
    if (section.audio) {
      return acc + section.audio.duration;
    }
    return acc + (section.endTime - section.startTime);
  }, 0);

  // Count generated sections
  const generatedCount = scriptSections.filter((s) => s.audio).length;

  if (isLoading) {
    return (
      <div className="flex flex-col min-h-screen">
        <Header
          title="Avatar & TTS"
          subtitle="Configure your AI presenter and voice settings"
        />
        <div className="flex flex-1 items-center justify-center p-6">
          <div className="h-12 w-12 animate-spin rounded-full border-4 border-primary border-t-transparent" />
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col min-h-screen">
      <Header
        title="Avatar & TTS"
        subtitle="Configure your AI presenter and voice settings"
      />

      <div className="flex-1 p-6">
        {/* Step Indicator */}
        <div className="mb-6">
          <EnhancedStepIndicator />
        </div>

        {/* Main Content */}
        <div className="grid gap-6 lg:grid-cols-3">
          {/* Left Column - Voice & Avatar Selection */}
          <div className="space-y-6 lg:col-span-2">
            <Tabs defaultValue="voice" className="w-full">
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="voice" className="flex items-center gap-2">
                  <Volume2 className="h-4 w-4" />
                  Voice Selection
                </TabsTrigger>
                <TabsTrigger value="avatar" className="flex items-center gap-2">
                  <User className="h-4 w-4" />
                  Avatar
                </TabsTrigger>
              </TabsList>

              <TabsContent value="voice" className="mt-4">
                <div className="rounded-xl border border-border bg-card p-6">
                  <VoiceSelector
                    voices={voices}
                    selectedVoiceId={selectedVoiceId}
                    onVoiceSelect={setSelectedVoiceId}
                    onUploadVoice={handleUploadVoice}
                  />
                </div>
              </TabsContent>

              <TabsContent value="avatar" className="mt-4">
                <div className="rounded-xl border border-border bg-card p-6">
                  <AvatarGenerator
                    avatars={avatars}
                    selectedAvatarId={selectedAvatarId}
                    onAvatarSelect={setSelectedAvatarId}
                    onCreateAvatar={handleCreateAvatar}
                    isEnabled={isAvatarEnabled}
                    onToggleEnabled={setIsAvatarEnabled}
                  />
                </div>
              </TabsContent>
            </Tabs>

            {/* Script Sections */}
            <div className="rounded-xl border border-border bg-card p-6">
              <div className="mb-4 flex items-center justify-between">
                <h3 className="flex items-center gap-2 text-lg font-semibold text-foreground">
                  <FileText className="h-5 w-5 text-primary" />
                  Script Sections
                </h3>
                <div className="flex items-center gap-2">
                  <span className="text-sm text-muted-foreground">
                    {generatedCount}/{scriptSections.length} generated
                  </span>
                  <Button
                    onClick={handleGenerateAll}
                    disabled={!selectedVoiceId || isGenerating}
                    size="sm"
                    className="gap-2"
                  >
                    <Wand2 className="h-4 w-4" />
                    Generate All
                  </Button>
                </div>
              </div>

              <div className="space-y-3">
                {scriptSections.map((section, index) => {
                  const isExpanded = expandedSections.has(section.id);
                  const isActive = activeSectionId === section.id;

                  return (
                    <div
                      key={section.id}
                      className={cn(
                        "rounded-lg border transition-all",
                        isActive
                          ? "border-primary bg-primary/5"
                          : "border-border hover:border-primary/50"
                      )}
                    >
                      {/* Section Header */}
                      <button
                        type="button"
                        onClick={() => toggleSection(section.id)}
                        className="flex w-full items-center justify-between p-3 text-left"
                      >
                        <div className="flex items-center gap-3">
                          <span className="flex h-6 w-6 items-center justify-center rounded-full bg-muted text-xs font-medium">
                            {index + 1}
                          </span>
                          <div>
                            <p className="text-sm font-medium text-foreground line-clamp-1">
                              {section.text.substring(0, 60)}...
                            </p>
                            <p className="text-xs text-muted-foreground">
                              {formatDuration(section.startTime)} -{" "}
                              {formatDuration(section.endTime)}
                              {section.audio && (
                                <span className="ml-2 text-green-500">
                                  Audio generated
                                </span>
                              )}
                            </p>
                          </div>
                        </div>
                        {isExpanded ? (
                          <ChevronUp className="h-4 w-4 text-muted-foreground" />
                        ) : (
                          <ChevronDown className="h-4 w-4 text-muted-foreground" />
                        )}
                      </button>

                      {/* Expanded Content */}
                      {isExpanded && (
                        <div className="border-t border-border p-3 space-y-3">
                          <p className="text-sm text-foreground">
                            {section.text}
                          </p>

                          {section.audio && (
                            <div className="space-y-2">
                              <AudioWaveform
                                data={section.audio.waveformData}
                                progress={100}
                                height={48}
                              />
                              <div className="flex items-center justify-between text-xs text-muted-foreground">
                                <span>
                                  Duration:{" "}
                                  {formatDuration(section.audio.duration)}
                                </span>
                                <span>
                                  Speed: {section.audio.config.speed}x
                                </span>
                              </div>
                            </div>
                          )}

                          <div className="flex gap-2">
                            <Button
                              onClick={() => handleGenerateAudio(section)}
                              disabled={!selectedVoiceId || isGenerating}
                              size="sm"
                              variant={section.audio ? "outline" : "default"}
                              className="gap-2"
                            >
                              {isGenerating && isActive ? (
                                <RefreshCw className="h-3 w-3 animate-spin" />
                              ) : (
                                <Wand2 className="h-3 w-3" />
                              )}
                              {section.audio ? "Regenerate" : "Generate"}
                            </Button>
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Right Column - TTS Controls & Preview */}
          <div className="space-y-6">
            {/* TTS Controls */}
            <div className="rounded-xl border border-border bg-card p-6">
              <TTSControls config={ttsConfig} onChange={setLocalTTSConfig} />
            </div>

            {/* Audio Preview */}
            <div className="rounded-xl border border-border bg-card p-6">
              <h3 className="mb-4 flex items-center gap-2 text-lg font-semibold text-foreground">
                <Volume2 className="h-5 w-5 text-secondary" />
                Preview
              </h3>
              <VoicePreview
                audio={generatedAudio}
                isLoading={isGenerating}
                onRegenerate={
                  activeSectionId
                    ? () => {
                        const section = scriptSections.find(
                          (s) => s.id === activeSectionId
                        );
                        if (section) handleGenerateAudio(section);
                      }
                    : undefined
                }
              />
            </div>

            {/* Summary */}
            <div className="rounded-xl border border-border bg-card p-6">
              <h3 className="mb-4 text-sm font-semibold text-foreground">
                Summary
              </h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Selected Voice</span>
                  <span className="font-medium text-foreground">
                    {selectedVoiceId
                      ? voices.find((v) => v.id === selectedVoiceId)?.name ||
                        "Unknown"
                      : "None"}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Avatar</span>
                  <span className="font-medium text-foreground">
                    {isAvatarEnabled
                      ? selectedAvatarId
                        ? avatars.find((a) => a.id === selectedAvatarId)
                            ?.name || "Unknown"
                        : "No Avatar"
                      : "Disabled"}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Total Duration</span>
                  <span className="font-medium text-foreground">
                    {formatDuration(totalDuration)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Sections</span>
                  <span className="font-medium text-foreground">
                    {generatedCount}/{scriptSections.length} ready
                  </span>
                </div>
              </div>
            </div>

            {/* Navigation */}
            <WorkflowNav />
          </div>
        </div>
      </div>
    </div>
  );
}
