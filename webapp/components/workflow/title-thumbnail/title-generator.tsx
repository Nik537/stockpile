"use client";

import { useState, useCallback } from "react";
import {
  Sparkles,
  RefreshCw,
  Check,
  Lightbulb,
  HelpCircle,
  List,
  BookOpen,
  Zap,
} from "lucide-react";
import { cn } from "@/lib/utils";
import type { TitleSuggestion, TitleStyle, TitleGenerationResponse } from "@/types/workflow";

const STYLE_INFO: Record<
  TitleStyle,
  { label: string; description: string; icon: React.ComponentType<{ className?: string }> }
> = {
  hook: {
    label: "Hook",
    description: "Attention-grabbing statements",
    icon: Zap,
  },
  curiosity: {
    label: "Curiosity",
    description: "Creates intrigue and questions",
    icon: HelpCircle,
  },
  howto: {
    label: "How-To",
    description: "Educational and instructional",
    icon: Lightbulb,
  },
  listicle: {
    label: "Listicle",
    description: "Numbered lists and tips",
    icon: List,
  },
  story: {
    label: "Story",
    description: "Personal narrative format",
    icon: BookOpen,
  },
};

interface TitleGeneratorProps {
  onTitleSelect: (title: string) => void;
  selectedTitle: string;
  topic: string;
  onTopicChange: (topic: string) => void;
}

export function TitleGenerator({
  onTitleSelect,
  selectedTitle,
  topic,
  onTopicChange,
}: TitleGeneratorProps) {
  const [suggestions, setSuggestions] = useState<TitleSuggestion[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [customTitle, setCustomTitle] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [filterStyle, setFilterStyle] = useState<TitleStyle | "all">("all");

  const generateTitles = useCallback(async () => {
    if (!topic.trim()) {
      setError("Please enter a topic first");
      return;
    }

    setIsGenerating(true);
    setError(null);

    try {
      const response = await fetch("/api/title-thumbnail/titles", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          topic: topic.trim(),
          count: 10,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to generate titles");
      }

      const data: TitleGenerationResponse = await response.json();
      setSuggestions(data.suggestions);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setIsGenerating(false);
    }
  }, [topic]);

  const handleCustomTitleChange = (value: string) => {
    setCustomTitle(value);
    if (value.trim()) {
      onTitleSelect(value);
    }
  };

  const handleSuggestionSelect = (title: string) => {
    setCustomTitle("");
    onTitleSelect(title);
  };

  const filteredSuggestions =
    filterStyle === "all"
      ? suggestions
      : suggestions.filter((s) => s.style === filterStyle);

  const getScoreColor = (score: number) => {
    if (score >= 90) return "text-green-500";
    if (score >= 80) return "text-yellow-500";
    return "text-muted-foreground";
  };

  return (
    <div className="space-y-4">
      {/* Topic Input */}
      <div className="rounded-xl border border-border bg-card p-6">
        <h3 className="mb-4 flex items-center gap-2 text-lg font-semibold text-foreground">
          <Sparkles className="h-5 w-5 text-primary" />
          Video Topic
        </h3>

        <div className="space-y-4">
          <div>
            <label
              htmlFor="topic"
              className="mb-2 block text-sm font-medium text-foreground"
            >
              What is your video about?
            </label>
            <textarea
              id="topic"
              value={topic}
              onChange={(e) => onTopicChange(e.target.value)}
              placeholder="e.g., How to build a successful startup, Best productivity tips for developers, My journey learning to code..."
              rows={3}
              className="w-full rounded-lg border border-input bg-background px-4 py-3 text-sm text-foreground placeholder:text-muted-foreground focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary resize-none"
            />
          </div>

          <button
            type="button"
            onClick={generateTitles}
            disabled={isGenerating || !topic.trim()}
            className="flex w-full items-center justify-center gap-2 rounded-lg bg-primary px-4 py-3 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {isGenerating ? (
              <>
                <div className="h-4 w-4 animate-spin rounded-full border-2 border-primary-foreground border-t-transparent" />
                Generating Titles...
              </>
            ) : (
              <>
                <Sparkles className="h-4 w-4" />
                Generate Title Suggestions
              </>
            )}
          </button>

          {error && (
            <p className="text-sm text-destructive">{error}</p>
          )}
        </div>
      </div>

      {/* Generated Titles */}
      {suggestions.length > 0 && (
        <div className="rounded-xl border border-border bg-card p-6">
          <div className="mb-4 flex items-center justify-between">
            <h3 className="flex items-center gap-2 text-lg font-semibold text-foreground">
              <Lightbulb className="h-5 w-5 text-yellow-500" />
              AI Suggestions
            </h3>
            <button
              type="button"
              onClick={generateTitles}
              disabled={isGenerating}
              className="flex items-center gap-2 rounded-lg border border-border px-3 py-1.5 text-sm font-medium text-muted-foreground transition-colors hover:bg-muted hover:text-foreground disabled:opacity-50"
            >
              <RefreshCw className={cn("h-4 w-4", isGenerating && "animate-spin")} />
              Regenerate
            </button>
          </div>

          {/* Style Filter */}
          <div className="mb-4 flex flex-wrap gap-2">
            <button
              type="button"
              onClick={() => setFilterStyle("all")}
              className={cn(
                "rounded-full px-3 py-1 text-xs font-medium transition-colors",
                filterStyle === "all"
                  ? "bg-primary text-primary-foreground"
                  : "bg-muted text-muted-foreground hover:bg-muted/80"
              )}
            >
              All
            </button>
            {(Object.keys(STYLE_INFO) as TitleStyle[]).map((style) => {
              const info = STYLE_INFO[style];
              const Icon = info.icon;
              return (
                <button
                  key={style}
                  type="button"
                  onClick={() => setFilterStyle(style)}
                  className={cn(
                    "flex items-center gap-1 rounded-full px-3 py-1 text-xs font-medium transition-colors",
                    filterStyle === style
                      ? "bg-primary text-primary-foreground"
                      : "bg-muted text-muted-foreground hover:bg-muted/80"
                  )}
                >
                  <Icon className="h-3 w-3" />
                  {info.label}
                </button>
              );
            })}
          </div>

          {/* Suggestions List */}
          <div className="space-y-2 max-h-[400px] overflow-y-auto">
            {filteredSuggestions.map((suggestion) => {
              const styleInfo = STYLE_INFO[suggestion.style];
              const StyleIcon = styleInfo.icon;

              return (
                <button
                  key={suggestion.id}
                  type="button"
                  onClick={() => handleSuggestionSelect(suggestion.title)}
                  className={cn(
                    "flex w-full items-center gap-3 rounded-lg border px-4 py-3 text-left transition-colors",
                    selectedTitle === suggestion.title
                      ? "border-primary bg-primary/10 text-foreground"
                      : "border-border bg-background text-muted-foreground hover:border-primary/50 hover:bg-muted"
                  )}
                >
                  <div className="flex-1">
                    <p className="text-sm font-medium">{suggestion.title}</p>
                    <div className="mt-1 flex items-center gap-2">
                      <span className="flex items-center gap-1 text-xs text-muted-foreground">
                        <StyleIcon className="h-3 w-3" />
                        {styleInfo.label}
                      </span>
                      <span className="text-muted-foreground/50">-</span>
                      <span className={cn("text-xs font-medium", getScoreColor(suggestion.score))}>
                        Score: {suggestion.score}%
                      </span>
                    </div>
                  </div>
                  {selectedTitle === suggestion.title && (
                    <Check className="h-5 w-5 flex-shrink-0 text-primary" />
                  )}
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* Custom Title */}
      <div className="rounded-xl border border-border bg-card p-6">
        <h3 className="mb-4 flex items-center gap-2 text-lg font-semibold text-foreground">
          Write Your Own
        </h3>
        <input
          type="text"
          value={customTitle}
          onChange={(e) => handleCustomTitleChange(e.target.value)}
          placeholder="Enter a custom title..."
          className="w-full rounded-lg border border-input bg-background px-4 py-3 text-sm text-foreground placeholder:text-muted-foreground focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
        />
        {customTitle && selectedTitle === customTitle && (
          <div className="mt-2 flex items-center gap-2 text-sm text-primary">
            <Check className="h-4 w-4" />
            Custom title selected
          </div>
        )}
      </div>
    </div>
  );
}
