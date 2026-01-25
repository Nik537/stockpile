"use client";

import { useState } from "react";
import {
  Sparkles,
  Search,
  Clock,
  ChevronRight,
  Loader2,
  RefreshCw,
  FileText,
  Lightbulb,
  Star,
  Copy,
  Check,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import type { AISuggestion, ScriptSection } from "@/types/broll";

interface AISuggestionsProps {
  suggestions: AISuggestion[];
  scriptSections?: ScriptSection[];
  onSearchSuggestion: (suggestion: AISuggestion) => void;
  onGenerateSuggestions?: () => void;
  isGenerating?: boolean;
  hasScript?: boolean;
  className?: string;
}

export function AISuggestions({
  suggestions,
  scriptSections = [],
  onSearchSuggestion,
  onGenerateSuggestions,
  isGenerating = false,
  hasScript = true,
  className,
}: AISuggestionsProps) {
  const [expandedSuggestion, setExpandedSuggestion] = useState<string | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const copySearchTerm = (suggestion: AISuggestion) => {
    navigator.clipboard.writeText(suggestion.searchTerm);
    setCopiedId(suggestion.id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  // Get confidence color
  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return "text-green-500";
    if (confidence >= 0.6) return "text-yellow-500";
    return "text-orange-500";
  };

  const getConfidenceBg = (confidence: number) => {
    if (confidence >= 0.8) return "bg-green-500/10";
    if (confidence >= 0.6) return "bg-yellow-500/10";
    return "bg-orange-500/10";
  };

  // No script state
  if (!hasScript) {
    return (
      <div className={cn("rounded-xl border border-border bg-card p-6", className)}>
        <div className="flex items-center gap-3 mb-4">
          <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary/10">
            <Sparkles className="h-5 w-5 text-primary" />
          </div>
          <div>
            <h3 className="font-semibold text-foreground">AI B-Roll Suggestions</h3>
            <p className="text-sm text-muted-foreground">Script analysis required</p>
          </div>
        </div>
        <div className="flex flex-col items-center justify-center py-8 text-center">
          <div className="mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-muted">
            <FileText className="h-6 w-6 text-muted-foreground" />
          </div>
          <p className="text-sm text-muted-foreground mb-4">
            Add a script to your project to get AI-powered B-roll suggestions for each section
          </p>
          <Button variant="outline" disabled>
            <FileText className="h-4 w-4 mr-2" />
            Go to Script Writer
          </Button>
        </div>
      </div>
    );
  }

  // Empty state (has script but no suggestions yet)
  if (suggestions.length === 0 && !isGenerating) {
    return (
      <div className={cn("rounded-xl border border-border bg-card p-6", className)}>
        <div className="flex items-center gap-3 mb-4">
          <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary/10">
            <Sparkles className="h-5 w-5 text-primary" />
          </div>
          <div>
            <h3 className="font-semibold text-foreground">AI B-Roll Suggestions</h3>
            <p className="text-sm text-muted-foreground">
              Analyze your script for B-roll opportunities
            </p>
          </div>
        </div>
        <div className="flex flex-col items-center justify-center py-8 text-center">
          <div className="mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-muted">
            <Lightbulb className="h-6 w-6 text-muted-foreground" />
          </div>
          <p className="text-sm text-muted-foreground mb-4">
            Let AI analyze your script and suggest relevant B-roll footage for each section
          </p>
          {onGenerateSuggestions && (
            <Button onClick={onGenerateSuggestions}>
              <Sparkles className="h-4 w-4 mr-2" />
              Generate Suggestions
            </Button>
          )}
        </div>
      </div>
    );
  }

  // Loading state
  if (isGenerating) {
    return (
      <div className={cn("rounded-xl border border-border bg-card p-6", className)}>
        <div className="flex items-center gap-3 mb-4">
          <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary/10">
            <Sparkles className="h-5 w-5 text-primary" />
          </div>
          <div>
            <h3 className="font-semibold text-foreground">AI B-Roll Suggestions</h3>
            <p className="text-sm text-muted-foreground">Analyzing script...</p>
          </div>
        </div>
        <div className="flex flex-col items-center justify-center py-8 text-center">
          <Loader2 className="h-8 w-8 animate-spin text-primary mb-4" />
          <p className="text-sm text-muted-foreground">
            Analyzing your script to find the best B-roll opportunities...
          </p>
          <p className="text-xs text-muted-foreground mt-2">
            This may take a few seconds
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className={cn("rounded-xl border border-border bg-card", className)}>
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-border">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary/10">
            <Sparkles className="h-5 w-5 text-primary" />
          </div>
          <div>
            <h3 className="font-semibold text-foreground">AI B-Roll Suggestions</h3>
            <p className="text-sm text-muted-foreground">
              {suggestions.length} suggestions for your script
            </p>
          </div>
        </div>
        {onGenerateSuggestions && (
          <Button
            variant="ghost"
            size="sm"
            onClick={onGenerateSuggestions}
            disabled={isGenerating}
          >
            <RefreshCw className={cn("h-4 w-4", isGenerating && "animate-spin")} />
            <span className="ml-1">Refresh</span>
          </Button>
        )}
      </div>

      {/* Suggestions List */}
      <div className="divide-y divide-border max-h-[500px] overflow-y-auto">
        {suggestions.map((suggestion) => (
          <div
            key={suggestion.id}
            className="p-4 hover:bg-muted/30 transition-colors"
          >
            <div className="flex items-start gap-3">
              {/* Timestamp */}
              <div className="flex flex-col items-center flex-shrink-0">
                <div className="flex h-8 w-14 items-center justify-center rounded bg-muted text-xs font-mono font-medium text-foreground">
                  <Clock className="h-3 w-3 mr-1 text-muted-foreground" />
                  {suggestion.timestamp}
                </div>
                {/* Confidence */}
                <div className={cn(
                  "mt-1 flex items-center gap-0.5 rounded px-1.5 py-0.5 text-xs",
                  getConfidenceBg(suggestion.confidence),
                  getConfidenceColor(suggestion.confidence)
                )}>
                  <Star className="h-3 w-3" />
                  {Math.round(suggestion.confidence * 100)}%
                </div>
              </div>

              {/* Content */}
              <div className="flex-1 min-w-0">
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <p className="text-sm font-medium text-foreground">
                      {suggestion.searchTerm}
                    </p>
                    <p className="text-xs text-muted-foreground mt-0.5 line-clamp-2">
                      {suggestion.description}
                    </p>
                  </div>

                  {/* Actions */}
                  <div className="flex items-center gap-1 flex-shrink-0">
                    <button
                      type="button"
                      onClick={() => copySearchTerm(suggestion)}
                      className="p-1.5 text-muted-foreground hover:text-foreground transition-colors"
                      title="Copy search term"
                    >
                      {copiedId === suggestion.id ? (
                        <Check className="h-4 w-4 text-green-500" />
                      ) : (
                        <Copy className="h-4 w-4" />
                      )}
                    </button>
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={() => onSearchSuggestion(suggestion)}
                    >
                      <Search className="h-3.5 w-3.5 mr-1" />
                      Search
                    </Button>
                  </div>
                </div>

                {/* Expandable Script Context */}
                {suggestion.scriptContext && (
                  <div className="mt-2">
                    <button
                      type="button"
                      onClick={() =>
                        setExpandedSuggestion(
                          expandedSuggestion === suggestion.id ? null : suggestion.id
                        )
                      }
                      className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
                    >
                      <ChevronRight
                        className={cn(
                          "h-3 w-3 transition-transform",
                          expandedSuggestion === suggestion.id && "rotate-90"
                        )}
                      />
                      View script context
                    </button>
                    {expandedSuggestion === suggestion.id && (
                      <div className="mt-2 rounded-lg bg-muted/50 p-3 text-xs text-muted-foreground italic">
                        &ldquo;{suggestion.scriptContext}&rdquo;
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Footer with Tip */}
      <div className="px-4 py-3 border-t border-border bg-muted/30">
        <div className="flex items-start gap-2">
          <Lightbulb className="h-4 w-4 text-amber-500 flex-shrink-0 mt-0.5" />
          <p className="text-xs text-muted-foreground">
            <span className="font-medium">Tip:</span> Click &ldquo;Search&rdquo; to find B-roll
            matching the suggestion. Higher confidence scores indicate better matches
            for your script context.
          </p>
        </div>
      </div>
    </div>
  );
}
