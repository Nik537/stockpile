"use client";

import { useState, useCallback } from "react";
import { cn } from "@/lib/utils";
import { formatDuration } from "@/lib/utils";
import type { BRollSuggestion, ScriptSegment } from "@/types/workflow";
import { Button } from "@/components/ui/button";
import {
  Sparkles,
  Check,
  X,
  ChevronDown,
  ChevronUp,
  Image,
  Clock,
  Loader2,
  RefreshCw,
} from "lucide-react";

interface BRollSuggestionsProps {
  suggestions: BRollSuggestion[];
  scriptSegments?: ScriptSegment[];
  isLoading?: boolean;
  onAccept: (suggestionId: string) => void;
  onReject: (suggestionId: string) => void;
  onAcceptAll: () => void;
  onRejectAll: () => void;
  onRefresh?: () => void;
  onGenerateSuggestions?: () => void;
}

interface SuggestionCardProps {
  suggestion: BRollSuggestion;
  onAccept: () => void;
  onReject: () => void;
}

function SuggestionCard({ suggestion, onAccept, onReject }: SuggestionCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div
      className={cn(
        "rounded-lg border transition-all",
        suggestion.accepted
          ? "border-green-500/50 bg-green-500/5"
          : "border-border bg-card"
      )}
    >
      {/* Header */}
      <div
        className="flex items-center gap-3 p-3 cursor-pointer"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        {/* Thumbnail or placeholder */}
        <div className="w-16 h-10 rounded bg-muted flex items-center justify-center overflow-hidden shrink-0">
          {suggestion.thumbnailUrl ? (
            <img
              src={suggestion.thumbnailUrl}
              alt={suggestion.searchQuery}
              className="w-full h-full object-cover"
            />
          ) : (
            <Image className="h-4 w-4 text-muted-foreground" />
          )}
        </div>

        {/* Info */}
        <div className="flex-1 min-w-0">
          <div className="text-sm font-medium text-foreground truncate">
            {suggestion.searchQuery}
          </div>
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <Clock className="h-3 w-3" />
            <span>At {formatDuration(suggestion.timestamp)}</span>
            <span>({formatDuration(suggestion.duration)})</span>
          </div>
        </div>

        {/* Status/Actions */}
        <div className="flex items-center gap-1">
          {suggestion.accepted ? (
            <span className="text-xs text-green-600 flex items-center gap-1">
              <Check className="h-3 w-3" />
              Added
            </span>
          ) : (
            <>
              <Button
                variant="ghost"
                size="sm"
                onClick={(e) => {
                  e.stopPropagation();
                  onAccept();
                }}
                className="h-7 w-7 p-0 text-green-600 hover:bg-green-500/10"
                title="Accept suggestion"
              >
                <Check className="h-4 w-4" />
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={(e) => {
                  e.stopPropagation();
                  onReject();
                }}
                className="h-7 w-7 p-0 text-destructive hover:bg-destructive/10"
                title="Reject suggestion"
              >
                <X className="h-4 w-4" />
              </Button>
            </>
          )}
          {isExpanded ? (
            <ChevronUp className="h-4 w-4 text-muted-foreground" />
          ) : (
            <ChevronDown className="h-4 w-4 text-muted-foreground" />
          )}
        </div>
      </div>

      {/* Expanded content */}
      {isExpanded && (
        <div className="px-3 pb-3 pt-0 border-t border-border">
          <p className="text-xs text-muted-foreground mt-2">{suggestion.reason}</p>
          {suggestion.mediaUrl && (
            <div className="mt-2 aspect-video rounded overflow-hidden bg-black">
              <video
                src={suggestion.mediaUrl}
                className="w-full h-full object-contain"
                controls
                preload="metadata"
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export function BRollSuggestions({
  suggestions,
  scriptSegments,
  isLoading = false,
  onAccept,
  onReject,
  onAcceptAll,
  onRejectAll,
  onRefresh,
  onGenerateSuggestions,
}: BRollSuggestionsProps) {
  const pendingSuggestions = suggestions.filter((s) => !s.accepted);
  const acceptedCount = suggestions.filter((s) => s.accepted).length;

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between p-3 border-b border-border">
        <div className="flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-primary" />
          <span className="text-sm font-medium text-foreground">
            AI B-Roll Suggestions
          </span>
        </div>
        {suggestions.length > 0 && (
          <span className="text-xs text-muted-foreground">
            {acceptedCount}/{suggestions.length} accepted
          </span>
        )}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-3">
        {isLoading ? (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <Loader2 className="h-8 w-8 animate-spin text-primary mb-2" />
            <p className="text-sm text-muted-foreground">
              Analyzing script and generating suggestions...
            </p>
          </div>
        ) : suggestions.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center p-4">
            <Sparkles className="h-12 w-12 text-muted-foreground/30 mb-4" />
            <p className="text-sm text-muted-foreground mb-2">
              No B-roll suggestions yet
            </p>
            <p className="text-xs text-muted-foreground/70 mb-4">
              Click the button below to analyze your script and get AI-powered
              suggestions for B-roll placement.
            </p>
            {onGenerateSuggestions && (
              <Button onClick={onGenerateSuggestions} size="sm">
                <Sparkles className="h-4 w-4 mr-2" />
                Auto-place B-Roll
              </Button>
            )}
          </div>
        ) : (
          <div className="space-y-2">
            {suggestions.map((suggestion) => (
              <SuggestionCard
                key={suggestion.id}
                suggestion={suggestion}
                onAccept={() => onAccept(suggestion.id)}
                onReject={() => onReject(suggestion.id)}
              />
            ))}
          </div>
        )}
      </div>

      {/* Actions footer */}
      {suggestions.length > 0 && !isLoading && (
        <div className="border-t border-border p-3 space-y-2">
          {pendingSuggestions.length > 0 && (
            <div className="grid grid-cols-2 gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={onAcceptAll}
                className="h-8 text-xs"
              >
                <Check className="h-3 w-3 mr-1" />
                Accept All ({pendingSuggestions.length})
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={onRejectAll}
                className="h-8 text-xs text-destructive hover:bg-destructive/10"
              >
                <X className="h-3 w-3 mr-1" />
                Reject All
              </Button>
            </div>
          )}
          {onRefresh && (
            <Button
              variant="ghost"
              size="sm"
              onClick={onRefresh}
              className="w-full h-8 text-xs"
            >
              <RefreshCw className="h-3 w-3 mr-1" />
              Regenerate Suggestions
            </Button>
          )}
        </div>
      )}
    </div>
  );
}
