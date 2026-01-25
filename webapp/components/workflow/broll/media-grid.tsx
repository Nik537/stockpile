"use client";

import { useMemo } from "react";
import { Loader2, Search, AlertCircle } from "lucide-react";
import { cn } from "@/lib/utils";
import { MediaCard } from "./media-card";
import type { BRollItem, MediaSource } from "@/types/broll";

interface MediaGridProps {
  items: BRollItem[];
  selectedItems?: BRollItem[];
  onSelectItem?: (item: BRollItem) => void;
  onDownloadItem?: (item: BRollItem) => void;
  onPreviewItem?: (item: BRollItem) => void;
  isLoading?: boolean;
  error?: string;
  emptyMessage?: string;
  showRelevanceScores?: boolean;
  isDraggable?: boolean;
  columns?: 2 | 3 | 4;
  sourceResults?: {
    source: MediaSource;
    count: number;
    error?: string;
  }[];
  className?: string;
}

export function MediaGrid({
  items,
  selectedItems = [],
  onSelectItem,
  onDownloadItem,
  onPreviewItem,
  isLoading = false,
  error,
  emptyMessage = "No results found. Try a different search term.",
  showRelevanceScores = false,
  isDraggable = false,
  columns = 3,
  sourceResults,
  className,
}: MediaGridProps) {
  // Check if item is selected
  const isItemSelected = (item: BRollItem) => {
    return selectedItems.some((selected) => selected.id === item.id);
  };

  // Grid column classes
  const gridCols = useMemo(() => {
    switch (columns) {
      case 2:
        return "grid-cols-1 sm:grid-cols-2";
      case 4:
        return "grid-cols-2 sm:grid-cols-3 lg:grid-cols-4";
      default:
        return "grid-cols-1 sm:grid-cols-2 lg:grid-cols-3";
    }
  }, [columns]);

  // Loading State
  if (isLoading) {
    return (
      <div className={cn("flex flex-col items-center justify-center py-16", className)}>
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        <p className="mt-4 text-sm text-muted-foreground">Searching across sources...</p>
        {sourceResults && (
          <div className="mt-4 flex flex-wrap gap-2 justify-center">
            {sourceResults.map((result) => (
              <span
                key={result.source}
                className={cn(
                  "inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs",
                  result.error
                    ? "bg-destructive/10 text-destructive"
                    : result.count > 0
                    ? "bg-green-500/10 text-green-600"
                    : "bg-muted text-muted-foreground"
                )}
              >
                <span className="capitalize">{result.source}</span>
                {result.error ? (
                  <AlertCircle className="h-3 w-3" />
                ) : (
                  <span>({result.count})</span>
                )}
              </span>
            ))}
          </div>
        )}
      </div>
    );
  }

  // Error State
  if (error) {
    return (
      <div className={cn("flex flex-col items-center justify-center py-16", className)}>
        <div className="flex h-12 w-12 items-center justify-center rounded-full bg-destructive/10">
          <AlertCircle className="h-6 w-6 text-destructive" />
        </div>
        <p className="mt-4 text-sm text-destructive">{error}</p>
      </div>
    );
  }

  // Empty State
  if (items.length === 0) {
    return (
      <div className={cn("flex flex-col items-center justify-center py-16", className)}>
        <div className="flex h-12 w-12 items-center justify-center rounded-full bg-muted">
          <Search className="h-6 w-6 text-muted-foreground" />
        </div>
        <p className="mt-4 text-sm text-muted-foreground">{emptyMessage}</p>
        {sourceResults && sourceResults.some((r) => r.error) && (
          <div className="mt-4 flex flex-col gap-1">
            {sourceResults
              .filter((r) => r.error)
              .map((result) => (
                <span
                  key={result.source}
                  className="text-xs text-destructive"
                >
                  {result.source}: {result.error}
                </span>
              ))}
          </div>
        )}
      </div>
    );
  }

  return (
    <div className={className}>
      {/* Source Results Summary */}
      {sourceResults && (
        <div className="mb-4 flex flex-wrap items-center gap-2">
          <span className="text-sm text-muted-foreground">Results:</span>
          {sourceResults.map((result) => (
            <span
              key={result.source}
              className={cn(
                "inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs",
                result.error
                  ? "bg-destructive/10 text-destructive"
                  : "bg-primary/10 text-foreground"
              )}
            >
              <span className="capitalize">{result.source}</span>
              {result.error ? (
                <AlertCircle className="h-3 w-3" aria-label={result.error} />
              ) : (
                <span>({result.count})</span>
              )}
            </span>
          ))}
          <span className="text-sm text-muted-foreground">
            Total: {items.length}
          </span>
        </div>
      )}

      {/* Grid */}
      <div className={cn("grid gap-4", gridCols)}>
        {items.map((item) => (
          <MediaCard
            key={`${item.source}-${item.id}`}
            item={item}
            isSelected={isItemSelected(item)}
            onSelect={onSelectItem}
            onDownload={onDownloadItem}
            onPreview={onPreviewItem}
            isDraggable={isDraggable}
            showRelevanceScore={showRelevanceScores}
          />
        ))}
      </div>

      {/* Load More (placeholder for pagination) */}
      {items.length >= 20 && (
        <div className="mt-6 flex justify-center">
          <button
            type="button"
            className="rounded-lg border border-border px-6 py-2 text-sm font-medium text-foreground transition-colors hover:bg-muted"
          >
            Load More Results
          </button>
        </div>
      )}
    </div>
  );
}
