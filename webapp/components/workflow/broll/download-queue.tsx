"use client";

import { useState } from "react";
import {
  Download,
  X,
  Check,
  AlertCircle,
  Loader2,
  Trash2,
  Play,
  Pause,
  ChevronDown,
  ChevronUp,
  ExternalLink,
  Film,
  ImageIcon,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";
import { formatDuration } from "@/lib/utils";
import type { DownloadQueueItem } from "@/types/broll";

interface DownloadQueueProps {
  items: DownloadQueueItem[];
  onRemoveItem: (id: string) => void;
  onRetryItem?: (id: string) => void;
  onClearCompleted?: () => void;
  onPauseAll?: () => void;
  onResumeAll?: () => void;
  isPaused?: boolean;
  className?: string;
}

export function DownloadQueue({
  items,
  onRemoveItem,
  onRetryItem,
  onClearCompleted,
  onPauseAll,
  onResumeAll,
  isPaused = false,
  className,
}: DownloadQueueProps) {
  const [isExpanded, setIsExpanded] = useState(true);

  // Count items by status
  const pendingCount = items.filter((i) => i.status === "pending").length;
  const downloadingCount = items.filter((i) => i.status === "downloading").length;
  const completeCount = items.filter((i) => i.status === "complete").length;
  const errorCount = items.filter((i) => i.status === "error").length;

  // Calculate overall progress
  const totalProgress = items.length > 0
    ? items.reduce((acc, item) => acc + item.progress, 0) / items.length
    : 0;

  // Get status icon
  const getStatusIcon = (status: DownloadQueueItem["status"]) => {
    switch (status) {
      case "pending":
        return <Loader2 className="h-4 w-4 text-muted-foreground" />;
      case "downloading":
        return <Loader2 className="h-4 w-4 animate-spin text-primary" />;
      case "complete":
        return <Check className="h-4 w-4 text-green-500" />;
      case "error":
        return <AlertCircle className="h-4 w-4 text-destructive" />;
    }
  };

  // Get status color
  const getStatusColor = (status: DownloadQueueItem["status"]) => {
    switch (status) {
      case "pending":
        return "text-muted-foreground";
      case "downloading":
        return "text-primary";
      case "complete":
        return "text-green-500";
      case "error":
        return "text-destructive";
    }
  };

  if (items.length === 0) {
    return (
      <div className={cn("rounded-xl border border-border bg-card p-6", className)}>
        <div className="flex items-center gap-3 mb-4">
          <div className="flex h-10 w-10 items-center justify-center rounded-full bg-muted">
            <Download className="h-5 w-5 text-muted-foreground" />
          </div>
          <div>
            <h3 className="font-semibold text-foreground">Download Queue</h3>
            <p className="text-sm text-muted-foreground">No downloads in queue</p>
          </div>
        </div>
        <div className="flex items-center justify-center py-8 text-center">
          <p className="text-sm text-muted-foreground">
            Select media items and click download to add them to the queue
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className={cn("rounded-xl border border-border bg-card", className)}>
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-border">
        <button
          type="button"
          onClick={() => setIsExpanded(!isExpanded)}
          className="flex items-center gap-3"
        >
          <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary/10">
            <Download className="h-5 w-5 text-primary" />
          </div>
          <div className="text-left">
            <h3 className="font-semibold text-foreground">Download Queue</h3>
            <p className="text-sm text-muted-foreground">
              {downloadingCount > 0
                ? `Downloading ${downloadingCount} of ${items.length}`
                : `${completeCount} complete, ${pendingCount} pending`}
            </p>
          </div>
          {isExpanded ? (
            <ChevronUp className="h-4 w-4 text-muted-foreground" />
          ) : (
            <ChevronDown className="h-4 w-4 text-muted-foreground" />
          )}
        </button>

        <div className="flex items-center gap-2">
          {/* Pause/Resume Button */}
          {(onPauseAll || onResumeAll) && (pendingCount > 0 || downloadingCount > 0) && (
            <Button
              variant="ghost"
              size="sm"
              onClick={isPaused ? onResumeAll : onPauseAll}
            >
              {isPaused ? (
                <>
                  <Play className="h-4 w-4" />
                  <span className="ml-1">Resume</span>
                </>
              ) : (
                <>
                  <Pause className="h-4 w-4" />
                  <span className="ml-1">Pause</span>
                </>
              )}
            </Button>
          )}

          {/* Clear Completed Button */}
          {onClearCompleted && completeCount > 0 && (
            <Button
              variant="ghost"
              size="sm"
              onClick={onClearCompleted}
              className="text-muted-foreground"
            >
              <Trash2 className="h-4 w-4" />
              <span className="ml-1">Clear Done</span>
            </Button>
          )}
        </div>
      </div>

      {/* Overall Progress Bar */}
      {(downloadingCount > 0 || pendingCount > 0) && (
        <div className="px-4 py-2 border-b border-border">
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs text-muted-foreground">Overall Progress</span>
            <span className="text-xs font-medium text-foreground">{Math.round(totalProgress)}%</span>
          </div>
          <Progress value={totalProgress} className="h-1.5" />
        </div>
      )}

      {/* Queue Items */}
      {isExpanded && (
        <div className="divide-y divide-border max-h-[400px] overflow-y-auto">
          {items.map((item) => (
            <div key={item.id} className="p-3 hover:bg-muted/50 transition-colors">
              <div className="flex items-start gap-3">
                {/* Thumbnail */}
                <div className="relative h-12 w-16 flex-shrink-0 overflow-hidden rounded bg-muted">
                  {item.item.thumbnailUrl ? (
                    <img
                      src={item.item.thumbnailUrl}
                      alt={item.item.title}
                      className="h-full w-full object-cover"
                    />
                  ) : (
                    <div className="flex h-full w-full items-center justify-center">
                      {item.item.type === "video" ? (
                        <Film className="h-5 w-5 text-muted-foreground" />
                      ) : (
                        <ImageIcon className="h-5 w-5 text-muted-foreground" />
                      )}
                    </div>
                  )}
                  {/* Status overlay */}
                  <div className="absolute inset-0 flex items-center justify-center bg-black/40">
                    {getStatusIcon(item.status)}
                  </div>
                </div>

                {/* Info */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-start justify-between gap-2">
                    <div className="min-w-0">
                      <p className="text-sm font-medium text-foreground truncate">
                        {item.item.title}
                      </p>
                      <div className="flex items-center gap-2 mt-0.5">
                        <span className={cn("text-xs capitalize", getStatusColor(item.status))}>
                          {item.status}
                        </span>
                        {item.item.duration && (
                          <span className="text-xs text-muted-foreground">
                            {formatDuration(item.item.duration)}
                          </span>
                        )}
                        <span className="text-xs text-muted-foreground capitalize">
                          {item.item.source}
                        </span>
                      </div>
                    </div>

                    {/* Actions */}
                    <div className="flex items-center gap-1">
                      {item.status === "error" && onRetryItem && (
                        <button
                          type="button"
                          onClick={() => onRetryItem(item.id)}
                          className="p-1 text-muted-foreground hover:text-foreground transition-colors"
                          title="Retry"
                        >
                          <Play className="h-4 w-4" />
                        </button>
                      )}
                      {item.status === "complete" && item.downloadedPath && (
                        <button
                          type="button"
                          className="p-1 text-muted-foreground hover:text-foreground transition-colors"
                          title="Open file"
                        >
                          <ExternalLink className="h-4 w-4" />
                        </button>
                      )}
                      <button
                        type="button"
                        onClick={() => onRemoveItem(item.id)}
                        className="p-1 text-muted-foreground hover:text-destructive transition-colors"
                        title="Remove"
                      >
                        <X className="h-4 w-4" />
                      </button>
                    </div>
                  </div>

                  {/* Progress Bar (for downloading items) */}
                  {item.status === "downloading" && (
                    <div className="mt-2">
                      <Progress value={item.progress} className="h-1" />
                    </div>
                  )}

                  {/* Error Message */}
                  {item.status === "error" && item.error && (
                    <p className="mt-1 text-xs text-destructive truncate">
                      {item.error}
                    </p>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Status Summary Footer */}
      <div className="flex items-center justify-between px-4 py-2 border-t border-border bg-muted/30">
        <div className="flex items-center gap-4">
          {pendingCount > 0 && (
            <span className="text-xs text-muted-foreground">
              {pendingCount} pending
            </span>
          )}
          {downloadingCount > 0 && (
            <span className="text-xs text-primary">
              {downloadingCount} downloading
            </span>
          )}
          {completeCount > 0 && (
            <span className="text-xs text-green-500">
              {completeCount} complete
            </span>
          )}
          {errorCount > 0 && (
            <span className="text-xs text-destructive">
              {errorCount} failed
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
