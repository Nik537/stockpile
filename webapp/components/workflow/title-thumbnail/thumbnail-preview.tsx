"use client";

import { useState } from "react";
import { Check, Download, RefreshCw, Maximize2 } from "lucide-react";
import { cn } from "@/lib/utils";
import type { GeneratedThumbnail } from "@/types/workflow";

interface ThumbnailPreviewProps {
  thumbnail: GeneratedThumbnail;
  isSelected: boolean;
  onSelect: () => void;
  onRegenerate: () => void;
  isRegenerating?: boolean;
}

export function ThumbnailPreview({
  thumbnail,
  isSelected,
  onSelect,
  onRegenerate,
  isRegenerating = false,
}: ThumbnailPreviewProps) {
  const [isHovered, setIsHovered] = useState(false);
  const [showFullPrompt, setShowFullPrompt] = useState(false);
  const [imageError, setImageError] = useState(false);

  const handleDownload = async () => {
    try {
      const response = await fetch(thumbnail.url);
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `thumbnail-${thumbnail.id}.png`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error("Download failed:", error);
    }
  };

  return (
    <div className="space-y-2">
      {/* Thumbnail Image */}
      <div
        className={cn(
          "group relative aspect-video overflow-hidden rounded-lg border-2 transition-all cursor-pointer",
          isSelected
            ? "border-primary ring-2 ring-primary ring-offset-2 ring-offset-background"
            : "border-border hover:border-primary/50"
        )}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        onClick={onSelect}
      >
        {/* Image */}
        {imageError ? (
          <div className="flex h-full items-center justify-center bg-muted">
            <p className="text-sm text-muted-foreground">Failed to load image</p>
          </div>
        ) : (
          <img
            src={thumbnail.url}
            alt={`Thumbnail option`}
            className={cn(
              "h-full w-full object-cover transition-transform duration-300",
              isHovered && !isRegenerating && "scale-105"
            )}
            onError={() => setImageError(true)}
          />
        )}

        {/* Overlay on hover */}
        {isHovered && !isRegenerating && (
          <div className="absolute inset-0 bg-black/50 transition-opacity">
            <div className="absolute bottom-2 left-2 right-2 flex items-center justify-between">
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  onSelect();
                }}
                className={cn(
                  "flex items-center gap-1 rounded-md px-3 py-1.5 text-xs font-medium transition-colors",
                  isSelected
                    ? "bg-primary text-primary-foreground"
                    : "bg-white/90 text-black hover:bg-white"
                )}
              >
                {isSelected ? (
                  <>
                    <Check className="h-3 w-3" />
                    Selected
                  </>
                ) : (
                  "Select"
                )}
              </button>
              <div className="flex items-center gap-1">
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    onRegenerate();
                  }}
                  className="rounded-md bg-white/90 p-1.5 text-black transition-colors hover:bg-white"
                  title="Regenerate"
                >
                  <RefreshCw className="h-4 w-4" />
                </button>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleDownload();
                  }}
                  className="rounded-md bg-white/90 p-1.5 text-black transition-colors hover:bg-white"
                  title="Download"
                >
                  <Download className="h-4 w-4" />
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Selection indicator */}
        {isSelected && (
          <div className="absolute right-2 top-2 rounded-full bg-primary p-1">
            <Check className="h-4 w-4 text-primary-foreground" />
          </div>
        )}

        {/* Loading state */}
        {isRegenerating && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/60">
            <div className="flex flex-col items-center gap-2">
              <div className="h-8 w-8 animate-spin rounded-full border-2 border-primary border-t-transparent" />
              <span className="text-xs text-white">Regenerating...</span>
            </div>
          </div>
        )}
      </div>

      {/* Style badge */}
      <div className="flex items-center justify-between">
        <span className="rounded-full bg-muted px-2 py-0.5 text-xs capitalize text-muted-foreground">
          {thumbnail.config.style} - {thumbnail.config.colorScheme}
        </span>
        <button
          type="button"
          onClick={() => setShowFullPrompt(!showFullPrompt)}
          className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
        >
          <Maximize2 className="h-3 w-3" />
          Prompt
        </button>
      </div>

      {/* Full prompt (expandable) */}
      {showFullPrompt && (
        <div className="rounded-lg bg-muted/50 p-3">
          <p className="text-xs text-muted-foreground">{thumbnail.prompt}</p>
        </div>
      )}
    </div>
  );
}
