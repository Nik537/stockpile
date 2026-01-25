"use client";

import { useState, useRef, useEffect } from "react";
import {
  Play,
  Pause,
  Plus,
  Check,
  Download,
  ExternalLink,
  Clock,
  Camera,
  Film,
  Youtube,
  ImageIcon,
  Star,
  GripVertical,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { formatDuration } from "@/lib/utils";
import type { BRollItem, MediaSource } from "@/types/broll";

interface MediaCardProps {
  item: BRollItem;
  isSelected?: boolean;
  onSelect?: (item: BRollItem) => void;
  onDownload?: (item: BRollItem) => void;
  onPreview?: (item: BRollItem) => void;
  isDraggable?: boolean;
  showRelevanceScore?: boolean;
  className?: string;
}

const SOURCE_ICONS: Record<MediaSource, React.ElementType> = {
  pexels: Camera,
  pixabay: ImageIcon,
  unsplash: Camera,
  youtube: Youtube,
};

const SOURCE_COLORS: Record<MediaSource, string> = {
  pexels: "bg-green-500/80",
  pixabay: "bg-blue-500/80",
  unsplash: "bg-amber-500/80",
  youtube: "bg-red-500/80",
};

export function MediaCard({
  item,
  isSelected = false,
  onSelect,
  onDownload,
  onPreview,
  isDraggable = false,
  showRelevanceScore = false,
  className,
}: MediaCardProps) {
  const [isHovering, setIsHovering] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [imageLoaded, setImageLoaded] = useState(false);
  const [imageError, setImageError] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);

  const SourceIcon = SOURCE_ICONS[item.source];
  const isVideo = item.type === "video";

  // Handle video preview on hover
  useEffect(() => {
    if (isVideo && isHovering && videoRef.current && item.previewUrl) {
      videoRef.current.play().catch(() => {
        // Autoplay may be blocked
      });
      setIsPlaying(true);
    } else if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.currentTime = 0;
      setIsPlaying(false);
    }
  }, [isHovering, isVideo, item.previewUrl]);

  const handleClick = () => {
    if (onSelect) {
      onSelect(item);
    }
  };

  const handleDownloadClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (onDownload) {
      onDownload(item);
    }
  };

  const handlePreviewClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (onPreview) {
      onPreview(item);
    }
  };

  const handleDragStart = (e: React.DragEvent) => {
    if (!isDraggable) return;
    e.dataTransfer.setData("application/json", JSON.stringify(item));
    e.dataTransfer.effectAllowed = "copy";
  };

  return (
    <div
      className={cn(
        "group relative overflow-hidden rounded-lg border transition-all cursor-pointer",
        isSelected
          ? "border-primary ring-2 ring-primary"
          : "border-border hover:border-primary/50",
        isDraggable && "cursor-grab active:cursor-grabbing",
        className
      )}
      onClick={handleClick}
      onMouseEnter={() => setIsHovering(true)}
      onMouseLeave={() => setIsHovering(false)}
      draggable={isDraggable}
      onDragStart={handleDragStart}
    >
      {/* Aspect Ratio Container */}
      <div className="relative aspect-video bg-muted">
        {/* Drag Handle */}
        {isDraggable && (
          <div className="absolute left-2 top-2 z-20 opacity-0 group-hover:opacity-100 transition-opacity">
            <div className="flex h-6 w-6 items-center justify-center rounded bg-black/60 text-white">
              <GripVertical className="h-4 w-4" />
            </div>
          </div>
        )}

        {/* Thumbnail/Preview */}
        {!imageError ? (
          <>
            {/* Image Thumbnail */}
            <img
              src={item.thumbnailUrl}
              alt={item.title}
              className={cn(
                "absolute inset-0 h-full w-full object-cover transition-opacity",
                imageLoaded ? "opacity-100" : "opacity-0",
                isVideo && isHovering && item.previewUrl ? "opacity-0" : "opacity-100"
              )}
              onLoad={() => setImageLoaded(true)}
              onError={() => setImageError(true)}
            />

            {/* Video Preview (on hover) */}
            {isVideo && item.previewUrl && (
              <video
                ref={videoRef}
                src={item.previewUrl}
                className={cn(
                  "absolute inset-0 h-full w-full object-cover transition-opacity",
                  isHovering ? "opacity-100" : "opacity-0"
                )}
                muted
                loop
                playsInline
              />
            )}
          </>
        ) : (
          /* Placeholder */
          <div className="flex h-full items-center justify-center">
            {isVideo ? (
              <Film className="h-8 w-8 text-muted-foreground" />
            ) : (
              <ImageIcon className="h-8 w-8 text-muted-foreground" />
            )}
          </div>
        )}

        {/* Loading Skeleton */}
        {!imageLoaded && !imageError && (
          <div className="absolute inset-0 animate-pulse bg-muted" />
        )}

        {/* Video Play Indicator */}
        {isVideo && !isHovering && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="flex h-10 w-10 items-center justify-center rounded-full bg-black/50 text-white backdrop-blur-sm">
              <Play className="h-5 w-5 ml-0.5" />
            </div>
          </div>
        )}

        {/* Selection Indicator */}
        {isSelected && (
          <div className="absolute right-2 top-2 z-20 flex h-6 w-6 items-center justify-center rounded-full bg-primary text-primary-foreground">
            <Check className="h-4 w-4" />
          </div>
        )}

        {/* Source Badge */}
        <div className={cn(
          "absolute left-2 bottom-2 z-10 flex items-center gap-1 rounded px-1.5 py-0.5 text-xs font-medium text-white",
          SOURCE_COLORS[item.source]
        )}>
          <SourceIcon className="h-3 w-3" />
          <span className="capitalize">{item.source}</span>
        </div>

        {/* Duration Badge (for videos) */}
        {isVideo && item.duration && (
          <div className="absolute right-2 bottom-2 z-10 flex items-center gap-1 rounded bg-black/70 px-1.5 py-0.5 text-xs font-medium text-white">
            <Clock className="h-3 w-3" />
            {formatDuration(item.duration)}
          </div>
        )}

        {/* Relevance Score */}
        {showRelevanceScore && item.relevanceScore !== undefined && (
          <div className="absolute right-2 top-2 z-10 flex items-center gap-1 rounded bg-black/70 px-1.5 py-0.5 text-xs font-medium text-white">
            <Star className="h-3 w-3 text-yellow-400" />
            {(item.relevanceScore * 10).toFixed(1)}
          </div>
        )}

        {/* Hover Overlay with Actions */}
        <div className={cn(
          "absolute inset-0 flex items-end bg-gradient-to-t from-black/80 via-black/40 to-transparent p-3 transition-opacity",
          isHovering ? "opacity-100" : "opacity-0"
        )}>
          <div className="flex w-full items-center justify-between gap-2">
            {/* Title */}
            <span className="flex-1 truncate text-sm font-medium text-white">
              {item.title}
            </span>

            {/* Action Buttons */}
            <div className="flex items-center gap-1">
              {onPreview && (
                <button
                  type="button"
                  onClick={handlePreviewClick}
                  className="flex h-7 w-7 items-center justify-center rounded bg-white/20 text-white backdrop-blur-sm transition-colors hover:bg-white/30"
                  title="Preview"
                >
                  <ExternalLink className="h-3.5 w-3.5" />
                </button>
              )}
              {onDownload && (
                <button
                  type="button"
                  onClick={handleDownloadClick}
                  className="flex h-7 w-7 items-center justify-center rounded bg-white/20 text-white backdrop-blur-sm transition-colors hover:bg-white/30"
                  title="Download"
                >
                  <Download className="h-3.5 w-3.5" />
                </button>
              )}
              {onSelect && (
                <button
                  type="button"
                  onClick={(e) => { e.stopPropagation(); handleClick(); }}
                  className={cn(
                    "flex h-7 w-7 items-center justify-center rounded backdrop-blur-sm transition-colors",
                    isSelected
                      ? "bg-primary text-primary-foreground"
                      : "bg-white/20 text-white hover:bg-white/30"
                  )}
                  title={isSelected ? "Remove from selection" : "Add to selection"}
                >
                  {isSelected ? (
                    <Check className="h-3.5 w-3.5" />
                  ) : (
                    <Plus className="h-3.5 w-3.5" />
                  )}
                </button>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Info Section (optional, below card) */}
      {(item.photographer || item.videographer) && (
        <div className="border-t border-border bg-card/50 px-2 py-1.5">
          <p className="truncate text-xs text-muted-foreground">
            by {item.photographer || item.videographer}
          </p>
        </div>
      )}
    </div>
  );
}
