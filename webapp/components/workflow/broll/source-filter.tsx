"use client";

import { useState } from "react";
import { Check, ChevronDown, Film, ImageIcon, Youtube, Camera } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import type { MediaSource, MediaType, BRollSearchFilters } from "@/types/broll";

interface SourceFilterProps {
  filters: BRollSearchFilters;
  onFiltersChange: (filters: BRollSearchFilters) => void;
  className?: string;
}

const SOURCE_CONFIG: Record<MediaSource, { label: string; icon: React.ElementType; color: string }> = {
  pexels: { label: "Pexels", icon: Camera, color: "text-green-500" },
  pixabay: { label: "Pixabay", icon: ImageIcon, color: "text-blue-500" },
  unsplash: { label: "Unsplash", icon: Camera, color: "text-amber-500" },
  youtube: { label: "YouTube", icon: Youtube, color: "text-red-500" },
};

const MEDIA_TYPES: { value: MediaType | 'all'; label: string; icon: React.ElementType }[] = [
  { value: "all", label: "All Media", icon: Film },
  { value: "video", label: "Videos", icon: Film },
  { value: "photo", label: "Photos", icon: ImageIcon },
];

const ORIENTATIONS: { value: 'landscape' | 'portrait' | 'square' | 'all'; label: string }[] = [
  { value: "all", label: "Any Orientation" },
  { value: "landscape", label: "Landscape" },
  { value: "portrait", label: "Portrait" },
  { value: "square", label: "Square" },
];

const DURATION_OPTIONS: { value: string; min?: number; max?: number; label: string }[] = [
  { value: "any", label: "Any Duration" },
  { value: "short", min: 0, max: 10, label: "0-10s" },
  { value: "medium", min: 10, max: 30, label: "10-30s" },
  { value: "long", min: 30, max: 60, label: "30-60s" },
  { value: "extended", min: 60, label: "60s+" },
];

export function SourceFilter({ filters, onFiltersChange, className }: SourceFilterProps) {
  const [showDurationDropdown, setShowDurationDropdown] = useState(false);
  const [showOrientationDropdown, setShowOrientationDropdown] = useState(false);

  const toggleSource = (source: MediaSource) => {
    const newSources = filters.sources.includes(source)
      ? filters.sources.filter((s) => s !== source)
      : [...filters.sources, source];

    // Don't allow empty sources
    if (newSources.length === 0) return;

    onFiltersChange({ ...filters, sources: newSources });
  };

  const setMediaType = (type: MediaType | 'all') => {
    onFiltersChange({ ...filters, type });
  };

  const setOrientation = (orientation: 'landscape' | 'portrait' | 'square' | 'all') => {
    onFiltersChange({ ...filters, orientation });
    setShowOrientationDropdown(false);
  };

  const setDuration = (min?: number, max?: number) => {
    onFiltersChange({ ...filters, minDuration: min, maxDuration: max });
    setShowDurationDropdown(false);
  };

  const getCurrentDurationLabel = () => {
    const option = DURATION_OPTIONS.find(
      (opt) => opt.min === filters.minDuration && opt.max === filters.maxDuration
    );
    return option?.label || "Any Duration";
  };

  const getCurrentOrientationLabel = () => {
    const option = ORIENTATIONS.find((opt) => opt.value === filters.orientation);
    return option?.label || "Any Orientation";
  };

  return (
    <div className={cn("space-y-4", className)}>
      {/* Source Toggles */}
      <div className="flex flex-wrap gap-2">
        <span className="flex items-center text-sm font-medium text-muted-foreground mr-2">
          Sources:
        </span>
        {(Object.keys(SOURCE_CONFIG) as MediaSource[]).map((source) => {
          const config = SOURCE_CONFIG[source];
          const Icon = config.icon;
          const isActive = filters.sources.includes(source);

          return (
            <button
              key={source}
              type="button"
              onClick={() => toggleSource(source)}
              className={cn(
                "flex items-center gap-1.5 rounded-full border px-3 py-1 text-sm transition-all",
                isActive
                  ? "border-primary bg-primary/10 text-foreground"
                  : "border-border text-muted-foreground hover:border-primary/50 hover:text-foreground"
              )}
            >
              <Icon className={cn("h-3.5 w-3.5", isActive && config.color)} />
              <span>{config.label}</span>
              {isActive && <Check className="h-3 w-3 text-primary" />}
            </button>
          );
        })}
      </div>

      {/* Media Type & Filters Row */}
      <div className="flex flex-wrap items-center gap-3">
        {/* Media Type Pills */}
        <div className="flex gap-1 rounded-lg border border-border p-1">
          {MEDIA_TYPES.map(({ value, label, icon: Icon }) => (
            <button
              key={value}
              type="button"
              onClick={() => setMediaType(value)}
              className={cn(
                "flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm transition-all",
                filters.type === value
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:bg-muted hover:text-foreground"
              )}
            >
              <Icon className="h-3.5 w-3.5" />
              <span>{label}</span>
            </button>
          ))}
        </div>

        {/* Orientation Dropdown */}
        <div className="relative">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowOrientationDropdown(!showOrientationDropdown)}
            className="min-w-[140px] justify-between"
          >
            {getCurrentOrientationLabel()}
            <ChevronDown className={cn(
              "h-4 w-4 transition-transform",
              showOrientationDropdown && "rotate-180"
            )} />
          </Button>
          {showOrientationDropdown && (
            <div className="absolute top-full left-0 z-50 mt-1 min-w-[140px] rounded-lg border border-border bg-card shadow-lg">
              {ORIENTATIONS.map((option) => (
                <button
                  key={option.value}
                  type="button"
                  onClick={() => setOrientation(option.value)}
                  className={cn(
                    "flex w-full items-center justify-between px-3 py-2 text-sm transition-colors",
                    filters.orientation === option.value
                      ? "bg-primary/10 text-foreground"
                      : "text-muted-foreground hover:bg-muted hover:text-foreground"
                  )}
                >
                  {option.label}
                  {filters.orientation === option.value && (
                    <Check className="h-4 w-4 text-primary" />
                  )}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Duration Dropdown (only show when videos are included) */}
        {(filters.type === 'video' || filters.type === 'all') && (
          <div className="relative">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowDurationDropdown(!showDurationDropdown)}
              className="min-w-[120px] justify-between"
            >
              {getCurrentDurationLabel()}
              <ChevronDown className={cn(
                "h-4 w-4 transition-transform",
                showDurationDropdown && "rotate-180"
              )} />
            </Button>
            {showDurationDropdown && (
              <div className="absolute top-full left-0 z-50 mt-1 min-w-[120px] rounded-lg border border-border bg-card shadow-lg">
                {DURATION_OPTIONS.map((option) => (
                  <button
                    key={option.value}
                    type="button"
                    onClick={() => setDuration(option.min, option.max)}
                    className={cn(
                      "flex w-full items-center justify-between px-3 py-2 text-sm transition-colors",
                      filters.minDuration === option.min && filters.maxDuration === option.max
                        ? "bg-primary/10 text-foreground"
                        : "text-muted-foreground hover:bg-muted hover:text-foreground"
                    )}
                  >
                    {option.label}
                    {filters.minDuration === option.min && filters.maxDuration === option.max && (
                      <Check className="h-4 w-4 text-primary" />
                    )}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Active Filter Count */}
        {(filters.sources.length < 4 || filters.type !== 'all' || filters.orientation !== 'all' || filters.minDuration !== undefined) && (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => onFiltersChange({
              type: 'all',
              sources: ['pexels', 'pixabay', 'unsplash', 'youtube'],
              orientation: 'all',
              minDuration: undefined,
              maxDuration: undefined,
            })}
            className="text-muted-foreground hover:text-foreground"
          >
            Clear Filters
          </Button>
        )}
      </div>
    </div>
  );
}
