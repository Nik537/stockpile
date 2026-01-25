"use client";

import { cn } from "@/lib/utils";
import type { ThumbnailStyle, ColorScheme } from "@/types/workflow";
import {
  Film,
  Minimize2,
  Zap,
  TrendingUp,
  Briefcase,
  Sun,
  Moon,
  Palette,
} from "lucide-react";

const THUMBNAIL_STYLES: {
  value: ThumbnailStyle;
  label: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
}[] = [
  {
    value: "cinematic",
    label: "Cinematic",
    description: "Dramatic lighting, movie-like",
    icon: Film,
  },
  {
    value: "minimal",
    label: "Minimal",
    description: "Clean, simple, modern",
    icon: Minimize2,
  },
  {
    value: "bold",
    label: "Bold",
    description: "Vibrant colors, high impact",
    icon: Zap,
  },
  {
    value: "viral",
    label: "Viral",
    description: "Eye-catching, clickable",
    icon: TrendingUp,
  },
  {
    value: "professional",
    label: "Professional",
    description: "Corporate, trustworthy",
    icon: Briefcase,
  },
];

const COLOR_SCHEMES: {
  value: ColorScheme;
  label: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
  previewColors: string[];
}[] = [
  {
    value: "dark",
    label: "Dark",
    description: "Moody, dramatic",
    icon: Moon,
    previewColors: ["bg-slate-900", "bg-slate-800", "bg-slate-700"],
  },
  {
    value: "light",
    label: "Light",
    description: "Bright, clean",
    icon: Sun,
    previewColors: ["bg-slate-100", "bg-slate-200", "bg-slate-300"],
  },
  {
    value: "vibrant",
    label: "Vibrant",
    description: "Colorful, energetic",
    icon: Palette,
    previewColors: ["bg-pink-500", "bg-purple-500", "bg-blue-500"],
  },
];

interface StyleSelectorProps {
  selectedStyle: ThumbnailStyle;
  selectedColorScheme: ColorScheme;
  onStyleChange: (style: ThumbnailStyle) => void;
  onColorSchemeChange: (scheme: ColorScheme) => void;
  overlayText: string;
  onOverlayTextChange: (text: string) => void;
}

export function StyleSelector({
  selectedStyle,
  selectedColorScheme,
  onStyleChange,
  onColorSchemeChange,
  overlayText,
  onOverlayTextChange,
}: StyleSelectorProps) {
  return (
    <div className="space-y-6">
      {/* Thumbnail Style */}
      <div>
        <h4 className="mb-3 text-sm font-medium text-foreground">
          Thumbnail Style
        </h4>
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-5">
          {THUMBNAIL_STYLES.map((style) => {
            const Icon = style.icon;
            const isSelected = selectedStyle === style.value;

            return (
              <button
                key={style.value}
                type="button"
                onClick={() => onStyleChange(style.value)}
                className={cn(
                  "flex flex-col items-center gap-2 rounded-lg border p-3 transition-all",
                  isSelected
                    ? "border-primary bg-primary/10 ring-1 ring-primary"
                    : "border-border bg-background hover:border-primary/50 hover:bg-muted"
                )}
              >
                <Icon
                  className={cn(
                    "h-5 w-5",
                    isSelected ? "text-primary" : "text-muted-foreground"
                  )}
                />
                <span
                  className={cn(
                    "text-xs font-medium",
                    isSelected ? "text-foreground" : "text-muted-foreground"
                  )}
                >
                  {style.label}
                </span>
              </button>
            );
          })}
        </div>
        <p className="mt-2 text-xs text-muted-foreground">
          {THUMBNAIL_STYLES.find((s) => s.value === selectedStyle)?.description}
        </p>
      </div>

      {/* Color Scheme */}
      <div>
        <h4 className="mb-3 text-sm font-medium text-foreground">
          Color Scheme
        </h4>
        <div className="grid grid-cols-3 gap-2">
          {COLOR_SCHEMES.map((scheme) => {
            const Icon = scheme.icon;
            const isSelected = selectedColorScheme === scheme.value;

            return (
              <button
                key={scheme.value}
                type="button"
                onClick={() => onColorSchemeChange(scheme.value)}
                className={cn(
                  "flex flex-col items-center gap-2 rounded-lg border p-4 transition-all",
                  isSelected
                    ? "border-primary bg-primary/10 ring-1 ring-primary"
                    : "border-border bg-background hover:border-primary/50 hover:bg-muted"
                )}
              >
                <div className="flex items-center gap-1">
                  {scheme.previewColors.map((color, i) => (
                    <div
                      key={i}
                      className={cn("h-4 w-4 rounded-full", color)}
                    />
                  ))}
                </div>
                <div className="flex items-center gap-1">
                  <Icon
                    className={cn(
                      "h-4 w-4",
                      isSelected ? "text-primary" : "text-muted-foreground"
                    )}
                  />
                  <span
                    className={cn(
                      "text-sm font-medium",
                      isSelected ? "text-foreground" : "text-muted-foreground"
                    )}
                  >
                    {scheme.label}
                  </span>
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Overlay Text */}
      <div>
        <h4 className="mb-3 text-sm font-medium text-foreground">
          Overlay Text (Optional)
        </h4>
        <input
          type="text"
          value={overlayText}
          onChange={(e) => onOverlayTextChange(e.target.value)}
          placeholder="Add text to display on thumbnail..."
          maxLength={50}
          className="w-full rounded-lg border border-input bg-background px-4 py-3 text-sm text-foreground placeholder:text-muted-foreground focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
        />
        <p className="mt-1 text-xs text-muted-foreground">
          Short text that will appear on the thumbnail ({overlayText.length}/50)
        </p>
      </div>
    </div>
  );
}
