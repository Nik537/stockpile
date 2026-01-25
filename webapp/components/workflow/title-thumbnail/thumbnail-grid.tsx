"use client";

import { useState, useCallback } from "react";
import { Sparkles, RefreshCw, Image as ImageIcon, Download } from "lucide-react";
import { cn } from "@/lib/utils";
import type {
  GeneratedThumbnail,
  ThumbnailConfig,
  ThumbnailGenerationResponse,
} from "@/types/workflow";
import { ThumbnailPreview } from "./thumbnail-preview";
import { StyleSelector } from "./style-selector";

interface ThumbnailGridProps {
  title: string;
  topic: string;
  selectedThumbnail: GeneratedThumbnail | null;
  onThumbnailSelect: (thumbnail: GeneratedThumbnail) => void;
}

export function ThumbnailGrid({
  title,
  topic,
  selectedThumbnail,
  onThumbnailSelect,
}: ThumbnailGridProps) {
  const [thumbnails, setThumbnails] = useState<GeneratedThumbnail[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [regeneratingId, setRegeneratingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Style configuration
  const [style, setStyle] = useState<ThumbnailConfig["style"]>("bold");
  const [colorScheme, setColorScheme] = useState<ThumbnailConfig["colorScheme"]>("dark");
  const [overlayText, setOverlayText] = useState("");

  const generateThumbnails = useCallback(
    async (count: number = 6) => {
      if (!title.trim()) {
        setError("Please select a title first");
        return;
      }

      if (!topic.trim()) {
        setError("Please enter a topic first");
        return;
      }

      setIsGenerating(true);
      setError(null);

      try {
        const response = await fetch("/api/title-thumbnail/thumbnail", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            title: title.trim(),
            topic: topic.trim(),
            config: {
              style,
              colorScheme,
              text: overlayText || undefined,
            },
            count,
          }),
        });

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error || "Failed to generate thumbnails");
        }

        const data: ThumbnailGenerationResponse = await response.json();
        setThumbnails(data.thumbnails);
      } catch (err) {
        setError(err instanceof Error ? err.message : "An error occurred");
      } finally {
        setIsGenerating(false);
      }
    },
    [title, topic, style, colorScheme, overlayText]
  );

  const regenerateSingleThumbnail = useCallback(
    async (thumbnailId: string) => {
      setRegeneratingId(thumbnailId);
      setError(null);

      try {
        const response = await fetch("/api/title-thumbnail/thumbnail", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            title: title.trim(),
            topic: topic.trim(),
            config: {
              style,
              colorScheme,
              text: overlayText || undefined,
            },
            count: 1,
          }),
        });

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error || "Failed to regenerate thumbnail");
        }

        const data: ThumbnailGenerationResponse = await response.json();
        const newThumbnail = data.thumbnails[0];

        // Replace the specific thumbnail in the array
        setThumbnails((prev) =>
          prev.map((t) => (t.id === thumbnailId ? newThumbnail : t))
        );
      } catch (err) {
        setError(err instanceof Error ? err.message : "An error occurred");
      } finally {
        setRegeneratingId(null);
      }
    },
    [title, topic, style, colorScheme, overlayText]
  );

  const downloadSelectedThumbnail = async () => {
    if (!selectedThumbnail) return;

    try {
      const response = await fetch(selectedThumbnail.url);
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `thumbnail-${selectedThumbnail.id}.png`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error("Download failed:", error);
    }
  };

  const canGenerate = title.trim() && topic.trim();

  return (
    <div className="space-y-6">
      {/* Style Configuration */}
      <div className="rounded-xl border border-border bg-card p-6">
        <h3 className="mb-4 flex items-center gap-2 text-lg font-semibold text-foreground">
          <ImageIcon className="h-5 w-5 text-secondary" />
          Thumbnail Style
        </h3>

        <StyleSelector
          selectedStyle={style}
          selectedColorScheme={colorScheme}
          onStyleChange={setStyle}
          onColorSchemeChange={setColorScheme}
          overlayText={overlayText}
          onOverlayTextChange={setOverlayText}
        />

        <button
          type="button"
          onClick={() => generateThumbnails(6)}
          disabled={isGenerating || !canGenerate}
          className="mt-6 flex w-full items-center justify-center gap-2 rounded-lg bg-secondary px-4 py-3 text-sm font-medium text-secondary-foreground transition-colors hover:bg-secondary/90 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {isGenerating ? (
            <>
              <div className="h-4 w-4 animate-spin rounded-full border-2 border-secondary-foreground border-t-transparent" />
              Generating Thumbnails...
            </>
          ) : (
            <>
              <Sparkles className="h-4 w-4" />
              Generate Thumbnails
            </>
          )}
        </button>

        {!canGenerate && (
          <p className="mt-2 text-center text-sm text-muted-foreground">
            {!title.trim()
              ? "Please select or enter a title first"
              : "Please enter a topic first"}
          </p>
        )}

        {error && (
          <p className="mt-2 text-center text-sm text-destructive">{error}</p>
        )}
      </div>

      {/* Generated Thumbnails Grid */}
      {thumbnails.length > 0 && (
        <div className="rounded-xl border border-border bg-card p-6">
          <div className="mb-4 flex items-center justify-between">
            <h3 className="flex items-center gap-2 text-lg font-semibold text-foreground">
              Generated Thumbnails
            </h3>
            <div className="flex items-center gap-2">
              {selectedThumbnail && (
                <button
                  type="button"
                  onClick={downloadSelectedThumbnail}
                  className="flex items-center gap-2 rounded-lg bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90"
                >
                  <Download className="h-4 w-4" />
                  Download Selected
                </button>
              )}
              <button
                type="button"
                onClick={() => generateThumbnails(6)}
                disabled={isGenerating}
                className="flex items-center gap-2 rounded-lg border border-border px-3 py-1.5 text-sm font-medium text-muted-foreground transition-colors hover:bg-muted hover:text-foreground disabled:opacity-50"
              >
                <RefreshCw
                  className={cn("h-4 w-4", isGenerating && "animate-spin")}
                />
                Regenerate All
              </button>
            </div>
          </div>

          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {thumbnails.map((thumbnail) => (
              <ThumbnailPreview
                key={thumbnail.id}
                thumbnail={thumbnail}
                isSelected={selectedThumbnail?.id === thumbnail.id}
                onSelect={() => onThumbnailSelect(thumbnail)}
                onRegenerate={() => regenerateSingleThumbnail(thumbnail.id)}
                isRegenerating={regeneratingId === thumbnail.id}
              />
            ))}
          </div>

          {selectedThumbnail && (
            <div className="mt-4 rounded-lg bg-primary/5 p-4">
              <p className="text-sm text-foreground">
                <span className="font-medium">Selected:</span> Thumbnail with{" "}
                <span className="capitalize">{selectedThumbnail.config.style}</span> style,{" "}
                <span className="capitalize">{selectedThumbnail.config.colorScheme}</span>{" "}
                color scheme
              </p>
            </div>
          )}
        </div>
      )}

      {/* Empty State */}
      {thumbnails.length === 0 && !isGenerating && (
        <div className="rounded-xl border border-dashed border-border bg-card/50 p-12">
          <div className="flex flex-col items-center justify-center text-center">
            <div className="mb-4 rounded-full bg-muted p-4">
              <ImageIcon className="h-8 w-8 text-muted-foreground" />
            </div>
            <h4 className="mb-2 text-lg font-medium text-foreground">
              No Thumbnails Yet
            </h4>
            <p className="mb-4 max-w-sm text-sm text-muted-foreground">
              Configure your style preferences above and click "Generate
              Thumbnails" to create thumbnail variations for your video.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
