"use client";

import { useState, useEffect } from "react";
import {
  Download,
  Settings,
  Monitor,
  Smartphone,
  Square,
  Play,
  CheckCircle,
  Loader2,
  ArrowLeft,
  FileText,
  Image,
  Film,
  Volume2,
  Layers,
  AlertCircle,
} from "lucide-react";
import Link from "next/link";
import { Header } from "@/components/layout/header";
import { EnhancedStepIndicator } from "@/components/workflow/enhanced-step-indicator";
import { WorkflowNav } from "@/components/workflow/workflow-nav";
import { Button } from "@/components/ui/button";
import {
  useWorkflowStore,
  useScriptData,
  useTitleThumbnailData,
  useBRollData,
  useAvatarTtsData,
  useEditorData,
  useExportData,
  useWorkflowProject,
} from "@/lib/workflow-store";
import { cn } from "@/lib/utils";

type ExportStatus = "idle" | "preparing" | "rendering" | "complete" | "error";

const resolutionOptions = [
  { id: "4k", label: "4K (2160p)", resolution: "3840x2160" },
  { id: "1080p", label: "Full HD (1080p)", resolution: "1920x1080" },
  { id: "720p", label: "HD (720p)", resolution: "1280x720" },
  { id: "480p", label: "SD (480p)", resolution: "854x480" },
] as const;

const aspectRatios = [
  { id: "16:9", label: "Landscape", icon: Monitor },
  { id: "9:16", label: "Portrait", icon: Smartphone },
  { id: "1:1", label: "Square", icon: Square },
] as const;

export default function ExportPage() {
  // Workflow data from store
  const project = useWorkflowProject();
  const scriptData = useScriptData();
  const titleThumbnailData = useTitleThumbnailData();
  const brollData = useBRollData();
  const avatarTtsData = useAvatarTtsData();
  const editorData = useEditorData();
  const exportData = useExportData();

  // Store actions
  const setExportConfig = useWorkflowStore((state) => state.setExportConfig);
  const setExportStatus = useWorkflowStore((state) => state.setExportStatus);
  const setExportProgress = useWorkflowStore((state) => state.setExportProgress);
  const markStepComplete = useWorkflowStore((state) => state.markStepComplete);

  // Local state for compatibility with original component
  const [resolution, setResolution] = useState<typeof resolutionOptions[number]["id"]>(
    exportData.resolution
  );
  const [aspectRatio, setAspectRatio] = useState<typeof aspectRatios[number]["id"]>(
    exportData.aspectRatio
  );
  const [format, setFormat] = useState(exportData.format);
  const [quality, setQuality] = useState(exportData.quality);
  const [exportStatus, setLocalExportStatus] = useState<ExportStatus>(
    exportData.exportStatus
  );
  const [progress, setProgress] = useState(exportData.progress);

  // Sync local state with store
  useEffect(() => {
    setExportConfig({
      resolution,
      aspectRatio,
      format,
      quality,
    });
  }, [resolution, aspectRatio, format, quality, setExportConfig]);

  const handleExport = () => {
    setLocalExportStatus("preparing");
    setExportStatus("preparing");
    setProgress(0);
    setExportProgress(0);

    // Simulate export process
    setTimeout(() => {
      setLocalExportStatus("rendering");
      setExportStatus("rendering");
      const interval = setInterval(() => {
        setProgress((prev) => {
          const newProgress = prev + 5;
          setExportProgress(newProgress);
          if (newProgress >= 100) {
            clearInterval(interval);
            setLocalExportStatus("complete");
            setExportStatus("complete");
            markStepComplete("export");
            return 100;
          }
          return newProgress;
        });
      }, 200);
    }, 1000);
  };

  const handleDownload = () => {
    // Simulate download
    alert("Download would start here!");
  };

  // Calculate workflow summary
  const workflowSummary = {
    hasScript: scriptData.script !== null,
    scriptSections: scriptData.script?.sections.length || 0,
    hasTitle: titleThumbnailData.finalTitle.length > 0,
    title: titleThumbnailData.finalTitle,
    hasThumbnail: titleThumbnailData.selectedThumbnail !== null,
    brollCount: brollData.selectedMedia.length,
    hasVoice: avatarTtsData.selectedVoice !== null,
    voiceName: avatarTtsData.selectedVoice?.name || "None",
    audioCount: avatarTtsData.generatedAudio.length,
    hasTimeline: editorData.timeline !== null,
    clipCount:
      editorData.timeline?.tracks.reduce((acc, t) => acc + t.clips.length, 0) || 0,
  };

  return (
    <div className="flex flex-col min-h-screen">
      <Header
        title="Export Video"
        subtitle="Configure and render your final video"
      />

      <div className="flex-1 p-6">
        {/* Step Indicator */}
        <div className="mb-6">
          <EnhancedStepIndicator />
        </div>

        <div className="grid gap-6 lg:grid-cols-2">
          {/* Left Column: Export Settings */}
          <div className="space-y-4">
            {/* Workflow Summary Card */}
            <div className="rounded-xl border border-border bg-card p-6">
              <h3 className="mb-4 flex items-center gap-2 text-lg font-semibold text-foreground">
                <CheckCircle className="h-5 w-5 text-green-500" />
                Workflow Summary
              </h3>

              <div className="space-y-3">
                {/* Script */}
                <SummaryItem
                  icon={FileText}
                  label="Script"
                  value={
                    workflowSummary.hasScript
                      ? `${workflowSummary.scriptSections} sections`
                      : "Not created"
                  }
                  isComplete={workflowSummary.hasScript}
                />

                {/* Title */}
                <SummaryItem
                  icon={FileText}
                  label="Title"
                  value={workflowSummary.hasTitle ? workflowSummary.title : "Not set"}
                  isComplete={workflowSummary.hasTitle}
                />

                {/* Thumbnail */}
                <SummaryItem
                  icon={Image}
                  label="Thumbnail"
                  value={workflowSummary.hasThumbnail ? "Selected" : "Not selected"}
                  isComplete={workflowSummary.hasThumbnail}
                />

                {/* B-Roll */}
                <SummaryItem
                  icon={Film}
                  label="B-Roll Media"
                  value={`${workflowSummary.brollCount} items selected`}
                  isComplete={workflowSummary.brollCount > 0}
                  optional
                />

                {/* Voice */}
                <SummaryItem
                  icon={Volume2}
                  label="Voice"
                  value={workflowSummary.hasVoice ? workflowSummary.voiceName : "Not selected"}
                  isComplete={workflowSummary.hasVoice}
                />

                {/* Audio */}
                <SummaryItem
                  icon={Volume2}
                  label="Generated Audio"
                  value={`${workflowSummary.audioCount} segments`}
                  isComplete={workflowSummary.audioCount > 0}
                />

                {/* Timeline */}
                <SummaryItem
                  icon={Layers}
                  label="Timeline"
                  value={
                    workflowSummary.hasTimeline
                      ? `${workflowSummary.clipCount} clips`
                      : "Not configured"
                  }
                  isComplete={workflowSummary.hasTimeline}
                />
              </div>
            </div>

            {/* Export Settings */}
            <div className="rounded-xl border border-border bg-card p-6">
              <h3 className="mb-4 flex items-center gap-2 text-lg font-semibold text-foreground">
                <Settings className="h-5 w-5 text-primary" />
                Export Settings
              </h3>

              {/* Resolution */}
              <div className="mb-6">
                <label className="mb-2 block text-sm font-medium text-foreground">
                  Resolution
                </label>
                <div className="grid grid-cols-2 gap-2">
                  {resolutionOptions.map((option) => (
                    <button
                      key={option.id}
                      type="button"
                      onClick={() => setResolution(option.id)}
                      className={cn(
                        "rounded-lg border p-3 text-left transition-colors",
                        resolution === option.id
                          ? "border-primary bg-primary/10"
                          : "border-border hover:border-primary/50 hover:bg-muted"
                      )}
                    >
                      <p className="text-sm font-medium text-foreground">
                        {option.label}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        {option.resolution}
                      </p>
                    </button>
                  ))}
                </div>
              </div>

              {/* Aspect Ratio */}
              <div className="mb-6">
                <label className="mb-2 block text-sm font-medium text-foreground">
                  Aspect Ratio
                </label>
                <div className="flex gap-2">
                  {aspectRatios.map((option) => (
                    <button
                      key={option.id}
                      type="button"
                      onClick={() => setAspectRatio(option.id)}
                      className={cn(
                        "flex flex-1 flex-col items-center gap-2 rounded-lg border p-3 transition-colors",
                        aspectRatio === option.id
                          ? "border-primary bg-primary/10"
                          : "border-border hover:border-primary/50 hover:bg-muted"
                      )}
                    >
                      <option.icon className="h-5 w-5 text-muted-foreground" />
                      <span className="text-xs text-foreground">{option.label}</span>
                    </button>
                  ))}
                </div>
              </div>

              {/* Format */}
              <div className="mb-6">
                <label className="mb-2 block text-sm font-medium text-foreground">
                  Format
                </label>
                <div className="flex gap-2">
                  {["mp4", "webm", "mov"].map((f) => (
                    <button
                      key={f}
                      type="button"
                      onClick={() => setFormat(f as typeof format)}
                      className={cn(
                        "flex-1 rounded-lg border px-4 py-2 text-sm uppercase transition-colors",
                        format === f
                          ? "border-primary bg-primary/10 text-foreground"
                          : "border-border text-muted-foreground hover:border-primary/50"
                      )}
                    >
                      {f}
                    </button>
                  ))}
                </div>
              </div>

              {/* Quality */}
              <div>
                <div className="mb-2 flex items-center justify-between">
                  <label className="text-sm font-medium text-foreground">
                    Quality
                  </label>
                  <span className="text-sm text-muted-foreground">{quality}%</span>
                </div>
                <input
                  type="range"
                  min="50"
                  max="100"
                  value={quality}
                  onChange={(e) => setQuality(parseInt(e.target.value))}
                  className="w-full accent-primary"
                />
                <div className="mt-1 flex justify-between text-xs text-muted-foreground">
                  <span>Smaller file</span>
                  <span>Better quality</span>
                </div>
              </div>
            </div>
          </div>

          {/* Right Column: Preview & Actions */}
          <div className="space-y-4">
            {/* Preview */}
            <div className="rounded-xl border border-border bg-card p-6">
              <h3 className="mb-4 text-lg font-semibold text-foreground">
                Preview
              </h3>

              <div className="aspect-video overflow-hidden rounded-lg border border-border bg-black mb-4">
                <div className="flex h-full items-center justify-center">
                  <div className="text-center">
                    <Play className="h-12 w-12 mx-auto mb-2 text-muted-foreground opacity-50" />
                    <p className="text-sm text-muted-foreground">
                      Click to preview
                    </p>
                  </div>
                </div>
              </div>

              {/* Export Summary */}
              <div className="rounded-lg bg-muted/50 p-4">
                <h4 className="mb-2 text-sm font-medium text-foreground">
                  Export Summary
                </h4>
                <div className="space-y-1 text-sm text-muted-foreground">
                  <p>
                    Resolution:{" "}
                    {resolutionOptions.find((r) => r.id === resolution)?.resolution}
                  </p>
                  <p>Aspect Ratio: {aspectRatio}</p>
                  <p>Format: {format.toUpperCase()}</p>
                  <p>Quality: {quality}%</p>
                  <p>Estimated size: ~250 MB</p>
                </div>
              </div>
            </div>

            {/* Export Progress / Actions */}
            <div className="rounded-xl border border-border bg-card p-6">
              {exportStatus === "idle" && (
                <Button onClick={handleExport} className="w-full gap-2" size="lg">
                  <Download className="h-5 w-5" />
                  Start Export
                </Button>
              )}

              {(exportStatus === "preparing" || exportStatus === "rendering") && (
                <div className="space-y-4">
                  <div className="flex items-center gap-3">
                    <Loader2 className="h-5 w-5 animate-spin text-primary" />
                    <span className="text-sm font-medium text-foreground">
                      {exportStatus === "preparing" ? "Preparing..." : "Rendering..."}
                    </span>
                  </div>
                  <div className="space-y-2">
                    <div className="h-2 overflow-hidden rounded-full bg-muted">
                      <div
                        className="h-full bg-primary transition-all duration-300"
                        style={{ width: `${progress}%` }}
                      />
                    </div>
                    <p className="text-xs text-muted-foreground text-right">
                      {progress}% complete
                    </p>
                  </div>
                </div>
              )}

              {exportStatus === "complete" && (
                <div className="space-y-4">
                  <div className="flex items-center gap-3 text-green-500">
                    <CheckCircle className="h-5 w-5" />
                    <span className="text-sm font-medium">Export Complete!</span>
                  </div>
                  <Button
                    onClick={handleDownload}
                    className="w-full gap-2 bg-green-600 hover:bg-green-700"
                    size="lg"
                  >
                    <Download className="h-5 w-5" />
                    Download Video
                  </Button>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Navigation */}
        <WorkflowNav
          className="mt-6"
          showSave={true}
          customPrevLabel="Back to Editor"
        />
      </div>
    </div>
  );
}

interface SummaryItemProps {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: string;
  isComplete: boolean;
  optional?: boolean;
}

function SummaryItem({
  icon: Icon,
  label,
  value,
  isComplete,
  optional,
}: SummaryItemProps) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-border/50 last:border-0">
      <div className="flex items-center gap-3">
        <Icon
          className={cn(
            "h-4 w-4",
            isComplete ? "text-green-500" : "text-muted-foreground"
          )}
        />
        <span className="text-sm text-foreground">{label}</span>
        {optional && (
          <span className="text-xs text-muted-foreground">(optional)</span>
        )}
      </div>
      <div className="flex items-center gap-2">
        <span className="text-sm text-muted-foreground">{value}</span>
        {isComplete ? (
          <CheckCircle className="h-4 w-4 text-green-500" />
        ) : !optional ? (
          <AlertCircle className="h-4 w-4 text-amber-500" />
        ) : null}
      </div>
    </div>
  );
}
