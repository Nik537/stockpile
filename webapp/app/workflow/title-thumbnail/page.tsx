"use client";

import { useState, useEffect } from "react";
import { CheckCircle2 } from "lucide-react";
import { Header } from "@/components/layout/header";
import { EnhancedStepIndicator } from "@/components/workflow/enhanced-step-indicator";
import { WorkflowNav } from "@/components/workflow/workflow-nav";
import { TitleGenerator, ThumbnailGrid } from "@/components/workflow/title-thumbnail";
import { useWorkflowStore, useTitleThumbnailData } from "@/lib/workflow-store";
import type { GeneratedThumbnail } from "@/types/workflow";

export default function TitleThumbnailPage() {
  // Workflow store
  const titleThumbnailData = useTitleThumbnailData();
  const setFinalTitle = useWorkflowStore((state) => state.setFinalTitle);
  const setSelectedThumbnail = useWorkflowStore((state) => state.setSelectedThumbnail);
  const markStepComplete = useWorkflowStore((state) => state.markStepComplete);

  // Local state synced with store
  const [topic, setTopic] = useState("");
  const [selectedTitle, setSelectedTitle] = useState(titleThumbnailData.finalTitle);
  const [selectedThumbnail, setLocalSelectedThumbnail] = useState<GeneratedThumbnail | null>(
    titleThumbnailData.selectedThumbnail
  );

  // Sync local state with store on mount
  useEffect(() => {
    if (titleThumbnailData.finalTitle) {
      setSelectedTitle(titleThumbnailData.finalTitle);
    }
    if (titleThumbnailData.selectedThumbnail) {
      setLocalSelectedThumbnail(titleThumbnailData.selectedThumbnail);
    }
  }, []);

  // Sync to workflow store when title changes
  useEffect(() => {
    if (selectedTitle) {
      setFinalTitle(selectedTitle);
    }
  }, [selectedTitle, setFinalTitle]);

  // Sync to workflow store when thumbnail changes
  useEffect(() => {
    if (selectedThumbnail) {
      setSelectedThumbnail(selectedThumbnail);
    }
  }, [selectedThumbnail, setSelectedThumbnail]);

  // Mark step complete when both title and thumbnail are selected
  useEffect(() => {
    if (selectedTitle.trim() && selectedThumbnail) {
      markStepComplete("title-thumbnail");
    }
  }, [selectedTitle, selectedThumbnail, markStepComplete]);

  // Check if step is complete
  const isStepComplete = selectedTitle.trim() && selectedThumbnail;

  return (
    <div className="flex flex-col min-h-screen">
      <Header
        title="Title & Thumbnail"
        subtitle="Create compelling titles and eye-catching thumbnails for your video"
      />

      <div className="flex-1 p-6">
        {/* Step Indicator */}
        <div className="mb-6">
          <EnhancedStepIndicator />
        </div>

        {/* Progress Summary */}
        {(selectedTitle || selectedThumbnail) && (
          <div className="mb-6 rounded-xl border border-border bg-card/50 p-4">
            <h4 className="mb-3 text-sm font-medium text-muted-foreground">Current Selection</h4>
            <div className="space-y-2">
              {selectedTitle && (
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4 text-green-500" />
                  <span className="text-sm text-foreground">
                    <span className="font-medium">Title:</span> {selectedTitle}
                  </span>
                </div>
              )}
              {selectedThumbnail && (
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4 text-green-500" />
                  <span className="text-sm text-foreground">
                    <span className="font-medium">Thumbnail:</span>{" "}
                    <span className="capitalize">{selectedThumbnail.config.style}</span> style,{" "}
                    <span className="capitalize">{selectedThumbnail.config.colorScheme}</span> scheme
                  </span>
                </div>
              )}
            </div>
          </div>
        )}

        <div className="grid gap-6 lg:grid-cols-2">
          {/* Left Column: Title Generation */}
          <div>
            <TitleGenerator
              topic={topic}
              onTopicChange={setTopic}
              selectedTitle={selectedTitle}
              onTitleSelect={setSelectedTitle}
            />
          </div>

          {/* Right Column: Thumbnail Generation */}
          <div>
            <ThumbnailGrid
              title={selectedTitle}
              topic={topic}
              selectedThumbnail={selectedThumbnail}
              onThumbnailSelect={setLocalSelectedThumbnail}
            />
          </div>
        </div>

        {/* Workflow Navigation */}
        <WorkflowNav className="mt-8" />
      </div>
    </div>
  );
}
