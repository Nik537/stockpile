"use client";

import { Clock, Film, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";
import type { Script, ScriptWriterSection } from "@/types/workflow";

interface ScriptTimelineProps {
  script: Script | null;
  onSectionClick?: (section: ScriptWriterSection) => void;
  activeSection?: string | null;
}

export function ScriptTimeline({
  script,
  onSectionClick,
  activeSection,
}: ScriptTimelineProps) {
  if (!script) {
    return (
      <div className="flex h-full items-center justify-center p-4">
        <p className="text-sm text-muted-foreground">
          Generate a script to see the timeline
        </p>
      </div>
    );
  }

  // Parse timestamp to seconds for positioning
  const parseTimestamp = (timestamp: string): number => {
    const [mins, secs] = timestamp.split(":").map(Number);
    return mins * 60 + secs;
  };

  // Get total duration in seconds
  const getTotalDurationSeconds = (): number => {
    const [mins, secs] = script.totalDuration.split(":").map(Number);
    return mins * 60 + secs;
  };

  const totalDuration = getTotalDurationSeconds();

  // Calculate section duration (time until next section or end)
  const getSectionDuration = (index: number): number => {
    const currentStart = parseTimestamp(script.sections[index].timestamp);
    const nextStart =
      index < script.sections.length - 1
        ? parseTimestamp(script.sections[index + 1].timestamp)
        : totalDuration;

    return nextStart - currentStart;
  };

  // Format seconds to display
  const formatSeconds = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  // Generate timeline markers (every 30 seconds)
  const generateMarkers = () => {
    const markers = [];
    for (let i = 0; i <= totalDuration; i += 30) {
      markers.push(i);
    }
    return markers;
  };

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="border-b border-border p-3">
        <div className="flex items-center justify-between">
          <h4 className="flex items-center gap-2 text-sm font-medium text-foreground">
            <Clock className="h-4 w-4 text-primary" />
            Script Timeline
          </h4>
          <span className="text-xs text-muted-foreground">
            Total: {script.totalDuration}
          </span>
        </div>
      </div>

      {/* Timeline View */}
      <div className="flex-1 overflow-y-auto p-4">
        {/* Time markers */}
        <div className="relative mb-2">
          <div className="flex justify-between text-xs text-muted-foreground">
            {generateMarkers().map((seconds) => (
              <span key={seconds}>{formatSeconds(seconds)}</span>
            ))}
          </div>
          <div className="absolute left-0 right-0 top-6 h-0.5 bg-border" />
        </div>

        {/* Timeline visualization */}
        <div className="relative mt-8">
          {/* Main timeline track */}
          <div className="relative h-2 w-full rounded-full bg-muted">
            {script.sections.map((section, index) => {
              const startSeconds = parseTimestamp(section.timestamp);
              const duration = getSectionDuration(index);
              const startPercent = (startSeconds / totalDuration) * 100;
              const widthPercent = (duration / totalDuration) * 100;

              // Color palette for sections
              const colors = [
                "bg-primary",
                "bg-blue-500",
                "bg-green-500",
                "bg-yellow-500",
                "bg-purple-500",
                "bg-pink-500",
                "bg-orange-500",
              ];
              const color = colors[index % colors.length];

              return (
                <div
                  key={section.id}
                  className={cn(
                    "absolute top-0 h-full cursor-pointer rounded-full transition-all hover:scale-y-150",
                    color,
                    activeSection === section.id && "ring-2 ring-white ring-offset-2"
                  )}
                  style={{
                    left: `${startPercent}%`,
                    width: `${widthPercent}%`,
                  }}
                  onClick={() => onSectionClick?.(section)}
                  title={`${section.timestamp} - ${section.content.substring(0, 50)}...`}
                />
              );
            })}
          </div>

          {/* Section markers */}
          <div className="relative mt-4">
            {script.sections.map((section, index) => {
              const startSeconds = parseTimestamp(section.timestamp);
              const startPercent = (startSeconds / totalDuration) * 100;

              return (
                <div
                  key={section.id}
                  className="absolute -translate-x-1/2"
                  style={{ left: `${startPercent}%` }}
                >
                  <div className="h-2 w-0.5 bg-muted-foreground/50" />
                </div>
              );
            })}
          </div>
        </div>

        {/* Section list */}
        <div className="mt-6 space-y-2">
          {script.sections.map((section, index) => {
            const duration = getSectionDuration(index);
            const colors = [
              "border-l-primary",
              "border-l-blue-500",
              "border-l-green-500",
              "border-l-yellow-500",
              "border-l-purple-500",
              "border-l-pink-500",
              "border-l-orange-500",
            ];
            const borderColor = colors[index % colors.length];

            return (
              <button
                key={section.id}
                type="button"
                onClick={() => onSectionClick?.(section)}
                className={cn(
                  "w-full rounded-lg border border-border bg-card p-3 text-left transition-colors hover:bg-muted/50",
                  "border-l-4",
                  borderColor,
                  activeSection === section.id && "ring-2 ring-primary"
                )}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="rounded bg-primary/10 px-1.5 py-0.5 font-mono text-xs text-primary">
                      {section.timestamp}
                    </span>
                    <span className="text-xs text-muted-foreground">
                      {formatSeconds(duration)} duration
                    </span>
                  </div>
                  <ChevronRight className="h-4 w-4 text-muted-foreground" />
                </div>
                <p className="mt-2 line-clamp-2 text-sm text-foreground">
                  {section.content}
                </p>
                {section.brollSuggestions && section.brollSuggestions.length > 0 && (
                  <div className="mt-2 flex items-center gap-1">
                    <Film className="h-3 w-3 text-muted-foreground" />
                    <span className="text-xs text-muted-foreground">
                      {section.brollSuggestions.length} B-roll suggestions
                    </span>
                  </div>
                )}
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
