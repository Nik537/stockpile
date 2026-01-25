"use client";

import { useState } from "react";
import {
  Clock,
  Film,
  Edit3,
  Trash2,
  Plus,
  GripVertical,
  ChevronDown,
  ChevronRight,
  Save,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";
import type { Script, ScriptWriterSection } from "@/types/workflow";

interface ScriptEditorProps {
  script: Script | null;
  onScriptChange: (script: Script) => void;
}

export function ScriptEditor({ script, onScriptChange }: ScriptEditorProps) {
  const [expandedSections, setExpandedSections] = useState<Set<string>>(
    new Set()
  );
  const [editingSection, setEditingSection] = useState<string | null>(null);
  const [editedContent, setEditedContent] = useState("");

  if (!script) {
    return (
      <div className="flex h-full flex-col items-center justify-center p-6 text-center">
        <Film className="mb-4 h-12 w-12 text-muted-foreground/50" />
        <h4 className="mb-2 font-medium text-foreground">No Script Yet</h4>
        <p className="max-w-sm text-sm text-muted-foreground">
          Start a conversation with the AI assistant to generate your video
          script. You can then edit and refine it here.
        </p>
      </div>
    );
  }

  const toggleSection = (sectionId: string) => {
    const newExpanded = new Set(expandedSections);
    if (newExpanded.has(sectionId)) {
      newExpanded.delete(sectionId);
    } else {
      newExpanded.add(sectionId);
    }
    setExpandedSections(newExpanded);
  };

  const startEditing = (section: ScriptWriterSection) => {
    setEditingSection(section.id);
    setEditedContent(section.content);
  };

  const saveEditing = (sectionId: string) => {
    if (!script) return;

    const updatedSections = script.sections.map((section) =>
      section.id === sectionId
        ? { ...section, content: editedContent }
        : section
    );

    onScriptChange({
      ...script,
      sections: updatedSections,
    });

    setEditingSection(null);
    setEditedContent("");
  };

  const cancelEditing = () => {
    setEditingSection(null);
    setEditedContent("");
  };

  const deleteSection = (sectionId: string) => {
    if (!script) return;

    const updatedSections = script.sections.filter(
      (section) => section.id !== sectionId
    );

    onScriptChange({
      ...script,
      sections: updatedSections,
    });
  };

  const addSection = (afterIndex: number) => {
    if (!script) return;

    const newSection: ScriptWriterSection = {
      id: Math.random().toString(36).substring(2, 15),
      timestamp: calculateNewTimestamp(afterIndex),
      content: "[New section - click edit to add content]",
      brollSuggestions: [],
    };

    const updatedSections = [
      ...script.sections.slice(0, afterIndex + 1),
      newSection,
      ...script.sections.slice(afterIndex + 1),
    ];

    onScriptChange({
      ...script,
      sections: updatedSections,
    });
  };

  const calculateNewTimestamp = (afterIndex: number): string => {
    if (!script || script.sections.length === 0) return "0:00";

    const currentSection = script.sections[afterIndex];
    const nextSection = script.sections[afterIndex + 1];

    if (!nextSection) {
      // Add 30 seconds to the last section
      const [mins, secs] = currentSection.timestamp.split(":").map(Number);
      const totalSecs = mins * 60 + secs + 30;
      return `${Math.floor(totalSecs / 60)}:${(totalSecs % 60)
        .toString()
        .padStart(2, "0")}`;
    }

    // Calculate midpoint between current and next
    const [curMins, curSecs] = currentSection.timestamp.split(":").map(Number);
    const [nextMins, nextSecs] = nextSection.timestamp.split(":").map(Number);

    const curTotal = curMins * 60 + curSecs;
    const nextTotal = nextMins * 60 + nextSecs;
    const midTotal = Math.floor((curTotal + nextTotal) / 2);

    return `${Math.floor(midTotal / 60)}:${(midTotal % 60)
      .toString()
      .padStart(2, "0")}`;
  };

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="border-b border-border p-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="font-semibold text-foreground">{script.title}</h3>
            <div className="mt-1 flex items-center gap-4 text-sm text-muted-foreground">
              <span className="flex items-center gap-1">
                <Clock className="h-4 w-4" />
                {script.totalDuration}
              </span>
              <span>{script.sections.length} sections</span>
            </div>
          </div>
        </div>
      </div>

      {/* Sections List */}
      <div className="flex-1 overflow-y-auto p-4">
        <div className="space-y-3">
          {script.sections.map((section, index) => (
            <div key={section.id}>
              <div
                className={cn(
                  "rounded-lg border border-border bg-card transition-colors",
                  editingSection === section.id && "ring-2 ring-primary"
                )}
              >
                {/* Section Header */}
                <div
                  className="flex cursor-pointer items-center gap-2 p-3"
                  onClick={() => toggleSection(section.id)}
                >
                  <GripVertical className="h-4 w-4 cursor-grab text-muted-foreground" />

                  <button
                    type="button"
                    className="shrink-0"
                    onClick={(e) => {
                      e.stopPropagation();
                      toggleSection(section.id);
                    }}
                  >
                    {expandedSections.has(section.id) ? (
                      <ChevronDown className="h-4 w-4 text-muted-foreground" />
                    ) : (
                      <ChevronRight className="h-4 w-4 text-muted-foreground" />
                    )}
                  </button>

                  <span className="shrink-0 rounded bg-primary/10 px-2 py-0.5 font-mono text-xs text-primary">
                    {section.timestamp}
                  </span>

                  <span className="flex-1 truncate text-sm text-foreground">
                    {section.content.substring(0, 60)}
                    {section.content.length > 60 && "..."}
                  </span>

                  <div className="flex shrink-0 items-center gap-1">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7"
                      onClick={(e) => {
                        e.stopPropagation();
                        startEditing(section);
                      }}
                    >
                      <Edit3 className="h-3.5 w-3.5" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7 text-destructive hover:text-destructive"
                      onClick={(e) => {
                        e.stopPropagation();
                        deleteSection(section.id);
                      }}
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </Button>
                  </div>
                </div>

                {/* Expanded Content */}
                {expandedSections.has(section.id) && (
                  <div className="border-t border-border p-3">
                    {editingSection === section.id ? (
                      <div className="space-y-3">
                        <Textarea
                          value={editedContent}
                          onChange={(e) => setEditedContent(e.target.value)}
                          className="min-h-[120px]"
                          autoFocus
                        />
                        <div className="flex justify-end gap-2">
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={cancelEditing}
                          >
                            Cancel
                          </Button>
                          <Button
                            size="sm"
                            onClick={() => saveEditing(section.id)}
                          >
                            <Save className="mr-1 h-3.5 w-3.5" />
                            Save
                          </Button>
                        </div>
                      </div>
                    ) : (
                      <>
                        <p className="whitespace-pre-wrap text-sm text-foreground">
                          {section.content}
                        </p>

                        {/* B-Roll Suggestions */}
                        {section.brollSuggestions &&
                          section.brollSuggestions.length > 0 && (
                            <div className="mt-3 border-t border-border pt-3">
                              <h5 className="mb-2 flex items-center gap-1 text-xs font-medium text-muted-foreground">
                                <Film className="h-3.5 w-3.5" />
                                B-Roll Suggestions
                              </h5>
                              <div className="flex flex-wrap gap-1">
                                {section.brollSuggestions.map(
                                  (suggestion, idx) => (
                                    <span
                                      key={idx}
                                      className="rounded-full bg-muted px-2 py-0.5 text-xs text-muted-foreground"
                                    >
                                      {suggestion}
                                    </span>
                                  )
                                )}
                              </div>
                            </div>
                          )}
                      </>
                    )}
                  </div>
                )}
              </div>

              {/* Add Section Button */}
              <div className="flex justify-center py-1">
                <button
                  type="button"
                  onClick={() => addSection(index)}
                  className="flex items-center gap-1 rounded-full border border-dashed border-muted-foreground/30 px-2 py-0.5 text-xs text-muted-foreground transition-colors hover:border-primary hover:text-primary"
                >
                  <Plus className="h-3 w-3" />
                  Add section
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
