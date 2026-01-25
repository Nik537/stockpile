"use client";

import { cn } from "@/lib/utils";
import {
  Scissors,
  Trash2,
  Copy,
  Gauge,
  Undo2,
  Redo2,
  Magnet,
  MousePointer,
  Move,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface ToolbarProps {
  hasSelection: boolean;
  isLocked: boolean;
  snapEnabled: boolean;
  canUndo: boolean;
  canRedo: boolean;
  onSplit: () => void;
  onDelete: () => void;
  onDuplicate: () => void;
  onUndo: () => void;
  onRedo: () => void;
  onToggleSnap: () => void;
  currentTool: "select" | "move";
  onToolChange: (tool: "select" | "move") => void;
}

interface ToolButtonProps {
  icon: React.ReactNode;
  label: string;
  shortcut?: string;
  onClick: () => void;
  disabled?: boolean;
  active?: boolean;
  variant?: "default" | "destructive";
}

function ToolButton({
  icon,
  label,
  shortcut,
  onClick,
  disabled = false,
  active = false,
  variant = "default",
}: ToolButtonProps) {
  return (
    <TooltipProvider delayDuration={300}>
      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            variant={variant === "destructive" ? "destructive" : "ghost"}
            size="sm"
            onClick={onClick}
            disabled={disabled}
            className={cn(
              "h-8 w-8 p-0",
              active && variant !== "destructive" && "bg-accent text-accent-foreground"
            )}
          >
            {icon}
          </Button>
        </TooltipTrigger>
        <TooltipContent side="bottom" className="flex items-center gap-2">
          <span>{label}</span>
          {shortcut && (
            <kbd className="text-[10px] bg-muted px-1.5 py-0.5 rounded text-muted-foreground">
              {shortcut}
            </kbd>
          )}
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

export function Toolbar({
  hasSelection,
  isLocked,
  snapEnabled,
  canUndo,
  canRedo,
  onSplit,
  onDelete,
  onDuplicate,
  onUndo,
  onRedo,
  onToggleSnap,
  currentTool,
  onToolChange,
}: ToolbarProps) {
  return (
    <div className="flex items-center gap-1 p-2 border-b border-border bg-card">
      {/* Tool selection */}
      <div className="flex items-center gap-0.5 border-r border-border pr-2 mr-1">
        <ToolButton
          icon={<MousePointer className="h-4 w-4" />}
          label="Select Tool"
          shortcut="V"
          onClick={() => onToolChange("select")}
          active={currentTool === "select"}
        />
        <ToolButton
          icon={<Move className="h-4 w-4" />}
          label="Move Tool"
          shortcut="M"
          onClick={() => onToolChange("move")}
          active={currentTool === "move"}
        />
      </div>

      {/* Snap toggle */}
      <ToolButton
        icon={<Magnet className="h-4 w-4" />}
        label="Snap to Grid/Clips"
        shortcut="S"
        onClick={onToggleSnap}
        active={snapEnabled}
      />

      <div className="border-l border-border pl-2 ml-1" />

      {/* Edit actions */}
      <ToolButton
        icon={<Scissors className="h-4 w-4" />}
        label="Split at Playhead"
        shortcut="S"
        onClick={onSplit}
        disabled={!hasSelection || isLocked}
      />
      <ToolButton
        icon={<Copy className="h-4 w-4" />}
        label="Duplicate Clip"
        shortcut="Ctrl+D"
        onClick={onDuplicate}
        disabled={!hasSelection || isLocked}
      />
      <ToolButton
        icon={<Trash2 className="h-4 w-4" />}
        label="Delete Clip"
        shortcut="Del"
        onClick={onDelete}
        disabled={!hasSelection || isLocked}
        variant="destructive"
      />

      <div className="border-l border-border pl-2 ml-1" />

      {/* Speed presets - shown when clip is selected */}
      {hasSelection && (
        <>
          <span className="text-xs text-muted-foreground px-2">Speed:</span>
          <div className="flex items-center gap-0.5">
            {[0.5, 0.75, 1, 1.25, 1.5, 2].map((speed) => (
              <button
                key={speed}
                type="button"
                disabled={isLocked}
                className={cn(
                  "px-2 py-1 text-xs rounded transition-colors",
                  "hover:bg-accent hover:text-accent-foreground",
                  "disabled:opacity-50 disabled:cursor-not-allowed"
                )}
              >
                {speed}x
              </button>
            ))}
          </div>
          <div className="border-l border-border pl-2 ml-1" />
        </>
      )}

      {/* Spacer */}
      <div className="flex-1" />

      {/* History actions */}
      <ToolButton
        icon={<Undo2 className="h-4 w-4" />}
        label="Undo"
        shortcut="Ctrl+Z"
        onClick={onUndo}
        disabled={!canUndo}
      />
      <ToolButton
        icon={<Redo2 className="h-4 w-4" />}
        label="Redo"
        shortcut="Ctrl+Y"
        onClick={onRedo}
        disabled={!canRedo}
      />
    </div>
  );
}
