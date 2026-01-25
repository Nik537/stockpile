"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  Check,
  ChevronLeft,
  ChevronRight,
  Circle,
  FileText,
  Image,
  Film,
  User,
  Layers,
  Download,
  Home,
  RotateCcw,
  Save,
  Loader2,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { WORKFLOW_STEPS, type WorkflowStep } from "@/types/workflow";
import { canNavigateToStep } from "@/types/workflow-state";
import {
  useCurrentStep,
  useCompletedSteps,
  useStepValidation,
  useWorkflowStore,
  useWorkflowProject,
} from "@/lib/workflow-store";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";

// Icon mapping for steps
const stepIcons: Record<WorkflowStep, React.ComponentType<{ className?: string }>> = {
  script: FileText,
  "title-thumbnail": Image,
  broll: Film,
  "avatar-tts": User,
  editor: Layers,
  export: Download,
};

interface WorkflowSidebarProps {
  className?: string;
}

/**
 * Workflow sidebar showing steps and completion status
 */
export function WorkflowSidebar({ className }: WorkflowSidebarProps) {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const router = useRouter();

  const currentStep = useCurrentStep();
  const completedSteps = useCompletedSteps();
  const stepValidation = useStepValidation();
  const project = useWorkflowProject();
  const isSaving = useWorkflowStore((state) => state.isSaving);
  const setCurrentStep = useWorkflowStore((state) => state.setCurrentStep);
  const saveProject = useWorkflowStore((state) => state.saveProject);
  const resetProject = useWorkflowStore((state) => state.resetProject);

  const handleStepClick = (step: WorkflowStep, path: string) => {
    if (canNavigateToStep(step, currentStep, completedSteps)) {
      setCurrentStep(step);
      router.push(path);
    }
  };

  const handleReset = () => {
    resetProject();
    router.push("/workflow/script");
  };

  return (
    <aside
      className={cn(
        "flex flex-col border-r border-border bg-card transition-all duration-300",
        isCollapsed ? "w-16" : "w-64",
        className
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-border">
        {!isCollapsed && (
          <div className="flex flex-col">
            <span className="text-sm font-semibold text-foreground">Workflow</span>
            <span className="text-xs text-muted-foreground truncate max-w-[180px]">
              {project.name}
            </span>
          </div>
        )}
        <Button
          variant="ghost"
          size="icon"
          onClick={() => setIsCollapsed(!isCollapsed)}
          className="h-8 w-8"
        >
          {isCollapsed ? (
            <ChevronRight className="h-4 w-4" />
          ) : (
            <ChevronLeft className="h-4 w-4" />
          )}
        </Button>
      </div>

      {/* Steps */}
      <nav className="flex-1 p-2 space-y-1 overflow-y-auto">
        {WORKFLOW_STEPS.map((step, index) => {
          const isCompleted = completedSteps.includes(step.id);
          const isCurrent = step.id === currentStep;
          const isValid = stepValidation[step.id];
          const canNavigate = canNavigateToStep(step.id, currentStep, completedSteps);
          const Icon = stepIcons[step.id];

          return (
            <button
              key={step.id}
              type="button"
              onClick={() => handleStepClick(step.id, step.path)}
              disabled={!canNavigate}
              className={cn(
                "w-full flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm transition-all",
                isCurrent
                  ? "bg-primary text-primary-foreground"
                  : isCompleted
                  ? "bg-green-500/10 text-green-500 hover:bg-green-500/20"
                  : canNavigate
                  ? "text-foreground hover:bg-muted"
                  : "text-muted-foreground/50 cursor-not-allowed",
                isCollapsed && "justify-center px-2"
              )}
            >
              {/* Step indicator */}
              <div
                className={cn(
                  "flex h-6 w-6 items-center justify-center rounded-full flex-shrink-0",
                  isCurrent
                    ? "bg-primary-foreground/20"
                    : isCompleted
                    ? "bg-green-500/20"
                    : "bg-muted"
                )}
              >
                {isCompleted ? (
                  <Check className="h-3.5 w-3.5" />
                ) : (
                  <span className="text-xs font-medium">{index + 1}</span>
                )}
              </div>

              {/* Label */}
              {!isCollapsed && (
                <div className="flex flex-col items-start flex-1 min-w-0">
                  <span className="font-medium truncate w-full text-left">
                    {step.name}
                  </span>
                  {isCurrent && (
                    <span className="text-xs opacity-70">Current step</span>
                  )}
                </div>
              )}

              {/* Status indicator */}
              {!isCollapsed && !isCurrent && !isCompleted && isValid && (
                <Circle className="h-2 w-2 fill-amber-500 text-amber-500" />
              )}
            </button>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="p-2 border-t border-border space-y-1">
        {/* Save Button */}
        <Button
          variant="ghost"
          onClick={saveProject}
          disabled={isSaving}
          className={cn(
            "w-full justify-start gap-3",
            isCollapsed && "justify-center px-2"
          )}
        >
          {isSaving ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Save className="h-4 w-4" />
          )}
          {!isCollapsed && (isSaving ? "Saving..." : "Save Project")}
        </Button>

        {/* Reset Button */}
        <AlertDialog>
          <AlertDialogTrigger asChild>
            <Button
              variant="ghost"
              className={cn(
                "w-full justify-start gap-3 text-destructive hover:text-destructive hover:bg-destructive/10",
                isCollapsed && "justify-center px-2"
              )}
            >
              <RotateCcw className="h-4 w-4" />
              {!isCollapsed && "Start Over"}
            </Button>
          </AlertDialogTrigger>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>Start Over?</AlertDialogTitle>
              <AlertDialogDescription>
                This will reset all your workflow progress and data. This action
                cannot be undone.
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>Cancel</AlertDialogCancel>
              <AlertDialogAction
                onClick={handleReset}
                className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              >
                Reset Everything
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>

        {/* Home Link */}
        <Link href="/">
          <Button
            variant="ghost"
            className={cn(
              "w-full justify-start gap-3",
              isCollapsed && "justify-center px-2"
            )}
          >
            <Home className="h-4 w-4" />
            {!isCollapsed && "Back to Home"}
          </Button>
        </Link>
      </div>
    </aside>
  );
}

/**
 * Mobile workflow navigation
 */
export function WorkflowMobileNav({ className }: { className?: string }) {
  const currentStep = useCurrentStep();
  const completedSteps = useCompletedSteps();
  const setCurrentStep = useWorkflowStore((state) => state.setCurrentStep);

  return (
    <div
      className={cn(
        "flex items-center gap-1 overflow-x-auto pb-2 scrollbar-hide",
        className
      )}
    >
      {WORKFLOW_STEPS.map((step, index) => {
        const isCompleted = completedSteps.includes(step.id);
        const isCurrent = step.id === currentStep;
        const canNavigate = canNavigateToStep(step.id, currentStep, completedSteps);

        return (
          <Link
            key={step.id}
            href={step.path}
            onClick={(e) => {
              if (!canNavigate) {
                e.preventDefault();
                return;
              }
              setCurrentStep(step.id);
            }}
            className={cn(
              "flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-medium whitespace-nowrap transition-all",
              isCurrent
                ? "bg-primary text-primary-foreground"
                : isCompleted
                ? "bg-green-500/20 text-green-500"
                : canNavigate
                ? "bg-muted text-muted-foreground hover:bg-muted/80"
                : "bg-muted/50 text-muted-foreground/50 cursor-not-allowed"
            )}
          >
            {isCompleted ? (
              <Check className="h-3 w-3" />
            ) : (
              <span>{index + 1}</span>
            )}
            <span>{step.name}</span>
          </Link>
        );
      })}
    </div>
  );
}
