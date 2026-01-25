"use client";

import { Check, ChevronRight, Circle, Lock } from "lucide-react";
import Link from "next/link";
import { cn } from "@/lib/utils";
import { WORKFLOW_STEPS, type WorkflowStep } from "@/types/workflow";
import { canNavigateToStep } from "@/types/workflow-state";
import {
  useCurrentStep,
  useCompletedSteps,
  useStepValidation,
  useWorkflowStore,
} from "@/lib/workflow-store";

interface EnhancedStepIndicatorProps {
  className?: string;
  showLabels?: boolean;
  compact?: boolean;
  interactive?: boolean;
}

/**
 * Enhanced step indicator with clickable navigation
 */
export function EnhancedStepIndicator({
  className,
  showLabels = true,
  compact = false,
  interactive = true,
}: EnhancedStepIndicatorProps) {
  const currentStep = useCurrentStep();
  const completedSteps = useCompletedSteps();
  const stepValidation = useStepValidation();
  const setCurrentStep = useWorkflowStore((state) => state.setCurrentStep);

  const handleStepClick = (step: WorkflowStep) => {
    if (!interactive) return;
    if (canNavigateToStep(step, currentStep, completedSteps)) {
      setCurrentStep(step);
    }
  };

  return (
    <div className={cn("w-full", className)}>
      <div
        className={cn(
          "flex items-center",
          compact ? "gap-1" : "gap-2 md:gap-4",
          "overflow-x-auto pb-2"
        )}
      >
        {WORKFLOW_STEPS.map((step, index) => {
          const isCompleted = completedSteps.includes(step.id);
          const isCurrent = step.id === currentStep;
          const isValid = stepValidation[step.id];
          const canNavigate = canNavigateToStep(step.id, currentStep, completedSteps);
          const isClickable = interactive && canNavigate;
          const isLast = index === WORKFLOW_STEPS.length - 1;

          return (
            <div key={step.id} className="flex items-center">
              {/* Step */}
              {isClickable ? (
                <Link
                  href={step.path}
                  className="flex items-center gap-2 group"
                  onClick={(e) => {
                    e.preventDefault();
                    handleStepClick(step.id);
                    // Navigate via Link
                    window.location.href = step.path;
                  }}
                >
                  <StepCircle
                    isCompleted={isCompleted}
                    isCurrent={isCurrent}
                    isClickable={isClickable}
                    compact={compact}
                    stepNumber={index + 1}
                  />
                  {showLabels && !compact && (
                    <StepLabel
                      name={step.name}
                      isCompleted={isCompleted}
                      isCurrent={isCurrent}
                      isClickable={isClickable}
                    />
                  )}
                </Link>
              ) : (
                <div className="flex items-center gap-2 opacity-50 cursor-not-allowed">
                  <StepCircle
                    isCompleted={isCompleted}
                    isCurrent={isCurrent}
                    isClickable={false}
                    compact={compact}
                    stepNumber={index + 1}
                  />
                  {showLabels && !compact && (
                    <StepLabel
                      name={step.name}
                      isCompleted={isCompleted}
                      isCurrent={isCurrent}
                      isClickable={false}
                    />
                  )}
                </div>
              )}

              {/* Connector */}
              {!isLast && (
                <ChevronRight
                  className={cn(
                    "mx-1 flex-shrink-0",
                    compact ? "h-3 w-3" : "h-4 w-4",
                    isCompleted ? "text-green-500" : "text-muted-foreground/30"
                  )}
                />
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

interface StepCircleProps {
  isCompleted: boolean;
  isCurrent: boolean;
  isClickable: boolean;
  compact: boolean;
  stepNumber: number;
}

function StepCircle({
  isCompleted,
  isCurrent,
  isClickable,
  compact,
  stepNumber,
}: StepCircleProps) {
  const size = compact ? "h-6 w-6" : "h-8 w-8 md:h-10 md:w-10";
  const iconSize = compact ? "h-3 w-3" : "h-4 w-4 md:h-5 md:w-5";

  return (
    <div
      className={cn(
        "flex items-center justify-center rounded-full border-2 transition-all",
        size,
        isCompleted
          ? "border-green-500 bg-green-500 text-white"
          : isCurrent
          ? "border-primary bg-primary text-primary-foreground"
          : isClickable
          ? "border-muted-foreground/50 bg-muted/50 text-muted-foreground hover:border-primary/50 hover:bg-primary/10"
          : "border-muted-foreground/30 bg-transparent text-muted-foreground/30"
      )}
    >
      {isCompleted ? (
        <Check className={iconSize} />
      ) : (
        <span className={cn(compact ? "text-xs" : "text-sm", "font-semibold")}>
          {stepNumber}
        </span>
      )}
    </div>
  );
}

interface StepLabelProps {
  name: string;
  isCompleted: boolean;
  isCurrent: boolean;
  isClickable: boolean;
}

function StepLabel({ name, isCompleted, isCurrent, isClickable }: StepLabelProps) {
  return (
    <span
      className={cn(
        "text-sm font-medium whitespace-nowrap",
        isCompleted
          ? "text-green-500"
          : isCurrent
          ? "text-foreground"
          : isClickable
          ? "text-muted-foreground group-hover:text-foreground"
          : "text-muted-foreground/30"
      )}
    >
      {name}
    </span>
  );
}

/**
 * Minimal step progress indicator
 */
export function StepProgressBar({ className }: { className?: string }) {
  const currentStep = useCurrentStep();
  const completedSteps = useCompletedSteps();

  const currentIndex = WORKFLOW_STEPS.findIndex((s) => s.id === currentStep);
  const progressPercentage = ((completedSteps.length) / WORKFLOW_STEPS.length) * 100;

  return (
    <div className={cn("w-full", className)}>
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm text-muted-foreground">
          Step {currentIndex + 1} of {WORKFLOW_STEPS.length}
        </span>
        <span className="text-sm text-muted-foreground">
          {completedSteps.length} completed
        </span>
      </div>
      <div className="h-2 w-full rounded-full bg-muted overflow-hidden">
        <div
          className="h-full bg-primary transition-all duration-300"
          style={{ width: `${progressPercentage}%` }}
        />
      </div>
    </div>
  );
}

/**
 * Workflow status badge showing current step
 */
export function WorkflowStatusBadge({ className }: { className?: string }) {
  const currentStep = useCurrentStep();
  const stepInfo = WORKFLOW_STEPS.find((s) => s.id === currentStep);

  return (
    <div
      className={cn(
        "inline-flex items-center gap-2 rounded-full bg-primary/10 px-3 py-1.5 text-sm",
        className
      )}
    >
      <Circle className="h-2 w-2 fill-primary text-primary animate-pulse" />
      <span className="font-medium text-primary">{stepInfo?.name || currentStep}</span>
    </div>
  );
}
