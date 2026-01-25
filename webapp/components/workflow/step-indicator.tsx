"use client";

import { Check } from "lucide-react";
import { cn } from "@/lib/utils";
import { WORKFLOW_STEPS, type WorkflowStep } from "@/types/workflow";

interface StepIndicatorProps {
  currentStep: WorkflowStep;
  completedSteps?: WorkflowStep[];
}

export function StepIndicator({
  currentStep,
  completedSteps = [],
}: StepIndicatorProps) {
  return (
    <div className="flex items-center justify-center gap-2 py-4">
      {WORKFLOW_STEPS.map((step, index) => {
        const isCompleted = completedSteps.includes(step.id);
        const isCurrent = step.id === currentStep;
        const isPending = !isCompleted && !isCurrent;

        return (
          <div key={step.id} className="flex items-center">
            {/* Step Circle */}
            <div
              className={cn(
                "flex h-10 w-10 items-center justify-center rounded-full border-2 transition-all",
                isCompleted
                  ? "border-green-500 bg-green-500 text-white"
                  : isCurrent
                  ? "border-primary bg-primary text-primary-foreground"
                  : "border-muted-foreground/30 bg-transparent text-muted-foreground/50"
              )}
            >
              {isCompleted ? (
                <Check className="h-5 w-5" />
              ) : (
                <span className="text-sm font-semibold">{index + 1}</span>
              )}
            </div>

            {/* Step Label (shown below on larger screens) */}
            <div className="ml-2 hidden md:block">
              <p
                className={cn(
                  "text-sm font-medium",
                  isCompleted
                    ? "text-green-500"
                    : isCurrent
                    ? "text-foreground"
                    : "text-muted-foreground/50"
                )}
              >
                {step.name}
              </p>
            </div>

            {/* Connector Line */}
            {index < WORKFLOW_STEPS.length - 1 && (
              <div
                className={cn(
                  "mx-2 h-0.5 w-8 md:w-16",
                  isCompleted || (isCurrent && index < WORKFLOW_STEPS.findIndex(s => s.id === currentStep))
                    ? "bg-green-500"
                    : "bg-muted-foreground/30"
                )}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}

// Compact version for smaller spaces
export function StepIndicatorCompact({
  currentStep,
  completedSteps = [],
}: StepIndicatorProps) {
  const currentIndex = WORKFLOW_STEPS.findIndex((s) => s.id === currentStep);
  const totalSteps = WORKFLOW_STEPS.length;

  return (
    <div className="flex items-center gap-4">
      <div className="flex items-center gap-1">
        {WORKFLOW_STEPS.map((step, index) => {
          const isCompleted = completedSteps.includes(step.id);
          const isCurrent = step.id === currentStep;

          return (
            <div
              key={step.id}
              className={cn(
                "h-2 w-2 rounded-full transition-all",
                isCompleted
                  ? "bg-green-500"
                  : isCurrent
                  ? "bg-primary"
                  : "bg-muted-foreground/30"
              )}
            />
          );
        })}
      </div>
      <span className="text-sm text-muted-foreground">
        Step {currentIndex + 1} of {totalSteps}
      </span>
    </div>
  );
}
