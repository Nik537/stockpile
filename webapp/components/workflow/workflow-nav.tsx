"use client";

import { ArrowLeft, ArrowRight, Loader2, Save, AlertCircle } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { WORKFLOW_STEPS, type WorkflowStep } from "@/types/workflow";
import { getNextStep, getPreviousStep, getStepIndex } from "@/types/workflow-state";
import {
  useCurrentStep,
  useCompletedSteps,
  useStepValidation,
  useWorkflowStore,
} from "@/lib/workflow-store";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface WorkflowNavProps {
  className?: string;
  showSave?: boolean;
  customNextLabel?: string;
  customPrevLabel?: string;
  onBeforeNavigate?: () => Promise<boolean> | boolean;
}

/**
 * Workflow navigation component with Previous/Next buttons
 * Validates step completion before allowing forward navigation
 */
export function WorkflowNav({
  className,
  showSave = true,
  customNextLabel,
  customPrevLabel,
  onBeforeNavigate,
}: WorkflowNavProps) {
  const router = useRouter();
  const currentStep = useCurrentStep();
  const completedSteps = useCompletedSteps();
  const stepValidation = useStepValidation();
  const isSaving = useWorkflowStore((state) => state.isSaving);
  const goToNextStep = useWorkflowStore((state) => state.goToNextStep);
  const goToPreviousStep = useWorkflowStore((state) => state.goToPreviousStep);
  const markStepComplete = useWorkflowStore((state) => state.markStepComplete);
  const saveProject = useWorkflowStore((state) => state.saveProject);
  const validateStep = useWorkflowStore((state) => state.validateStep);

  const prevStep = getPreviousStep(currentStep);
  const nextStep = getNextStep(currentStep);
  const currentStepValid = stepValidation[currentStep];
  const currentStepInfo = WORKFLOW_STEPS.find((s) => s.id === currentStep);
  const nextStepInfo = nextStep ? WORKFLOW_STEPS.find((s) => s.id === nextStep) : null;
  const prevStepInfo = prevStep ? WORKFLOW_STEPS.find((s) => s.id === prevStep) : null;

  const handlePrevious = async () => {
    if (onBeforeNavigate) {
      const canNavigate = await onBeforeNavigate();
      if (!canNavigate) return;
    }
    saveProject();
    if (prevStepInfo) {
      router.push(prevStepInfo.path);
    }
  };

  const handleNext = async () => {
    // Validate current step first
    const isValid = validateStep(currentStep);

    if (!isValid) {
      return;
    }

    if (onBeforeNavigate) {
      const canNavigate = await onBeforeNavigate();
      if (!canNavigate) return;
    }

    // Mark current step as complete and save
    markStepComplete(currentStep);
    saveProject();

    if (nextStepInfo) {
      router.push(nextStepInfo.path);
    }
  };

  const handleSave = () => {
    saveProject();
  };

  return (
    <TooltipProvider>
      <div
        className={cn(
          "flex items-center justify-between border-t border-border pt-4",
          className
        )}
      >
        {/* Previous Button */}
        <div>
          {prevStep ? (
            <Button
              variant="outline"
              onClick={handlePrevious}
              disabled={isSaving}
              className="gap-2"
            >
              <ArrowLeft className="h-4 w-4" />
              {customPrevLabel || `Back to ${prevStepInfo?.name}`}
            </Button>
          ) : (
            <div /> // Empty div for layout
          )}
        </div>

        {/* Center Actions */}
        <div className="flex items-center gap-2">
          {showSave && (
            <Button
              variant="ghost"
              size="sm"
              onClick={handleSave}
              disabled={isSaving}
              className="gap-2"
            >
              {isSaving ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Save className="h-4 w-4" />
              )}
              {isSaving ? "Saving..." : "Save"}
            </Button>
          )}
        </div>

        {/* Next Button */}
        <div className="flex items-center gap-2">
          {!currentStepValid && nextStep && (
            <Tooltip>
              <TooltipTrigger asChild>
                <div className="flex items-center gap-1 text-amber-500 text-sm">
                  <AlertCircle className="h-4 w-4" />
                  <span>Complete this step to continue</span>
                </div>
              </TooltipTrigger>
              <TooltipContent>
                <p>Please complete all required fields before proceeding</p>
              </TooltipContent>
            </Tooltip>
          )}

          {nextStep ? (
            <Button
              onClick={handleNext}
              disabled={!currentStepValid || isSaving}
              className="gap-2"
            >
              {isSaving ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <>
                  {customNextLabel || `Continue to ${nextStepInfo?.name}`}
                  <ArrowRight className="h-4 w-4" />
                </>
              )}
            </Button>
          ) : (
            // Final step - no next button, show complete message
            <div className="text-sm text-muted-foreground">
              Final step of the workflow
            </div>
          )}
        </div>
      </div>
    </TooltipProvider>
  );
}

/**
 * Compact navigation for smaller spaces
 */
export function WorkflowNavCompact({ className }: { className?: string }) {
  const router = useRouter();
  const currentStep = useCurrentStep();
  const stepValidation = useStepValidation();
  const saveProject = useWorkflowStore((state) => state.saveProject);
  const markStepComplete = useWorkflowStore((state) => state.markStepComplete);
  const validateStep = useWorkflowStore((state) => state.validateStep);

  const prevStep = getPreviousStep(currentStep);
  const nextStep = getNextStep(currentStep);
  const currentStepValid = stepValidation[currentStep];
  const nextStepInfo = nextStep ? WORKFLOW_STEPS.find((s) => s.id === nextStep) : null;
  const prevStepInfo = prevStep ? WORKFLOW_STEPS.find((s) => s.id === prevStep) : null;

  const handlePrevious = () => {
    saveProject();
    if (prevStepInfo) {
      router.push(prevStepInfo.path);
    }
  };

  const handleNext = () => {
    const isValid = validateStep(currentStep);
    if (!isValid) return;

    markStepComplete(currentStep);
    saveProject();

    if (nextStepInfo) {
      router.push(nextStepInfo.path);
    }
  };

  return (
    <div className={cn("flex items-center gap-2", className)}>
      {prevStep && (
        <Button variant="outline" size="sm" onClick={handlePrevious}>
          <ArrowLeft className="h-4 w-4" />
        </Button>
      )}
      {nextStep && (
        <Button size="sm" onClick={handleNext} disabled={!currentStepValid}>
          <ArrowRight className="h-4 w-4" />
        </Button>
      )}
    </div>
  );
}

/**
 * Link-based navigation that uses Next.js Link component
 */
export function WorkflowNavLinks({ className }: { className?: string }) {
  const currentStep = useCurrentStep();
  const stepValidation = useStepValidation();
  const markStepComplete = useWorkflowStore((state) => state.markStepComplete);

  const prevStep = getPreviousStep(currentStep);
  const nextStep = getNextStep(currentStep);
  const currentStepValid = stepValidation[currentStep];
  const nextStepInfo = nextStep ? WORKFLOW_STEPS.find((s) => s.id === nextStep) : null;
  const prevStepInfo = prevStep ? WORKFLOW_STEPS.find((s) => s.id === prevStep) : null;

  return (
    <div className={cn("flex items-center justify-between", className)}>
      {prevStepInfo ? (
        <Link
          href={prevStepInfo.path}
          className="flex items-center gap-2 rounded-lg border border-border px-4 py-2.5 text-sm font-medium text-foreground transition-colors hover:bg-muted"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to {prevStepInfo.name}
        </Link>
      ) : (
        <div />
      )}

      {nextStepInfo ? (
        <Link
          href={nextStepInfo.path}
          onClick={(e) => {
            if (currentStepValid) {
              markStepComplete(currentStep);
            } else {
              e.preventDefault();
            }
          }}
          className={cn(
            "flex items-center gap-2 rounded-lg px-4 py-2.5 text-sm font-medium transition-colors",
            currentStepValid
              ? "bg-primary text-primary-foreground hover:bg-primary/90"
              : "bg-muted text-muted-foreground cursor-not-allowed"
          )}
        >
          Continue to {nextStepInfo.name}
          <ArrowRight className="h-4 w-4" />
        </Link>
      ) : (
        <div className="text-sm text-muted-foreground">Final step</div>
      )}
    </div>
  );
}
