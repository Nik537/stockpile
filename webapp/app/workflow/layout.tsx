"use client";

import { useEffect } from "react";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { WORKFLOW_STEPS, type WorkflowStep } from "@/types/workflow";
import { useWorkflowStore, useCurrentStep, useCompletedSteps } from "@/lib/workflow-store";
import { WorkflowSidebar } from "@/components/workflow/workflow-sidebar";

interface WorkflowLayoutProps {
  children: React.ReactNode;
}

/**
 * Shared layout for all workflow pages
 * Provides WorkflowProvider context and workflow sidebar
 */
export default function WorkflowLayout({ children }: WorkflowLayoutProps) {
  const pathname = usePathname();
  const setCurrentStep = useWorkflowStore((state) => state.setCurrentStep);
  const validateAllSteps = useWorkflowStore((state) => state.validateAllSteps);
  const currentStep = useCurrentStep();
  const completedSteps = useCompletedSteps();

  // Sync current step with URL
  useEffect(() => {
    const stepFromUrl = getStepFromPath(pathname);
    if (stepFromUrl && stepFromUrl !== currentStep) {
      setCurrentStep(stepFromUrl);
    }
  }, [pathname, currentStep, setCurrentStep]);

  // Validate all steps on mount
  useEffect(() => {
    validateAllSteps();
  }, [validateAllSteps]);

  return (
    <div className="flex h-full min-h-screen">
      {/* Workflow Sidebar */}
      <WorkflowSidebar className="hidden lg:block" />

      {/* Main Content */}
      <div className="flex-1 lg:ml-0">
        {children}
      </div>
    </div>
  );
}

/**
 * Extract workflow step from URL path
 */
function getStepFromPath(pathname: string): WorkflowStep | null {
  const pathParts = pathname.split("/").filter(Boolean);
  const workflowIndex = pathParts.indexOf("workflow");

  if (workflowIndex === -1 || workflowIndex >= pathParts.length - 1) {
    return null;
  }

  const stepSlug = pathParts[workflowIndex + 1];

  // Map URL slugs to step IDs
  const slugToStep: Record<string, WorkflowStep> = {
    script: "script",
    "title-thumbnail": "title-thumbnail",
    broll: "broll",
    "avatar-tts": "avatar-tts",
    editor: "editor",
    export: "export",
  };

  return slugToStep[stepSlug] || null;
}
