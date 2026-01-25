"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  FileText,
  Image,
  Film,
  User,
  Layers,
  Download,
  Home,
  ChevronRight,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { WORKFLOW_STEPS, type WorkflowStep } from "@/types/workflow";

const iconMap = {
  FileText,
  Image,
  Film,
  User,
  Layers,
  Download,
};

interface SidebarProps {
  currentStep?: WorkflowStep;
}

export function Sidebar({ currentStep }: SidebarProps) {
  const pathname = usePathname();

  const getStepStatus = (stepId: WorkflowStep) => {
    if (!currentStep) return "pending";
    const currentIndex = WORKFLOW_STEPS.findIndex((s) => s.id === currentStep);
    const stepIndex = WORKFLOW_STEPS.findIndex((s) => s.id === stepId);

    if (stepIndex < currentIndex) return "completed";
    if (stepIndex === currentIndex) return "current";
    return "pending";
  };

  return (
    <aside className="fixed left-0 top-0 z-40 h-screen w-64 border-r border-border bg-card">
      <div className="flex h-full flex-col">
        {/* Logo */}
        <div className="flex h-16 items-center border-b border-border px-6">
          <Link href="/" className="flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary">
              <Film className="h-5 w-5 text-primary-foreground" />
            </div>
            <span className="text-xl font-bold text-foreground">Stockpile</span>
          </Link>
        </div>

        {/* Navigation */}
        <nav className="flex-1 space-y-1 px-3 py-4">
          {/* Dashboard Link */}
          <Link
            href="/"
            className={cn(
              "flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
              pathname === "/"
                ? "bg-primary/10 text-primary"
                : "text-muted-foreground hover:bg-muted hover:text-foreground"
            )}
          >
            <Home className="h-5 w-5" />
            Dashboard
          </Link>

          {/* Workflow Steps Divider */}
          <div className="my-4 px-3">
            <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              Workflow Steps
            </p>
          </div>

          {/* Workflow Steps */}
          {WORKFLOW_STEPS.map((step, index) => {
            const Icon = iconMap[step.icon as keyof typeof iconMap];
            const status = getStepStatus(step.id);
            const isActive = pathname === step.path;

            return (
              <Link
                key={step.id}
                href={step.path}
                className={cn(
                  "group flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
                  isActive
                    ? "bg-primary/10 text-primary"
                    : status === "completed"
                    ? "text-muted-foreground hover:bg-muted hover:text-foreground"
                    : status === "current"
                    ? "text-foreground hover:bg-muted"
                    : "text-muted-foreground/50 hover:bg-muted hover:text-muted-foreground"
                )}
              >
                <div
                  className={cn(
                    "flex h-6 w-6 items-center justify-center rounded-full text-xs font-bold",
                    isActive
                      ? "bg-primary text-primary-foreground"
                      : status === "completed"
                      ? "bg-green-500/20 text-green-500"
                      : status === "current"
                      ? "bg-secondary/20 text-secondary"
                      : "bg-muted text-muted-foreground"
                  )}
                >
                  {index + 1}
                </div>
                <span className="flex-1">{step.name}</span>
                {isActive && (
                  <ChevronRight className="h-4 w-4 text-primary" />
                )}
              </Link>
            );
          })}
        </nav>

        {/* Footer */}
        <div className="border-t border-border p-4">
          <div className="rounded-lg bg-muted/50 p-3">
            <p className="text-xs text-muted-foreground">
              Stockpile Web v0.1.0
            </p>
            <p className="mt-1 text-xs text-muted-foreground/70">
              AI-powered video creation
            </p>
          </div>
        </div>
      </div>
    </aside>
  );
}
