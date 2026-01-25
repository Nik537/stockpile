"use client";

import { Plus, FolderOpen, Clock, ArrowRight } from "lucide-react";
import Link from "next/link";
import { Header } from "@/components/layout/header";

export default function DashboardPage() {
  return (
    <div className="flex flex-col">
      <Header
        title="Dashboard"
        subtitle="Welcome to Stockpile Web"
      />

      <div className="flex-1 p-6">
        {/* Quick Actions */}
        <section className="mb-8">
          <h2 className="mb-4 text-lg font-semibold text-foreground">
            Quick Actions
          </h2>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {/* New Project Card */}
            <Link
              href="/workflow/script"
              className="group relative flex flex-col rounded-xl border border-border bg-card p-6 transition-all hover:border-primary/50 hover:shadow-lg hover:shadow-primary/5"
            >
              <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
                <Plus className="h-6 w-6 text-primary" />
              </div>
              <h3 className="mb-2 text-lg font-semibold text-foreground">
                New Project
              </h3>
              <p className="text-sm text-muted-foreground">
                Start creating a new video from scratch
              </p>
              <ArrowRight className="absolute bottom-6 right-6 h-5 w-5 text-muted-foreground opacity-0 transition-all group-hover:opacity-100 group-hover:translate-x-1" />
            </Link>

            {/* Open Project Card */}
            <button
              type="button"
              className="group relative flex flex-col rounded-xl border border-border bg-card p-6 text-left transition-all hover:border-secondary/50 hover:shadow-lg hover:shadow-secondary/5"
            >
              <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-lg bg-secondary/10">
                <FolderOpen className="h-6 w-6 text-secondary" />
              </div>
              <h3 className="mb-2 text-lg font-semibold text-foreground">
                Open Project
              </h3>
              <p className="text-sm text-muted-foreground">
                Continue working on an existing project
              </p>
              <ArrowRight className="absolute bottom-6 right-6 h-5 w-5 text-muted-foreground opacity-0 transition-all group-hover:opacity-100 group-hover:translate-x-1" />
            </button>

            {/* Recent Activity Card */}
            <button
              type="button"
              className="group relative flex flex-col rounded-xl border border-border bg-card p-6 text-left transition-all hover:border-muted-foreground/30 hover:shadow-lg"
            >
              <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-lg bg-muted">
                <Clock className="h-6 w-6 text-muted-foreground" />
              </div>
              <h3 className="mb-2 text-lg font-semibold text-foreground">
                Recent Activity
              </h3>
              <p className="text-sm text-muted-foreground">
                View your recent projects and activity
              </p>
              <ArrowRight className="absolute bottom-6 right-6 h-5 w-5 text-muted-foreground opacity-0 transition-all group-hover:opacity-100 group-hover:translate-x-1" />
            </button>
          </div>
        </section>

        {/* Recent Projects */}
        <section>
          <h2 className="mb-4 text-lg font-semibold text-foreground">
            Recent Projects
          </h2>
          <div className="rounded-xl border border-border bg-card">
            <div className="flex flex-col items-center justify-center py-12 text-center">
              <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-muted">
                <FolderOpen className="h-8 w-8 text-muted-foreground" />
              </div>
              <h3 className="mb-2 text-lg font-medium text-foreground">
                No projects yet
              </h3>
              <p className="mb-4 max-w-sm text-sm text-muted-foreground">
                Create your first project to get started with Stockpile Web
              </p>
              <Link
                href="/workflow/script"
                className="inline-flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90"
              >
                <Plus className="h-4 w-4" />
                Create Project
              </Link>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
