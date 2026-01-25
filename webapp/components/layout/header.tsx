"use client";

import { Bell, Settings, User } from "lucide-react";

interface HeaderProps {
  title?: string;
  subtitle?: string;
}

export function Header({ title, subtitle }: HeaderProps) {
  return (
    <header className="sticky top-0 z-30 flex h-16 items-center justify-between border-b border-border bg-background/95 px-6 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div>
        {title && (
          <h1 className="text-xl font-semibold text-foreground">{title}</h1>
        )}
        {subtitle && (
          <p className="text-sm text-muted-foreground">{subtitle}</p>
        )}
      </div>

      <div className="flex items-center gap-4">
        {/* Notifications */}
        <button
          type="button"
          className="flex h-9 w-9 items-center justify-center rounded-lg text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
        >
          <Bell className="h-5 w-5" />
        </button>

        {/* Settings */}
        <button
          type="button"
          className="flex h-9 w-9 items-center justify-center rounded-lg text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
        >
          <Settings className="h-5 w-5" />
        </button>

        {/* User Avatar */}
        <button
          type="button"
          className="flex h-9 w-9 items-center justify-center rounded-full bg-primary text-primary-foreground"
        >
          <User className="h-5 w-5" />
        </button>
      </div>
    </header>
  );
}
