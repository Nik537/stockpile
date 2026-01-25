"use client";

import { useState } from "react";
import { User, Plus, Check, Video, Sparkles } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import type { Avatar } from "@/types/workflow";

interface AvatarGeneratorProps {
  avatars: Avatar[];
  selectedAvatarId: string | null;
  onAvatarSelect: (avatarId: string | null) => void;
  onCreateAvatar?: (name: string, style: Avatar["style"]) => void;
  isEnabled: boolean;
  onToggleEnabled: (enabled: boolean) => void;
}

export function AvatarGenerator({
  avatars,
  selectedAvatarId,
  onAvatarSelect,
  onCreateAvatar,
  isEnabled,
  onToggleEnabled,
}: AvatarGeneratorProps) {
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);
  const [newAvatarName, setNewAvatarName] = useState("");
  const [newAvatarStyle, setNewAvatarStyle] = useState<Avatar["style"]>("realistic");

  const handleCreateAvatar = () => {
    if (newAvatarName && onCreateAvatar) {
      onCreateAvatar(newAvatarName, newAvatarStyle);
      setNewAvatarName("");
      setNewAvatarStyle("realistic");
      setIsCreateDialogOpen(false);
    }
  };

  const getStyleIcon = (style: Avatar["style"]) => {
    switch (style) {
      case "realistic":
        return <User className="h-4 w-4" />;
      case "animated":
        return <Sparkles className="h-4 w-4" />;
      case "professional":
        return <Video className="h-4 w-4" />;
    }
  };

  const getStyleColor = (style: Avatar["style"]) => {
    switch (style) {
      case "realistic":
        return "bg-blue-500/20 text-blue-400 border-blue-500/30";
      case "animated":
        return "bg-purple-500/20 text-purple-400 border-purple-500/30";
      case "professional":
        return "bg-emerald-500/20 text-emerald-400 border-emerald-500/30";
    }
  };

  return (
    <div className="space-y-4">
      {/* Enable/Disable Toggle */}
      <div className="flex items-center justify-between rounded-lg border border-border bg-card p-4">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary/20">
            <Video className="h-5 w-5 text-primary" />
          </div>
          <div>
            <p className="text-sm font-medium text-foreground">
              Avatar Video Generation
            </p>
            <p className="text-xs text-muted-foreground">
              Generate talking head videos with lip-sync
            </p>
          </div>
        </div>
        <button
          type="button"
          onClick={() => onToggleEnabled(!isEnabled)}
          className={cn(
            "relative h-6 w-11 rounded-full transition-colors",
            isEnabled ? "bg-primary" : "bg-muted"
          )}
        >
          <span
            className={cn(
              "absolute top-0.5 block h-5 w-5 rounded-full bg-white shadow transition-transform",
              isEnabled ? "translate-x-5" : "translate-x-0.5"
            )}
          />
        </button>
      </div>

      {/* Avatar Selection */}
      <div
        className={cn(
          "transition-opacity",
          !isEnabled && "pointer-events-none opacity-50"
        )}
      >
        <div className="mb-3 flex items-center justify-between">
          <h3 className="text-sm font-medium text-foreground">
            Select Avatar
          </h3>
          <Dialog open={isCreateDialogOpen} onOpenChange={setIsCreateDialogOpen}>
            <DialogTrigger asChild>
              <Button variant="outline" size="sm" className="gap-2">
                <Plus className="h-4 w-4" />
                Create New
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Create New Avatar</DialogTitle>
                <DialogDescription>
                  Enter a name and choose a style for your new avatar.
                </DialogDescription>
              </DialogHeader>
              <div className="space-y-4 py-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground">
                    Avatar Name
                  </label>
                  <input
                    type="text"
                    value={newAvatarName}
                    onChange={(e) => setNewAvatarName(e.target.value)}
                    placeholder="Enter avatar name"
                    className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground">
                    Style
                  </label>
                  <div className="grid grid-cols-3 gap-2">
                    {(["realistic", "animated", "professional"] as const).map(
                      (style) => (
                        <button
                          key={style}
                          type="button"
                          onClick={() => setNewAvatarStyle(style)}
                          className={cn(
                            "flex flex-col items-center gap-2 rounded-lg border p-3 transition-colors",
                            newAvatarStyle === style
                              ? "border-primary bg-primary/10"
                              : "border-border hover:border-primary/50"
                          )}
                        >
                          {getStyleIcon(style)}
                          <span className="text-xs capitalize">{style}</span>
                        </button>
                      )
                    )}
                  </div>
                </div>
              </div>
              <div className="flex justify-end gap-2">
                <Button
                  variant="outline"
                  onClick={() => setIsCreateDialogOpen(false)}
                >
                  Cancel
                </Button>
                <Button
                  onClick={handleCreateAvatar}
                  disabled={!newAvatarName}
                >
                  Create Avatar
                </Button>
              </div>
            </DialogContent>
          </Dialog>
        </div>

        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4">
          {/* No Avatar Option */}
          <button
            type="button"
            onClick={() => onAvatarSelect(null)}
            className={cn(
              "group relative flex flex-col items-center rounded-xl border p-4 transition-all",
              selectedAvatarId === null
                ? "border-primary bg-primary/10"
                : "border-border hover:border-primary/50 hover:bg-muted/50"
            )}
          >
            <div className="mb-2 flex h-16 w-16 items-center justify-center rounded-full bg-muted">
              <span className="text-2xl text-muted-foreground">-</span>
            </div>
            <span className="text-xs font-medium text-foreground">
              No Avatar
            </span>
            <span className="text-[10px] text-muted-foreground">
              Audio only
            </span>
            {selectedAvatarId === null && (
              <div className="absolute -right-1 -top-1 flex h-5 w-5 items-center justify-center rounded-full bg-primary text-primary-foreground">
                <Check className="h-3 w-3" />
              </div>
            )}
          </button>

          {/* Avatar Options */}
          {avatars.map((avatar) => (
            <button
              key={avatar.id}
              type="button"
              onClick={() => onAvatarSelect(avatar.id)}
              className={cn(
                "group relative flex flex-col items-center rounded-xl border p-4 transition-all",
                selectedAvatarId === avatar.id
                  ? "border-primary bg-primary/10"
                  : "border-border hover:border-primary/50 hover:bg-muted/50"
              )}
            >
              {/* Avatar Thumbnail */}
              {avatar.thumbnailUrl ? (
                <img
                  src={avatar.thumbnailUrl}
                  alt={avatar.name}
                  className="mb-2 h-16 w-16 rounded-full object-cover"
                />
              ) : (
                <div className="mb-2 flex h-16 w-16 items-center justify-center rounded-full bg-muted">
                  <User className="h-8 w-8 text-muted-foreground" />
                </div>
              )}
              <span className="text-xs font-medium text-foreground truncate max-w-full">
                {avatar.name}
              </span>
              <span
                className={cn(
                  "mt-1 rounded-full border px-2 py-0.5 text-[10px] font-medium",
                  getStyleColor(avatar.style)
                )}
              >
                {avatar.style}
              </span>
              {selectedAvatarId === avatar.id && (
                <div className="absolute -right-1 -top-1 flex h-5 w-5 items-center justify-center rounded-full bg-primary text-primary-foreground">
                  <Check className="h-3 w-3" />
                </div>
              )}
            </button>
          ))}
        </div>

        {avatars.length === 0 && (
          <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-border py-8 text-center">
            <User className="h-12 w-12 text-muted-foreground/50" />
            <p className="mt-2 text-sm text-muted-foreground">
              No avatars created yet
            </p>
            <p className="mt-1 text-xs text-muted-foreground">
              Create an avatar to generate talking head videos
            </p>
          </div>
        )}
      </div>

      {/* Info Box */}
      {isEnabled && selectedAvatarId && (
        <div className="rounded-lg border border-border bg-muted/50 p-4">
          <h4 className="text-sm font-medium text-foreground">
            Avatar Video Preview
          </h4>
          <p className="mt-1 text-xs text-muted-foreground">
            Your avatar video will be generated after audio is created.
            The lip-sync will be automatically synchronized with the generated speech.
          </p>
        </div>
      )}
    </div>
  );
}
