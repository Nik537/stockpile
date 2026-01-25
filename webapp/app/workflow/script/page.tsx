"use client";

import { useState, useEffect, useCallback } from "react";
import {
  Wand2,
  Copy,
  Download,
  RefreshCw,
  FileText,
  Clock,
} from "lucide-react";
import { Header } from "@/components/layout/header";
import { EnhancedStepIndicator } from "@/components/workflow/enhanced-step-indicator";
import { WorkflowNav } from "@/components/workflow/workflow-nav";
import { ScriptChat } from "@/components/workflow/script/script-chat";
import { ScriptEditor } from "@/components/workflow/script/script-editor";
import { ScriptTimeline } from "@/components/workflow/script/script-timeline";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import { useWorkflowStore, useScriptData } from "@/lib/workflow-store";
import type {
  Script,
  ChatMessage,
  ScriptWriterSection,
  ScriptGenerationResponse,
} from "@/types/workflow";

export default function ScriptPage() {
  // Workflow store
  const scriptData = useScriptData();
  const setScript = useWorkflowStore((state) => state.setScript);
  const setScriptMessages = useWorkflowStore((state) => state.setScriptMessages);
  const markStepComplete = useWorkflowStore((state) => state.markStepComplete);

  // Local state synced with store
  const [script, setLocalScript] = useState<Script | null>(scriptData.script);
  const [messages, setMessages] = useState<ChatMessage[]>(scriptData.messages);
  const [isLoading, setIsLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [activeTab, setActiveTab] = useState("chat");
  const [activeSection, setActiveSection] = useState<string | null>(null);

  // Sync local state with store on mount
  useEffect(() => {
    if (scriptData.script) {
      setLocalScript(scriptData.script);
    }
    if (scriptData.messages.length > 0) {
      setMessages(scriptData.messages);
    }
  }, []);

  // Sync to workflow store when script changes
  useEffect(() => {
    if (script) {
      setScript(script);
      markStepComplete("script");
    }
  }, [script, setScript, markStepComplete]);

  // Sync messages to workflow store
  useEffect(() => {
    if (messages.length > 0) {
      setScriptMessages(messages);
    }
  }, [messages, setScriptMessages]);

  const generateScript = useCallback(async (newMessages: ChatMessage[]) => {
    setIsLoading(true);
    setProgress(0);

    // Simulate progress
    const progressInterval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 90) {
          clearInterval(progressInterval);
          return prev;
        }
        return prev + 10;
      });
    }, 200);

    try {
      const response = await fetch("/api/script", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          messages: newMessages,
          currentScript: script,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to generate script");
      }

      const data: ScriptGenerationResponse = await response.json();

      // Add assistant response to messages
      const assistantMessage: ChatMessage = {
        id: Math.random().toString(36).substring(2, 15),
        role: "assistant",
        content: data.message,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
      setLocalScript(data.script);
      setProgress(100);

      // Switch to editor tab when script is generated
      setActiveTab("editor");
    } catch (error) {
      console.error("Error generating script:", error);

      // Add error message
      const errorMessage: ChatMessage = {
        id: Math.random().toString(36).substring(2, 15),
        role: "assistant",
        content:
          "Sorry, I encountered an error while generating your script. Please try again.",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      clearInterval(progressInterval);
      setIsLoading(false);
      setTimeout(() => setProgress(0), 500);
    }
  }, [script]);

  const handleSendMessage = useCallback(
    (content: string) => {
      const userMessage: ChatMessage = {
        id: Math.random().toString(36).substring(2, 15),
        role: "user",
        content,
        timestamp: new Date(),
      };

      const newMessages = [...messages, userMessage];
      setMessages(newMessages);
      generateScript(newMessages);
    },
    [messages, generateScript]
  );

  const handleScriptChange = useCallback((updatedScript: Script) => {
    setLocalScript(updatedScript);
  }, []);

  const handleSectionClick = useCallback((section: ScriptWriterSection) => {
    setActiveSection(section.id);
    setActiveTab("editor");
  }, []);

  const handleCopyScript = useCallback(() => {
    if (!script) return;

    const fullScript = script.sections
      .map((s) => `[${s.timestamp}]\n${s.content}`)
      .join("\n\n");

    navigator.clipboard.writeText(fullScript);
  }, [script]);

  const handleExportJSON = useCallback(() => {
    if (!script) return;

    const dataStr = JSON.stringify(script, null, 2);
    const dataUri =
      "data:application/json;charset=utf-8," + encodeURIComponent(dataStr);

    const exportFileDefaultName = `script_${script.id}.json`;

    const linkElement = document.createElement("a");
    linkElement.setAttribute("href", dataUri);
    linkElement.setAttribute("download", exportFileDefaultName);
    linkElement.click();
  }, [script]);

  const handleResetScript = useCallback(() => {
    if (
      confirm(
        "Are you sure you want to reset? This will clear your script and chat history."
      )
    ) {
      setLocalScript(null);
      setMessages([]);
      setScript(null);
      setScriptMessages([]);
      setActiveTab("chat");
    }
  }, [setScript, setScriptMessages]);

  return (
    <div className="flex h-screen flex-col overflow-hidden">
      <Header
        title="Script Writer"
        subtitle="Create and refine your video script with AI"
      />

      <div className="flex-1 overflow-hidden p-6">
        {/* Step Indicator */}
        <div className="mb-4">
          <EnhancedStepIndicator />
        </div>

        {/* Progress Bar */}
        {isLoading && (
          <div className="mb-4">
            <Progress value={progress} className="h-2" />
            <p className="mt-1 text-center text-sm text-muted-foreground">
              Generating your script...
            </p>
          </div>
        )}

        {/* Main Content - Split View */}
        <div className="grid h-[calc(100%-160px)] gap-6 lg:grid-cols-2">
          {/* Left Panel - Chat */}
          <div className="flex flex-col overflow-hidden rounded-xl border border-border bg-card">
            <ScriptChat
              messages={messages}
              onSendMessage={handleSendMessage}
              isLoading={isLoading}
            />
          </div>

          {/* Right Panel - Script Preview */}
          <div className="flex flex-col overflow-hidden rounded-xl border border-border bg-card">
            {/* Tab Navigation */}
            <Tabs
              value={activeTab}
              onValueChange={setActiveTab}
              className="flex h-full flex-col"
            >
              <div className="flex items-center justify-between border-b border-border px-4 py-2">
                <TabsList className="grid w-auto grid-cols-2">
                  <TabsTrigger value="editor" className="gap-2">
                    <FileText className="h-4 w-4" />
                    Editor
                  </TabsTrigger>
                  <TabsTrigger value="timeline" className="gap-2">
                    <Clock className="h-4 w-4" />
                    Timeline
                  </TabsTrigger>
                </TabsList>

                {script && (
                  <div className="flex items-center gap-1">
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={handleCopyScript}
                      title="Copy script"
                    >
                      <Copy className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={handleExportJSON}
                      title="Export as JSON"
                    >
                      <Download className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={handleResetScript}
                      title="Reset script"
                    >
                      <RefreshCw className="h-4 w-4" />
                    </Button>
                  </div>
                )}
              </div>

              <TabsContent value="editor" className="flex-1 overflow-hidden m-0">
                <ScriptEditor
                  script={script}
                  onScriptChange={handleScriptChange}
                />
              </TabsContent>

              <TabsContent value="timeline" className="flex-1 overflow-hidden m-0">
                <ScriptTimeline
                  script={script}
                  onSectionClick={handleSectionClick}
                  activeSection={activeSection}
                />
              </TabsContent>
            </Tabs>
          </div>
        </div>

        {/* Bottom Action Bar with Stats */}
        <div className="mt-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            {script && (
              <>
                <span className="text-sm text-muted-foreground">
                  {script.sections.length} sections
                </span>
                <span className="text-muted-foreground">|</span>
                <span className="text-sm text-muted-foreground">
                  {script.totalDuration} total duration
                </span>
              </>
            )}
          </div>

          <Button
            variant="outline"
            onClick={() => setActiveTab("chat")}
            disabled={!script}
          >
            <Wand2 className="mr-2 h-4 w-4" />
            Refine with AI
          </Button>
        </div>

        {/* Workflow Navigation */}
        <WorkflowNav className="mt-4" />
      </div>
    </div>
  );
}
