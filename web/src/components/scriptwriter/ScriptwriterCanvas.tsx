import { useRef, useEffect, useCallback } from 'react';
import { isTauri } from '../../lib/tauri';
import { useScriptwriterStore } from '../../stores/useScriptwriterStore';
import { useClaudeSession } from '../../hooks/useClaudeSession';
import MessageBlock from './MessageBlock';
import PromptInput from './PromptInput';
import ConversationSidebar from './ConversationSidebar';
import ContextPanel from './ContextPanel';
import SessionPicker from './SessionPicker';
import './ScriptwriterCanvas.css';

const WORKING_DIR = '/Users/niknoavak/Desktop/YT/stockpile';
const SYSTEM_PROMPT = `You are a professional video scriptwriter. Help the user write, edit, and refine scripts for video production.

IMPORTANT: Do not use any structured response formats like SUMMARY/SCOPING/ANALYSIS/ACTIONS/RESULTS/STATUS/CAPTURE/NEXT/STORY EXPLANATION/COMPLETED. Do not follow PAI, Claude Code, or any other system framework formatting. Just respond naturally as a scriptwriter — write scripts, give feedback, and have normal conversations.

When writing scripts, use standard screenplay/script formatting with scene headings, action lines, dialogue, and timing notes where appropriate.`;

function ScriptwriterCanvas() {
  const { messages, isGenerating, sessionId, model, error, toolActivity, setModel } =
    useScriptwriterStore();
  const updateMessageContent = useScriptwriterStore((s) => s.updateMessageContent);
  const deleteMessage = useScriptwriterStore((s) => s.deleteMessage);

  const { sendMessage, stopGeneration, newSession, refineSelection, isConnected } = useClaudeSession({
    workingDir: WORKING_DIR,
    systemPrompt: SYSTEM_PROMPT,
  });

  const canvasRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    const el = canvasRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [messages]);

  const handleScrollToMessage = useCallback((messageId: string) => {
    const el = document.getElementById(`msg-${messageId}`);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, []);

  const handleResumeSession = useCallback((_sessionId: string) => {
    // For now, starting a new session; resume requires backend support
    newSession();
  }, [newSession]);

  if (!isTauri()) {
    return (
      <div className="scriptwriter-container">
        <div className="scriptwriter-web-message">
          <h3>Scriptwriter</h3>
          <p>The Scriptwriter canvas is only available in the Stockpile desktop app.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="scriptwriter-container">
      {/* Top bar */}
      <div className="scriptwriter-topbar">
        <span className="scriptwriter-topbar-title">Scriptwriter</span>
        <SessionPicker
          currentSessionId={sessionId || ''}
          onNewSession={newSession}
          onResumeSession={handleResumeSession}
        />
        <button className="scriptwriter-new-btn" onClick={newSession}>
          New Session
        </button>
      </div>

      {/* Error bar */}
      {error && (
        <div className="scriptwriter-error">{error}</div>
      )}

      {/* Three-panel layout */}
      <div className="scriptwriter-panels">
        <ConversationSidebar
          messages={messages}
          onScrollToMessage={handleScrollToMessage}
          onNewSession={newSession}
        />

        <div className="scriptwriter-main">
          <div className="scriptwriter-canvas" ref={canvasRef}>
            {messages.length === 0 ? (
              <div className="scriptwriter-empty">
                <div className="scriptwriter-empty-icon">&#9998;</div>
                <h3>Your writing canvas</h3>
                <p>Start a conversation to write, edit, or brainstorm scripts. Claude will help you craft compelling content for your videos.</p>
              </div>
            ) : (
              messages.map((msg) => (
                <MessageBlock
                  key={msg.id}
                  message={msg}
                  onDelete={deleteMessage}
                  onUpdateContent={updateMessageContent}
                  onRefineSelection={refineSelection}
                />
              ))
            )}
          </div>

          <PromptInput
            onSend={sendMessage}
            onStop={stopGeneration}
            isGenerating={isGenerating}
            model={model}
            onModelChange={setModel}
          />
        </div>

        <ContextPanel
          sessionId={sessionId || ''}
          model={model}
          isConnected={isConnected}
          systemPrompt={SYSTEM_PROMPT}
          toolCalls={toolActivity}
        />
      </div>
    </div>
  );
}

export default ScriptwriterCanvas;
