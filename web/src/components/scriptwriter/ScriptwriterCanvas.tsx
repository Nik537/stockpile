import { useRef, useEffect, useCallback, useState } from 'react';
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
  const [showScrollBtn, setShowScrollBtn] = useState(false);
  const [unreadCount, setUnreadCount] = useState(0);
  const prevMessageCountRef = useRef(messages.length);

  const handleCanvasScroll = useCallback(() => {
    const el = canvasRef.current;
    if (!el) return;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    setShowScrollBtn(distanceFromBottom > 200);
  }, []);

  const scrollToBottom = useCallback(() => {
    const el = canvasRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
    setUnreadCount(0);
  }, []);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    const el = canvasRef.current;
    if (!el) return;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    const isNearBottom = distanceFromBottom < 200;

    if (isNearBottom) {
      el.scrollTop = el.scrollHeight;
      setUnreadCount(0);
    } else if (messages.length > prevMessageCountRef.current) {
      setUnreadCount((c) => c + (messages.length - prevMessageCountRef.current));
    }
    prevMessageCountRef.current = messages.length;
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
          <div className="scriptwriter-canvas" ref={canvasRef} onScroll={handleCanvasScroll}>
            {messages.length === 0 ? (
              <div className="scriptwriter-empty">
                <div className="scriptwriter-empty-orb" />
                <div className="scriptwriter-empty-icon">
                  <svg width="48" height="48" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <defs>
                      <linearGradient id="penGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stopColor="var(--color-primary-600)" />
                        <stop offset="100%" stopColor="var(--color-accent)" />
                      </linearGradient>
                    </defs>
                    <path d="M33.6 6.4a4 4 0 0 1 5.66 0l2.34 2.34a4 4 0 0 1 0 5.66L16.4 39.6a2 2 0 0 1-.9.52l-8 2.4a1 1 0 0 1-1.22-1.22l2.4-8a2 2 0 0 1 .52-.9L33.6 6.4Z" stroke="url(#penGradient)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" fill="none" />
                    <path d="M28 12l8 8" stroke="url(#penGradient)" strokeWidth="2.5" strokeLinecap="round" />
                  </svg>
                </div>
                <h3>Your writing canvas</h3>
                <p>Start a conversation to write, edit, or brainstorm scripts. Claude will help you craft compelling content for your videos.</p>
                <div className="scriptwriter-quickstart-grid">
                  {[
                    { icon: '\uD83C\uDFAC', text: 'Write a YouTube script' },
                    { icon: '\uD83D\uDCA1', text: 'Brainstorm video ideas' },
                    { icon: '\uD83D\uDCDD', text: 'Create a script outline' },
                    { icon: '\u2728', text: 'Edit and refine a draft' },
                  ].map((item) => (
                    <button
                      key={item.text}
                      className="scriptwriter-quickstart-card"
                      onClick={() => sendMessage(item.text)}
                    >
                      <span className="scriptwriter-quickstart-icon">{item.icon}</span>
                      <span className="scriptwriter-quickstart-text">{item.text}</span>
                      <span className="scriptwriter-quickstart-arrow">&rarr;</span>
                    </button>
                  ))}
                </div>
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

          {showScrollBtn && (
            <div className="scriptwriter-scroll-bottom">
              <button
                className="scriptwriter-scroll-bottom-btn"
                onClick={scrollToBottom}
                title="Scroll to bottom"
              >
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                  <path d="M4 6l4 4 4-4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
                {unreadCount > 0 && (
                  <span className="scriptwriter-scroll-bottom-badge">{unreadCount}</span>
                )}
              </button>
            </div>
          )}

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
