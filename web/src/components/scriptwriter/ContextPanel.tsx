import { useState, useMemo } from 'react';
import type { ToolCall } from '../../lib/claude-messages';
import { useScriptwriterStore } from '../../stores/useScriptwriterStore';
import ToolActivityFeed from './ToolActivityFeed';
import './ContextPanel.css';

interface ContextPanelProps {
  sessionId: string;
  model: string;
  isConnected: boolean;
  systemPrompt: string;
  toolCalls: ToolCall[];
}

function formatDuration(ms: number): string {
  const seconds = Math.floor(ms / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);

  if (hours > 0) {
    return `${hours}h ${minutes % 60}m`;
  }
  if (minutes > 0) {
    return `${minutes}m ${seconds % 60}s`;
  }
  return `${seconds}s`;
}

function countWords(text: string): number {
  return text.trim().split(/\s+/).filter(Boolean).length;
}

function ContextPanel({ sessionId, model, isConnected, systemPrompt, toolCalls }: ContextPanelProps) {
  const [collapsed, setCollapsed] = useState(false);
  const [showSystemPrompt, setShowSystemPrompt] = useState(false);

  const messages = useScriptwriterStore((s) => s.messages);

  const metrics = useMemo(() => {
    const userMessages = messages.filter((m) => m.role === 'user');
    const assistantMessages = messages.filter((m) => m.role === 'assistant');
    const totalWords = messages.reduce((sum, m) => sum + countWords(m.content), 0);

    let sessionDuration = 0;
    if (messages.length > 0) {
      const firstTimestamp = messages[0].timestamp;
      const now = Date.now();
      sessionDuration = now - firstTimestamp;
    }

    return {
      totalWords,
      userCount: userMessages.length,
      assistantCount: assistantMessages.length,
      totalCount: messages.length,
      sessionDuration,
    };
  }, [messages]);

  return (
    <div className={`context-panel ${collapsed ? 'context-panel--collapsed' : ''}`}>
      {collapsed ? (
        <button
          className="context-panel-toggle"
          onClick={() => setCollapsed(false)}
          title="Expand panel"
        >
          &laquo;
        </button>
      ) : (
        <>
          <div className="context-panel-header">
            <span className="context-panel-title">Context</span>
            <button
              className="context-panel-toggle"
              onClick={() => setCollapsed(true)}
              title="Collapse panel"
            >
              &raquo;
            </button>
          </div>

          <div className="context-panel-body">
            {/* Session Info */}
            <div className="context-section">
              <div className="context-section-label">Session</div>
              <div className="context-session-info">
                <div className="context-row">
                  <span className="context-key">ID</span>
                  <span className="context-value" title={sessionId}>
                    {sessionId ? sessionId.slice(0, 12) + '...' : 'None'}
                  </span>
                </div>
                <div className="context-row">
                  <span className="context-key">Model</span>
                  <span className="context-value">{model || 'sonnet'}</span>
                </div>
                <div className="context-row">
                  <span className="context-key">Status</span>
                  <span className="context-value">
                    <span className={`context-status-dot ${isConnected ? 'connected' : 'disconnected'}`} />
                    {isConnected ? 'Connected' : 'Disconnected'}
                  </span>
                </div>
              </div>
            </div>

            {/* Script Metrics */}
            {metrics.totalCount > 0 && (
              <div className="context-section">
                <div className="context-section-label">Script Metrics</div>
                <div className="context-metrics">
                  <div className="context-metric">
                    <span className="context-metric-value">{metrics.totalWords.toLocaleString()}</span>
                    <span className="context-metric-label">words</span>
                  </div>
                  <div className="context-metric">
                    <span className="context-metric-value">{metrics.userCount}/{metrics.assistantCount}</span>
                    <span className="context-metric-label">user/ai msgs</span>
                  </div>
                  <div className="context-metric">
                    <span className="context-metric-value">{formatDuration(metrics.sessionDuration)}</span>
                    <span className="context-metric-label">duration</span>
                  </div>
                </div>
              </div>
            )}

            {/* System Prompt - collapsed by default */}
            {systemPrompt && (
              <div className="context-section">
                <button
                  className="context-section-label context-section-label--toggle"
                  onClick={() => setShowSystemPrompt(!showSystemPrompt)}
                >
                  <span>System Prompt</span>
                  <svg
                    className={`context-toggle-chevron ${showSystemPrompt ? 'context-toggle-chevron--open' : ''}`}
                    width="12"
                    height="12"
                    viewBox="0 0 16 16"
                    fill="none"
                  >
                    <path d="M6 4l4 4-4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                </button>
                {showSystemPrompt && (
                  <div className="context-prompt-display">
                    {systemPrompt.length > 200
                      ? systemPrompt.slice(0, 200) + '...'
                      : systemPrompt}
                  </div>
                )}
              </div>
            )}

            {/* Tool Activity */}
            <div className="context-section context-section--tools">
              <div className="context-section-label">
                Tool Activity
                {toolCalls.length > 0 && (
                  <span className="context-tool-count">{toolCalls.length}</span>
                )}
              </div>
              <ToolActivityFeed toolCalls={toolCalls} />
            </div>
          </div>
        </>
      )}
    </div>
  );
}

export default ContextPanel;
