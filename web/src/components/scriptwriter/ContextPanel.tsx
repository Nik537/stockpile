import { useState } from 'react';
import type { ToolCall } from '../../lib/claude-messages';
import ToolActivityFeed from './ToolActivityFeed';
import './ContextPanel.css';

interface ContextPanelProps {
  sessionId: string;
  model: string;
  isConnected: boolean;
  systemPrompt: string;
  toolCalls: ToolCall[];
}

function ContextPanel({ sessionId, model, isConnected, systemPrompt, toolCalls }: ContextPanelProps) {
  const [collapsed, setCollapsed] = useState(false);

  if (collapsed) {
    return (
      <div className="context-panel context-panel--collapsed">
        <button
          className="context-panel-toggle"
          onClick={() => setCollapsed(false)}
          title="Expand panel"
        >
          &laquo;
        </button>
      </div>
    );
  }

  return (
    <div className="context-panel">
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

        {/* System Prompt */}
        {systemPrompt && (
          <div className="context-section">
            <div className="context-section-label">System Prompt</div>
            <div className="context-prompt-display">
              {systemPrompt.length > 200
                ? systemPrompt.slice(0, 200) + '...'
                : systemPrompt}
            </div>
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
    </div>
  );
}

export default ContextPanel;
