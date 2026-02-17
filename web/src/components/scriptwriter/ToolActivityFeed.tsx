import { useState } from 'react';
import type { ToolCall } from '../../lib/claude-messages';
import './ToolActivityFeed.css';

interface ToolActivityFeedProps {
  toolCalls: ToolCall[];
}

function getToolIcon(name: string): string {
  const lower = name.toLowerCase();
  if (lower.includes('bash') || lower.includes('terminal')) return '\u25B6';
  if (lower.includes('read') || lower.includes('edit') || lower.includes('write')) return '\u2B1C';
  if (lower.includes('grep') || lower.includes('glob') || lower.includes('search')) return '\u{1F50D}';
  return '\u2699';
}

function ToolActivityFeed({ toolCalls }: ToolActivityFeedProps) {
  const [expandedId, setExpandedId] = useState<string | null>(null);

  if (toolCalls.length === 0) {
    return (
      <div className="tool-feed-empty">
        No tool activity yet
      </div>
    );
  }

  return (
    <div className="tool-feed">
      {toolCalls.map((tc) => {
        const isExpanded = expandedId === tc.id;
        const inputSummary = tc.input
          ? Object.values(tc.input).join(' ').slice(0, 60)
          : '';

        return (
          <div key={tc.id} className="tool-feed-item">
            <button
              className="tool-feed-header"
              onClick={() => setExpandedId(isExpanded ? null : tc.id)}
            >
              <span className="tool-feed-icon">{getToolIcon(tc.name)}</span>
              <span className="tool-feed-name">{tc.name}</span>
              <span className="tool-feed-summary">{inputSummary}</span>
              <span className={`tool-feed-status tool-feed-status--${tc.status}`}>
                {tc.status === 'running' && <span className="tool-feed-spinner" />}
                {tc.status === 'completed' && '\u2713'}
                {tc.status === 'error' && '\u2717'}
              </span>
            </button>

            {isExpanded && (
              <div className="tool-feed-detail">
                <div className="tool-feed-section">
                  <span className="tool-feed-section-label">Input</span>
                  <pre className="tool-feed-code">
                    {JSON.stringify(tc.input, null, 2)}
                  </pre>
                </div>
                {tc.output && (
                  <div className="tool-feed-section">
                    <span className="tool-feed-section-label">Output</span>
                    <pre className="tool-feed-code">
                      {tc.output.length > 500 ? tc.output.slice(0, 500) + '...' : tc.output}
                    </pre>
                  </div>
                )}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

export default ToolActivityFeed;
