import { useState, useMemo } from 'react';
import type { Message } from '../../lib/claude-messages';
import './ConversationSidebar.css';

interface ConversationSidebarProps {
  messages: Message[];
  onScrollToMessage: (messageId: string) => void;
  onNewSession: () => void;
  currentMessageId?: string;
}

function formatTime(timestamp: number): string {
  const date = new Date(timestamp);
  return date.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
}

function ConversationSidebar({ messages, onScrollToMessage, onNewSession, currentMessageId }: ConversationSidebarProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [collapsed, setCollapsed] = useState(false);

  const userMessages = messages.filter((m) => m.role === 'user');

  const filteredMessages = useMemo(() => {
    if (!searchQuery.trim()) return userMessages;
    const query = searchQuery.toLowerCase();
    return userMessages.filter((msg) =>
      msg.content.toLowerCase().includes(query)
    );
  }, [userMessages, searchQuery]);

  if (collapsed) {
    return (
      <div className="conv-sidebar conv-sidebar--collapsed">
        <div className="conv-sidebar-header">
          <button
            className="conv-sidebar-collapse-btn"
            onClick={() => setCollapsed(false)}
            title="Expand sidebar"
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <path d="M6 3l5 5-5 5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </button>
        </div>
        <div className="conv-sidebar-list">
          {userMessages.map((msg, index) => (
            <button
              key={msg.id}
              className={`conv-sidebar-item-icon ${currentMessageId === msg.id ? 'conv-sidebar-item--active' : ''}`}
              onClick={() => onScrollToMessage(msg.id)}
              title={msg.content}
            >
              <span className="conv-sidebar-index">{index + 1}</span>
            </button>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="conv-sidebar">
      <div className="conv-sidebar-header">
        <span className="conv-sidebar-title">Conversations</span>
        <div className="conv-sidebar-header-actions">
          <button className="conv-sidebar-new" onClick={onNewSession} title="New Session">
            +
          </button>
          <button
            className="conv-sidebar-collapse-btn"
            onClick={() => setCollapsed(true)}
            title="Collapse sidebar"
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <path d="M10 3l-5 5 5 5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </button>
        </div>
      </div>

      <div className="conv-sidebar-search">
        <svg className="conv-sidebar-search-icon" width="14" height="14" viewBox="0 0 16 16" fill="none">
          <circle cx="7" cy="7" r="5.5" stroke="currentColor" strokeWidth="1.5"/>
          <path d="M11 11l3.5 3.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
        </svg>
        <input
          type="text"
          className="conv-sidebar-search-input"
          placeholder="Search messages..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
        />
      </div>

      <div className="conv-sidebar-list">
        {filteredMessages.length === 0 && userMessages.length === 0 ? (
          <div className="conv-sidebar-empty">
            <svg className="conv-sidebar-empty-icon" width="24" height="24" viewBox="0 0 24 24" fill="none">
              <path d="M17 3a2.83 2.83 0 0 1 4 4L7.5 20.5 2 22l1.5-5.5L17 3z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            <span>Start writing to see your conversation here</span>
          </div>
        ) : filteredMessages.length === 0 ? (
          <div className="conv-sidebar-empty">
            <span>No matching messages</span>
          </div>
        ) : (
          filteredMessages.map((msg) => {
            const preview = msg.content.length > 40
              ? msg.content.slice(0, 40) + '...'
              : msg.content;

            const originalIndex = userMessages.indexOf(msg);

            return (
              <button
                key={msg.id}
                className={`conv-sidebar-item ${currentMessageId === msg.id ? 'conv-sidebar-item--active' : ''}`}
                onClick={() => onScrollToMessage(msg.id)}
                title={msg.content}
              >
                <span className="conv-sidebar-index">{originalIndex + 1}</span>
                <span className="conv-sidebar-preview">{preview}</span>
                <span className="conv-sidebar-time">{formatTime(msg.timestamp)}</span>
              </button>
            );
          })
        )}
      </div>
    </div>
  );
}

export default ConversationSidebar;
