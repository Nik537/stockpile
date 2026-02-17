import { useState, useRef, useEffect, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { Message, ToolCall } from '../../lib/claude-messages';
import StreamingCursor from './StreamingCursor';
import './MessageBlock.css';

interface MessageBlockProps {
  message: Message;
  onDelete?: (id: string) => void;
  onUpdateContent?: (id: string, content: string) => void;
  onRefineSelection?: (messageId: string, selectedText: string) => void;
}

function ToolCallBlock({ toolCall }: { toolCall: ToolCall }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className={`msg-tool msg-tool--${toolCall.status}`}>
      <button className="msg-tool-header" onClick={() => setExpanded(!expanded)}>
        <span className={`msg-tool-status-icon msg-tool-status-icon--${toolCall.status}`}>
          {toolCall.status === 'running' && <span className="msg-tool-spinner" />}
          {toolCall.status === 'completed' && '\u2713'}
          {toolCall.status === 'error' && '\u2717'}
        </span>
        <span className="msg-tool-name">{toolCall.name}</span>
        <span className="msg-tool-chevron">{expanded ? '\u25B2' : '\u25BC'}</span>
      </button>
      {expanded && (
        <div className="msg-tool-body">
          <pre className="msg-tool-code">{JSON.stringify(toolCall.input, null, 2)}</pre>
          {toolCall.output && (
            <>
              <div className="msg-tool-divider" />
              <pre className="msg-tool-code msg-tool-output">
                {toolCall.output.length > 1000
                  ? toolCall.output.slice(0, 1000) + '\n... (truncated)'
                  : toolCall.output}
              </pre>
            </>
          )}
        </div>
      )}
    </div>
  );
}

function MessageBlock({ message, onDelete, onUpdateContent, onRefineSelection }: MessageBlockProps) {
  const isUser = message.role === 'user';
  const time = new Date(message.timestamp).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
  });

  const [isEditing, setIsEditing] = useState(false);
  const [editContent, setEditContent] = useState('');
  const [copied, setCopied] = useState(false);
  const [selectionToolbar, setSelectionToolbar] = useState<{ x: number; y: number; text: string } | null>(null);
  const blockRef = useRef<HTMLDivElement>(null);
  const editTextareaRef = useRef<HTMLTextAreaElement>(null);

  const isThinking = !isUser && message.isStreaming && !message.content;

  // Auto-focus and auto-size textarea when entering edit mode
  useEffect(() => {
    if (isEditing && editTextareaRef.current) {
      editTextareaRef.current.focus();
      editTextareaRef.current.style.height = 'auto';
      editTextareaRef.current.style.height = editTextareaRef.current.scrollHeight + 'px';
    }
  }, [isEditing]);

  const handleEdit = useCallback(() => {
    setEditContent(message.content);
    setIsEditing(true);
  }, [message.content]);

  const handleSave = useCallback(() => {
    if (onUpdateContent) {
      onUpdateContent(message.id, editContent);
    }
    setIsEditing(false);
  }, [message.id, editContent, onUpdateContent]);

  const handleCancel = useCallback(() => {
    setIsEditing(false);
  }, []);

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [message.content]);

  const handleDelete = useCallback(() => {
    if (onDelete) {
      onDelete(message.id);
    }
  }, [message.id, onDelete]);

  // Text selection handling
  const handleMouseUp = useCallback(() => {
    const selection = window.getSelection();
    if (!selection || selection.isCollapsed || !blockRef.current) {
      setSelectionToolbar(null);
      return;
    }

    const text = selection.toString().trim();
    if (!text) {
      setSelectionToolbar(null);
      return;
    }

    const range = selection.getRangeAt(0);
    const rect = range.getBoundingClientRect();
    const blockRect = blockRef.current.getBoundingClientRect();

    setSelectionToolbar({
      x: rect.left - blockRect.left + rect.width / 2,
      y: rect.top - blockRect.top - 8,
      text,
    });
  }, []);

  // Clear selection toolbar on click outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (blockRef.current && !blockRef.current.contains(e.target as Node)) {
        setSelectionToolbar(null);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleRefine = useCallback(() => {
    if (selectionToolbar && onRefineSelection) {
      onRefineSelection(message.id, selectionToolbar.text);
      setSelectionToolbar(null);
    }
  }, [selectionToolbar, message.id, onRefineSelection]);

  const handleCopySelection = useCallback(() => {
    if (selectionToolbar) {
      navigator.clipboard.writeText(selectionToolbar.text);
      setSelectionToolbar(null);
    }
  }, [selectionToolbar]);

  return (
    <div
      className={`msg-block msg-block--${message.role}`}
      id={`msg-${message.id}`}
      ref={blockRef}
    >
      {/* Role indicator */}
      <div className="msg-role-label">
        <span className={`msg-role-dot msg-role-dot--${message.role}`} />
        <span className="msg-role-name">{isUser ? 'You' : 'Claude'}</span>
      </div>

      <div className="msg-bubble-wrapper">
        {/* Action buttons on hover - positioned on the right side */}
        <div className="msg-actions">
          <button className="msg-action-btn" onClick={handleEdit} title="Edit">
            &#9998;
          </button>
          <button
            className={`msg-action-btn ${copied ? 'msg-action-btn--copied' : ''}`}
            onClick={handleCopy}
            title={copied ? 'Copied!' : 'Copy'}
          >
            {copied ? '\u2713' : '\u{1F4CB}'}
          </button>
          <button className="msg-action-btn msg-action-btn--delete" onClick={handleDelete} title="Delete">
            &#128465;
          </button>
        </div>

        {/* Selection toolbar */}
        {selectionToolbar && (
          <div
            className="msg-selection-toolbar"
            style={{
              left: selectionToolbar.x,
              top: selectionToolbar.y,
              transform: 'translate(-50%, -100%)',
            }}
          >
            <button className="msg-selection-btn msg-selection-btn--refine" onClick={handleRefine}>
              Refine with AI
            </button>
            <button className="msg-selection-btn" onClick={handleCopySelection}>
              Copy
            </button>
          </div>
        )}

        <div className="msg-bubble" onMouseUp={handleMouseUp}>
          {isEditing ? (
            <div className="msg-edit-container">
              <textarea
                ref={editTextareaRef}
                className="msg-edit-textarea"
                value={editContent}
                onChange={(e) => {
                  setEditContent(e.target.value);
                  e.target.style.height = 'auto';
                  e.target.style.height = e.target.scrollHeight + 'px';
                }}
              />
              <div className="msg-edit-actions">
                <button className="msg-edit-btn msg-edit-btn--save" onClick={handleSave}>
                  Save
                </button>
                <button className="msg-edit-btn msg-edit-btn--cancel" onClick={handleCancel}>
                  Cancel
                </button>
              </div>
            </div>
          ) : isThinking ? (
            <div className="msg-markdown">
              <StreamingCursor isThinking />
            </div>
          ) : isUser ? (
            <div className="msg-text">{message.content}</div>
          ) : (
            <div className="msg-markdown">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {message.content}
              </ReactMarkdown>
              {message.isStreaming && <StreamingCursor />}
            </div>
          )}

          {message.toolCalls && message.toolCalls.length > 0 && (
            <div className="msg-tools">
              {message.toolCalls.map((tc) => (
                <ToolCallBlock key={tc.id} toolCall={tc} />
              ))}
            </div>
          )}
        </div>
      </div>
      <span className="msg-time">{time}</span>
    </div>
  );
}

export default MessageBlock;
