import { useState, useRef, useCallback, useEffect, useMemo } from 'react';
import './PromptInput.css';

interface PromptInputProps {
  onSend: (text: string) => void;
  onStop: () => void;
  isGenerating: boolean;
  model: string;
  onModelChange: (model: string) => void;
}

const MODELS = [
  { value: 'haiku', label: 'Haiku' },
  { value: 'sonnet', label: 'Sonnet' },
  { value: 'opus', label: 'Opus' },
];

function PromptInput({ onSend, onStop, isGenerating, model, onModelChange }: PromptInputProps) {
  const [text, setText] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const wordCount = useMemo(() => {
    const trimmed = text.trim();
    if (!trimmed) return 0;
    return trimmed.split(/\s+/).length;
  }, [text]);

  const autoResize = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 200) + 'px';
  }, []);

  useEffect(() => {
    autoResize();
  }, [text, autoResize]);

  const handleSend = useCallback(() => {
    const trimmed = text.trim();
    if (!trimmed || isGenerating) return;
    onSend(trimmed);
    setText('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  }, [text, isGenerating, onSend]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="prompt-input">
      <div className="prompt-input-row">
        <textarea
          ref={textareaRef}
          className="prompt-textarea"
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask Claude anything... (Cmd+Enter to send)"
          rows={1}
          disabled={false}
        />
        <div className="prompt-actions">
          <button
            className="prompt-attach-btn"
            onClick={() => {}}
            title="Attach file (Coming soon)"
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <path
                d="M13.354 7.354l-5.5 5.5a3.182 3.182 0 01-4.5-4.5l5.5-5.5a2.121 2.121 0 013 3l-5.5 5.5a1.06 1.06 0 01-1.5-1.5l5-5"
                stroke="currentColor"
                strokeWidth="1.2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </button>
          {isGenerating ? (
            <button className="prompt-stop-btn" onClick={onStop} title="Stop generation">
              <span className="prompt-stop-icon" />
            </button>
          ) : (
            <button
              className="prompt-send-btn"
              onClick={handleSend}
              disabled={!text.trim()}
              title="Send message (Cmd+Enter)"
            >
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                <path d="M2 14L14.5 8L2 2V6.5L10 8L2 9.5V14Z" fill="currentColor" />
              </svg>
            </button>
          )}
        </div>
      </div>
      <div className="prompt-footer">
        <div className="prompt-model-segmented">
          {MODELS.map((m) => (
            <button
              key={m.value}
              className={`prompt-model-segment${model === m.value ? ' prompt-model-segment--active' : ''}`}
              onClick={() => onModelChange(m.value)}
            >
              {m.label}
            </button>
          ))}
        </div>
        <div className="prompt-footer-right">
          {wordCount > 0 && (
            <span className="prompt-word-count">
              {wordCount} {wordCount === 1 ? 'word' : 'words'}
            </span>
          )}
          <span className="prompt-hint">Cmd+Enter to send</span>
        </div>
      </div>
    </div>
  );
}

export default PromptInput;
