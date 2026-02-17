import { useState, useRef, useCallback, useEffect } from 'react';
import './PromptInput.css';

interface PromptInputProps {
  onSend: (text: string) => void;
  onStop: () => void;
  isGenerating: boolean;
  model: string;
  onModelChange: (model: string) => void;
}

const MODELS = [
  { value: 'sonnet', label: 'Sonnet' },
  { value: 'opus', label: 'Opus' },
  { value: 'haiku', label: 'Haiku' },
];

function PromptInput({ onSend, onStop, isGenerating, model, onModelChange }: PromptInputProps) {
  const [text, setText] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

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
        <select
          className="prompt-model-picker"
          value={model}
          onChange={(e) => onModelChange(e.target.value)}
        >
          {MODELS.map((m) => (
            <option key={m.value} value={m.value}>{m.label}</option>
          ))}
        </select>
        <span className="prompt-hint">Cmd+Enter to send</span>
      </div>
    </div>
  );
}

export default PromptInput;
