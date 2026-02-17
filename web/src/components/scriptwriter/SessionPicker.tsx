import { useState, useRef, useEffect } from 'react';

interface SessionPickerProps {
  currentSessionId: string;
  onNewSession: () => void;
  onResumeSession: (sessionId: string) => void;
}

function SessionPicker({ currentSessionId, onNewSession, onResumeSession }: SessionPickerProps) {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const savedSessions = getSavedSessions();

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  function getSavedSessions(): string[] {
    try {
      const raw = localStorage.getItem('scriptwriter-sessions');
      return raw ? JSON.parse(raw) : [];
    } catch {
      return [];
    }
  }

  const shortId = currentSessionId ? currentSessionId.slice(0, 8) + '...' : 'No session';

  return (
    <div className="session-picker" ref={dropdownRef}>
      <button
        className="session-picker-trigger"
        onClick={() => setIsOpen(!isOpen)}
        title={currentSessionId}
      >
        <span className="session-picker-label">Session:</span>
        <span className="session-picker-id">{shortId}</span>
        <span className="session-picker-chevron">{isOpen ? '\u25B2' : '\u25BC'}</span>
      </button>

      {isOpen && (
        <div className="session-picker-dropdown">
          <button
            className="session-picker-item session-picker-new"
            onClick={() => { onNewSession(); setIsOpen(false); }}
          >
            + New Session
          </button>

          {savedSessions.length > 0 && (
            <div className="session-picker-divider" />
          )}

          {savedSessions.map((sid) => (
            <button
              key={sid}
              className={`session-picker-item ${sid === currentSessionId ? 'active' : ''}`}
              onClick={() => { onResumeSession(sid); setIsOpen(false); }}
            >
              {sid.slice(0, 12)}...
              {sid === currentSessionId && <span className="session-picker-active-dot" />}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

export default SessionPicker;
