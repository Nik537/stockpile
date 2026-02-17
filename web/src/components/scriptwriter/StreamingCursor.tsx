import './StreamingCursor.css';

interface StreamingCursorProps {
  isThinking?: boolean;
}

function StreamingCursor({ isThinking }: StreamingCursorProps) {
  if (isThinking) {
    return (
      <span className="streaming-thinking">
        <span className="streaming-thinking-text">Claude is thinking</span>
        <span className="streaming-thinking-dots">
          <span className="streaming-thinking-dot" />
          <span className="streaming-thinking-dot" />
          <span className="streaming-thinking-dot" />
        </span>
      </span>
    );
  }

  return <span className="streaming-cursor">|</span>;
}

export default StreamingCursor;
