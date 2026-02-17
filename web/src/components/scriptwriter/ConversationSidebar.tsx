import type { Message } from '../../lib/claude-messages';
import './ConversationSidebar.css';

interface ConversationSidebarProps {
  messages: Message[];
  onScrollToMessage: (messageId: string) => void;
  onNewSession: () => void;
}

function ConversationSidebar({ messages, onScrollToMessage, onNewSession }: ConversationSidebarProps) {
  const userMessages = messages.filter((m) => m.role === 'user');

  return (
    <div className="conv-sidebar">
      <div className="conv-sidebar-header">
        <span className="conv-sidebar-title">Conversations</span>
        <button className="conv-sidebar-new" onClick={onNewSession} title="New Session">
          +
        </button>
      </div>

      <div className="conv-sidebar-list">
        {userMessages.length === 0 ? (
          <div className="conv-sidebar-empty">No messages yet</div>
        ) : (
          userMessages.map((msg, index) => {
            const preview = msg.content.length > 40
              ? msg.content.slice(0, 40) + '...'
              : msg.content;

            return (
              <button
                key={msg.id}
                className="conv-sidebar-item"
                onClick={() => onScrollToMessage(msg.id)}
                title={msg.content}
              >
                <span className="conv-sidebar-index">{index + 1}</span>
                <span className="conv-sidebar-preview">{preview}</span>
              </button>
            );
          })
        )}
      </div>
    </div>
  );
}

export default ConversationSidebar;
