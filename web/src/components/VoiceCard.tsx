import './VoiceCard.css'

interface VoiceCardProps {
  voice: {
    id: string
    name: string
    is_preset: boolean
    duration_seconds: number
  }
  isSelected: boolean
  onSelect: () => void
  onDelete?: () => void
  onPreview?: () => void
  disabled?: boolean
}

function VoiceCard({ voice, isSelected, onSelect, onDelete, onPreview, disabled }: VoiceCardProps) {
  return (
    <div
      className={`voice-card ${isSelected ? 'selected' : ''} ${disabled ? 'disabled' : ''}`}
      onClick={() => !disabled && onSelect()}
    >
      <div className="voice-card-icon">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
          <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
          <line x1="12" x2="12" y1="19" y2="22"/>
        </svg>
      </div>
      <span className="voice-card-name">{voice.name}</span>
      <span className={`voice-card-badge ${voice.is_preset ? 'preset' : 'custom'}`}>
        {voice.is_preset ? 'Preset' : 'Custom'}
      </span>
      {voice.duration_seconds > 0 && (
        <span className="voice-card-duration">{voice.duration_seconds.toFixed(1)}s</span>
      )}
      <div className="voice-card-actions">
        {onPreview && voice.duration_seconds > 0 && (
          <button
            className="voice-card-btn preview"
            onClick={(e) => { e.stopPropagation(); onPreview() }}
            disabled={disabled}
            title="Preview voice"
          >
            &#x25B6;
          </button>
        )}
        {onDelete && !voice.is_preset && (
          <button
            className="voice-card-btn delete"
            onClick={(e) => { e.stopPropagation(); onDelete() }}
            disabled={disabled}
            title="Delete voice"
          >
            &#x2715;
          </button>
        )}
      </div>
    </div>
  )
}

export default VoiceCard
