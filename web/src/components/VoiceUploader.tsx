import { useCallback, useRef, useState } from 'react'
import './VoiceUploader.css'

interface VoiceUploaderProps {
  voiceFile: File | null
  onVoiceChange: (file: File | null) => void
  disabled?: boolean
}

function VoiceUploader({ voiceFile, onVoiceChange, disabled }: VoiceUploaderProps) {
  const [isDragging, setIsDragging] = useState(false)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFile = useCallback(
    (file: File | null) => {
      if (!file) {
        onVoiceChange(null)
        setPreviewUrl(null)
        return
      }

      // Validate file type
      const validTypes = ['audio/mpeg', 'audio/wav', 'audio/mp3', 'audio/x-wav', 'audio/ogg']
      if (!validTypes.includes(file.type) && !file.name.match(/\.(mp3|wav|ogg)$/i)) {
        alert('Please upload an audio file (MP3, WAV, or OGG)')
        return
      }

      // Create preview URL
      const url = URL.createObjectURL(file)
      setPreviewUrl(url)
      onVoiceChange(file)
    },
    [onVoiceChange]
  )

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setIsDragging(false)

      if (disabled) return

      const file = e.dataTransfer.files[0]
      if (file) {
        handleFile(file)
      }
    },
    [disabled, handleFile]
  )

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }, [])

  const handleClick = () => {
    if (!disabled) {
      fileInputRef.current?.click()
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] || null
    handleFile(file)
  }

  const handleRemove = (e: React.MouseEvent) => {
    e.stopPropagation()
    handleFile(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  return (
    <div className="voice-uploader">
      <label className="voice-label">
        Voice Reference (optional)
        <span className="voice-hint">5-10 seconds of clear speech for voice cloning</span>
      </label>

      {voiceFile ? (
        <div className="voice-preview">
          <div className="voice-info">
            <span className="voice-icon">&#x1F3A4;</span>
            <div className="voice-details">
              <span className="voice-name">{voiceFile.name}</span>
              <span className="voice-size">
                {(voiceFile.size / 1024).toFixed(1)} KB
              </span>
            </div>
            <button
              onClick={handleRemove}
              className="btn-remove-voice"
              disabled={disabled}
              title="Remove voice file"
            >
              &#x2715;
            </button>
          </div>
          {previewUrl && (
            <audio controls src={previewUrl} className="voice-audio-preview">
              Your browser does not support the audio element.
            </audio>
          )}
        </div>
      ) : (
        <div
          className={`voice-dropzone ${isDragging ? 'dragging' : ''} ${disabled ? 'disabled' : ''}`}
          onClick={handleClick}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
        >
          <span className="dropzone-icon">&#x1F3A4;</span>
          <span className="dropzone-text">
            Drop an audio file here or click to upload
          </span>
          <span className="dropzone-hint">
            MP3, WAV, or OGG (5-10 seconds recommended)
          </span>
        </div>
      )}

      <input
        ref={fileInputRef}
        type="file"
        accept="audio/*,.mp3,.wav,.ogg"
        onChange={handleFileChange}
        style={{ display: 'none' }}
        disabled={disabled}
      />
    </div>
  )
}

export default VoiceUploader
