import { useCallback, useEffect, useRef, useState } from 'react'
import { Voice } from '../types'
import VoiceCard from './VoiceCard'
import './VoiceLibrary.css'

const API_BASE = ''

interface VoiceLibraryProps {
  selectedVoiceId: string | null
  onSelectVoice: (voiceId: string | null) => void
  disabled?: boolean
}

function VoiceLibrary({ selectedVoiceId, onSelectVoice, disabled }: VoiceLibraryProps) {
  const [voices, setVoices] = useState<Voice[]>([])
  const [loading, setLoading] = useState(true)
  const [showUpload, setShowUpload] = useState(false)
  const [uploadName, setUploadName] = useState('')
  const [uploadFile, setUploadFile] = useState<File | null>(null)
  const [uploading, setUploading] = useState(false)
  const audioRef = useRef<HTMLAudioElement | null>(null)

  // Fetch voices on mount
  const fetchVoices = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/tts/voices`)
      if (response.ok) {
        const data: Voice[] = await response.json()
        setVoices(data)
      }
    } catch (e) {
      console.error('Failed to fetch voices:', e)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchVoices()
  }, [fetchVoices])

  // Preview voice audio
  const handlePreview = (voiceId: string) => {
    if (audioRef.current) {
      audioRef.current.pause()
    }
    const audio = new Audio(`${API_BASE}/api/tts/voices/${voiceId}/audio`)
    audioRef.current = audio
    audio.play().catch(e => console.error('Failed to play preview:', e))
  }

  // Delete voice
  const handleDelete = async (voiceId: string) => {
    try {
      const response = await fetch(`${API_BASE}/api/tts/voices/${voiceId}`, { method: 'DELETE' })
      if (response.ok) {
        setVoices(prev => prev.filter(v => v.id !== voiceId))
        if (selectedVoiceId === voiceId) {
          onSelectVoice(null)
        }
      }
    } catch (e) {
      console.error('Failed to delete voice:', e)
    }
  }

  // Upload new voice
  const handleUpload = async () => {
    if (!uploadFile || !uploadName.trim()) return
    setUploading(true)

    try {
      const formData = new FormData()
      formData.append('name', uploadName.trim())
      formData.append('audio', uploadFile)

      const response = await fetch(`${API_BASE}/api/tts/voices`, {
        method: 'POST',
        body: formData,
      })

      if (response.ok) {
        const newVoice: Voice = await response.json()
        setVoices(prev => [...prev, newVoice])
        setShowUpload(false)
        setUploadName('')
        setUploadFile(null)
        onSelectVoice(newVoice.id)
      }
    } catch (e) {
      console.error('Failed to upload voice:', e)
    } finally {
      setUploading(false)
    }
  }

  if (loading) {
    return <div className="voice-library loading">Loading voices...</div>
  }

  return (
    <div className="voice-library">
      <div className="voice-library-header">
        <h3>Choose a Voice</h3>
        <span className="voice-count">{voices.length} voices</span>
      </div>

      <div className="voice-library-scroll">
        {/* No voice selected = default (no cloning) */}
        <div
          className={`voice-card ${selectedVoiceId === null ? 'selected' : ''} ${disabled ? 'disabled' : ''}`}
          onClick={() => !disabled && onSelectVoice(null)}
        >
          <div className="voice-card-icon default-icon">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10"/>
              <path d="M8 12h8M12 8v8"/>
            </svg>
          </div>
          <span className="voice-card-name">No Clone</span>
          <span className="voice-card-badge preset">Default</span>
        </div>

        {voices.map(voice => (
          <VoiceCard
            key={voice.id}
            voice={voice}
            isSelected={selectedVoiceId === voice.id}
            onSelect={() => onSelectVoice(voice.id)}
            onDelete={() => handleDelete(voice.id)}
            onPreview={() => handlePreview(voice.id)}
            disabled={disabled}
          />
        ))}

        {/* Add Voice card */}
        <div
          className={`voice-card add-voice-card ${disabled ? 'disabled' : ''}`}
          onClick={() => !disabled && setShowUpload(true)}
        >
          <div className="voice-card-icon add-icon">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="12" x2="12" y1="5" y2="19"/>
              <line x1="5" x2="19" y1="12" y2="12"/>
            </svg>
          </div>
          <span className="voice-card-name">Add Voice</span>
        </div>
      </div>

      {/* Upload Modal */}
      {showUpload && (
        <div className="voice-upload-overlay" onClick={() => setShowUpload(false)}>
          <div className="voice-upload-modal" onClick={e => e.stopPropagation()}>
            <h3>Add New Voice</h3>
            <p className="upload-hint">Upload 5-10 seconds of clear speech for voice cloning</p>

            <div className="upload-form">
              <div className="form-group">
                <label htmlFor="voice-name">Voice Name</label>
                <input
                  id="voice-name"
                  type="text"
                  value={uploadName}
                  onChange={e => setUploadName(e.target.value)}
                  placeholder="e.g., My Voice, Deep Narrator"
                  disabled={uploading}
                />
              </div>

              <div className="form-group">
                <label>Audio File</label>
                {uploadFile ? (
                  <div className="upload-file-preview">
                    <span>{uploadFile.name}</span>
                    <span className="file-size">({(uploadFile.size / 1024).toFixed(1)} KB)</span>
                    <button onClick={() => setUploadFile(null)} disabled={uploading}>Remove</button>
                  </div>
                ) : (
                  <label className="upload-dropzone">
                    <input
                      type="file"
                      accept="audio/*,.mp3,.wav,.ogg"
                      onChange={e => setUploadFile(e.target.files?.[0] || null)}
                      style={{ display: 'none' }}
                      disabled={uploading}
                    />
                    <span>Click or drag to upload audio</span>
                    <span className="dropzone-hint">MP3, WAV, or OGG</span>
                  </label>
                )}
              </div>
            </div>

            <div className="upload-actions">
              <button
                className="btn-cancel"
                onClick={() => { setShowUpload(false); setUploadName(''); setUploadFile(null) }}
                disabled={uploading}
              >
                Cancel
              </button>
              <button
                className="btn-save-voice"
                onClick={handleUpload}
                disabled={uploading || !uploadName.trim() || !uploadFile}
              >
                {uploading ? 'Saving...' : 'Save Voice'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default VoiceLibrary
