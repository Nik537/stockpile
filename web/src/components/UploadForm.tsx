import { useState } from 'react'
import { useDropzone } from 'react-dropzone'
import './UploadForm.css'
import { UserPreferences } from '../types'

interface UploadFormProps {
  onJobCreated: () => void
}

function UploadForm({ onJobCreated }: UploadFormProps) {
  const [uploading, setUploading] = useState(false)
  const [preferencesOpen, setPreferencesOpen] = useState(false)
  const [preferences, setPreferences] = useState<UserPreferences>({
    style: '',
    avoid: '',
    time_of_day: '',
    preferred_sources: '',
  })

  const onDrop = async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return

    const file = acceptedFiles[0]
    setUploading(true)

    try {
      const formData = new FormData()
      formData.append('file', file)

      // Add preferences if provided
      const prefsToSend: UserPreferences = {}
      if (preferences.style) prefsToSend.style = preferences.style
      if (preferences.avoid) prefsToSend.avoid = preferences.avoid
      if (preferences.time_of_day) prefsToSend.time_of_day = preferences.time_of_day
      if (preferences.preferred_sources) prefsToSend.preferred_sources = preferences.preferred_sources

      if (Object.keys(prefsToSend).length > 0) {
        formData.append('preferences', JSON.stringify(prefsToSend))
      }

      const response = await fetch('/api/process', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error('Upload failed')
      }

      await response.json()
      onJobCreated()

      // Reset form
      setPreferences({
        style: '',
        avoid: '',
        time_of_day: '',
        preferred_sources: '',
      })
    } catch (error) {
      console.error('Upload failed:', error)
      alert('Failed to upload video. Please try again.')
    } finally {
      setUploading(false)
    }
  }

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/*': ['.mp4', '.mov', '.avi', '.mkv', '.webm'],
    },
    maxFiles: 1,
    disabled: uploading,
  })

  const formats = ['MP4', 'MOV', 'AVI', 'MKV', 'WebM']

  return (
    <div className="upload-form">
      <div
        {...getRootProps()}
        className={`dropzone ${isDragActive ? 'active' : ''} ${uploading ? 'uploading' : ''}`}
      >
        <input {...getInputProps()} />
        {uploading ? (
          <div className="upload-status">
            <div className="upload-spinner">
              <div className="upload-spinner-ring"></div>
              <div className="upload-spinner-ring"></div>
              <span className="upload-spinner-icon">&#x1F3AC;</span>
            </div>
            <p className="upload-status-text">Uploading and processing...</p>
            <p className="upload-status-subtext">This may take a moment</p>
          </div>
        ) : isDragActive ? (
          <div className="drag-active-content">
            <span className="drag-active-icon">&#x1F4E5;</span>
            <p className="drag-active-text">Drop your video here!</p>
          </div>
        ) : (
          <div className="upload-prompt">
            <div className="upload-icon">&#x1F3AC;</div>
            <p className="upload-title">Drag & drop your video here</p>
            <p className="upload-subtitle">or click to browse files</p>
            <div className="upload-hint">
              <span>Supported formats:</span>
              <div className="format-list">
                {formats.map((format) => (
                  <span key={format} className="format-badge">{format}</span>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="preferences-form">
        <div
          className="preferences-header"
          onClick={() => setPreferencesOpen(!preferencesOpen)}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => e.key === 'Enter' && setPreferencesOpen(!preferencesOpen)}
        >
          <div className="preferences-header-left">
            <div className="preferences-icon">&#x2699;</div>
            <h3>B-roll Preferences</h3>
            <span className="preferences-badge">Optional</span>
          </div>
          <span className={`preferences-toggle ${preferencesOpen ? 'open' : ''}`}>&#x25BC;</span>
        </div>

        {preferencesOpen && (
          <div className="preferences-content">
            <div className="form-grid">
              <div className="form-group">
                <label htmlFor="style">
                  <span className="label-icon">&#x1F3A8;</span>
                  B-roll Style
                </label>
                <input
                  id="style"
                  type="text"
                  placeholder="e.g., cinematic, documentary, raw"
                  value={preferences.style}
                  onChange={(e) => setPreferences({ ...preferences, style: e.target.value })}
                  disabled={uploading}
                />
              </div>

              <div className="form-group">
                <label htmlFor="avoid">
                  <span className="label-icon">&#x26D4;</span>
                  Content to Avoid
                </label>
                <input
                  id="avoid"
                  type="text"
                  placeholder="e.g., text overlays, logos"
                  value={preferences.avoid}
                  onChange={(e) => setPreferences({ ...preferences, avoid: e.target.value })}
                  disabled={uploading}
                />
              </div>

              <div className="form-group">
                <label htmlFor="time_of_day">
                  <span className="label-icon">&#x1F305;</span>
                  Preferred Time of Day
                </label>
                <input
                  id="time_of_day"
                  type="text"
                  placeholder="e.g., golden hour, night"
                  value={preferences.time_of_day}
                  onChange={(e) => setPreferences({ ...preferences, time_of_day: e.target.value })}
                  disabled={uploading}
                />
              </div>

              <div className="form-group">
                <label htmlFor="preferred_sources">
                  <span className="label-icon">&#x1F4F9;</span>
                  Preferred Sources
                </label>
                <input
                  id="preferred_sources"
                  type="text"
                  placeholder="e.g., nature footage, city aerials"
                  value={preferences.preferred_sources}
                  onChange={(e) =>
                    setPreferences({ ...preferences, preferred_sources: e.target.value })
                  }
                  disabled={uploading}
                />
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default UploadForm
