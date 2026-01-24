import { useState } from 'react'
import { useDropzone } from 'react-dropzone'
import './UploadForm.css'
import { UserPreferences } from '../types'

interface UploadFormProps {
  onJobCreated: (jobId: string) => void
}

function UploadForm({ onJobCreated }: UploadFormProps) {
  const [uploading, setUploading] = useState(false)
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

      const data = await response.json()
      onJobCreated(data.job_id)

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

  return (
    <div className="upload-form">
      <div
        {...getRootProps()}
        className={`dropzone ${isDragActive ? 'active' : ''} ${uploading ? 'uploading' : ''}`}
      >
        <input {...getInputProps()} />
        {uploading ? (
          <div className="upload-status">
            <div className="spinner"></div>
            <p>Uploading and processing...</p>
          </div>
        ) : isDragActive ? (
          <p>Drop the video here...</p>
        ) : (
          <div className="upload-prompt">
            <p>📹 Drag & drop a video file here, or click to select</p>
            <p className="upload-hint">Supported formats: MP4, MOV, AVI, MKV, WebM</p>
          </div>
        )}
      </div>

      <div className="preferences-form">
        <h3>User Preferences (Optional)</h3>
        <div className="form-grid">
          <div className="form-group">
            <label htmlFor="style">B-roll Style</label>
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
            <label htmlFor="avoid">Content to Avoid</label>
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
            <label htmlFor="time_of_day">Preferred Time of Day</label>
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
            <label htmlFor="preferred_sources">Preferred Sources</label>
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
    </div>
  )
}

export default UploadForm
