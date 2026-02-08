import { useCallback, useEffect, useState } from 'react'
import { TTSStatus } from '../types'
import VoiceLibrary from './VoiceLibrary'
import './TTSGenerator.css'

const API_BASE = ''  // Same origin

interface RunpodStatus {
  configured: boolean
  available: boolean
  endpoint_id?: string
  error?: string
}

function TTSGenerator() {
  // Voice selection state
  const [selectedVoiceId, setSelectedVoiceId] = useState<string | null>(null)

  // RunPod status
  const [runpodStatus, setRunpodStatus] = useState<RunpodStatus | null>(null)

  // Generation state
  const [text, setText] = useState('')
  const [status, setStatus] = useState<TTSStatus>('idle')
  const [error, setError] = useState<string | null>(null)

  // Parameters
  const [exaggeration, setExaggeration] = useState(0.5)
  const [cfgWeight, setCfgWeight] = useState(0.5)
  const [temperature, setTemperature] = useState(0.8)

  // Result
  const [audioUrl, setAudioUrl] = useState<string | null>(null)
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null)

  // Check RunPod status on mount
  const checkStatus = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/tts/status`)
      const data = await response.json()
      setRunpodStatus(data.runpod)
    } catch (e) {
      console.error('Failed to check TTS status:', e)
    }
  }, [])

  useEffect(() => {
    checkStatus()
  }, [checkStatus])

  // Generate TTS
  const handleGenerate = async () => {
    if (!text.trim()) {
      setError('Please enter some text')
      return
    }

    if (!runpodStatus?.configured || !runpodStatus?.available) {
      setError('RunPod is not configured. Check your .env settings.')
      return
    }

    setStatus('generating')
    setError(null)
    setAudioUrl(null)
    setAudioBlob(null)

    try {
      const formData = new FormData()
      formData.append('text', text.trim())
      formData.append('mode', 'runpod')
      formData.append('exaggeration', exaggeration.toString())
      formData.append('cfg_weight', cfgWeight.toString())
      formData.append('temperature', temperature.toString())

      if (selectedVoiceId) {
        formData.append('voice_id', selectedVoiceId)
      }

      const response = await fetch(`${API_BASE}/api/tts/generate`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Generation failed')
      }

      const blob = await response.blob()
      const url = URL.createObjectURL(blob)

      setAudioBlob(blob)
      setAudioUrl(url)
      setStatus('completed')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Generation failed')
      setStatus('error')
    }
  }

  // Download audio
  const handleDownload = () => {
    if (!audioBlob || !audioUrl) return

    const a = document.createElement('a')
    a.href = audioUrl
    a.download = `tts_${Date.now()}.mp3`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
  }

  // Character count with estimated duration
  const charCount = text.length
  const estimatedMinutes = Math.ceil(charCount / 150)  // ~150 chars per minute speaking

  return (
    <section className="tts-section">
      {/* Header */}
      <div className="section-header">
        <div className="section-icon">&#x1F3A4;</div>
        <div>
          <h2>Text-to-Speech Generator</h2>
          <p className="section-description">
            Generate natural speech using Chatterbox TTS with optional voice cloning
          </p>
        </div>
      </div>

      {/* Step 1: Voice Selection */}
      <VoiceLibrary
        selectedVoiceId={selectedVoiceId}
        onSelectVoice={setSelectedVoiceId}
        disabled={status === 'generating'}
      />

      {/* Step 2: Text Input + Parameters */}
      <div className="tts-form">
        <div className="form-group">
          <label htmlFor="tts-text">
            Enter Text
            <span className="char-count">
              {charCount.toLocaleString()} chars · ~{estimatedMinutes} min
            </span>
          </label>
          <textarea
            id="tts-text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Enter the text you want to convert to speech. The server handles chunking automatically, so you can paste long texts (up to 12+ minutes)."
            rows={8}
            disabled={status === 'generating'}
          />
        </div>

        {/* Step 3: Fine-tune (collapsed) */}
        <details className="advanced-params">
          <summary>Fine-tune (optional)</summary>
          <div className="params-grid">
            <div className="param-group">
              <label htmlFor="exaggeration">
                Exaggeration: {exaggeration.toFixed(2)}
              </label>
              <input
                id="exaggeration"
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={exaggeration}
                onChange={(e) => setExaggeration(parseFloat(e.target.value))}
                disabled={status === 'generating'}
              />
              <span className="param-hint">Voice expressiveness (0=neutral, 1=dramatic)</span>
            </div>

            <div className="param-group">
              <label htmlFor="cfg-weight">
                CFG Weight: {cfgWeight.toFixed(2)}
              </label>
              <input
                id="cfg-weight"
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={cfgWeight}
                onChange={(e) => setCfgWeight(parseFloat(e.target.value))}
                disabled={status === 'generating'}
              />
              <span className="param-hint">Adherence to text (higher=more accurate)</span>
            </div>

            <div className="param-group">
              <label htmlFor="temperature">
                Temperature: {temperature.toFixed(2)}
              </label>
              <input
                id="temperature"
                type="range"
                min="0.1"
                max="1"
                step="0.05"
                value={temperature}
                onChange={(e) => setTemperature(parseFloat(e.target.value))}
                disabled={status === 'generating'}
              />
              <span className="param-hint">Variation in speech (lower=more consistent)</span>
            </div>
          </div>
        </details>

        {/* Error */}
        {error && (
          <div className="tts-error">
            <span className="error-icon">&#x26A0;</span>
            <span>{error}</span>
          </div>
        )}

        {/* Generate Button */}
        <button
          onClick={handleGenerate}
          disabled={status === 'generating' || !runpodStatus?.configured || !runpodStatus?.available || !text.trim()}
          className="btn-generate"
        >
          {status === 'generating' ? (
            <>
              <span className="loading-spinner-sm"></span>
              Generating... (this may take a while)
            </>
          ) : (
            <>
              <span className="btn-icon">&#x1F3A4;</span>
              Generate Speech
            </>
          )}
        </button>
      </div>

      {/* Audio Result */}
      {audioUrl && (
        <div className="tts-result">
          <h3>Generated Audio</h3>
          <audio controls src={audioUrl} className="audio-player">
            Your browser does not support the audio element.
          </audio>
          <button onClick={handleDownload} className="btn-download">
            <span className="btn-icon">&#x2B07;</span>
            Download MP3
          </button>
        </div>
      )}

      {/* Simplified Setup */}
      <div className="tts-help">
        <details className="setup-instructions">
          <summary>
            <h4>Setup</h4>
          </summary>
          <div className="setup-content">
            <p>Set <code>RUNPOD_API_KEY</code> and <code>RUNPOD_ENDPOINT_ID</code> in your <code>.env</code> file.</p>
            {runpodStatus?.configured ? (
              <div className="connection-status success">
                <span className="status-dot connected"></span>
                <span>RunPod configured (Endpoint: {runpodStatus.endpoint_id})</span>
              </div>
            ) : (
              <div className="connection-status error">
                <span className="status-dot"></span>
                <span>RunPod not configured</span>
              </div>
            )}
            <div className="setup-tips">
              <h5>Tips</h5>
              <ul>
                <li>First generation may be slow (cold start ~30s), subsequent ones are fast</li>
                <li>Voice cloning works best with 5-10 seconds of clear speech</li>
                <li>Upload voices via the voice library above for reuse across sessions</li>
              </ul>
            </div>
          </div>
        </details>
      </div>
    </section>
  )
}

export default TTSGenerator
