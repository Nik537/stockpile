import { useCallback, useEffect, useRef, useState } from 'react'
import { TTSServerStatus, TTSStatus } from '../types'
import VoiceUploader from './VoiceUploader'
import './TTSGenerator.css'

const API_BASE = ''  // Same origin
const TTS_URL_KEY = 'stockpile_tts_server_url'
const TTS_MODE_KEY = 'stockpile_tts_mode'

type TTSMode = 'runpod' | 'colab'

interface TTSStatusResponse {
  colab: TTSServerStatus
  runpod: {
    configured: boolean
    available: boolean
    endpoint_id?: string
    error?: string
  }
}

function TTSGenerator() {
  // Mode selection state
  const [ttsMode, setTtsMode] = useState<TTSMode>(() => {
    const saved = localStorage.getItem(TTS_MODE_KEY)
    return (saved === 'colab' || saved === 'runpod') ? saved : 'runpod'
  })

  // Server connection state (for colab mode)
  const [serverUrl, setServerUrl] = useState('')
  const [serverStatus, setServerStatus] = useState<TTSServerStatus | null>(null)
  const [runpodStatus, setRunpodStatus] = useState<TTSStatusResponse['runpod'] | null>(null)
  const [isConnecting, setIsConnecting] = useState(false)

  // Auto-connect tracking (prevents infinite retry loops)
  const hasAttemptedAutoConnect = useRef(false)

  // Generation state
  const [text, setText] = useState('')
  const [voiceFile, setVoiceFile] = useState<File | null>(null)
  const [status, setStatus] = useState<TTSStatus>('idle')
  const [error, setError] = useState<string | null>(null)

  // Parameters
  const [exaggeration, setExaggeration] = useState(0.5)
  const [cfgWeight, setCfgWeight] = useState(0.5)
  const [temperature, setTemperature] = useState(0.8)

  // Result
  const [audioUrl, setAudioUrl] = useState<string | null>(null)
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null)

  // Save mode to localStorage
  useEffect(() => {
    localStorage.setItem(TTS_MODE_KEY, ttsMode)
  }, [ttsMode])

  // Check server status on mount
  const checkServerStatus = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/tts/status`)
      const data: TTSStatusResponse = await response.json()

      // Update colab status
      setServerStatus(data.colab)
      // Only update URL from backend if we don't already have one from localStorage
      if (data.colab.server_url && !serverUrl) {
        setServerUrl(data.colab.server_url)
      }

      // Update runpod status
      setRunpodStatus(data.runpod)
    } catch (e) {
      console.error('Failed to check TTS status:', e)
    }
  }, [serverUrl])

  // Load from localStorage first, then check backend status
  useEffect(() => {
    const savedUrl = localStorage.getItem(TTS_URL_KEY)
    if (savedUrl) {
      setServerUrl(savedUrl)
    }
    checkServerStatus()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Connect to server (colab mode)
  const handleConnect = useCallback(async () => {
    if (!serverUrl.trim()) {
      setError('Please enter a server URL')
      return
    }

    setIsConnecting(true)
    setError(null)

    try {
      const response = await fetch(`${API_BASE}/api/tts/endpoint`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: serverUrl.trim() }),
      })

      const data = await response.json()
      setServerStatus({
        connected: data.connected,
        server_url: data.server_url,
        error: data.error,
      })

      if (data.connected) {
        // Save to localStorage on successful connect
        localStorage.setItem(TTS_URL_KEY, serverUrl.trim())
      } else {
        setError(data.error || 'Failed to connect to server')
      }
    } catch (e) {
      setError('Failed to connect: ' + (e instanceof Error ? e.message : 'Unknown error'))
    } finally {
      setIsConnecting(false)
    }
  }, [serverUrl])

  // Auto-connect effect: if URL exists but not connected, try once (colab mode only)
  useEffect(() => {
    if (
      ttsMode === 'colab' &&
      serverUrl &&
      serverStatus &&
      !serverStatus.connected &&
      !isConnecting &&
      !hasAttemptedAutoConnect.current
    ) {
      hasAttemptedAutoConnect.current = true
      handleConnect()
    }
  }, [ttsMode, serverUrl, serverStatus, isConnecting, handleConnect])

  // Check if ready to generate
  const isReadyToGenerate = ttsMode === 'runpod'
    ? runpodStatus?.configured && runpodStatus?.available
    : serverStatus?.connected

  // Generate TTS
  const handleGenerate = async () => {
    if (!text.trim()) {
      setError('Please enter some text')
      return
    }

    if (!isReadyToGenerate) {
      if (ttsMode === 'runpod') {
        setError('RunPod is not configured. Check your .env settings.')
      } else {
        setError('Please connect to a TTS server first')
      }
      return
    }

    setStatus('generating')
    setError(null)
    setAudioUrl(null)
    setAudioBlob(null)

    try {
      const formData = new FormData()
      formData.append('text', text.trim())
      formData.append('mode', ttsMode)
      formData.append('exaggeration', exaggeration.toString())
      formData.append('cfg_weight', cfgWeight.toString())
      formData.append('temperature', temperature.toString())

      if (voiceFile) {
        formData.append('voice', voiceFile)
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
      <div className="section-header">
        <div className="section-icon">&#x1F3A4;</div>
        <div>
          <h2>Text-to-Speech Generator</h2>
          <p className="section-description">
            Generate natural speech using Chatterbox TTS with optional voice cloning
          </p>
        </div>
      </div>

      {/* Mode Selection */}
      <div className="tts-mode-selector">
        <h3>TTS Mode</h3>
        <div className="mode-options">
          <label className={`mode-option ${ttsMode === 'runpod' ? 'selected' : ''}`}>
            <input
              type="radio"
              name="tts-mode"
              value="runpod"
              checked={ttsMode === 'runpod'}
              onChange={() => setTtsMode('runpod')}
            />
            <div className="mode-content">
              <div className="mode-header">
                <span className="mode-name">RunPod Serverless</span>
                {runpodStatus?.configured && runpodStatus?.available && (
                  <span className="mode-badge ready">Ready</span>
                )}
                {runpodStatus?.configured === false && (
                  <span className="mode-badge not-configured">Not Configured</span>
                )}
              </div>
              <span className="mode-desc">Always available, pay-per-use (~$0.03/10min audio)</span>
            </div>
          </label>

          <label className={`mode-option ${ttsMode === 'colab' ? 'selected' : ''}`}>
            <input
              type="radio"
              name="tts-mode"
              value="colab"
              checked={ttsMode === 'colab'}
              onChange={() => setTtsMode('colab')}
            />
            <div className="mode-content">
              <div className="mode-header">
                <span className="mode-name">Custom Server (Colab)</span>
                {serverStatus?.connected && (
                  <span className="mode-badge connected">Connected</span>
                )}
              </div>
              <span className="mode-desc">Free, requires Colab notebook running</span>
            </div>
          </label>
        </div>
      </div>

      {/* RunPod Status (when RunPod mode selected) */}
      {ttsMode === 'runpod' && (
        <div className="tts-connection-panel runpod-panel">
          <h3>RunPod Status</h3>
          {runpodStatus?.configured ? (
            <div className="connection-status success">
              <span className="status-dot connected"></span>
              <span>RunPod configured (Endpoint: {runpodStatus.endpoint_id})</span>
            </div>
          ) : (
            <div className="connection-status error">
              <span className="status-dot"></span>
              <span>RunPod not configured</span>
              <p className="runpod-setup-hint">
                Set <code>RUNPOD_API_KEY</code> and <code>RUNPOD_ENDPOINT_ID</code> in your <code>.env</code> file.
                See setup instructions below for deployment guide.
              </p>
            </div>
          )}
        </div>
      )}

      {/* Server Connection (when Colab mode selected) */}
      {ttsMode === 'colab' && (
        <div className="tts-connection-panel">
          <h3>Server Connection</h3>
          <p className="connection-help">
            Paste the URL from your{' '}
            <a
              href="https://github.com/devnen/Chatterbox-TTS-Server"
              target="_blank"
              rel="noopener noreferrer"
            >
              Chatterbox-TTS-Server
            </a>{' '}
            Colab notebook
          </p>

          <div className="connection-form">
            <input
              type="url"
              value={serverUrl}
              onChange={(e) => setServerUrl(e.target.value)}
              placeholder="https://xxx.ngrok.io or https://xxx.loca.lt"
              disabled={isConnecting}
            />
            <button
              onClick={handleConnect}
              disabled={isConnecting || !serverUrl.trim()}
              className={`btn-connect ${serverStatus?.connected ? 'connected' : ''}`}
            >
              {isConnecting ? (
                <>
                  <span className="loading-spinner-sm"></span>
                  Connecting...
                </>
              ) : serverStatus?.connected ? (
                <>
                  <span className="status-dot connected"></span>
                  Connected
                </>
              ) : (
                'Connect'
              )}
            </button>
          </div>

          {serverStatus && (
            <div className={`connection-status ${serverStatus.connected ? 'success' : 'error'}`}>
              {serverStatus.connected ? (
                <span>Connected to {serverStatus.server_url}</span>
              ) : (
                <span>Not connected{serverStatus.error && `: ${serverStatus.error}`}</span>
              )}
            </div>
          )}
        </div>
      )}

      {/* Main Generation Form */}
      <div className="tts-form">
        {/* Text Input */}
        <div className="form-group">
          <label htmlFor="tts-text">
            Text to speak
            <span className="char-count">
              {charCount.toLocaleString()} characters (~{estimatedMinutes} min)
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

        {/* Voice Cloning */}
        <VoiceUploader
          voiceFile={voiceFile}
          onVoiceChange={setVoiceFile}
          disabled={status === 'generating'}
        />

        {/* Advanced Parameters */}
        <details className="advanced-params">
          <summary>Advanced Parameters</summary>
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

        {/* Error Message */}
        {error && (
          <div className="tts-error">
            <span className="error-icon">&#x26A0;</span>
            <span>{error}</span>
          </div>
        )}

        {/* Generate Button */}
        <button
          onClick={handleGenerate}
          disabled={status === 'generating' || !isReadyToGenerate || !text.trim()}
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

      {/* Setup Instructions */}
      <div className="tts-help">
        <details className="setup-instructions" open>
          <summary>
            <h4>Setup Instructions</h4>
          </summary>

          <div className="setup-tabs">
            {/* RunPod Setup */}
            <div className="setup-option">
              <h5>Option A: RunPod Serverless (Recommended)</h5>
              <p className="setup-desc">Always available, pay only when generating audio</p>
              <ol>
                <li>
                  Create account at{' '}
                  <a href="https://runpod.io" target="_blank" rel="noopener noreferrer">
                    RunPod.io
                  </a>{' '}
                  and add ~$10 credits
                </li>
                <li>
                  Build and push Docker image:
                  <pre className="code-block">
{`cd stockpile/runpod-tts-worker
docker build --platform linux/amd64 -t YOUR_USER/chatterbox-tts:latest .
docker push YOUR_USER/chatterbox-tts:latest`}
                  </pre>
                </li>
                <li>
                  Go to{' '}
                  <a href="https://runpod.io/console/serverless" target="_blank" rel="noopener noreferrer">
                    RunPod Serverless
                  </a>{' '}
                  &rarr; New Endpoint &rarr; Import from Docker Registry
                </li>
                <li>Configure: 16GB GPU (T4/RTX 3090), Max Workers: 1, Idle Timeout: 5s</li>
                <li>
                  Copy Endpoint ID and create API key at{' '}
                  <a href="https://runpod.io/console/user/settings" target="_blank" rel="noopener noreferrer">
                    User Settings
                  </a>
                </li>
                <li>
                  Add to your <code>.env</code>:
                  <pre className="code-block">
{`RUNPOD_API_KEY=your_api_key
RUNPOD_ENDPOINT_ID=your_endpoint_id`}
                  </pre>
                </li>
                <li>Restart the backend server</li>
              </ol>
              <p className="cost-estimate">
                <strong>Cost:</strong> ~$0.03-0.05 per 10 minutes of audio
              </p>
            </div>

            <div className="setup-divider">
              <span>or</span>
            </div>

            {/* Colab Setup */}
            <div className="setup-option">
              <h5>Option B: Colab Server (Free)</h5>
              <p className="setup-desc">Free but requires keeping Colab tab open</p>
              <ol>
                <li>
                  Go to{' '}
                  <a
                    href="https://colab.research.google.com/"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    Google Colab
                  </a>
                  {' '}&rarr; File &rarr; Upload notebook &rarr; select <code>notebooks/Chatterbox_TTS_Extended_Server.ipynb</code>
                </li>
                <li>Enable GPU: Runtime &rarr; Change runtime type &rarr; T4 GPU</li>
                <li>Run all cells: Runtime &rarr; Run all (Ctrl+F9)</li>
                <li>Copy the URL from the output (looks like https://xxxx.ngrok.io)</li>
                <li>Select "Custom Server (Colab)" mode above, paste URL, and click Connect</li>
              </ol>
            </div>
          </div>

          <div className="setup-tips">
            <h5>Tips</h5>
            <ul>
              <li><strong>RunPod:</strong> First generation may be slow (cold start ~30s), subsequent ones are fast</li>
              <li><strong>Colab:</strong> Keep the notebook running while generating; disconnects after ~90 min idle</li>
              <li>Voice cloning works best with 5-10 seconds of clear speech</li>
            </ul>
          </div>
        </details>
      </div>
    </section>
  )
}

export default TTSGenerator
