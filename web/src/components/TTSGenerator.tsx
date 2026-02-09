import { useCallback, useEffect, useRef, useState } from 'react'
import { TTSStatus } from '../types'
import VoiceLibrary from './VoiceLibrary'
import './TTSGenerator.css'

const API_BASE = ''  // Same origin

type TTSModel = 'chatterbox' | 'chatterbox-ext' | 'qwen3'

const MODEL_OPTIONS: { value: TTSModel; label: string; description: string }[] = [
  { value: 'chatterbox', label: 'Chatterbox', description: 'Original — fast, reliable voice cloning' },
  { value: 'chatterbox-ext', label: 'Chatterbox Extended', description: 'Enhanced — denoising, multi-candidate, Whisper validation' },
  { value: 'qwen3', label: 'Qwen3-TTS', description: 'Higher quality — custom voice presets, multilingual' },
]

const SPEED_PRESETS = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

const QWEN3_SPEAKERS = [
  '', 'Vivian', 'Serena', 'Aiden', 'Dylan', 'Eric', 'Ryan', 'Ono_Anna', 'Sohee', 'Uncle_Fu',
]

const QWEN3_LANGUAGES = [
  { value: 'auto', label: 'Auto-detect' },
  { value: 'en', label: 'English' },
  { value: 'zh', label: 'Chinese' },
  { value: 'ja', label: 'Japanese' },
  { value: 'ko', label: 'Korean' },
  { value: 'fr', label: 'French' },
  { value: 'de', label: 'German' },
  { value: 'es', label: 'Spanish' },
]

interface EndpointStatus {
  configured: boolean
  available: boolean
  endpoint_id?: string
  error?: string
}

interface AllStatus {
  runpod: EndpointStatus
  qwen3: EndpointStatus
  chatterbox_ext: EndpointStatus
  colab: any
}

function TTSGenerator() {
  // Voice selection state
  const [selectedVoiceId, setSelectedVoiceId] = useState<string | null>(null)

  // Model selection
  const [selectedModel, setSelectedModel] = useState<TTSModel>('chatterbox')

  // Status for all endpoints
  const [allStatus, setAllStatus] = useState<AllStatus | null>(null)

  // Generation state
  const [text, setText] = useState('')
  const [status, setStatus] = useState<TTSStatus>('idle')
  const [error, setError] = useState<string | null>(null)

  // Chatterbox params (shared between original + extended)
  const [exaggeration, setExaggeration] = useState(0.5)
  const [cfgWeight, setCfgWeight] = useState(0.5)
  const [temperature, setTemperature] = useState(0.8)

  // Chatterbox Extended-only params
  const [numCandidates, setNumCandidates] = useState(1)
  const [enableDenoising, setEnableDenoising] = useState(false)
  const [enableWhisperValidation, setEnableWhisperValidation] = useState(false)

  // Qwen3-specific params
  const [qwen3Language, setQwen3Language] = useState('auto')
  const [qwen3Speaker, setQwen3Speaker] = useState('')
  const [qwen3Instruction, setQwen3Instruction] = useState('')
  const [qwen3Temperature, setQwen3Temperature] = useState(0.7)
  const [qwen3TopP, setQwen3TopP] = useState(0.9)
  const [voiceReferenceTranscript, setVoiceReferenceTranscript] = useState('')

  // Playback speed
  const [playbackSpeed, setPlaybackSpeed] = useState(1.0)
  const audioRef = useRef<HTMLAudioElement>(null)

  // Result
  const [audioUrl, setAudioUrl] = useState<string | null>(null)
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null)
  const [downloadFilename, setDownloadFilename] = useState<string>('tts_output.wav')

  // Check all endpoint statuses
  const checkStatus = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/tts/status`)
      const data = await response.json()
      setAllStatus(data)
    } catch (e) {
      console.error('Failed to check TTS status:', e)
    }
  }, [])

  useEffect(() => {
    checkStatus()
  }, [checkStatus])

  // Sync playback speed to audio element
  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.playbackRate = playbackSpeed
    }
  }, [playbackSpeed, audioUrl])

  // Determine which mode string to send based on selected model
  const getModeForModel = (model: TTSModel): string => {
    switch (model) {
      case 'chatterbox': return 'runpod'
      case 'chatterbox-ext': return 'chatterbox-ext'
      case 'qwen3': return 'qwen3'
    }
  }

  // Check if the selected model's endpoint is configured
  const isModelConfigured = (): boolean => {
    if (!allStatus) return false
    switch (selectedModel) {
      case 'chatterbox': return allStatus.runpod?.configured && allStatus.runpod?.available
      case 'chatterbox-ext': return allStatus.chatterbox_ext?.configured && allStatus.chatterbox_ext?.available
      case 'qwen3': return allStatus.qwen3?.configured && allStatus.qwen3?.available
    }
  }

  // Generate TTS
  const handleGenerate = async () => {
    if (!text.trim()) {
      setError('Please enter some text')
      return
    }

    if (!isModelConfigured()) {
      setError(`${MODEL_OPTIONS.find(m => m.value === selectedModel)?.label} is not configured. Check your .env settings.`)
      return
    }

    setStatus('generating')
    setError(null)
    setAudioUrl(null)
    setAudioBlob(null)
    setDownloadFilename('tts_output.wav')

    try {
      const formData = new FormData()
      formData.append('text', text.trim())
      formData.append('mode', getModeForModel(selectedModel))

      if (selectedVoiceId) {
        formData.append('voice_id', selectedVoiceId)
      }

      // Model-specific params
      if (selectedModel === 'chatterbox' || selectedModel === 'chatterbox-ext') {
        formData.append('exaggeration', exaggeration.toString())
        formData.append('cfg_weight', cfgWeight.toString())
        formData.append('temperature', temperature.toString())
      }

      if (selectedModel === 'chatterbox-ext') {
        formData.append('num_candidates', numCandidates.toString())
        formData.append('enable_denoising', enableDenoising.toString())
        formData.append('enable_whisper_validation', enableWhisperValidation.toString())
      }

      if (selectedModel === 'qwen3') {
        formData.append('language', qwen3Language)
        formData.append('temperature', qwen3Temperature.toString())
        formData.append('top_p', qwen3TopP.toString())
        if (qwen3Speaker) formData.append('speaker_name', qwen3Speaker)
        if (qwen3Instruction) formData.append('instruction', qwen3Instruction)
        if (voiceReferenceTranscript) formData.append('voice_reference_transcript', voiceReferenceTranscript)
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
      const contentDisposition = response.headers.get('content-disposition')
      const filenameMatch = contentDisposition?.match(/filename="?([^"]+)"?/i)
      const filename = filenameMatch?.[1] || `tts_${Date.now()}.wav`
      const url = URL.createObjectURL(blob)

      setAudioBlob(blob)
      setAudioUrl(url)
      setDownloadFilename(filename)
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
    a.download = downloadFilename
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
  }

  // Character count with estimated duration
  const charCount = text.length
  const estimatedMinutes = Math.ceil(charCount / 150)

  const isChatterbox = selectedModel === 'chatterbox' || selectedModel === 'chatterbox-ext'

  return (
    <section className="tts-section">
      {/* Header */}
      <div className="section-header">
        <div className="section-icon">&#x1F3A4;</div>
        <div>
          <h2>Text-to-Speech Generator</h2>
          <p className="section-description">
            Generate natural speech with voice cloning — choose from 3 TTS backends
          </p>
        </div>
      </div>

      {/* Model Selector */}
      <div className="tts-model-selector">
        <label>TTS Model</label>
        <div className="model-options">
          {MODEL_OPTIONS.map(opt => (
            <button
              key={opt.value}
              className={`model-option ${selectedModel === opt.value ? 'active' : ''}`}
              onClick={() => setSelectedModel(opt.value)}
              disabled={status === 'generating'}
            >
              <span className="model-name">{opt.label}</span>
              <span className="model-desc">{opt.description}</span>
              {allStatus && (() => {
                const s = opt.value === 'chatterbox' ? allStatus.runpod
                  : opt.value === 'chatterbox-ext' ? allStatus.chatterbox_ext
                  : allStatus.qwen3
                return s?.configured ? (
                  <span className="model-status configured">Configured</span>
                ) : (
                  <span className="model-status not-configured">Not configured</span>
                )
              })()}
            </button>
          ))}
        </div>
      </div>

      {/* Voice Selection */}
      <VoiceLibrary
        selectedVoiceId={selectedVoiceId}
        onSelectVoice={setSelectedVoiceId}
        disabled={status === 'generating'}
      />

      {/* Text Input + Parameters */}
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
            placeholder="Enter the text you want to convert to speech. The server handles chunking automatically, so you can paste long texts."
            rows={8}
            disabled={status === 'generating'}
          />
        </div>

        {/* Qwen3 voice reference transcript */}
        {selectedModel === 'qwen3' && selectedVoiceId && (
          <div className="form-group">
            <label htmlFor="voice-ref-transcript">
              Voice Reference Transcript (optional)
              <span className="param-hint">What is said in the reference audio — improves cloning accuracy</span>
            </label>
            <input
              id="voice-ref-transcript"
              type="text"
              value={voiceReferenceTranscript}
              onChange={(e) => setVoiceReferenceTranscript(e.target.value)}
              placeholder="Transcript of the voice reference audio..."
              disabled={status === 'generating'}
            />
          </div>
        )}

        {/* Fine-tune parameters */}
        <details className="advanced-params">
          <summary>Fine-tune (optional)</summary>
          <div className="params-grid">
            {/* Chatterbox params */}
            {isChatterbox && (
              <>
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
              </>
            )}

            {/* Chatterbox Extended-only params */}
            {selectedModel === 'chatterbox-ext' && (
              <>
                <div className="param-group">
                  <label htmlFor="num-candidates">
                    Candidates: {numCandidates}
                  </label>
                  <input
                    id="num-candidates"
                    type="range"
                    min="1"
                    max="5"
                    step="1"
                    value={numCandidates}
                    onChange={(e) => setNumCandidates(parseInt(e.target.value))}
                    disabled={status === 'generating'}
                  />
                  <span className="param-hint">Generate multiple and pick the best (slower but higher quality)</span>
                </div>

                <div className="param-group param-checkbox">
                  <label>
                    <input
                      type="checkbox"
                      checked={enableDenoising}
                      onChange={(e) => setEnableDenoising(e.target.checked)}
                      disabled={status === 'generating'}
                    />
                    Enable Denoising
                  </label>
                  <span className="param-hint">Apply audio denoising post-processing</span>
                </div>

                <div className="param-group param-checkbox">
                  <label>
                    <input
                      type="checkbox"
                      checked={enableWhisperValidation}
                      onChange={(e) => setEnableWhisperValidation(e.target.checked)}
                      disabled={status === 'generating'}
                    />
                    Whisper Validation
                  </label>
                  <span className="param-hint">Validate output accuracy using Whisper transcription</span>
                </div>
              </>
            )}

            {/* Qwen3 params */}
            {selectedModel === 'qwen3' && (
              <>
                <div className="param-group">
                  <label htmlFor="qwen3-language">Language</label>
                  <select
                    id="qwen3-language"
                    value={qwen3Language}
                    onChange={(e) => setQwen3Language(e.target.value)}
                    disabled={status === 'generating'}
                  >
                    {QWEN3_LANGUAGES.map(l => (
                      <option key={l.value} value={l.value}>{l.label}</option>
                    ))}
                  </select>
                </div>

                <div className="param-group">
                  <label htmlFor="qwen3-speaker">Speaker Preset</label>
                  <select
                    id="qwen3-speaker"
                    value={qwen3Speaker}
                    onChange={(e) => setQwen3Speaker(e.target.value)}
                    disabled={status === 'generating'}
                  >
                    <option value="">None (use voice reference)</option>
                    {QWEN3_SPEAKERS.filter(Boolean).map(s => (
                      <option key={s} value={s}>{s}</option>
                    ))}
                  </select>
                  <span className="param-hint">Built-in voice preset (overrides voice reference)</span>
                </div>

                <div className="param-group">
                  <label htmlFor="qwen3-instruction">
                    Style Instruction (optional)
                  </label>
                  <input
                    id="qwen3-instruction"
                    type="text"
                    value={qwen3Instruction}
                    onChange={(e) => setQwen3Instruction(e.target.value)}
                    placeholder='e.g. "Speak with excitement" or "Whisper softly"'
                    disabled={status === 'generating'}
                  />
                </div>

                <div className="param-group">
                  <label htmlFor="qwen3-temperature">
                    Temperature: {qwen3Temperature.toFixed(2)}
                  </label>
                  <input
                    id="qwen3-temperature"
                    type="range"
                    min="0.1"
                    max="1.5"
                    step="0.05"
                    value={qwen3Temperature}
                    onChange={(e) => setQwen3Temperature(parseFloat(e.target.value))}
                    disabled={status === 'generating'}
                  />
                </div>

                <div className="param-group">
                  <label htmlFor="qwen3-top-p">
                    Top-P: {qwen3TopP.toFixed(2)}
                  </label>
                  <input
                    id="qwen3-top-p"
                    type="range"
                    min="0.1"
                    max="1"
                    step="0.05"
                    value={qwen3TopP}
                    onChange={(e) => setQwen3TopP(parseFloat(e.target.value))}
                    disabled={status === 'generating'}
                  />
                  <span className="param-hint">Nucleus sampling threshold</span>
                </div>
              </>
            )}
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
          disabled={status === 'generating' || !isModelConfigured() || !text.trim()}
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
          <audio
            ref={audioRef}
            controls
            src={audioUrl}
            className="audio-player"
          >
            Your browser does not support the audio element.
          </audio>

          {/* Playback Speed Control */}
          <div className="speed-control">
            <label className="speed-label">
              Speed: {playbackSpeed.toFixed(2)}x
            </label>
            <div className="speed-slider-row">
              <span className="speed-bound">0.5x</span>
              <input
                type="range"
                min="0.5"
                max="2"
                step="0.05"
                value={playbackSpeed}
                onChange={(e) => setPlaybackSpeed(parseFloat(e.target.value))}
                className="speed-slider"
              />
              <span className="speed-bound">2.0x</span>
            </div>
            <div className="speed-presets">
              {SPEED_PRESETS.map(speed => (
                <button
                  key={speed}
                  className={`speed-preset ${playbackSpeed === speed ? 'active' : ''}`}
                  onClick={() => setPlaybackSpeed(speed)}
                >
                  {speed === 1.0 ? '1x' : `${speed}x`}
                </button>
              ))}
            </div>
          </div>

          <button onClick={handleDownload} className="btn-download">
            <span className="btn-icon">&#x2B07;</span>
            Download MP3
          </button>
        </div>
      )}

      {/* Setup Help */}
      <div className="tts-help">
        <details className="setup-instructions">
          <summary>
            <h4>Setup</h4>
          </summary>
          <div className="setup-content">
            <p>Configure endpoints in your <code>.env</code> file:</p>
            <ul className="setup-env-list">
              <li>
                <code>RUNPOD_API_KEY</code> — Required for all RunPod backends
              </li>
              <li>
                <code>RUNPOD_ENDPOINT_ID</code> — Chatterbox (original)
                {allStatus?.runpod?.configured ? (
                  <span className="status-dot connected" title="Configured"></span>
                ) : (
                  <span className="status-dot" title="Not configured"></span>
                )}
              </li>
              <li>
                <code>RUNPOD_CHATTERBOX_EXT_ENDPOINT_ID</code> — Chatterbox Extended
                {allStatus?.chatterbox_ext?.configured ? (
                  <span className="status-dot connected" title="Configured"></span>
                ) : (
                  <span className="status-dot" title="Not configured"></span>
                )}
              </li>
              <li>
                <code>RUNPOD_QWEN3_ENDPOINT_ID</code> — Qwen3-TTS
                {allStatus?.qwen3?.configured ? (
                  <span className="status-dot connected" title="Configured"></span>
                ) : (
                  <span className="status-dot" title="Not configured"></span>
                )}
              </li>
            </ul>
            <div className="setup-tips">
              <h5>Tips</h5>
              <ul>
                <li>First generation may be slow (cold start ~30s), subsequent ones are fast</li>
                <li>Voice cloning works best with 5-10 seconds of clear speech</li>
                <li>Qwen3 voice cloning needs only ~3 seconds of reference audio</li>
                <li>Chatterbox Extended can generate multiple candidates and pick the best</li>
              </ul>
            </div>
          </div>
        </details>
      </div>
    </section>
  )
}

export default TTSGenerator
