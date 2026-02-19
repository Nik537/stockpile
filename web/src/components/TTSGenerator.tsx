import { useCallback, useEffect, useRef, useState } from 'react'
import { useJobQueueStore } from '../stores/useJobQueueStore'
import VoiceLibrary from './VoiceLibrary'
import './TTSGenerator.css'
import { API_BASE } from '../config'

type TTSModel = 'chatterbox' | 'chatterbox-ext' | 'qwen3' | 'moss-ttsd'

const MODEL_OPTIONS: { value: TTSModel; label: string; description: string }[] = [
  { value: 'chatterbox-ext', label: 'Chatterbox Extended', description: 'Enhanced — denoising, multi-candidate, Whisper validation' },
  { value: 'chatterbox', label: 'Chatterbox', description: 'Original — fast, reliable voice cloning' },
  { value: 'qwen3', label: 'Qwen3-TTS', description: 'Higher quality — custom voice presets, multilingual' },
  { value: 'moss-ttsd', label: 'MOSS-TTSD', description: 'Multi-speaker dialogue — up to 5 speakers, 20 languages, voice cloning' },
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

const MOSS_TTSD_LANGUAGES = [
  { value: 'en', label: 'English' },
  { value: 'zh', label: 'Chinese' },
  { value: 'ja', label: 'Japanese' },
  { value: 'ko', label: 'Korean' },
  { value: 'de', label: 'German' },
  { value: 'fr', label: 'French' },
  { value: 'es', label: 'Spanish' },
  { value: 'it', label: 'Italian' },
  { value: 'ru', label: 'Russian' },
  { value: 'ar', label: 'Arabic' },
  { value: 'fa', label: 'Persian' },
  { value: 'he', label: 'Hebrew' },
  { value: 'pl', label: 'Polish' },
  { value: 'pt', label: 'Portuguese' },
  { value: 'cs', label: 'Czech' },
  { value: 'da', label: 'Danish' },
  { value: 'sv', label: 'Swedish' },
  { value: 'hu', label: 'Hungarian' },
  { value: 'el', label: 'Greek' },
  { value: 'tr', label: 'Turkish' },
]

const MOSS_TTSD_MODES = [
  { value: 'generation', label: 'Generation', description: 'Model uses its own voices' },
  { value: 'voice_clone', label: 'Voice Clone', description: 'Clone from reference audio' },
  { value: 'continuation', label: 'Continuation', description: 'Continue from prompt audio' },
  { value: 'voice_clone_and_continuation', label: 'Clone + Continue', description: 'Best quality — clone and continue' },
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
  moss_ttsd: EndpointStatus
  colab: any
}

interface TTSResult {
  jobId: string
  audioUrl: string
  filename: string
}

function TTSGenerator() {
  // Voice selection state
  const [selectedVoiceId, setSelectedVoiceId] = useState<string | null>(null)

  // Model selection
  const [selectedModel, setSelectedModel] = useState<TTSModel>('chatterbox-ext')

  // Status for all endpoints
  const [allStatus, setAllStatus] = useState<AllStatus | null>(null)

  // Generation state
  const [text, setText] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [submitFlash, setSubmitFlash] = useState(false)

  // Chatterbox params (shared between original + extended)
  const [exaggeration, setExaggeration] = useState(0.4)
  const [cfgWeight, setCfgWeight] = useState(0.3)
  const [temperature, setTemperature] = useState(0.8)

  // Chatterbox Extended-only params
  const [numCandidates, setNumCandidates] = useState(3)
  const [enableDenoising, setEnableDenoising] = useState(true)
  const [enableWhisperValidation, setEnableWhisperValidation] = useState(true)

  // Qwen3-specific params
  const [qwen3Language, setQwen3Language] = useState('auto')
  const [qwen3Speaker, setQwen3Speaker] = useState('')
  const [qwen3Instruction, setQwen3Instruction] = useState('')
  const [qwen3Temperature, setQwen3Temperature] = useState(0.7)
  const [qwen3TopP, setQwen3TopP] = useState(0.9)
  const [voiceReferenceTranscript, setVoiceReferenceTranscript] = useState('')
  const [isTranscribing, setIsTranscribing] = useState(false)

  // MOSS-TTSD-specific params
  const [mossLanguage, setMossLanguage] = useState('en')
  const [mossSpeakers, setMossSpeakers] = useState(1)
  const [mossMode, setMossMode] = useState('generation')
  const [mossTemperature, setMossTemperature] = useState(0.9)
  const [mossMaxTokens, setMossMaxTokens] = useState(2000)

  // Completed results array
  const [results, setResults] = useState<TTSResult[]>([])

  // Playback speed per result
  const [playbackSpeeds, setPlaybackSpeeds] = useState<Record<string, number>>({})
  const audioRefs = useRef<Record<string, HTMLAudioElement | null>>({})

  // Job queue store
  const { jobs, addJob, connectWebSocket } = useJobQueueStore()

  // Track which job IDs we've already fetched results for
  const fetchedJobIds = useRef<Set<string>>(new Set())

  // Subscribe to job completions and fetch audio
  useEffect(() => {
    const ttsJobs = jobs.filter(j => j.type === 'tts' && j.status === 'completed')
    for (const job of ttsJobs) {
      if (fetchedJobIds.current.has(job.id)) continue
      fetchedJobIds.current.add(job.id)

      // Fetch audio from the job endpoint
      fetch(`${API_BASE}/api/tts/jobs/${job.id}/audio`)
        .then(res => {
          if (!res.ok) throw new Error('Failed to fetch audio')
          const contentDisposition = res.headers.get('content-disposition')
          const filenameMatch = contentDisposition?.match(/filename="?([^"]+)"?/i)
          const filename = filenameMatch?.[1] || `tts_${job.id}.wav`
          return res.blob().then(blob => ({ blob, filename }))
        })
        .then(({ blob, filename }) => {
          const url = URL.createObjectURL(blob)
          setResults(prev => [{ jobId: job.id, audioUrl: url, filename }, ...prev])
        })
        .catch(err => {
          console.error(`Failed to fetch TTS result for job ${job.id}:`, err)
        })
    }
  }, [jobs])

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

  // Sync playback speed to audio elements
  useEffect(() => {
    for (const [jobId, speed] of Object.entries(playbackSpeeds)) {
      const el = audioRefs.current[jobId]
      if (el) el.playbackRate = speed
    }
  }, [playbackSpeeds])

  // Determine which mode string to send based on selected model
  const getModeForModel = (model: TTSModel): string => {
    switch (model) {
      case 'chatterbox': return 'runpod'
      case 'chatterbox-ext': return 'chatterbox-ext'
      case 'qwen3': return 'qwen3'
      case 'moss-ttsd': return 'moss-ttsd'
    }
  }

  // Check if the selected model's endpoint is configured
  const isModelConfigured = (): boolean => {
    if (!allStatus) return false
    switch (selectedModel) {
      case 'chatterbox': return allStatus.runpod?.configured && allStatus.runpod?.available
      case 'chatterbox-ext': return allStatus.chatterbox_ext?.configured && allStatus.chatterbox_ext?.available
      case 'qwen3': return allStatus.qwen3?.configured && allStatus.qwen3?.available
      case 'moss-ttsd': return allStatus.moss_ttsd?.configured && allStatus.moss_ttsd?.available
    }
  }

  // Auto-transcribe voice reference using Gemini
  const handleAutoTranscribe = async () => {
    if (!selectedVoiceId) return
    setIsTranscribing(true)
    try {
      const res = await fetch(`${API_BASE}/api/tts/voices/${selectedVoiceId}/transcribe`, {
        method: 'POST',
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Transcription failed' }))
        setError(err.detail || 'Transcription failed')
        return
      }
      const data = await res.json()
      if (data.transcript) {
        setVoiceReferenceTranscript(data.transcript)
      }
    } catch (e) {
      setError('Failed to auto-transcribe voice reference')
    } finally {
      setIsTranscribing(false)
    }
  }

  // Generate TTS (async job)
  const handleGenerate = async () => {
    if (!text.trim()) {
      setError('Please enter some text')
      return
    }

    if (!isModelConfigured()) {
      setError(`${MODEL_OPTIONS.find(m => m.value === selectedModel)?.label} is not configured. Check your .env settings.`)
      return
    }

    setSubmitting(true)
    setError(null)

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

      if (selectedModel === 'moss-ttsd') {
        formData.append('language', mossLanguage)
        formData.append('temperature', mossTemperature.toString())
        formData.append('num_speakers', mossSpeakers.toString())
        formData.append('moss_ttsd_mode', mossMode)
        formData.append('moss_ttsd_max_tokens', mossMaxTokens.toString())
      }

      const response = await fetch(`${API_BASE}/api/tts/generate`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Generation failed')
      }

      const data = await response.json()
      const jobId = data.job_id

      // Add to job queue
      addJob({
        id: jobId,
        type: 'tts',
        status: 'processing',
        label: `TTS: ${text.trim().slice(0, 40)}...`,
        createdAt: new Date().toISOString(),
      })

      // Connect WebSocket for real-time updates
      connectWebSocket(jobId, 'tts')

      // Brief flash feedback
      setSubmitFlash(true)
      setTimeout(() => setSubmitFlash(false), 1500)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Generation failed')
    } finally {
      setSubmitting(false)
    }
  }

  // Download audio
  const handleDownload = (result: TTSResult) => {
    const a = document.createElement('a')
    a.href = result.audioUrl
    a.download = result.filename
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
  }

  const getSpeed = (jobId: string) => playbackSpeeds[jobId] ?? 1.0
  const setSpeed = (jobId: string, speed: number) => {
    setPlaybackSpeeds(prev => ({ ...prev, [jobId]: speed }))
    const el = audioRefs.current[jobId]
    if (el) el.playbackRate = speed
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
            Generate natural speech with voice cloning — choose from multiple TTS backends
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
            >
              <span className="model-name">{opt.label}</span>
              <span className="model-desc">{opt.description}</span>
              {allStatus && (() => {
                const s = opt.value === 'chatterbox' ? allStatus.runpod
                  : opt.value === 'chatterbox-ext' ? allStatus.chatterbox_ext
                  : opt.value === 'moss-ttsd' ? allStatus.moss_ttsd
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
        disabled={false}
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
            placeholder={selectedModel === 'moss-ttsd'
              ? "Enter dialogue with speaker tags:\n[S1] Hello, how are you today?\n[S2] I'm doing great, thanks for asking!\n[S1] That's wonderful to hear."
              : "Enter the text you want to convert to speech. The server handles chunking automatically, so you can paste long texts."}
            rows={8}
          />
        </div>

        {/* Qwen3 voice reference transcript */}
        {selectedModel === 'qwen3' && selectedVoiceId && (
          <div className="form-group">
            <label htmlFor="voice-ref-transcript">
              Voice Reference Transcript (optional)
              <span className="param-hint">What is said in the reference audio — improves cloning accuracy</span>
            </label>
            <div className="transcript-input-row">
              <input
                id="voice-ref-transcript"
                type="text"
                value={voiceReferenceTranscript}
                onChange={(e) => setVoiceReferenceTranscript(e.target.value)}
                placeholder="Transcript of the voice reference audio..."
              />
              <button
                type="button"
                className="btn-auto-transcribe"
                onClick={handleAutoTranscribe}
                disabled={isTranscribing || !selectedVoiceId}
                title="Auto-transcribe using Gemini (free)"
              >
                {isTranscribing ? (
                  <span className="loading-spinner-sm" />
                ) : (
                  'Auto Generate'
                )}
              </button>
            </div>
          </div>
        )}

        {/* Fine-tune parameters */}
        <details className="advanced-params" open>
          <summary>Fine-tune</summary>
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
                  />
                  <span className="param-hint">Generate multiple and pick the best (slower but higher quality)</span>
                </div>

                <div className="param-group param-checkbox">
                  <label>
                    <input
                      type="checkbox"
                      checked={enableDenoising}
                      onChange={(e) => setEnableDenoising(e.target.checked)}
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
                  />
                  <span className="param-hint">Nucleus sampling threshold</span>
                </div>
              </>
            )}

            {/* MOSS-TTSD params */}
            {selectedModel === 'moss-ttsd' && (
              <>
                <div className="param-group">
                  <label htmlFor="moss-language">Language</label>
                  <select
                    id="moss-language"
                    value={mossLanguage}
                    onChange={(e) => setMossLanguage(e.target.value)}
                  >
                    {MOSS_TTSD_LANGUAGES.map(l => (
                      <option key={l.value} value={l.value}>{l.label}</option>
                    ))}
                  </select>
                </div>

                <div className="param-group">
                  <label htmlFor="moss-speakers">
                    Speakers: {mossSpeakers}
                  </label>
                  <input
                    id="moss-speakers"
                    type="range"
                    min="1"
                    max="5"
                    step="1"
                    value={mossSpeakers}
                    onChange={(e) => setMossSpeakers(parseInt(e.target.value))}
                  />
                  <span className="param-hint">Use [S1]-[S5] tags in text to assign speakers</span>
                </div>

                <div className="param-group">
                  <label htmlFor="moss-mode">Inference Mode</label>
                  <select
                    id="moss-mode"
                    value={mossMode}
                    onChange={(e) => setMossMode(e.target.value)}
                  >
                    {MOSS_TTSD_MODES.map(m => (
                      <option key={m.value} value={m.value}>{m.label} — {m.description}</option>
                    ))}
                  </select>
                </div>

                <div className="param-group">
                  <label htmlFor="moss-temperature">
                    Temperature: {mossTemperature.toFixed(2)}
                  </label>
                  <input
                    id="moss-temperature"
                    type="range"
                    min="0.1"
                    max="1.5"
                    step="0.05"
                    value={mossTemperature}
                    onChange={(e) => setMossTemperature(parseFloat(e.target.value))}
                  />
                  <span className="param-hint">Higher values produce more varied speech</span>
                </div>

                <div className="param-group">
                  <label htmlFor="moss-max-tokens">
                    Max Tokens: {mossMaxTokens}
                  </label>
                  <input
                    id="moss-max-tokens"
                    type="range"
                    min="500"
                    max="8000"
                    step="500"
                    value={mossMaxTokens}
                    onChange={(e) => setMossMaxTokens(parseInt(e.target.value))}
                  />
                  <span className="param-hint">~12.5 tokens per second of audio (2000 ≈ 160s)</span>
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

        {/* Submit flash */}
        {submitFlash && (
          <div className="tts-submit-flash">Job submitted! Check the queue bar below.</div>
        )}

        {/* Generate Button */}
        <button
          onClick={handleGenerate}
          disabled={submitting || !isModelConfigured() || !text.trim()}
          className="btn-generate"
        >
          {submitting ? (
            <>
              <span className="loading-spinner-sm"></span>
              Submitting...
            </>
          ) : (
            <>
              <span className="btn-icon">&#x1F3A4;</span>
              Generate Speech
            </>
          )}
        </button>
      </div>

      {/* Completed Results */}
      {results.length > 0 && (
        <div className="tts-results-list">
          <h3>Generated Audio ({results.length})</h3>
          <div className="tts-results-scroll">
            {results.map((result) => (
              <div key={result.jobId} className="tts-result">
                <audio
                  ref={(el) => { audioRefs.current[result.jobId] = el }}
                  controls
                  src={result.audioUrl}
                  className="audio-player"
                >
                  Your browser does not support the audio element.
                </audio>

                {/* Playback Speed Control */}
                <div className="speed-control">
                  <label className="speed-label">
                    Speed: {getSpeed(result.jobId).toFixed(2)}x
                  </label>
                  <div className="speed-presets">
                    {SPEED_PRESETS.map(speed => (
                      <button
                        key={speed}
                        className={`speed-preset ${getSpeed(result.jobId) === speed ? 'active' : ''}`}
                        onClick={() => setSpeed(result.jobId, speed)}
                      >
                        {speed === 1.0 ? '1x' : `${speed}x`}
                      </button>
                    ))}
                  </div>
                </div>

                <button onClick={() => handleDownload(result)} className="btn-download">
                  <span className="btn-icon">&#x2B07;</span>
                  Download
                </button>
              </div>
            ))}
          </div>
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
              <li>
                <code>RUNPOD_MOSS_TTSD_ENDPOINT_ID</code> — MOSS-TTSD (RunPod)
                {allStatus?.moss_ttsd?.configured ? (
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
                <li>MOSS-TTSD runs on RunPod serverless (GPU with 24+ GB VRAM)</li>
                <li>Use [S1]-[S5] speaker tags for multi-speaker dialogue with MOSS-TTSD</li>
              </ul>
            </div>
          </div>
        </details>
      </div>
    </section>
  )
}

export default TTSGenerator
