import { useCallback, useEffect, useRef, useState } from 'react'
import { useJobQueueStore } from '../stores/useJobQueueStore'
import './MusicGenerator.css'
import { API_BASE } from '../config'

const MAX_DURATION = 190

const GENRE_PRESETS = [
  { label: 'YouTube BG', genres: 'upbeat, corporate, background music, positive, 110 BPM', duration: 120 },
  { label: 'Lo-fi Chill', genres: 'lo-fi, chill hop, mellow, jazzy, relaxing, vinyl crackle, 80 BPM', duration: 180 },
  { label: 'Epic Cinematic', genres: 'orchestral, cinematic, epic, dramatic, film score, 90 BPM', duration: 60 },
  { label: 'TikTok Viral', genres: 'pop, catchy, energetic, dance, electronic, viral, 128 BPM', duration: 30 },
  { label: 'Podcast Intro', genres: 'upbeat, electronic, modern, clean, podcast intro, 120 BPM', duration: 15 },
  { label: 'Funk Groove', genres: 'funk, soul, groovy, bass-heavy, 115 BPM', duration: 90 },
  { label: 'Ambient', genres: 'ambient, atmospheric, ethereal, spacious, drone, pad, 70 BPM', duration: 190 },
  { label: 'Hip Hop Beat', genres: 'hip hop, trap, 808, dark, hard-hitting, 140 BPM', duration: 90 },
]

interface MusicResult {
  jobId: string
  audioUrl: string
}

function MusicGenerator() {
  // Service status
  const [isConfigured, setIsConfigured] = useState<boolean | null>(null)
  const [statusError, setStatusError] = useState<string | null>(null)

  // Generation state
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [submitFlash, setSubmitFlash] = useState(false)

  // Form state
  const [genres, setGenres] = useState('')
  const [duration, setDuration] = useState(60)

  // Advanced params
  const [steps, setSteps] = useState(8)
  const [cfg, setCfg] = useState(1)
  const [seed, setSeed] = useState<string>('')

  // Completed results (accumulating)
  const [results, setResults] = useState<MusicResult[]>([])

  // Job queue store
  const { jobs, addJob, connectWebSocket } = useJobQueueStore()

  // Track which job IDs we've already fetched results for
  const fetchedJobIds = useRef<Set<string>>(new Set())

  // Subscribe to job completions and fetch audio
  useEffect(() => {
    const musicJobs = jobs.filter(j => j.type === 'music' && j.status === 'completed')
    for (const job of musicJobs) {
      if (fetchedJobIds.current.has(job.id)) continue
      fetchedJobIds.current.add(job.id)

      // Fetch audio from the job endpoint
      fetch(`${API_BASE}/api/music/jobs/${job.id}/audio`)
        .then(res => {
          if (!res.ok) throw new Error('Failed to fetch audio')
          return res.blob()
        })
        .then(blob => {
          const url = URL.createObjectURL(blob)
          setResults(prev => [{ jobId: job.id, audioUrl: url }, ...prev])
        })
        .catch(err => {
          console.error(`Failed to fetch music result for job ${job.id}:`, err)
        })
    }
  }, [jobs])

  // Handle preset click
  const handlePresetClick = (preset: typeof GENRE_PRESETS[number]) => {
    setGenres(preset.genres)
    setDuration(Math.min(preset.duration, MAX_DURATION))
  }

  // Check service status on mount
  const checkStatus = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/music/status`)
      const data = await response.json()
      setIsConfigured(data.configured ?? false)
      setStatusError(data.error ?? null)
    } catch (e) {
      console.error('Failed to check music service status:', e)
      setIsConfigured(false)
    }
  }, [])

  useEffect(() => {
    checkStatus()
  }, [checkStatus])

  // Generate music (async job)
  const handleGenerate = async () => {
    if (!genres.trim()) {
      setError('Please enter a genre/description')
      return
    }

    if (!isConfigured) {
      setError('Replicate is not configured. Add REPLICATE_API_KEY to your .env file.')
      return
    }

    setSubmitting(true)
    setError(null)

    try {
      const response = await fetch(`${API_BASE}/api/music/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          genres: genres.trim(),
          output_seconds: duration,
          seed: seed ? parseInt(seed, 10) : null,
          steps,
          cfg,
        }),
      })

      if (!response.ok) {
        const errorData = await response.json()
        const detail = errorData.detail
        const msg = typeof detail === 'string' ? detail
          : Array.isArray(detail) ? detail.map((e: Record<string, unknown>) => e.msg ?? String(e)).join(', ')
          : 'Generation failed'
        throw new Error(msg)
      }

      const data = await response.json()
      const jobId = data.job_id

      addJob({
        id: jobId,
        type: 'music',
        status: 'processing',
        label: `Music: ${genres.trim().slice(0, 40)}...`,
        createdAt: new Date().toISOString(),
      })

      connectWebSocket(jobId, 'music')

      setSubmitFlash(true)
      setTimeout(() => setSubmitFlash(false), 1500)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Generation failed')
    } finally {
      setSubmitting(false)
    }
  }

  // Download audio
  const handleDownload = (result: MusicResult) => {
    const a = document.createElement('a')
    a.href = result.audioUrl
    a.download = `music_${result.jobId}.mp3`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
  }

  // Format duration display
  const formatDuration = (seconds: number): string => {
    const m = Math.floor(seconds / 60)
    const s = seconds % 60
    return m > 0 ? `${m}m ${s}s` : `${s}s`
  }

  return (
    <section className="music-section">
      {/* Header */}
      <div className="section-header">
        <div className="section-icon">&#x1F3B5;</div>
        <div>
          <h2>Music Generator</h2>
          <p className="section-description">
            Generate original instrumental music with Stable Audio 2.5
          </p>
        </div>
      </div>

      {/* Service Status */}
      <div className="music-status-bar">
        <span className={`status-dot ${isConfigured ? 'connected' : 'disconnected'}`}></span>
        {isConfigured
          ? 'Stable Audio 2.5 ready'
          : 'Stable Audio 2.5 not configured'}
        {statusError && ` — ${statusError}`}
      </div>

      {/* Form */}
      <div className="music-form">
        {/* Genre Presets */}
        <div className="music-presets">
          <label>Presets</label>
          <div className="preset-chips">
            {GENRE_PRESETS.map(preset => (
              <button
                key={preset.label}
                className={`preset-chip ${genres === preset.genres ? 'active' : ''}`}
                onClick={() => handlePresetClick(preset)}
              >
                {preset.label}
              </button>
            ))}
          </div>
        </div>

        {/* Genre / Description */}
        <div className="form-group">
          <label htmlFor="music-genres">
            Genre / Description
          </label>
          <textarea
            id="music-genres"
            value={genres}
            onChange={(e) => setGenres(e.target.value)}
            placeholder="funk, pop, soul, rock, energetic, groovy, 120 BPM"
            rows={3}
          />
        </div>

        {/* Duration */}
        <div className="music-duration-control">
          <label htmlFor="music-duration">
            Duration
            <span className="duration-value">{formatDuration(duration)}</span>
          </label>
          <input
            id="music-duration"
            type="range"
            min="10"
            max={MAX_DURATION}
            step="5"
            value={duration}
            onChange={(e) => setDuration(parseInt(e.target.value))}
          />
          <div className="duration-bounds">
            <span>10s</span>
            <span>{MAX_DURATION}s</span>
          </div>
        </div>

        {/* Advanced Parameters */}
        <details className="music-advanced-params">
          <summary>Advanced Parameters</summary>
          <div className="music-params-grid">
            <div className="param-group">
              <label htmlFor="music-steps">
                Steps: {steps}
              </label>
              <input
                id="music-steps"
                type="range"
                min={4}
                max={8}
                step={1}
                value={steps}
                onChange={(e) => setSteps(parseInt(e.target.value))}
              />
              <span className="param-hint">More steps = higher quality, slower generation</span>
            </div>

            <div className="param-group">
              <label htmlFor="music-cfg">
                CFG: {cfg}
              </label>
              <input
                id="music-cfg"
                type="range"
                min={1}
                max={25}
                step={0.5}
                value={cfg}
                onChange={(e) => setCfg(parseFloat(e.target.value))}
              />
              <span className="param-hint">Classifier-free guidance strength</span>
            </div>

            <div className="param-group">
              <label htmlFor="music-seed">
                Seed (optional)
              </label>
              <input
                id="music-seed"
                type="number"
                value={seed}
                onChange={(e) => setSeed(e.target.value)}
                placeholder="Random"
              />
              <span className="param-hint">Set for reproducible results</span>
            </div>
          </div>
        </details>

        {/* Error */}
        {error && (
          <div className="music-error">
            <span className="error-icon">&#x26A0;</span>
            <span>{error}</span>
          </div>
        )}

        {/* Submit flash */}
        {submitFlash && (
          <div className="music-submit-flash">Job submitted! Check the queue bar below.</div>
        )}

        {/* Generate Button */}
        <button
          onClick={handleGenerate}
          disabled={submitting || !isConfigured || !genres.trim()}
          className="btn-generate"
        >
          {submitting ? (
            <>
              <span className="loading-spinner-sm"></span>
              Submitting...
            </>
          ) : (
            <>
              <span className="btn-icon">&#x1F3B5;</span>
              Generate Music
            </>
          )}
        </button>
      </div>

      {/* Completed Results */}
      {results.length > 0 && (
        <div className="music-results-list">
          <h3>Generated Music ({results.length})</h3>
          <div className="music-results-scroll">
            {results.map((result) => (
              <div key={result.jobId} className="music-result">
                <audio
                  controls
                  src={result.audioUrl}
                  className="audio-player"
                >
                  Your browser does not support the audio element.
                </audio>
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
      <div className="music-help">
        <details className="setup-instructions">
          <summary>
            <h4>Setup</h4>
          </summary>
          <div className="setup-content">
            <p>Configure Stable Audio 2.5 music generation in your <code>.env</code> file:</p>
            <ul className="setup-env-list">
              <li>
                <code>REPLICATE_API_KEY</code> — Stable Audio 2.5 (~$0.015/song)
                {isConfigured ? (
                  <span className="status-dot connected" title="Configured"></span>
                ) : (
                  <span className="status-dot" title="Not configured"></span>
                )}
              </li>
            </ul>
            <div className="setup-tips">
              <h5>Tips</h5>
              <ul>
                <li>First generation may be slow due to cold start (~60s), subsequent ones are faster</li>
                <li>Use descriptive genre tags: "funk, pop, soul, energetic, groovy, 120 BPM"</li>
                <li>Maximum duration is 190 seconds</li>
                <li>All output is instrumental — perfect for background music and B-roll</li>
              </ul>
            </div>
          </div>
        </details>
      </div>
    </section>
  )
}

export default MusicGenerator
