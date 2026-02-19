import { useCallback, useEffect, useRef, useState } from 'react'
import { useJobQueueStore } from '../stores'
import { API_BASE } from '../config'
import './VideoGenerator.css'

interface StoryboardImage {
  job_id: string
  title: string
  scenes: { scene_number: number; image_url: string }[]
  reference_images: Record<string, string>
}

interface VideoResult {
  job_id: string
  video_url: string
  prompt: string
  seed: number | null
  generation_time_ms: number | null
}

// Valid num_frames values: 8n+1
const FRAME_PRESETS = [
  { frames: 25, label: '~1s' },
  { frames: 49, label: '~2s' },
  { frames: 97, label: '~4s' },
  { frames: 121, label: '~5s' },
  { frames: 193, label: '~8s' },
  { frames: 257, label: '~11s' },
]

const PROMPT_PRESETS = [
  'Slow dolly in, cinematic lighting, shallow depth of field',
  'Camera pans left to right across a wide landscape',
  'Close-up portrait with gentle wind blowing hair',
  'Time-lapse of clouds moving across a dramatic sky',
]

function VideoGenerator() {
  // Service status
  const [isConfigured, setIsConfigured] = useState<boolean | null>(null)
  const [statusError, setStatusError] = useState<string | null>(null)

  // Form state
  const [prompt, setPrompt] = useState('')
  const [negativePrompt, setNegativePrompt] = useState('')
  const [width, setWidth] = useState(768)
  const [height, setHeight] = useState(512)
  const [numFrames, setNumFrames] = useState(97)
  const [fps, setFps] = useState(24)
  const [numInferenceSteps, setNumInferenceSteps] = useState(30)
  const [guidanceScale, setGuidanceScale] = useState(3.0)
  const [seed, setSeed] = useState<string>('')
  const [conditioningStrength, setConditioningStrength] = useState(1.0)

  // Reference images
  const [storyboardImages, setStoryboardImages] = useState<StoryboardImage[]>([])
  const [selectedImages, setSelectedImages] = useState<string[]>([]) // ordered list of URLs/base64
  const [loadingStoryboards, setLoadingStoryboards] = useState(false)

  // Generation state
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [submitFlash, setSubmitFlash] = useState(false)

  // Results
  const [results, setResults] = useState<VideoResult[]>([])

  // Job queue
  const { jobs, addJob, connectWebSocket } = useJobQueueStore()
  const fetchedJobIds = useRef<Set<string>>(new Set())

  // Duration hint
  const durationHint = numFrames > 0 && fps > 0
    ? `~${(numFrames / fps).toFixed(1)}s at ${fps}fps`
    : ''

  // Check service status
  const checkStatus = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/video/status`)
      const data = await response.json()
      setIsConfigured(data.configured && data.available !== false)
      if (data.error) setStatusError(data.error)
      else setStatusError(null)
    } catch {
      setIsConfigured(false)
      setStatusError('Cannot reach backend')
    }
  }, [])

  // Load storyboard images
  const loadStoryboardImages = useCallback(async () => {
    setLoadingStoryboards(true)
    try {
      const response = await fetch(`${API_BASE}/api/video/storyboard-images`)
      if (response.ok) {
        const data = await response.json()
        setStoryboardImages(data)
      }
    } catch {
      // Storyboard images are optional
    } finally {
      setLoadingStoryboards(false)
    }
  }, [])

  useEffect(() => {
    checkStatus()
    loadStoryboardImages()
  }, [checkStatus, loadStoryboardImages])

  // Watch for completed video jobs
  useEffect(() => {
    const videoJobs = jobs.filter(
      (j) => j.type === 'video' && j.status === 'completed' && !fetchedJobIds.current.has(j.id)
    )

    for (const job of videoJobs) {
      fetchedJobIds.current.add(job.id)

      // Extract result from job
      const result = job.result as { video_url?: string; seed?: number; generation_time_ms?: number } | undefined
      if (result?.video_url) {
        setResults((prev) => [
          {
            job_id: job.id,
            video_url: result.video_url!,
            prompt: job.label.replace('Video: ', ''),
            seed: result.seed ?? null,
            generation_time_ms: result.generation_time_ms ?? null,
          },
          ...prev,
        ])
      }
    }
  }, [jobs])

  // Toggle storyboard image selection
  const toggleStoryboardImage = (imageUrl: string) => {
    setSelectedImages((prev) => {
      if (prev.includes(imageUrl)) {
        return prev.filter((u) => u !== imageUrl)
      }
      return [...prev, imageUrl]
    })
  }

  // Remove a selected image
  const removeSelectedImage = (index: number) => {
    setSelectedImages((prev) => prev.filter((_, i) => i !== index))
  }

  // Handle file upload for custom reference images
  const handleFileUpload = (file: File) => {
    const reader = new FileReader()
    reader.onload = (e) => {
      const dataUrl = e.target?.result as string
      setSelectedImages((prev) => [...prev, dataUrl])
    }
    reader.readAsDataURL(file)
  }

  // Generate video
  const handleGenerate = async () => {
    if (!prompt.trim()) {
      setError('Please enter a prompt')
      return
    }

    setSubmitting(true)
    setError(null)

    try {
      const response = await fetch(`${API_BASE}/api/video/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: prompt.trim(),
          negative_prompt: negativePrompt.trim(),
          width,
          height,
          num_frames: numFrames,
          num_inference_steps: numInferenceSteps,
          guidance_scale: guidanceScale,
          seed: seed ? parseInt(seed, 10) : null,
          fps,
          conditioning_images: selectedImages.length > 0 ? selectedImages : undefined,
          conditioning_strength: selectedImages.length > 0 ? conditioningStrength : undefined,
        }),
      })

      if (!response.ok) {
        const data = await response.json()
        throw new Error(data.detail || 'Failed to start generation')
      }

      const data = await response.json()
      const jobId = data.job_id

      addJob({
        id: jobId,
        type: 'video',
        status: 'processing',
        label: `Video: ${prompt.trim().slice(0, 50)}...`,
        createdAt: new Date().toISOString(),
      })

      connectWebSocket(jobId, 'video')

      setSubmitFlash(true)
      setTimeout(() => setSubmitFlash(false), 2000)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start generation')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <section className="video-section">
      <h2>Video Generator</h2>
      <p className="section-description">
        Generate videos using LTX-Video 2 via RunPod. Supports text-to-video and image-to-video with reference images from your storyboards.
      </p>

      {/* Status Bar */}
      <div className="video-status-bar">
        <span className={`status-dot ${isConfigured ? 'connected' : 'disconnected'}`} />
        {isConfigured === null
          ? 'Checking...'
          : isConfigured
            ? 'LTX-Video 2 Ready'
            : statusError || 'Not configured'}
      </div>

      {/* Form */}
      <div className="video-form">
        {/* Prompt */}
        <div className="form-group">
          <label htmlFor="video-prompt">Prompt</label>
          <textarea
            id="video-prompt"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Describe the video you want to generate..."
            rows={3}
          />
          <div className="video-presets">
            <div className="preset-chips">
              {PROMPT_PRESETS.map((p, i) => (
                <button
                  key={i}
                  className="preset-chip"
                  onClick={() => setPrompt(p)}
                >
                  {p.slice(0, 40)}...
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Reference Images from Storyboard */}
        <div className="video-ref-section">
          <label>
            Reference Images
            {selectedImages.length > 0 && (
              <span className="ref-count-badge">{selectedImages.length}</span>
            )}
          </label>

          {/* Storyboard image picker */}
          <div className="storyboard-picker">
            {loadingStoryboards ? (
              <div className="storyboard-picker-empty">Loading storyboards...</div>
            ) : storyboardImages.length === 0 ? (
              <div className="storyboard-picker-empty">
                No completed storyboards yet. Generate a storyboard first, or upload images below.
              </div>
            ) : (
              storyboardImages.map((sb) => (
                <div key={sb.job_id} className="storyboard-picker-job">
                  <div className="storyboard-picker-title">{sb.title}</div>
                  <div className="storyboard-picker-grid">
                    {/* Reference images */}
                    {Object.entries(sb.reference_images).map(([name, url]) => (
                      <div
                        key={`ref-${name}`}
                        className={`storyboard-thumb ${selectedImages.includes(url) ? 'selected' : ''}`}
                        onClick={() => toggleStoryboardImage(url)}
                        title={`${name} (reference)`}
                      >
                        <img src={url} alt={name} />
                        {selectedImages.includes(url) && (
                          <span className="thumb-check">
                            {selectedImages.indexOf(url) + 1}
                          </span>
                        )}
                      </div>
                    ))}
                    {/* Scene images */}
                    {sb.scenes.map((scene) => (
                      <div
                        key={`scene-${scene.scene_number}`}
                        className={`storyboard-thumb ${selectedImages.includes(scene.image_url) ? 'selected' : ''}`}
                        onClick={() => toggleStoryboardImage(scene.image_url)}
                        title={`Scene ${scene.scene_number}`}
                      >
                        <img src={scene.image_url} alt={`Scene ${scene.scene_number}`} />
                        {selectedImages.includes(scene.image_url) && (
                          <span className="thumb-check">
                            {selectedImages.indexOf(scene.image_url) + 1}
                          </span>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              ))
            )}
          </div>

          {/* Upload custom images */}
          <label className="video-upload-zone">
            <input
              type="file"
              accept="image/*"
              multiple
              onChange={(e) => {
                const files = e.target.files
                if (files) {
                  Array.from(files).forEach((f) => handleFileUpload(f))
                }
                e.target.value = ''
              }}
            />
            <span className="upload-label">+ Upload custom reference images</span>
          </label>

          {/* Selected images strip */}
          {selectedImages.length > 0 && (
            <>
              <div className="selected-images-strip">
                {selectedImages.map((url, i) => (
                  <div key={i} className="selected-image-item">
                    <img src={url} alt={`Ref ${i + 1}`} />
                    <span className="order-badge">{i + 1}</span>
                    <button
                      className="btn-remove-selected"
                      onClick={() => removeSelectedImage(i)}
                    >
                      &times;
                    </button>
                  </div>
                ))}
              </div>
              <div className="form-group" style={{ marginTop: 'var(--space-3)' }}>
                <label>Conditioning Strength</label>
                <input
                  type="number"
                  value={conditioningStrength}
                  onChange={(e) => setConditioningStrength(parseFloat(e.target.value) || 1.0)}
                  min={0}
                  max={1}
                  step={0.1}
                  style={{ width: 80 }}
                />
              </div>
            </>
          )}
        </div>

        {/* Frame Presets */}
        <div className="form-group">
          <label>Duration</label>
          <div className="preset-chips">
            {FRAME_PRESETS.map((p) => (
              <button
                key={p.frames}
                className={`preset-chip ${numFrames === p.frames ? 'selected' : ''}`}
                onClick={() => setNumFrames(p.frames)}
                style={numFrames === p.frames ? {
                  borderColor: 'var(--color-primary-500)',
                  color: 'var(--color-text-primary)',
                  background: 'rgba(79, 70, 229, 0.1)',
                } : {}}
              >
                {p.label} ({p.frames}f)
              </button>
            ))}
          </div>
          {durationHint && <div className="frame-duration-hint">{durationHint}</div>}
        </div>

        {/* Advanced Params */}
        <details className="video-advanced-params">
          <summary>Advanced Settings</summary>
          <div className="video-params-grid" style={{ marginTop: 'var(--space-3)' }}>
            <div className="form-group">
              <label>Width</label>
              <input
                type="number"
                value={width}
                onChange={(e) => setWidth(parseInt(e.target.value, 10) || 768)}
                min={256}
                max={1920}
                step={32}
              />
            </div>
            <div className="form-group">
              <label>Height</label>
              <input
                type="number"
                value={height}
                onChange={(e) => setHeight(parseInt(e.target.value, 10) || 512)}
                min={256}
                max={1920}
                step={32}
              />
            </div>
            <div className="form-group">
              <label>FPS</label>
              <input
                type="number"
                value={fps}
                onChange={(e) => setFps(parseInt(e.target.value, 10) || 24)}
                min={8}
                max={60}
              />
            </div>
            <div className="form-group">
              <label>Steps</label>
              <input
                type="number"
                value={numInferenceSteps}
                onChange={(e) => setNumInferenceSteps(parseInt(e.target.value, 10) || 30)}
                min={1}
                max={100}
              />
            </div>
            <div className="form-group">
              <label>CFG Scale</label>
              <input
                type="number"
                value={guidanceScale}
                onChange={(e) => setGuidanceScale(parseFloat(e.target.value) || 3.0)}
                min={1}
                max={20}
                step={0.5}
              />
            </div>
            <div className="form-group">
              <label>Seed</label>
              <input
                type="number"
                value={seed}
                onChange={(e) => setSeed(e.target.value)}
                placeholder="Random"
              />
            </div>
          </div>

          <div className="form-group" style={{ marginTop: 'var(--space-3)' }}>
            <label htmlFor="neg-prompt">Negative Prompt</label>
            <textarea
              id="neg-prompt"
              value={negativePrompt}
              onChange={(e) => setNegativePrompt(e.target.value)}
              placeholder="What to avoid (e.g. blurry, low quality, watermark)"
              rows={2}
            />
          </div>
        </details>

        {/* Error */}
        {error && (
          <div className="video-error">
            <span>{error}</span>
          </div>
        )}

        {/* Submit Flash */}
        {submitFlash && (
          <div className="video-submit-flash">
            Video generation started! Check the job queue for progress.
          </div>
        )}

        {/* Generate Button */}
        <button
          className="btn-generate-video"
          onClick={handleGenerate}
          disabled={submitting || !prompt.trim()}
        >
          {submitting ? 'Starting...' : 'Generate Video'}
        </button>
      </div>

      {/* Results */}
      {results.length > 0 && (
        <div className="video-results-list">
          <h3>Generated Videos ({results.length})</h3>
          {results.map((r) => (
            <div key={r.job_id} className="video-result-card">
              <video controls preload="metadata" src={r.video_url} />
              <div className="video-result-meta">
                <span className="meta-text">
                  {r.prompt.slice(0, 60)}...
                </span>
                <span className="meta-text">
                  {r.seed !== null && <span>Seed: {r.seed} | </span>}
                  {r.generation_time_ms !== null && (
                    <span>{(r.generation_time_ms / 1000).toFixed(1)}s</span>
                  )}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Setup Help */}
      <div className="video-help">
        <details className="setup-instructions">
          <summary><h4>Setup Instructions</h4></summary>
          <p>LTX-Video 2 requires a custom RunPod serverless endpoint.</p>
          <p>1. Deploy LTX-Video 2 as a RunPod serverless worker</p>
          <p>2. Add to your <code>.env</code> file:</p>
          <p>
            <code>RUNPOD_API_KEY=your_key</code><br />
            <code>RUNPOD_LTX_VIDEO_ENDPOINT_ID=your_endpoint_id</code>
          </p>
          <p>3. Restart the backend server</p>
        </details>
      </div>
    </section>
  )
}

export default VideoGenerator
