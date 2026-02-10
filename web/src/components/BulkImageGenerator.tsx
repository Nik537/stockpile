import { useCallback, useEffect, useRef, useState } from 'react'
import {
  BulkImageModel,
  BulkImagePrompt,
  BulkImagePromptsResponse,
  BulkImageResult,
  BulkImageStatus,
  BulkImageWSMessage,
} from '../types'
import './BulkImageGenerator.css'
import { API_BASE, getWsUrl } from '../config'

// Step type for the workflow
type WorkflowStep = 'input' | 'review' | 'generating' | 'complete'

function BulkImageGenerator() {
  // Workflow state
  const [step, setStep] = useState<WorkflowStep>('input')

  // Step 1: Input state
  const [metaPrompt, setMetaPrompt] = useState('')
  const [count, setCount] = useState(50)
  const [isGeneratingPrompts, setIsGeneratingPrompts] = useState(false)

  // Step 2: Review state
  const [jobId, setJobId] = useState<string | null>(null)
  const [prompts, setPrompts] = useState<BulkImagePrompt[]>([])
  const [model, setModel] = useState<BulkImageModel>('runware-flux-klein-4b')
  const [width, setWidth] = useState(1024)
  const [height, setHeight] = useState(1024)
  const [estimatedCost, setEstimatedCost] = useState(0)
  const [estimatedTime, setEstimatedTime] = useState(0)

  // Step 3: Generation state
  const [_status, setStatus] = useState<BulkImageStatus>('pending')
  const [completedCount, setCompletedCount] = useState(0)
  const [failedCount, setFailedCount] = useState(0)
  const [totalCount, setTotalCount] = useState(0)
  const [results, setResults] = useState<BulkImageResult[]>([])
  const [totalCost, setTotalCost] = useState(0)

  // Error state
  const [error, setError] = useState<string | null>(null)

  // WebSocket ref
  const wsRef = useRef<WebSocket | null>(null)

  // Model pricing per megapixel
  const MODEL_PRICING: Record<BulkImageModel, number> = {
    'gemini-flash': 0,  // FREE 500/day
    'runware-flux-klein-4b': 0.0006,
    'runware-flux-klein-9b': 0.00078,
    'runware-z-image': 0.0006,
    'nano-banana-pro': 0.04,
  }

  // Recalculate cost estimate when model/dimensions change
  useEffect(() => {
    if (prompts.length > 0) {
      const pricePerImage = MODEL_PRICING[model] || 0.001
      setEstimatedCost(prompts.length * pricePerImage)

      // Estimate time: Klein is fastest, then Schnell, Gemini is medium
      const fastModels = ['runware-flux-klein-4b', 'runware-z-image', 'runware-flux-klein-9b']
      const timePerImage = fastModels.includes(model) ? 3 : 6
      const batches = Math.ceil(prompts.length / 10)
      setEstimatedTime(batches * timePerImage)
    }
  }, [prompts.length, model, width, height])

  // Handle WebSocket messages
  const handleWSMessage = useCallback((message: BulkImageWSMessage) => {
    switch (message.type) {
      case 'status':
        if (message.status) setStatus(message.status)
        if (message.total_count !== undefined) setTotalCount(message.total_count)
        if (message.completed_count !== undefined) setCompletedCount(message.completed_count)
        if (message.failed_count !== undefined) setFailedCount(message.failed_count)
        break

      case 'image_complete':
      case 'image_failed':
        if (message.completed_count !== undefined) setCompletedCount(message.completed_count)
        if (message.failed_count !== undefined) setFailedCount(message.failed_count)
        if (message.index !== undefined && message.prompt) {
          const newResult: BulkImageResult = {
            index: message.index,
            prompt: message.prompt,
            image_url: message.image_url || null,
            width: width,
            height: height,
            generation_time_ms: 0,
            status: message.type === 'image_complete' ? 'completed' : 'failed',
            error: message.error,
          }
          setResults((prev) => {
            // Replace if exists, otherwise add
            const existing = prev.findIndex((r) => r.index === message.index)
            if (existing >= 0) {
              const updated = [...prev]
              updated[existing] = newResult
              return updated
            }
            return [...prev, newResult].sort((a, b) => a.index - b.index)
          })
        }
        break

      case 'complete':
        setStatus('completed')
        setStep('complete')
        if (message.total_cost !== undefined) setTotalCost(message.total_cost)
        if (message.results) setResults(message.results)
        break

      case 'error':
        setStatus('failed')
        setError(message.message || 'Generation failed')
        break
    }
  }, [width, height])

  // Connect to WebSocket when generating
  useEffect(() => {
    if (!jobId || step !== 'generating') return

    const wsUrl = getWsUrl(`/ws/bulk-image/${jobId}`)
    const ws = new WebSocket(wsUrl)

    ws.onopen = () => {
      console.log(`WebSocket connected for bulk image job ${jobId}`)
    }

    ws.onmessage = (event) => {
      try {
        const message: BulkImageWSMessage = JSON.parse(event.data)
        handleWSMessage(message)
      } catch (e) {
        console.error('Failed to parse WebSocket message:', e)
      }
    }

    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
      setError('WebSocket connection error')
    }

    ws.onclose = () => {
      console.log(`WebSocket disconnected for bulk image job ${jobId}`)
    }

    wsRef.current = ws

    return () => {
      ws.close()
      wsRef.current = null
    }
  }, [jobId, step, handleWSMessage])

  // Step 1: Generate prompts
  const handleGeneratePrompts = async () => {
    if (!metaPrompt.trim()) {
      setError('Please enter a creative concept')
      return
    }

    setIsGeneratingPrompts(true)
    setError(null)

    try {
      const response = await fetch(`${API_BASE}/api/bulk-image/prompts`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          meta_prompt: metaPrompt.trim(),
          count,
        }),
      })

      if (!response.ok) {
        const data = await response.json()
        throw new Error(data.detail || 'Failed to generate prompts')
      }

      const data: BulkImagePromptsResponse = await response.json()
      setJobId(data.job_id)
      setPrompts(data.prompts)
      setEstimatedCost(data.estimated_cost)
      setEstimatedTime(data.estimated_time_seconds)
      setStep('review')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate prompts')
    } finally {
      setIsGeneratingPrompts(false)
    }
  }

  // Step 2: Start image generation
  const handleStartGeneration = async () => {
    if (!jobId || prompts.length === 0) {
      setError('No prompts to generate')
      return
    }

    setError(null)
    setResults([])
    setCompletedCount(0)
    setFailedCount(0)
    setTotalCount(prompts.length)
    setStep('generating')

    try {
      const response = await fetch(`${API_BASE}/api/bulk-image/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          job_id: jobId,
          prompts: prompts.map((p) => ({
            index: p.index,
            prompt: p.prompt,
            rendering_style: p.rendering_style,
            mood: p.mood,
            composition: p.composition,
            has_text_space: p.has_text_space,
          })),
          model,
          width,
          height,
        }),
      })

      if (!response.ok) {
        const data = await response.json()
        throw new Error(data.detail || 'Failed to start generation')
      }

      setStatus('generating_images')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start generation')
      setStep('review')
    }
  }

  // Update a single prompt
  const handlePromptChange = (index: number, newPrompt: string) => {
    setPrompts((prev) =>
      prev.map((p) => (p.index === index ? { ...p, prompt: newPrompt } : p))
    )
  }

  // Delete a prompt
  const handleDeletePrompt = (index: number) => {
    setPrompts((prev) => prev.filter((p) => p.index !== index))
  }

  // Download ZIP
  const handleDownloadZip = async () => {
    if (!jobId) return

    try {
      const response = await fetch(`${API_BASE}/api/bulk-image/${jobId}/download`)
      if (!response.ok) {
        throw new Error('Download failed')
      }

      const blob = await response.blob()
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `bulk_images_${jobId.slice(0, 8)}.zip`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Download failed')
    }
  }

  // Reset to start new batch
  const handleStartNew = () => {
    setStep('input')
    setJobId(null)
    setPrompts([])
    setResults([])
    setMetaPrompt('')
    setCount(50)
    setCompletedCount(0)
    setFailedCount(0)
    setTotalCount(0)
    setTotalCost(0)
    setError(null)
    setStatus('pending')
  }

  // Dimension presets with exact ratios (RunPod max is 1536)
  const dimensionPresets = [
    { label: '1:1 (1024)', w: 1024, h: 1024 },
    { label: '16:9 (1280x720)', w: 1280, h: 720 },
    { label: '9:16 (720x1280)', w: 720, h: 1280 },
    { label: '4:3 (1024x768)', w: 1024, h: 768 },
    { label: '3:4 (768x1024)', w: 768, h: 1024 },
    { label: '21:9 (1512x648)', w: 1512, h: 648 },
    { label: '4:5 (1024x1280)', w: 1024, h: 1280 },
  ]

  // Example prompts
  const examplePrompts = [
    'Creative ad concepts for spring cleaning products with 3D printing theme',
    'Tech startup logo variations with modern minimalist aesthetic',
    'Food photography styles for artisan coffee brand',
    'Lifestyle photography for eco-friendly fashion brand',
    'Abstract backgrounds for meditation app marketing',
  ]

  return (
    <section className="bulkimage-section">
      <h2>Bulk Image Generator</h2>
      <p className="section-description">
        Generate 10-200 unique images from a single creative concept. AI creates diverse prompts, you review and edit, then generate all images in parallel.
      </p>

      {/* Error Display */}
      {error && (
        <div className="bulkimage-error">
          <span className="error-icon">!</span>
          <span>{error}</span>
          <button onClick={() => setError(null)} className="btn-dismiss">Dismiss</button>
        </div>
      )}

      {/* Step 1: Input */}
      {step === 'input' && (
        <div className="bulkimage-step step-input">
          <div className="step-header">
            <span className="step-number">1</span>
            <h3>Describe Your Creative Concept</h3>
          </div>

          <div className="form-group">
            <label htmlFor="meta-prompt">Creative Concept</label>
            <textarea
              id="meta-prompt"
              value={metaPrompt}
              onChange={(e) => setMetaPrompt(e.target.value)}
              placeholder="Describe the theme, style, or concept for your images. Be specific about the subject, mood, and use case."
              rows={4}
            />
            <div className="example-prompts">
              <span className="examples-label">Examples:</span>
              {examplePrompts.map((example, i) => (
                <button
                  key={i}
                  className="example-btn"
                  onClick={() => setMetaPrompt(example)}
                >
                  {example.slice(0, 40)}...
                </button>
              ))}
            </div>
          </div>

          <div className="form-group">
            <label htmlFor="count">Number of Images: {count}</label>
            <input
              type="range"
              id="count"
              min={10}
              max={200}
              step={10}
              value={count}
              onChange={(e) => setCount(parseInt(e.target.value, 10))}
            />
            <div className="count-labels">
              <span>10</span>
              <span>50</span>
              <span>100</span>
              <span>150</span>
              <span>200</span>
            </div>
          </div>

          <button
            className="btn-primary btn-generate-prompts"
            onClick={handleGeneratePrompts}
            disabled={isGeneratingPrompts || !metaPrompt.trim()}
          >
            {isGeneratingPrompts ? (
              <>
                <span className="loading-spinner-sm" />
                Generating Prompts...
              </>
            ) : (
              <>Generate {count} Prompts</>
            )}
          </button>
        </div>
      )}

      {/* Step 2: Review */}
      {step === 'review' && (
        <div className="bulkimage-step step-review">
          <div className="step-header">
            <span className="step-number">2</span>
            <h3>Review &amp; Edit Prompts</h3>
          </div>

          {/* Settings Panel */}
          <div className="settings-panel">
            <div className="settings-row">
              <div className="setting-group">
                <label>Model</label>
                <div className="model-options">
                  {/* FREE tier */}
                  <label className={`model-option ${model === 'gemini-flash' ? 'selected' : ''}`}>
                    <input
                      type="radio"
                      name="model"
                      value="gemini-flash"
                      checked={model === 'gemini-flash'}
                      onChange={(e) => setModel(e.target.value as BulkImageModel)}
                    />
                    <div className="model-info">
                      <span className="model-name">Gemini Flash</span>
                      <span className="model-badge free">FREE</span>
                      <span className="model-price">500/day</span>
                    </div>
                  </label>
                  {/* Budget tier - Runware */}
                  <label className={`model-option ${model === 'runware-flux-klein-4b' ? 'selected' : ''}`}>
                    <input
                      type="radio"
                      name="model"
                      value="runware-flux-klein-4b"
                      checked={model === 'runware-flux-klein-4b'}
                      onChange={(e) => setModel(e.target.value as BulkImageModel)}
                    />
                    <div className="model-info">
                      <span className="model-name">Flux Klein 4B</span>
                      <span className="model-badge budget">Cheapest</span>
                      <span className="model-price">$0.0006/img</span>
                    </div>
                  </label>
                  <label className={`model-option ${model === 'runware-z-image' ? 'selected' : ''}`}>
                    <input
                      type="radio"
                      name="model"
                      value="runware-z-image"
                      checked={model === 'runware-z-image'}
                      onChange={(e) => setModel(e.target.value as BulkImageModel)}
                    />
                    <div className="model-info">
                      <span className="model-name">Z-Image</span>
                      <span className="model-badge budget">Fast</span>
                      <span className="model-price">$0.0006/img</span>
                    </div>
                  </label>
                  <label className={`model-option ${model === 'runware-flux-klein-9b' ? 'selected' : ''}`}>
                    <input
                      type="radio"
                      name="model"
                      value="runware-flux-klein-9b"
                      checked={model === 'runware-flux-klein-9b'}
                      onChange={(e) => setModel(e.target.value as BulkImageModel)}
                    />
                    <div className="model-info">
                      <span className="model-name">Flux Klein 9B</span>
                      <span className="model-badge quality">Quality</span>
                      <span className="model-price">$0.00078/img</span>
                    </div>
                  </label>
                  {/* RunPod tier */}
                  <label className={`model-option ${model === 'nano-banana-pro' ? 'selected' : ''}`}>
                    <input
                      type="radio"
                      name="model"
                      value="nano-banana-pro"
                      checked={model === 'nano-banana-pro'}
                      onChange={(e) => setModel(e.target.value as BulkImageModel)}
                    />
                    <div className="model-info">
                      <span className="model-name">Nano Banana Pro</span>
                      <span className="model-badge quality">Edit/Inpaint</span>
                      <span className="model-price">$0.04/img</span>
                    </div>
                  </label>
                </div>
              </div>

              <div className="setting-group">
                <label>Dimensions</label>
                <div className="dimension-presets">
                  {dimensionPresets.map((preset) => (
                    <button
                      key={preset.label}
                      className={`preset-btn ${width === preset.w && height === preset.h ? 'selected' : ''}`}
                      onClick={() => {
                        setWidth(preset.w)
                        setHeight(preset.h)
                      }}
                    >
                      {preset.label}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            <div className="estimate-panel">
              <div className="estimate-item">
                <span className="estimate-label">Images</span>
                <span className="estimate-value">{prompts.length}</span>
              </div>
              <div className="estimate-item">
                <span className="estimate-label">Est. Cost</span>
                <span className="estimate-value">${estimatedCost.toFixed(2)}</span>
              </div>
              <div className="estimate-item">
                <span className="estimate-label">Est. Time</span>
                <span className="estimate-value">~{estimatedTime}s</span>
              </div>
            </div>
          </div>

          {/* Prompts List */}
          <div className="prompts-list">
            <div className="prompts-header">
              <span>{prompts.length} prompts</span>
              <button className="btn-text" onClick={() => setStep('input')}>
                Regenerate All
              </button>
            </div>
            <div className="prompts-scroll">
              {prompts.map((prompt) => (
                <div key={prompt.index} className="prompt-card">
                  <div className="prompt-meta">
                    <span className="prompt-index">#{prompt.index + 1}</span>
                    <span className={`prompt-tag tag-style`}>{prompt.rendering_style}</span>
                    <span className={`prompt-tag tag-${prompt.mood}`}>{prompt.mood}</span>
                    <span className={`prompt-tag tag-composition`}>{prompt.composition}</span>
                    {prompt.has_text_space && (
                      <span className="prompt-tag tag-text">has text</span>
                    )}
                    <button
                      className="btn-delete-prompt"
                      onClick={() => handleDeletePrompt(prompt.index)}
                      title="Delete prompt"
                    >
                      x
                    </button>
                  </div>
                  <textarea
                    value={prompt.prompt}
                    onChange={(e) => handlePromptChange(prompt.index, e.target.value)}
                    rows={2}
                  />
                </div>
              ))}
            </div>
          </div>

          <div className="step-actions">
            <button className="btn-secondary" onClick={handleStartNew}>
              Start Over
            </button>
            <button
              className="btn-primary btn-start-generation"
              onClick={handleStartGeneration}
              disabled={prompts.length === 0}
            >
              Generate {prompts.length} Images
            </button>
          </div>
        </div>
      )}

      {/* Step 3: Generating */}
      {step === 'generating' && (
        <div className="bulkimage-step step-generating">
          <div className="step-header">
            <span className="step-number">3</span>
            <h3>Generating Images</h3>
          </div>

          <div className="progress-panel">
            <div className="progress-bar">
              <div
                className="progress-fill"
                style={{
                  width: `${totalCount > 0 ? ((completedCount + failedCount) / totalCount) * 100 : 0}%`,
                }}
              />
            </div>
            <div className="progress-stats">
              <span className="stat completed">{completedCount} completed</span>
              {failedCount > 0 && <span className="stat failed">{failedCount} failed</span>}
              <span className="stat total">/ {totalCount}</span>
            </div>
          </div>

          {/* Results Grid */}
          <div className="results-grid">
            {prompts.map((prompt) => {
              const result = results.find((r) => r.index === prompt.index)
              return (
                <div key={prompt.index} className={`result-card ${result?.status || 'pending'}`}>
                  {result?.status === 'completed' && result.image_url ? (
                    <img src={result.image_url} alt={`Generated image ${prompt.index + 1}`} loading="lazy" />
                  ) : result?.status === 'failed' ? (
                    <div className="result-failed">
                      <span className="failed-icon">!</span>
                      <span className="failed-text">{result.error || 'Failed'}</span>
                    </div>
                  ) : (
                    <div className="result-pending">
                      <span className="loading-spinner-sm" />
                    </div>
                  )}
                  <div className="result-prompt" title={prompt.prompt}>
                    #{prompt.index + 1}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Step 4: Complete */}
      {step === 'complete' && (
        <div className="bulkimage-step step-complete">
          <div className="step-header">
            <span className="step-number check">OK</span>
            <h3>Generation Complete</h3>
          </div>

          <div className="complete-stats">
            <div className="stat-card">
              <span className="stat-value">{completedCount}</span>
              <span className="stat-label">Successful</span>
            </div>
            {failedCount > 0 && (
              <div className="stat-card failed">
                <span className="stat-value">{failedCount}</span>
                <span className="stat-label">Failed</span>
              </div>
            )}
            <div className="stat-card">
              <span className="stat-value">${totalCost.toFixed(3)}</span>
              <span className="stat-label">Total Cost</span>
            </div>
          </div>

          {/* Results Grid */}
          <div className="results-grid completed">
            {results
              .filter((r) => r.status === 'completed' && r.image_url)
              .map((result) => (
                <div key={result.index} className="result-card completed">
                  <img src={result.image_url!} alt={`Generated image ${result.index + 1}`} loading="lazy" />
                  <div className="result-overlay">
                    <span className="result-prompt" title={result.prompt.prompt}>
                      {result.prompt.prompt.slice(0, 50)}...
                    </span>
                    <a
                      href={result.image_url!}
                      download={`image_${result.index + 1}.jpg`}
                      className="btn-download-single"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      Download
                    </a>
                  </div>
                </div>
              ))}
          </div>

          <div className="step-actions">
            <button className="btn-secondary" onClick={handleStartNew}>
              Start New Batch
            </button>
            <button className="btn-primary" onClick={handleDownloadZip}>
              Download All as ZIP
            </button>
          </div>
        </div>
      )}
    </section>
  )
}

export default BulkImageGenerator
