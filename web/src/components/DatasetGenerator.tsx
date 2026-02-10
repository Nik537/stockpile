import { useCallback, useEffect, useRef, useState } from 'react'
import { DatasetGenMode, ImageGenModel } from '../types'
import { useDatasetStore } from '../stores/useDatasetStore'
import './DatasetGenerator.css'
import { API_BASE, getWsUrl } from '../config'

const MODEL_INFO = [
  { id: 'runware-flux-klein-4b' as ImageGenModel, name: 'Flux Klein 4B', price: 0.0006, priceLabel: '$0.0006/img', badge: 'Cheapest', badgeClass: 'budget' },
  { id: 'runware-z-image' as ImageGenModel, name: 'Z-Image Turbo', price: 0.0006, priceLabel: '$0.0006/img', badge: 'Fast', badgeClass: '' },
  { id: 'runware-flux-klein-9b' as ImageGenModel, name: 'Flux Klein 9B', price: 0.0008, priceLabel: '$0.0008/img', badge: 'Quality', badgeClass: 'quality' },
  { id: 'gemini-flash' as ImageGenModel, name: 'Gemini Flash', price: 0, priceLabel: 'FREE', badge: 'Free', badgeClass: 'free' },
  { id: 'nano-banana-pro' as ImageGenModel, name: 'Nano Banana Pro', price: 0.04, priceLabel: '~$0.04/img', badge: 'Best', badgeClass: 'premium' },
]

const ASPECT_RATIOS = [
  { label: '1:1', width: 1024, height: 1024 },
  { label: '16:9', width: 1920, height: 1080 },
  { label: '9:16', width: 1080, height: 1920 },
  { label: '4:3', width: 1024, height: 768 },
  { label: '3:4', width: 768, height: 1024 },
]

const LAYERED_PRESETS: Record<string, { name: string; triggerWord: string; numSets: number }> = {
  character: { name: 'Character Design', triggerWord: 'CHARDESIGN', numSets: 5 },
  architecture: { name: 'Architecture', triggerWord: 'ARCHVIZ', numSets: 5 },
  food: { name: 'Food Composition', triggerWord: 'FOODPHOTO', numSets: 5 },
  interior: { name: 'Interior Design', triggerWord: 'INTERIOR', numSets: 5 },
  fashion: { name: 'Fashion/Outfit', triggerWord: 'FASHION', numSets: 5 },
  product: { name: 'Product Photography', triggerWord: 'PRODUCT', numSets: 5 },
  custom: { name: 'Custom', triggerWord: '', numSets: 3 },
}

function DatasetGenerator() {
  // Mode and model
  const [mode, setMode] = useState<DatasetGenMode>('pair')
  const [model, setModel] = useState<ImageGenModel>('runware-flux-klein-4b')
  const [llmModel] = useState('gemini-flash')

  // Settings
  const [aspectRatio, setAspectRatio] = useState('1:1')
  const [triggerWord, setTriggerWord] = useState('')
  const [useVisionCaption, setUseVisionCaption] = useState(true)
  const [customSystemPrompt, setCustomSystemPrompt] = useState('')
  const [numItems, setNumItems] = useState(10)
  const [maxConcurrent, setMaxConcurrent] = useState(3)

  // Mode-specific inputs
  const [theme, setTheme] = useState('')
  const [transformation, setTransformation] = useState('')
  const [actionName, setActionName] = useState('')
  const [referenceImage, setReferenceImage] = useState<File | null>(null)
  const [referencePreview, setReferencePreview] = useState<string | null>(null)
  const [layeredUseCase, setLayeredUseCase] = useState('character')
  const [elementsDescription, setElementsDescription] = useState('')
  const [finalImageDescription, setFinalImageDescription] = useState('')

  // UI state
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Store
  const store = useDatasetStore()

  // WebSocket ref
  const wsRef = useRef<WebSocket | null>(null)
  const logsEndRef = useRef<HTMLDivElement>(null)

  // Auto-scroll logs
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [store.logs])

  // Update trigger word when layered use case changes
  useEffect(() => {
    if (mode === 'layered' && layeredUseCase !== 'custom') {
      const preset = LAYERED_PRESETS[layeredUseCase]
      if (preset) setTriggerWord(preset.triggerWord)
    }
  }, [layeredUseCase, mode])

  // Calculate cost estimate
  const calculateCost = useCallback(() => {
    const modelInfo = MODEL_INFO.find((m) => m.id === model)
    if (!modelInfo) return 0
    const price = modelInfo.price
    switch (mode) {
      case 'pair':
        return numItems * 2 * price
      case 'single':
      case 'reference':
        return numItems * price
      case 'layered':
        return numItems * 6 * price
      default:
        return 0
    }
  }, [model, mode, numItems])

  const estimatedCost = calculateCost()

  // Get selected dimensions
  const getSelectedDimensions = () => {
    const ratio = ASPECT_RATIOS.find((r) => r.label === aspectRatio)
    return ratio || ASPECT_RATIOS[0]
  }

  // File to base64
  const fileToBase64 = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => {
        const result = reader.result as string
        // Strip data URL prefix to get raw base64
        const base64 = result.split(',')[1] || result
        resolve(base64)
      }
      reader.onerror = reject
      reader.readAsDataURL(file)
    })
  }

  // Handle reference image upload
  const handleRefImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setReferenceImage(file)
      setReferencePreview(URL.createObjectURL(file))
    }
  }

  // Connect WebSocket for job progress
  const connectWebSocket = useCallback((jobId: string) => {
    if (wsRef.current) {
      wsRef.current.close()
    }

    const wsUrl = getWsUrl(`/ws/dataset/${jobId}`)
    const ws = new WebSocket(wsUrl)

    ws.onopen = () => {
      store.addLog('Connected to server')
    }

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data)
        switch (msg.type) {
          case 'status':
            store.updateStatus(msg.status)
            if (msg.completed_count !== undefined) {
              store.updateProgress(
                msg.completed_count,
                msg.failed_count || 0,
                msg.total_cost || 0
              )
            }
            {
              const s = useDatasetStore.getState()
              store.addLog(`Status: ${msg.status} (${msg.completed_count ?? s.completedCount}/${msg.total_count ?? s.totalCount})`)
            }
            break

          case 'item_complete': {
            const item = msg.item || msg
            store.addItem({
              index: item.index,
              status: item.status || 'completed',
              startUrl: item.start_url,
              endUrl: item.end_url,
              imageUrl: item.image_url,
              caption: item.caption || '',
              cost: item.cost || 0,
            })
            store.updateProgress(
              msg.completed_count || 0,
              msg.failed_count || 0,
              msg.total_cost || 0
            )
            store.addLog(`Item ${(item.index ?? 0) + 1} completed`)
            break
          }

          case 'item_failed': {
            const failedItem = msg.item || msg
            store.addItem({
              index: failedItem.index,
              status: 'failed',
              caption: '',
              error: failedItem.error || msg.error,
              cost: 0,
            })
            store.updateProgress(
              msg.completed_count || 0,
              msg.failed_count || 0,
              msg.total_cost || 0
            )
            store.addLog(`Item ${(failedItem.index ?? 0) + 1} failed: ${failedItem.error || msg.error}`)
          }
            break

          case 'complete':
            store.setComplete(msg.total_cost || 0, msg.zip_path || '')
            store.updateProgress(
              msg.completed_count || 0,
              msg.failed_count || 0,
              msg.total_cost || 0
            )
            store.addLog(`Generation complete! ${msg.completed_count}/${msg.total_count} items, cost: $${(msg.total_cost || 0).toFixed(4)}`)
            break

          case 'error':
            store.setError(msg.message || 'Unknown error')
            store.addLog(`Error: ${msg.message}`)
            break
        }
      } catch (e) {
        console.error('Failed to parse WebSocket message:', e)
      }
    }

    ws.onerror = () => {
      store.addLog('WebSocket connection error')
    }

    ws.onclose = () => {
      store.addLog('Disconnected from server')
    }

    wsRef.current = ws
  }, [store])

  // Cleanup WebSocket on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [])

  // Start generation
  const handleGenerate = async () => {
    if (!theme.trim()) {
      setError('Please enter a theme')
      return
    }

    if (mode === 'reference' && !referenceImage) {
      setError('Please upload a reference image')
      return
    }

    setSubmitting(true)
    setError(null)

    try {
      const dims = getSelectedDimensions()

      const body: Record<string, unknown> = {
        mode,
        theme: theme.trim(),
        model,
        llm_model: llmModel,
        num_items: numItems,
        max_concurrent: maxConcurrent,
        aspect_ratio: aspectRatio,
        trigger_word: triggerWord,
        use_vision_caption: useVisionCaption,
        custom_system_prompt: customSystemPrompt || '',
        width: dims.width,
        height: dims.height,
      }

      // Mode-specific fields
      if (mode === 'pair') {
        body.transformation = transformation
        body.action_name = actionName
      }

      if (mode === 'reference' && referenceImage) {
        body.reference_image_base64 = await fileToBase64(referenceImage)
      }

      if (mode === 'layered') {
        body.layered_use_case = layeredUseCase
        body.elements_description = elementsDescription
        body.final_image_description = finalImageDescription
      }

      const response = await fetch(`${API_BASE}/api/dataset/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })

      if (!response.ok) {
        const data = await response.json()
        throw new Error(data.detail || 'Generation failed')
      }

      const data = await response.json()
      store.setJob(data.job_id, data.total_count, data.estimated_cost || estimatedCost)
      connectWebSocket(data.job_id)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Generation failed')
    } finally {
      setSubmitting(false)
    }
  }

  // Cancel job
  const handleCancel = async () => {
    if (!store.jobId) return

    try {
      await fetch(`${API_BASE}/api/dataset/${store.jobId}`, {
        method: 'DELETE',
      })
      store.addLog('Job cancelled')
      store.updateStatus('cancelled')
      if (wsRef.current) {
        wsRef.current.close()
      }
    } catch (e) {
      console.error('Failed to cancel job:', e)
    }
  }

  // Download ZIP
  const handleDownload = async () => {
    if (!store.jobId) return

    try {
      const response = await fetch(`${API_BASE}/api/dataset/${store.jobId}/download`)
      if (!response.ok) throw new Error('Download failed')

      const blob = await response.blob()
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `dataset_${store.jobId.slice(0, 8)}.zip`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Download failed')
    }
  }

  // Reset
  const handleReset = () => {
    if (wsRef.current) {
      wsRef.current.close()
    }
    store.reset()
    setError(null)
  }

  const isGenerating = ['pending', 'generating_prompts', 'generating_images', 'captioning', 'packaging'].includes(store.status)
  const isComplete = store.status === 'completed'
  const progress = store.totalCount > 0
    ? Math.round(((store.completedCount + store.failedCount) / store.totalCount) * 100)
    : 0

  const statusLabel: Record<string, string> = {
    pending: 'Starting...',
    generating_prompts: 'Generating prompts...',
    generating_images: 'Generating images...',
    captioning: 'Captioning images...',
    packaging: 'Packaging ZIP...',
    completed: 'Complete',
    failed: 'Failed',
    cancelled: 'Cancelled',
  }

  return (
    <section className="dataset-section">
      <h2>LoRA Dataset Generator</h2>
      <p className="section-description">
        Generate training datasets for LoRA fine-tuning with paired images, captions, and ZIP packaging.
      </p>

      {/* Error */}
      {error && (
        <div className="dataset-error">
          <span className="error-icon">!</span>
          <span>{error}</span>
          <button onClick={() => setError(null)} className="btn-dismiss">Dismiss</button>
        </div>
      )}

      {/* Mode Selector */}
      <div className="dataset-mode-selector">
        <h3>Generation Mode</h3>
        <div className="dataset-mode-tabs">
          <button
            className={`dataset-mode-tab ${mode === 'pair' ? 'active' : ''}`}
            onClick={() => setMode('pair')}
            disabled={isGenerating}
          >
            <span className="mode-icon">&#x21C4;</span>
            Pair Mode
          </button>
          <button
            className={`dataset-mode-tab ${mode === 'single' ? 'active' : ''}`}
            onClick={() => setMode('single')}
            disabled={isGenerating}
          >
            <span className="mode-icon">&#x1F5BC;</span>
            Single Image
          </button>
          <button
            className={`dataset-mode-tab ${mode === 'reference' ? 'active' : ''}`}
            onClick={() => setMode('reference')}
            disabled={isGenerating}
          >
            <span className="mode-icon">&#x1F4F7;</span>
            Reference Image
          </button>
          <button
            className={`dataset-mode-tab ${mode === 'layered' ? 'active' : ''}`}
            onClick={() => setMode('layered')}
            disabled={isGenerating}
          >
            <span className="mode-icon">&#x1F9E9;</span>
            Layered Grid
          </button>
        </div>
      </div>

      {/* Settings */}
      <div className="dataset-settings">
        <div className="dataset-settings-grid">
          {/* Left column: Model + Aspect Ratio */}
          <div>
            <div className="dataset-model-selector">
              <label>Image Model</label>
              <div className="dataset-model-cards">
                {MODEL_INFO.map((m) => (
                  <button
                    key={m.id}
                    className={`dataset-model-card ${model === m.id ? 'selected' : ''}`}
                    onClick={() => setModel(m.id)}
                    disabled={isGenerating}
                  >
                    <div className="model-card-header">
                      <span className="model-name">{m.name}</span>
                      {m.badge && <span className={`model-badge ${m.badgeClass}`}>{m.badge}</span>}
                    </div>
                    <span className="model-price">{m.priceLabel}</span>
                  </button>
                ))}
              </div>
            </div>

            <div className="form-group">
              <label>Aspect Ratio</label>
              <div className="dimension-presets">
                {ASPECT_RATIOS.map((r) => (
                  <button
                    key={r.label}
                    className={`preset-btn ${aspectRatio === r.label ? 'selected' : ''}`}
                    onClick={() => setAspectRatio(r.label)}
                    disabled={isGenerating}
                  >
                    {r.label}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Right column: Other settings */}
          <div>
            <div className="form-group">
              <label htmlFor="ds-trigger">Trigger Word</label>
              <input
                id="ds-trigger"
                type="text"
                value={triggerWord}
                onChange={(e) => setTriggerWord(e.target.value)}
                placeholder="e.g. MYSTYLE"
                disabled={isGenerating}
              />
            </div>

            <div className="form-group">
              <label htmlFor="ds-num-items">Number of Items: {numItems}</label>
              <input
                id="ds-num-items"
                type="range"
                min={1}
                max={40}
                value={numItems}
                onChange={(e) => setNumItems(parseInt(e.target.value, 10))}
                disabled={isGenerating}
              />
            </div>

            <div className="form-group">
              <label htmlFor="ds-concurrent">Parallel Requests: {maxConcurrent}</label>
              <input
                id="ds-concurrent"
                type="range"
                min={1}
                max={10}
                value={maxConcurrent}
                onChange={(e) => setMaxConcurrent(parseInt(e.target.value, 10))}
                disabled={isGenerating}
              />
            </div>

            <div className="form-group">
              <label className="dataset-checkbox">
                <input
                  type="checkbox"
                  checked={useVisionCaption}
                  onChange={(e) => setUseVisionCaption(e.target.checked)}
                  disabled={isGenerating}
                />
                <span>Use vision-based captions</span>
              </label>
            </div>
          </div>
        </div>
      </div>

      {/* Mode-Specific Inputs */}
      <div className="dataset-inputs">
        <h3>
          {mode === 'pair' && 'Pair Mode Settings'}
          {mode === 'single' && 'Single Image Settings'}
          {mode === 'reference' && 'Reference Image Settings'}
          {mode === 'layered' && 'Layered Grid Settings'}
        </h3>

        {/* Theme (all modes) */}
        <div className="form-group">
          <label htmlFor="ds-theme">Theme</label>
          <input
            id="ds-theme"
            type="text"
            value={theme}
            onChange={(e) => setTheme(e.target.value)}
            placeholder={
              mode === 'pair'
                ? 'e.g. anime character transformations'
                : mode === 'single'
                  ? 'e.g. watercolor landscape paintings'
                  : mode === 'reference'
                    ? 'e.g. variations of product photography'
                    : 'e.g. character design sheets'
            }
            disabled={isGenerating}
          />
        </div>

        {/* Pair mode extras */}
        {mode === 'pair' && (
          <>
            <div className="form-group">
              <label htmlFor="ds-transform">Transformation</label>
              <input
                id="ds-transform"
                type="text"
                value={transformation}
                onChange={(e) => setTransformation(e.target.value)}
                placeholder="e.g. line art to colored, day to night"
                disabled={isGenerating}
              />
            </div>
            <div className="form-group">
              <label htmlFor="ds-action">Action Name</label>
              <input
                id="ds-action"
                type="text"
                value={actionName}
                onChange={(e) => setActionName(e.target.value)}
                placeholder="e.g. colorize, stylize"
                disabled={isGenerating}
              />
            </div>
          </>
        )}

        {/* Reference mode: file upload */}
        {mode === 'reference' && (
          <div className="form-group">
            <label>Reference Image</label>
            {referencePreview ? (
              <div className="dataset-ref-preview">
                <img src={referencePreview} alt="Reference preview" />
                <button
                  className="btn-remove"
                  onClick={() => {
                    setReferenceImage(null)
                    setReferencePreview(null)
                  }}
                >
                  Remove
                </button>
              </div>
            ) : (
              <label className="dataset-upload-area">
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleRefImageSelect}
                  disabled={isGenerating}
                />
                <div className="upload-content">
                  <span className="upload-icon">&#x1F4F7;</span>
                  <span>Click to upload reference image</span>
                </div>
              </label>
            )}
          </div>
        )}

        {/* Layered mode extras */}
        {mode === 'layered' && (
          <>
            <div className="form-group">
              <label htmlFor="ds-usecase">Use Case</label>
              <select
                id="ds-usecase"
                value={layeredUseCase}
                onChange={(e) => setLayeredUseCase(e.target.value)}
                disabled={isGenerating}
              >
                {Object.entries(LAYERED_PRESETS).map(([key, preset]) => (
                  <option key={key} value={key}>{preset.name}</option>
                ))}
              </select>
            </div>
            <div className="form-group">
              <label htmlFor="ds-elements">Elements Description</label>
              <textarea
                id="ds-elements"
                value={elementsDescription}
                onChange={(e) => setElementsDescription(e.target.value)}
                placeholder="Describe the individual elements/layers"
                rows={3}
                disabled={isGenerating}
              />
            </div>
            <div className="form-group">
              <label htmlFor="ds-final">Final Image Description</label>
              <input
                id="ds-final"
                type="text"
                value={finalImageDescription}
                onChange={(e) => setFinalImageDescription(e.target.value)}
                placeholder="Describe the final composed image"
                disabled={isGenerating}
              />
            </div>
          </>
        )}

        {/* Custom system prompt (collapsible, all modes) */}
        <details>
          <summary>Custom System Prompt</summary>
          <div className="form-group">
            <textarea
              value={customSystemPrompt}
              onChange={(e) => setCustomSystemPrompt(e.target.value)}
              placeholder="Override the default system prompt for prompt generation..."
              rows={3}
              disabled={isGenerating}
            />
          </div>
        </details>
      </div>

      {/* Cost Estimate */}
      <div className="dataset-cost-panel">
        <div className="dataset-cost-item">
          <span className="dataset-cost-label">Images</span>
          <span className="dataset-cost-value">{numItems}</span>
        </div>
        <div className="dataset-cost-item">
          <span className="dataset-cost-label">Est. Cost</span>
          <span className={`dataset-cost-value ${estimatedCost === 0 ? 'free' : ''}`}>
            {estimatedCost === 0 ? 'FREE' : `$${estimatedCost.toFixed(4)}`}
          </span>
        </div>
        <div className="dataset-cost-item">
          <span className="dataset-cost-label">Mode</span>
          <span className="dataset-cost-value" style={{ fontSize: 'var(--text-sm)' }}>
            {mode === 'pair' ? `${numItems} pairs (${numItems * 2} imgs)` :
             mode === 'layered' ? `${numItems} sets (${numItems * 6} imgs)` :
             `${numItems} images`}
          </span>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="dataset-actions">
        {!isGenerating && !isComplete && (
          <button
            className="dataset-btn-generate"
            onClick={handleGenerate}
            disabled={submitting || !theme.trim() || (mode === 'reference' && !referenceImage)}
          >
            {submitting ? (
              <>
                <span className="loading-spinner-sm" />
                Starting...
              </>
            ) : (
              <>Generate Dataset</>
            )}
          </button>
        )}

        {isGenerating && (
          <button className="dataset-btn-cancel" onClick={handleCancel}>
            Cancel
          </button>
        )}

        {isComplete && (
          <>
            <button className="dataset-btn-download" onClick={handleDownload}>
              Download ZIP
            </button>
            <button className="dataset-btn-reset" onClick={handleReset}>
              Start New
            </button>
          </>
        )}

        {(store.status === 'failed' || store.status === 'cancelled') && (
          <button className="dataset-btn-reset" onClick={handleReset}>
            Start Over
          </button>
        )}
      </div>

      {/* Progress Panel */}
      {(isGenerating || isComplete || store.status === 'failed' || store.status === 'cancelled') && (
        <div className="dataset-progress">
          <h3>Progress</h3>
          <div className="dataset-progress-bar">
            <div
              className="dataset-progress-fill"
              style={{ width: `${progress}%` }}
            />
          </div>
          <div className="dataset-progress-info">
            <span className="dataset-progress-text">
              {statusLabel[store.status] || store.status} - {progress}%
            </span>
            <div className="dataset-progress-stats">
              <span className="stat-completed">{store.completedCount} done</span>
              {store.failedCount > 0 && (
                <span className="stat-failed">{store.failedCount} failed</span>
              )}
              <span className="stat-cost">${store.totalCost.toFixed(4)}</span>
            </div>
          </div>

          {/* Logs */}
          {store.logs.length > 0 && (
            <div className="dataset-logs">
              {store.logs.map((log, i) => (
                <p key={i}>{log}</p>
              ))}
              <div ref={logsEndRef} />
            </div>
          )}
        </div>
      )}

      {/* Complete Stats */}
      {isComplete && (
        <div className="dataset-complete-stats">
          <div className="dataset-stat-card success">
            <span className="dataset-stat-value">{store.completedCount}</span>
            <span className="dataset-stat-label">Successful</span>
          </div>
          {store.failedCount > 0 && (
            <div className="dataset-stat-card error">
              <span className="dataset-stat-value">{store.failedCount}</span>
              <span className="dataset-stat-label">Failed</span>
            </div>
          )}
          <div className="dataset-stat-card">
            <span className="dataset-stat-value">${store.totalCost.toFixed(4)}</span>
            <span className="dataset-stat-label">Total Cost</span>
          </div>
        </div>
      )}

      {/* Results Grid */}
      {store.items.length > 0 && (
        <div className="dataset-results">
          <h3>Generated Items ({store.items.length})</h3>
          <div className="dataset-results-grid">
            {store.items.map((item) => (
              <div
                key={item.index}
                className={`dataset-result-card ${item.status}`}
              >
                {item.status === 'completed' ? (
                  <>
                    {/* Pair mode: side by side */}
                    {mode === 'pair' && item.startUrl && item.endUrl ? (
                      <div className="dataset-pair-images" style={{ position: 'relative' }}>
                        <img src={item.startUrl} alt={`Start ${item.index + 1}`} loading="lazy" />
                        <img src={item.endUrl} alt={`End ${item.index + 1}`} loading="lazy" />
                        <span className="dataset-pair-arrow">&#x2192;</span>
                      </div>
                    ) : (
                      <div className="dataset-single-image">
                        <img
                          src={item.imageUrl || item.startUrl || ''}
                          alt={`Generated ${item.index + 1}`}
                          loading="lazy"
                        />
                      </div>
                    )}
                    {item.caption && (
                      <div className="dataset-result-caption">{item.caption}</div>
                    )}
                  </>
                ) : item.status === 'failed' ? (
                  <div className="dataset-result-error">
                    <span className="failed-icon">!</span>
                    <span className="failed-text">{item.error || 'Failed'}</span>
                  </div>
                ) : (
                  <div className="dataset-result-status">
                    <span className="loading-spinner-sm" />
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </section>
  )
}

export default DatasetGenerator
