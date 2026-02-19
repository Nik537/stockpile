import { useCallback, useEffect, useRef, useState } from 'react'
import {
  StoryboardCharacter,
  StoryboardScene,
  StoryboardPlan,
  StoryboardSceneResult,
  StoryboardJobStatus,
  StoryboardWSMessage,
} from '../types'
import './StoryboardGenerator.css'
import { API_BASE, getWsUrl } from '../config'

type WorkflowStep = 'input' | 'review' | 'generating' | 'complete'

const ASPECT_RATIOS: Record<string, { w: number; h: number; label: string }> = {
  '9:16': { w: 1080, h: 1920, label: '9:16 (Reels/TikTok/Shorts)' },
  '16:9': { w: 1920, h: 1080, label: '16:9 (YouTube)' },
  '1:1': { w: 1080, h: 1080, label: '1:1 (Instagram)' },
}

const MODEL_PRICING: Record<string, number> = {
  'gemini-flash': 0,
  'runware-flux-klein-4b': 0.0006,
  'runware-flux-klein-9b': 0.00078,
  'runware-z-image': 0.0006,
  'qwen-image': 0.02,
  'flux-dev': 0.02,
  'flux-schnell': 0.0024,
  'flux-kontext': 0.02,
}

const T2I_MODELS = [
  { value: 'gemini-flash', name: 'Gemini Flash', badge: 'FREE', badgeClass: 'free', price: 'FREE 500/day' },
  { value: 'runware-flux-klein-4b', name: 'Flux Klein 4B', badge: 'Cheapest', badgeClass: 'budget', price: '$0.0006/img' },
  { value: 'runware-flux-klein-9b', name: 'Flux Klein 9B', badge: 'Quality', badgeClass: 'quality', price: '$0.00078/img' },
  { value: 'runware-z-image', name: 'Z-Image', badge: 'Fast', badgeClass: 'budget', price: '$0.0006/img' },
  { value: 'flux-dev', name: 'Flux Dev', badge: 'Premium', badgeClass: 'quality', price: '$0.02/img' },
  { value: 'flux-schnell', name: 'Flux Schnell', badge: 'Fast', badgeClass: 'budget', price: '$0.0024/img' },
  { value: 'qwen-image', name: 'Qwen Image', badge: 'Premium', badgeClass: 'quality', price: '$0.02/img' },
]

const SCENE_MODELS = [
  { value: 'flux-kontext', name: 'Flux Kontext', badge: 'Recommended', badgeClass: 'quality', price: '$0.02/img' },
  ...T2I_MODELS,
]

const EXAMPLE_IDEAS = [
  'How to train forearms',
  'Morning routine for productivity',
  '5 healthy meal prep ideas',
  'Day in the life of a programmer',
]

function StoryboardGenerator() {
  // Workflow state
  const [step, setStep] = useState<WorkflowStep>('input')

  // Step 1: Input state
  const [videoIdea, setVideoIdea] = useState('')
  const [sceneCount, setSceneCount] = useState(6)
  const [aspectRatio, setAspectRatio] = useState('9:16')
  const [isPlanning, setIsPlanning] = useState(false)

  // Step 2: Review state
  const [jobId, setJobId] = useState<string | null>(null)
  const [plan, setPlan] = useState<StoryboardPlan | null>(null)
  const [refModel, setRefModel] = useState('flux-dev')
  const [sceneModel, setSceneModel] = useState('flux-kontext')

  // Step 3: Generation state
  const [_status, setStatus] = useState<StoryboardJobStatus>('pending')
  const [phase, setPhase] = useState('')
  const [referenceImages, setReferenceImages] = useState<Record<string, string>>({})
  const [sceneResults, setSceneResults] = useState<StoryboardSceneResult[]>([])
  const [completedCount, setCompletedCount] = useState(0)
  const [totalCount, setTotalCount] = useState(0)
  const [totalCost, setTotalCost] = useState(0)

  // Error state
  const [error, setError] = useState<string | null>(null)

  // WebSocket ref
  const wsRef = useRef<WebSocket | null>(null)

  // Cost estimate (skip reference cost for characters with user-uploaded images)
  const charsNeedingGeneration = plan
    ? plan.characters.filter((c) => !userReferenceImages[c.name]).length
    : 0
  const estimatedCost = plan
    ? (charsNeedingGeneration * (MODEL_PRICING[refModel] || 0)) +
      (plan.scenes.length * (MODEL_PRICING[sceneModel] || 0))
    : 0

  // Aspect ratio class helper
  const getAspectClass = () => {
    if (aspectRatio === '16:9') return 'landscape'
    if (aspectRatio === '1:1') return 'square'
    return ''
  }

  // Handle WebSocket messages
  // Backend sends 'character' field; we map to character_name in the handler
  const handleWSMessage = useCallback((message: StoryboardWSMessage & { character?: string }) => {
    // Map backend 'character' to 'character_name' for consistency
    const charName = message.character_name || message.character

    switch (message.type) {
      case 'status':
        if (message.status) setStatus(message.status)
        if (message.phase) setPhase(message.phase)
        if (message.step !== undefined && message.total_steps) {
          setCompletedCount(message.step)
          setTotalCount(message.total_steps)
        }
        break

      case 'reference_start':
        if (charName) setPhase(`Generating reference for ${charName}...`)
        if (message.total_steps) setTotalCount(message.total_steps)
        break

      case 'reference_complete':
        if (charName && message.image_url) {
          setReferenceImages((prev) => ({
            ...prev,
            [charName!]: message.image_url!,
          }))
        }
        if (message.step !== undefined) setCompletedCount(message.step)
        break

      case 'reference_failed':
        // Reference failed, continue
        if (message.step !== undefined) setCompletedCount(message.step)
        break

      case 'scene_start':
        if (message.scene_number !== undefined) {
          setPhase(`Generating scene ${message.scene_number}...`)
          setSceneResults((prev) => {
            const updated = [...prev]
            const idx = updated.findIndex((s) => s.scene_number === message.scene_number)
            if (idx >= 0) {
              updated[idx] = { ...updated[idx], status: 'generating' }
            }
            return updated
          })
        }
        break

      case 'scene_complete':
        if (message.scene_number !== undefined) {
          setSceneResults((prev) => {
            const updated = [...prev]
            const idx = updated.findIndex((s) => s.scene_number === message.scene_number)
            if (idx >= 0) {
              updated[idx] = {
                ...updated[idx],
                status: 'completed',
                image_url: message.image_url || null,
              }
            }
            return updated
          })
          if (message.step !== undefined) setCompletedCount(message.step)
          if (message.total_steps) setTotalCount(message.total_steps)
        }
        break

      case 'scene_failed':
        if (message.scene_number !== undefined) {
          setSceneResults((prev) => {
            const updated = [...prev]
            const idx = updated.findIndex((s) => s.scene_number === message.scene_number)
            if (idx >= 0) {
              updated[idx] = {
                ...updated[idx],
                status: 'failed',
                error: message.error,
              }
            }
            return updated
          })
          if (message.step !== undefined) setCompletedCount(message.step)
        }
        break

      case 'complete':
        setStatus('completed')
        setStep('complete')
        if (message.total_cost !== undefined) setTotalCost(message.total_cost)
        break

      case 'error':
        setStatus('failed')
        setError(message.message || 'Generation failed')
        break
    }
  }, [])

  // Connect to WebSocket when generating
  useEffect(() => {
    if (!jobId || step !== 'generating') return

    const wsUrl = getWsUrl(`/ws/storyboard/${jobId}`)
    const ws = new WebSocket(wsUrl)

    ws.onopen = () => {
      console.log(`WebSocket connected for storyboard job ${jobId}`)
    }

    ws.onmessage = (event) => {
      try {
        const message: StoryboardWSMessage = JSON.parse(event.data)
        handleWSMessage(message)
      } catch (e) {
        console.error('Failed to parse WebSocket message:', e)
      }
    }

    ws.onerror = (err) => {
      console.error('WebSocket error:', err)
      setError('WebSocket connection error')
    }

    ws.onclose = () => {
      console.log(`WebSocket disconnected for storyboard job ${jobId}`)
    }

    wsRef.current = ws

    return () => {
      ws.close()
      wsRef.current = null
    }
  }, [jobId, step, handleWSMessage])

  // Step 1: Generate plan
  const handleGeneratePlan = async () => {
    if (!videoIdea.trim()) {
      setError('Please describe your video concept')
      return
    }

    setIsPlanning(true)
    setError(null)

    try {
      const response = await fetch(`${API_BASE}/api/storyboard/plan`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          idea: videoIdea.trim(),
          num_scenes: sceneCount,
          aspect_ratio: aspectRatio,
        }),
      })

      if (!response.ok) {
        const data = await response.json()
        throw new Error(data.detail || 'Failed to generate plan')
      }

      const data = await response.json()
      setJobId(data.job_id)
      setPlan(data.plan)
      setStep('review')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate plan')
    } finally {
      setIsPlanning(false)
    }
  }

  // Step 2: Start image generation
  const handleStartGeneration = async () => {
    if (!jobId || !plan) {
      setError('No plan to generate')
      return
    }

    setError(null)
    setReferenceImages({})
    setCompletedCount(0)
    setTotalCount(plan.scenes.length)
    setSceneResults(
      plan.scenes.map((s) => ({
        scene_number: s.scene_number,
        image_url: null,
        status: 'pending' as const,
      }))
    )
    setStep('generating')

    try {
      const dims = ASPECT_RATIOS[aspectRatio]
      const response = await fetch(`${API_BASE}/api/storyboard/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          job_id: jobId,
          plan,
          ref_model: refModel,
          scene_model: sceneModel,
          width: dims.w,
          height: dims.h,
          user_reference_images: Object.keys(userReferenceImages).length > 0
            ? userReferenceImages
            : undefined,
        }),
      })

      if (!response.ok) {
        const data = await response.json()
        throw new Error(data.detail || 'Failed to start generation')
      }

      setStatus('generating_references')
      setPhase('Generating character references...')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start generation')
      setStep('review')
    }
  }

  // Character editing
  const handleCharacterChange = (idx: number, field: keyof StoryboardCharacter, value: string) => {
    if (!plan) return
    setPlan({
      ...plan,
      characters: plan.characters.map((c, i) =>
        i === idx ? { ...c, [field]: value } : c
      ),
    })
  }

  // Scene editing
  const handleSceneChange = (idx: number, field: keyof StoryboardScene, value: string) => {
    if (!plan) return
    setPlan({
      ...plan,
      scenes: plan.scenes.map((s, i) =>
        i === idx ? { ...s, [field]: value } : s
      ),
    })
  }

  // User-provided reference images (character name -> base64 data URL)
  const [userReferenceImages, setUserReferenceImages] = useState<Record<string, string>>({})

  const handleReferenceImageUpload = (characterName: string, file: File) => {
    const reader = new FileReader()
    reader.onload = (e) => {
      const dataUrl = e.target?.result as string
      setUserReferenceImages((prev) => ({ ...prev, [characterName]: dataUrl }))
    }
    reader.readAsDataURL(file)
  }

  const handleRemoveReferenceImage = (characterName: string) => {
    setUserReferenceImages((prev) => {
      const next = { ...prev }
      delete next[characterName]
      return next
    })
  }

  // Download ZIP
  const handleDownloadZip = async () => {
    if (!jobId) return

    try {
      const response = await fetch(`${API_BASE}/api/storyboard/${jobId}/download`)
      if (!response.ok) {
        throw new Error('Download failed')
      }

      const blob = await response.blob()
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `storyboard_${jobId.slice(0, 8)}.zip`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Download failed')
    }
  }

  // Reset
  const handleStartNew = () => {
    setStep('input')
    setJobId(null)
    setPlan(null)
    setVideoIdea('')
    setSceneCount(6)
    setAspectRatio('9:16')
    setReferenceImages({})
    setUserReferenceImages({})
    setSceneResults([])
    setCompletedCount(0)
    setTotalCount(0)
    setTotalCost(0)
    setError(null)
    setStatus('pending')
    setPhase('')
  }

  return (
    <section className="storyboard-section">
      <h2>Storyboard Generator</h2>
      <p className="section-description">
        Create consistent character storyboards for short-form video content. AI plans scenes, generates reference images, then produces each scene with character consistency.
      </p>

      {/* Error Display */}
      {error && (
        <div className="storyboard-error">
          <span className="error-icon">!</span>
          <span>{error}</span>
          <button onClick={() => setError(null)} className="btn-dismiss">Dismiss</button>
        </div>
      )}

      {/* Step 1: Input */}
      {step === 'input' && (
        <div className="storyboard-step step-input">
          <div className="step-header">
            <span className="step-number">1</span>
            <h3>Describe Your Video Concept</h3>
          </div>

          <div className="form-group">
            <label htmlFor="video-idea">Video Idea</label>
            <textarea
              id="video-idea"
              value={videoIdea}
              onChange={(e) => setVideoIdea(e.target.value)}
              placeholder="Describe your short-form video concept..."
              rows={4}
            />
            <div className="example-prompts">
              <span className="examples-label">Quick fill:</span>
              {EXAMPLE_IDEAS.map((idea, i) => (
                <button
                  key={i}
                  className="example-btn"
                  onClick={() => setVideoIdea(idea)}
                >
                  {idea}
                </button>
              ))}
            </div>
          </div>

          <div className="form-group">
            <label htmlFor="scene-count">Number of Scenes</label>
            <select
              id="scene-count"
              value={sceneCount}
              onChange={(e) => setSceneCount(parseInt(e.target.value, 10))}
            >
              {Array.from({ length: 9 }, (_, i) => i + 4).map((n) => (
                <option key={n} value={n}>{n} scenes</option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label>Aspect Ratio</label>
            <div className="aspect-ratio-options">
              {Object.entries(ASPECT_RATIOS).map(([key, val]) => (
                <label
                  key={key}
                  className={`aspect-ratio-option ${aspectRatio === key ? 'selected' : ''}`}
                >
                  <input
                    type="radio"
                    name="aspect-ratio"
                    value={key}
                    checked={aspectRatio === key}
                    onChange={(e) => setAspectRatio(e.target.value)}
                  />
                  <div>
                    <span className="aspect-ratio-label">{val.label}</span>
                    <span className="aspect-ratio-dims">{val.w}x{val.h}</span>
                  </div>
                </label>
              ))}
            </div>
          </div>

          <button
            className="btn-primary btn-generate-prompts"
            onClick={handleGeneratePlan}
            disabled={isPlanning || !videoIdea.trim()}
          >
            {isPlanning ? (
              <>
                <span className="loading-spinner-sm" />
                Planning Storyboard...
              </>
            ) : (
              <>Generate Storyboard Plan</>
            )}
          </button>
        </div>
      )}

      {/* Step 2: Review Plan */}
      {step === 'review' && plan && (
        <div className="storyboard-step step-review">
          <div className="step-header">
            <span className="step-number">2</span>
            <h3>Review &amp; Edit Plan</h3>
          </div>

          <div className="plan-title">{plan.title}</div>

          {/* Characters */}
          <div className="character-section-header">Characters</div>
          <div className="character-cards">
            {plan.characters.map((char, i) => (
              <div key={i} className="character-card">
                <div className="character-card-name">{char.name}</div>
                <div className="character-field">
                  <label>Appearance</label>
                  <input
                    value={char.appearance}
                    onChange={(e) => handleCharacterChange(i, 'appearance', e.target.value)}
                  />
                </div>
                <div className="character-field">
                  <label>Clothing</label>
                  <input
                    value={char.clothing}
                    onChange={(e) => handleCharacterChange(i, 'clothing', e.target.value)}
                  />
                </div>
                <div className="character-field">
                  <label>Accessories</label>
                  <input
                    value={char.accessories}
                    onChange={(e) => handleCharacterChange(i, 'accessories', e.target.value)}
                  />
                </div>
                <div className="character-reference-upload">
                  <label>Reference Image</label>
                  {userReferenceImages[char.name] ? (
                    <div className="reference-preview">
                      <img src={userReferenceImages[char.name]} alt={`${char.name} reference`} />
                      <button
                        className="btn-remove-ref"
                        onClick={() => handleRemoveReferenceImage(char.name)}
                        title="Remove reference image"
                      >
                        &times;
                      </button>
                    </div>
                  ) : (
                    <label className="upload-zone">
                      <input
                        type="file"
                        accept="image/*"
                        onChange={(e) => {
                          const file = e.target.files?.[0]
                          if (file) handleReferenceImageUpload(char.name, file)
                          e.target.value = ''
                        }}
                      />
                      <span className="upload-label">Upload your own reference</span>
                    </label>
                  )}
                </div>
              </div>
            ))}
          </div>

          {/* Scenes */}
          <div className="scenes-section-header">Scenes ({plan.scenes.length})</div>
          <div className="scene-cards">
            {plan.scenes.map((scene, i) => (
              <div key={scene.scene_number} className="scene-card">
                <div className="scene-card-header">
                  <span className="scene-number-badge">{scene.scene_number}</span>
                  <span>Scene {scene.scene_number}</span>
                </div>
                <div className="scene-edit-fields">
                  <div className="scene-edit-field full-width">
                    <label>Description</label>
                    <textarea
                      value={scene.description}
                      onChange={(e) => handleSceneChange(i, 'description', e.target.value)}
                      rows={2}
                    />
                  </div>
                  <div className="scene-edit-field">
                    <label>Camera Angle</label>
                    <input
                      value={scene.camera_angle}
                      onChange={(e) => handleSceneChange(i, 'camera_angle', e.target.value)}
                    />
                  </div>
                  <div className="scene-edit-field">
                    <label>Character Action</label>
                    <input
                      value={scene.character_action}
                      onChange={(e) => handleSceneChange(i, 'character_action', e.target.value)}
                    />
                  </div>
                  <div className="scene-edit-field full-width">
                    <label>Image Prompt</label>
                    <textarea
                      value={scene.image_prompt}
                      onChange={(e) => handleSceneChange(i, 'image_prompt', e.target.value)}
                      rows={3}
                      placeholder="The exact prompt sent to the image generation model..."
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Style Guide */}
          <div className="style-guide-display">
            <h4>Style Guide</h4>
            <p>{plan.style_guide}</p>
          </div>

          {/* Model Selector Panel */}
          <div className="model-selector-panel">
            <div className="model-selector-group">
              <label>Reference Image Model</label>
              <div className="model-options">
                {T2I_MODELS.map((m) => (
                  <label key={m.value} className={`model-option ${refModel === m.value ? 'selected' : ''}`}>
                    <input
                      type="radio"
                      name="ref-model"
                      value={m.value}
                      checked={refModel === m.value}
                      onChange={(e) => setRefModel(e.target.value)}
                    />
                    <div className="model-info">
                      <span className="model-name">{m.name}</span>
                      <span className={`model-badge ${m.badgeClass}`}>{m.badge}</span>
                      <span className="model-price">{m.price}</span>
                    </div>
                  </label>
                ))}
              </div>
            </div>

            <div className="model-selector-group">
              <label>Scene Image Model</label>
              <div className="model-options">
                {SCENE_MODELS.map((m) => (
                  <label key={m.value} className={`model-option ${sceneModel === m.value ? 'selected' : ''}`}>
                    <input
                      type="radio"
                      name="scene-model"
                      value={m.value}
                      checked={sceneModel === m.value}
                      onChange={(e) => setSceneModel(e.target.value)}
                    />
                    <div className="model-info">
                      <span className="model-name">{m.name}</span>
                      <span className={`model-badge ${m.badgeClass}`}>{m.badge}</span>
                      <span className="model-price">{m.price}</span>
                    </div>
                  </label>
                ))}
              </div>
            </div>

            <div className="estimate-panel">
              <div className="estimate-item">
                <span className="estimate-label">Reference Images</span>
                <span className="estimate-value">
                  {charsNeedingGeneration} generated
                  {Object.keys(userReferenceImages).length > 0 && (
                    <> + {Object.keys(userReferenceImages).length} uploaded</>
                  )}
                </span>
              </div>
              <div className="estimate-item">
                <span className="estimate-label">Scenes</span>
                <span className="estimate-value">{plan.scenes.length}</span>
              </div>
              <div className="estimate-item">
                <span className="estimate-label">Est. Cost</span>
                <span className="estimate-value">${estimatedCost.toFixed(4)}</span>
              </div>
            </div>
          </div>

          <div className="step-actions">
            <button className="btn-secondary" onClick={handleStartNew}>
              Start Over
            </button>
            <button
              className="btn-primary btn-start-generation"
              onClick={handleStartGeneration}
              disabled={!plan || plan.scenes.length === 0}
            >
              Generate Images
            </button>
          </div>
        </div>
      )}

      {/* Step 3: Generating */}
      {step === 'generating' && plan && (
        <div className="storyboard-step step-generating">
          <div className="step-header">
            <span className="step-number">3</span>
            <h3>Generating Storyboard</h3>
          </div>

          {/* Phase Indicator */}
          <div className="phase-indicator">
            <span className="loading-spinner-sm" />
            <span className="phase-text">
              {phase || 'Starting...'}
              {totalCount > 0 && (
                <span className="phase-detail">
                  ({completedCount} / {totalCount})
                </span>
              )}
            </span>
          </div>

          {/* Progress Bar */}
          <div className="progress-panel">
            <div className="progress-bar">
              <div
                className="progress-fill"
                style={{
                  width: `${totalCount > 0 ? (completedCount / totalCount) * 100 : 0}%`,
                }}
              />
            </div>
            <div className="progress-stats">
              <span className="stat completed">{completedCount} completed</span>
              <span className="stat total">/ {totalCount} scenes</span>
            </div>
          </div>

          {/* Reference Images */}
          {plan.characters.length > 0 && (
            <div className="reference-images-row">
              {plan.characters.map((char) => {
                const refUrl = referenceImages[char.name]
                return (
                  <div key={char.name} className={`reference-card ${refUrl ? 'completed' : ''}`}>
                    {refUrl ? (
                      <img src={refUrl} alt={`${char.name} reference`} />
                    ) : (
                      <div className="reference-card-pending">
                        <span className="loading-spinner-sm" />
                      </div>
                    )}
                    <div className="reference-name">{char.name}</div>
                  </div>
                )
              })}
            </div>
          )}

          {/* Scene Grid */}
          <div className="scene-gen-grid">
            {sceneResults.map((result) => {
              const scene = plan.scenes.find((s) => s.scene_number === result.scene_number)
              return (
                <div key={result.scene_number} className={`scene-gen-card ${result.status}`}>
                  {result.status === 'completed' && result.image_url ? (
                    <img
                      className={`scene-gen-image ${getAspectClass()}`}
                      src={result.image_url}
                      alt={`Scene ${result.scene_number}`}
                      loading="lazy"
                    />
                  ) : result.status === 'failed' ? (
                    <div className={`scene-gen-pending ${getAspectClass()}`}>
                      <span style={{ color: 'var(--color-error)', fontSize: 'var(--text-xs)' }}>Failed</span>
                    </div>
                  ) : (
                    <div className={`scene-gen-pending ${getAspectClass()}`}>
                      <span className="loading-spinner-sm" />
                    </div>
                  )}
                  <div className="scene-gen-label">
                    <span className="scene-number-badge">{result.scene_number}</span>
                    <span>{scene?.description.slice(0, 30) || `Scene ${result.scene_number}`}...</span>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Step 4: Complete */}
      {step === 'complete' && plan && (
        <div className="storyboard-step step-complete">
          <div className="step-header">
            <span className="step-number check">OK</span>
            <h3>Storyboard Complete</h3>
          </div>

          <div className="complete-stats">
            <div className="stat-card">
              <span className="stat-value">
                {sceneResults.filter((r) => r.status === 'completed').length}
              </span>
              <span className="stat-label">Scenes Completed</span>
            </div>
            {sceneResults.some((r) => r.status === 'failed') && (
              <div className="stat-card failed">
                <span className="stat-value">
                  {sceneResults.filter((r) => r.status === 'failed').length}
                </span>
                <span className="stat-label">Failed</span>
              </div>
            )}
            <div className="stat-card">
              <span className="stat-value">${totalCost.toFixed(3)}</span>
              <span className="stat-label">Total Cost</span>
            </div>
          </div>

          {/* Filmstrip Gallery */}
          <div className="filmstrip-gallery">
            {sceneResults
              .filter((r) => r.status === 'completed' && r.image_url)
              .map((result) => {
                const scene = plan.scenes.find((s) => s.scene_number === result.scene_number)
                return (
                  <div key={result.scene_number} className="filmstrip-scene">
                    <div className="filmstrip-scene-content">
                      <div className="filmstrip-scene-image">
                        <img
                          src={result.image_url!}
                          alt={`Scene ${result.scene_number}`}
                        />
                      </div>
                      <div className="filmstrip-scene-info">
                        <span className="filmstrip-scene-number">{result.scene_number}</span>
                        <p className="filmstrip-scene-description">
                          {scene?.description || `Scene ${result.scene_number}`}
                        </p>
                        <div className="filmstrip-scene-tags">
                          {scene?.camera_angle && (
                            <span className="filmstrip-tag camera">{scene.camera_angle}</span>
                          )}
                          {scene?.character_action && (
                            <span className="filmstrip-tag">{scene.character_action}</span>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                )
              })}
          </div>

          <div className="step-actions">
            <button className="btn-secondary" onClick={handleStartNew}>
              Start New
            </button>
            <button className="btn-primary" onClick={handleDownloadZip}>
              Download All
            </button>
          </div>
        </div>
      )}
    </section>
  )
}

export default StoryboardGenerator
