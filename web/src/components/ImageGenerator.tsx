import { useCallback, useEffect, useState } from 'react'
import {
  GeneratedImage,
  ImageEditParams,
  ImageGenerationParams,
  ImageGenerationResult,
  ImageGenMode,
  ImageGenModel,
  ImageGenServerStatus,
  ImageGenStatus,
} from '../types'
import './ImageGenerator.css'

const API_BASE = ''  // Same origin
const IMAGE_GEN_MODEL_KEY = 'stockpile_image_gen_model'

// Helper to determine provider from model
const getProvider = (model: ImageGenModel): 'fal' | 'runpod' => {
  return model.startsWith('runpod-') ? 'runpod' : 'fal'
}

function ImageGenerator() {
  // Mode and model state
  const [mode, setMode] = useState<ImageGenMode>('generate')
  const [model, setModel] = useState<ImageGenModel>(() => {
    const saved = localStorage.getItem(IMAGE_GEN_MODEL_KEY)
    const validModels: ImageGenModel[] = ['flux-klein', 'z-image', 'runpod-flux-schnell', 'runpod-flux-dev']
    return validModels.includes(saved as ImageGenModel) ? (saved as ImageGenModel) : 'runpod-flux-schnell'
  })

  // Server status (for fal.ai and RunPod)
  const [serverStatus, setServerStatus] = useState<ImageGenServerStatus | null>(null)
  const [runpodStatus, setRunpodStatus] = useState<{ configured: boolean; available: boolean; error?: string } | null>(null)

  // Generation state
  const [prompt, setPrompt] = useState('')
  const [status, setStatus] = useState<ImageGenStatus>('idle')
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<ImageGenerationResult | null>(null)

  // Generation parameters
  const [width, setWidth] = useState(1024)
  const [height, setHeight] = useState(1024)
  const [numImages, setNumImages] = useState(1)
  const [guidanceScale, setGuidanceScale] = useState(7.5)
  const [seed, setSeed] = useState<string>('')

  // Edit mode parameters
  const [inputImage, setInputImage] = useState<File | null>(null)
  const [inputImagePreview, setInputImagePreview] = useState<string | null>(null)
  const [strength, setStrength] = useState(0.75)

  // Save model preference
  useEffect(() => {
    localStorage.setItem(IMAGE_GEN_MODEL_KEY, model)
  }, [model])

  // Check server status on mount (both fal.ai and RunPod)
  const checkServerStatus = useCallback(async () => {
    // Check fal.ai status
    try {
      const response = await fetch(`${API_BASE}/api/image-generation/status`)
      const data: ImageGenServerStatus = await response.json()
      setServerStatus(data)
    } catch (e) {
      console.error('Failed to check fal.ai image generation status:', e)
      setServerStatus({ configured: false, available: false, error: 'Failed to connect to server' })
    }

    // Check RunPod status
    try {
      const response = await fetch(`${API_BASE}/api/runpod-image/status`)
      const data = await response.json()
      setRunpodStatus(data)
    } catch (e) {
      console.error('Failed to check RunPod image generation status:', e)
      setRunpodStatus({ configured: false, available: false, error: 'Failed to connect to server' })
    }
  }, [])

  useEffect(() => {
    checkServerStatus()
  }, [checkServerStatus])

  // Handle image file selection for edit mode
  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setInputImage(file)
      // Create preview URL
      const url = URL.createObjectURL(file)
      setInputImagePreview(url)
    }
  }

  // Convert file to base64 data URL
  const fileToDataUrl = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => resolve(reader.result as string)
      reader.onerror = reject
      reader.readAsDataURL(file)
    })
  }

  // Generate images
  const handleGenerate = async () => {
    if (!prompt.trim()) {
      setError('Please enter a prompt')
      return
    }

    const provider = getProvider(model)

    // Check provider-specific configuration
    if (provider === 'fal' && !serverStatus?.configured) {
      setError('fal.ai not configured. Set FAL_API_KEY in .env')
      return
    }
    if (provider === 'runpod' && !runpodStatus?.configured) {
      setError('RunPod not configured. Set RUNPOD_API_KEY in .env')
      return
    }

    setStatus('generating')
    setError(null)
    setResult(null)

    try {
      let response: Response

      if (provider === 'runpod') {
        // Use RunPod API
        const params = {
          prompt: prompt.trim(),
          model,
          width,
          height,
          seed: seed ? parseInt(seed, 10) : null,
        }
        response = await fetch(`${API_BASE}/api/runpod-image/generate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(params),
        })
      } else {
        // Use fal.ai API
        const params: ImageGenerationParams = {
          prompt: prompt.trim(),
          model,
          width,
          height,
          num_images: numImages,
          guidance_scale: guidanceScale,
          seed: seed ? parseInt(seed, 10) : null,
        }
        response = await fetch(`${API_BASE}/api/generate-image`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(params),
        })
      }

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Generation failed')
      }

      const data: ImageGenerationResult = await response.json()
      setResult(data)
      setStatus('completed')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Generation failed')
      setStatus('error')
    }
  }

  // Edit image
  const handleEdit = async () => {
    if (!prompt.trim()) {
      setError('Please enter a prompt')
      return
    }

    if (!inputImage) {
      setError('Please select an image to edit')
      return
    }

    if (!serverStatus?.configured) {
      setError('Image generation not configured. Set FAL_API_KEY in .env')
      return
    }

    setStatus('generating')
    setError(null)
    setResult(null)

    try {
      // Convert image to data URL
      const imageDataUrl = await fileToDataUrl(inputImage)

      const params: ImageEditParams = {
        prompt: prompt.trim(),
        image_url: imageDataUrl,
        model,
        strength,
        guidance_scale: guidanceScale,
        seed: seed ? parseInt(seed, 10) : null,
      }

      const response = await fetch(`${API_BASE}/api/edit-image`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Editing failed')
      }

      const data: ImageGenerationResult = await response.json()
      setResult(data)
      setStatus('completed')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Editing failed')
      setStatus('error')
    }
  }

  // Handle form submission
  const handleSubmit = () => {
    if (mode === 'generate') {
      handleGenerate()
    } else {
      handleEdit()
    }
  }

  // Download image
  const handleDownload = (image: GeneratedImage, index: number) => {
    const a = document.createElement('a')
    a.href = image.url
    a.download = `generated_${Date.now()}_${index + 1}.png`
    a.target = '_blank'
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
  }

  // Preset dimensions with exact ratios
  const dimensionPresets = [
    { label: 'Square (1:1)', width: 1024, height: 1024 },
    { label: 'Landscape (16:9)', width: 1920, height: 1080 },
    { label: 'Portrait (9:16)', width: 1080, height: 1920 },
    { label: 'Standard (4:3)', width: 1024, height: 768 },
    { label: 'Portrait (3:4)', width: 768, height: 1024 },
    { label: 'Wide (21:9)', width: 2016, height: 864 },
    { label: 'Social (4:5)', width: 1080, height: 1350 },
  ]

  // Check if ready based on selected model's provider
  const currentProvider = getProvider(model)
  const isReadyToGenerate = currentProvider === 'runpod'
    ? runpodStatus?.configured && runpodStatus?.available
    : serverStatus?.configured && serverStatus?.available
  const isGenerating = status === 'generating'

  return (
    <section className="imagegen-section">
      <div className="section-header">
        <div className="section-icon">&#x1F3A8;</div>
        <div>
          <h2>AI Image Generator</h2>
          <p className="section-description">
            Generate images using Flux models via RunPod or fal.ai
          </p>
        </div>
      </div>

      {/* Server Status */}
      <div className="imagegen-status-panel">
        <h3>Status</h3>
        <div className="status-grid">
          {/* RunPod Status */}
          <div className="status-item">
            <span className="status-label">RunPod:</span>
            {runpodStatus?.configured ? (
              <span className="connection-status success">
                <span className="status-dot connected"></span>
                <span>Ready</span>
              </span>
            ) : (
              <span className="connection-status error">
                <span className="status-dot"></span>
                <span>Not configured</span>
              </span>
            )}
          </div>

          {/* fal.ai Status */}
          <div className="status-item">
            <span className="status-label">fal.ai:</span>
            {serverStatus?.configured ? (
              <span className="connection-status success">
                <span className="status-dot connected"></span>
                <span>Ready</span>
              </span>
            ) : (
              <span className="connection-status error">
                <span className="status-dot"></span>
                <span>Not configured</span>
              </span>
            )}
          </div>
        </div>

        {!runpodStatus?.configured && !serverStatus?.configured && (
          <p className="setup-hint">
            Set <code>RUNPOD_API_KEY</code> or <code>FAL_API_KEY</code> in your <code>.env</code> file.
          </p>
        )}
      </div>

      {/* Mode Selector */}
      <div className="imagegen-mode-selector">
        <h3>Mode</h3>
        <div className="mode-options">
          <label className={`mode-option ${mode === 'generate' ? 'selected' : ''}`}>
            <input
              type="radio"
              name="imagegen-mode"
              value="generate"
              checked={mode === 'generate'}
              onChange={() => setMode('generate')}
            />
            <div className="mode-content">
              <div className="mode-header">
                <span className="mode-name">Text to Image</span>
              </div>
              <span className="mode-desc">Generate new images from text prompts</span>
            </div>
          </label>

          <label className={`mode-option ${mode === 'edit' ? 'selected' : ''}`}>
            <input
              type="radio"
              name="imagegen-mode"
              value="edit"
              checked={mode === 'edit'}
              onChange={() => setMode('edit')}
            />
            <div className="mode-content">
              <div className="mode-header">
                <span className="mode-name">Image Editing</span>
              </div>
              <span className="mode-desc">Transform existing images with prompts</span>
            </div>
          </label>
        </div>
      </div>

      {/* Model Selector */}
      <div className="imagegen-model-selector">
        <h3>Model</h3>

        {/* RunPod Models */}
        <h4 className="model-group-header">RunPod (Recommended)</h4>
        <div className="model-options">
          <label className={`model-option ${model === 'runpod-flux-schnell' ? 'selected' : ''}`}>
            <input
              type="radio"
              name="imagegen-model"
              value="runpod-flux-schnell"
              checked={model === 'runpod-flux-schnell'}
              onChange={() => setModel('runpod-flux-schnell')}
              disabled={!runpodStatus?.configured}
            />
            <div className="model-content">
              <div className="model-header">
                <span className="model-name">Flux Schnell</span>
                <span className="model-badge">Fast</span>
              </div>
              <span className="model-desc">~3 seconds, ~$0.0024/MP - Best for quick iterations</span>
            </div>
          </label>

          <label className={`model-option ${model === 'runpod-flux-dev' ? 'selected' : ''}`}>
            <input
              type="radio"
              name="imagegen-model"
              value="runpod-flux-dev"
              checked={model === 'runpod-flux-dev'}
              onChange={() => setModel('runpod-flux-dev')}
              disabled={!runpodStatus?.configured}
            />
            <div className="model-content">
              <div className="model-header">
                <span className="model-name">Flux Dev</span>
                <span className="model-badge quality">Quality</span>
              </div>
              <span className="model-desc">~6 seconds, ~$0.02/MP - Best quality results</span>
            </div>
          </label>
        </div>

        {/* fal.ai Models */}
        <h4 className="model-group-header">fal.ai</h4>
        <div className="model-options">
          <label className={`model-option ${model === 'flux-klein' ? 'selected' : ''}`}>
            <input
              type="radio"
              name="imagegen-model"
              value="flux-klein"
              checked={model === 'flux-klein'}
              onChange={() => setModel('flux-klein')}
              disabled={!serverStatus?.configured}
            />
            <div className="model-content">
              <div className="model-header">
                <span className="model-name">Flux 2 Klein</span>
              </div>
              <span className="model-desc">Black Forest Labs - High quality, ~$0.012/MP</span>
            </div>
          </label>

          <label className={`model-option ${model === 'z-image' ? 'selected' : ''}`}>
            <input
              type="radio"
              name="imagegen-model"
              value="z-image"
              checked={model === 'z-image'}
              onChange={() => setModel('z-image')}
              disabled={!serverStatus?.configured}
            />
            <div className="model-content">
              <div className="model-header">
                <span className="model-name">Z-Image Turbo</span>
                <span className="model-badge budget">Budget</span>
              </div>
              <span className="model-desc">Alibaba - Fast & affordable, ~$0.005/MP</span>
            </div>
          </label>
        </div>
      </div>

      {/* Main Form */}
      <div className="imagegen-form">
        {/* Prompt Input */}
        <div className="form-group">
          <label htmlFor="imagegen-prompt">
            {mode === 'generate' ? 'Prompt' : 'Edit Prompt'}
          </label>
          <textarea
            id="imagegen-prompt"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder={
              mode === 'generate'
                ? 'Describe the image you want to generate...\nExample: A majestic lion in a sunlit savanna, golden hour lighting, photorealistic'
                : 'Describe how you want to transform the image...\nExample: Make it look like a watercolor painting'
            }
            rows={4}
            disabled={isGenerating}
          />
        </div>

        {/* Image Upload (Edit Mode) */}
        {mode === 'edit' && (
          <div className="form-group">
            <label>Input Image</label>
            <div className="image-upload-area">
              {inputImagePreview ? (
                <div className="image-preview">
                  <img src={inputImagePreview} alt="Input preview" />
                  <button
                    type="button"
                    className="btn-remove-image"
                    onClick={() => {
                      setInputImage(null)
                      setInputImagePreview(null)
                    }}
                  >
                    Remove
                  </button>
                </div>
              ) : (
                <label className="upload-label">
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleImageSelect}
                    disabled={isGenerating}
                  />
                  <span className="upload-icon">&#x1F4F7;</span>
                  <span>Click or drag to upload an image</span>
                </label>
              )}
            </div>
          </div>
        )}

        {/* Dimension Presets (Generate Mode) */}
        {mode === 'generate' && (
          <div className="form-group">
            <label>Dimensions</label>
            <div className="dimension-presets">
              {dimensionPresets.map((preset) => (
                <button
                  key={preset.label}
                  type="button"
                  className={`preset-btn ${width === preset.width && height === preset.height ? 'selected' : ''}`}
                  onClick={() => {
                    setWidth(preset.width)
                    setHeight(preset.height)
                  }}
                  disabled={isGenerating}
                >
                  {preset.label}
                </button>
              ))}
            </div>
            <div className="dimension-inputs">
              <div className="dimension-input">
                <label htmlFor="width">Width</label>
                <input
                  id="width"
                  type="number"
                  value={width}
                  onChange={(e) => setWidth(parseInt(e.target.value, 10) || 1024)}
                  min={256}
                  max={2048}
                  step={64}
                  disabled={isGenerating}
                />
              </div>
              <span className="dimension-x">x</span>
              <div className="dimension-input">
                <label htmlFor="height">Height</label>
                <input
                  id="height"
                  type="number"
                  value={height}
                  onChange={(e) => setHeight(parseInt(e.target.value, 10) || 1024)}
                  min={256}
                  max={2048}
                  step={64}
                  disabled={isGenerating}
                />
              </div>
            </div>
          </div>
        )}

        {/* Number of Images (Generate Mode) */}
        {mode === 'generate' && (
          <div className="form-group">
            <label htmlFor="num-images">Number of Images: {numImages}</label>
            <input
              id="num-images"
              type="range"
              value={numImages}
              onChange={(e) => setNumImages(parseInt(e.target.value, 10))}
              min={1}
              max={4}
              step={1}
              disabled={isGenerating}
            />
          </div>
        )}

        {/* Advanced Parameters */}
        <details className="advanced-params">
          <summary>Advanced Parameters</summary>
          <div className="params-grid">
            <div className="param-group">
              <label htmlFor="guidance-scale">
                Guidance Scale: {guidanceScale.toFixed(1)}
              </label>
              <input
                id="guidance-scale"
                type="range"
                min="1"
                max="20"
                step="0.5"
                value={guidanceScale}
                onChange={(e) => setGuidanceScale(parseFloat(e.target.value))}
                disabled={isGenerating}
              />
              <span className="param-hint">How closely to follow the prompt (higher = more literal)</span>
            </div>

            {mode === 'edit' && (
              <div className="param-group">
                <label htmlFor="strength">
                  Strength: {strength.toFixed(2)}
                </label>
                <input
                  id="strength"
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={strength}
                  onChange={(e) => setStrength(parseFloat(e.target.value))}
                  disabled={isGenerating}
                />
                <span className="param-hint">How much to transform (0=none, 1=complete)</span>
              </div>
            )}

            <div className="param-group">
              <label htmlFor="seed">Seed (optional)</label>
              <input
                id="seed"
                type="text"
                value={seed}
                onChange={(e) => setSeed(e.target.value.replace(/\D/g, ''))}
                placeholder="Random"
                disabled={isGenerating}
              />
              <span className="param-hint">For reproducible results</span>
            </div>
          </div>
        </details>

        {/* Error Message */}
        {error && (
          <div className="imagegen-error">
            <span className="error-icon">&#x26A0;</span>
            <span>{error}</span>
          </div>
        )}

        {/* Generate Button */}
        <button
          onClick={handleSubmit}
          disabled={isGenerating || !isReadyToGenerate || !prompt.trim() || (mode === 'edit' && !inputImage)}
          className="btn-generate"
        >
          {isGenerating ? (
            <>
              <span className="loading-spinner-sm"></span>
              {mode === 'generate' ? 'Generating...' : 'Editing...'}
            </>
          ) : (
            <>
              <span className="btn-icon">&#x1F3A8;</span>
              {mode === 'generate' ? 'Generate Image' : 'Edit Image'}
            </>
          )}
        </button>
      </div>

      {/* Results */}
      {result && (
        <div className="imagegen-result">
          <div className="result-header">
            <h3>Generated Images</h3>
            <div className="result-meta">
              <span>Model: {result.model}</span>
              <span>Time: {(result.generation_time_ms / 1000).toFixed(1)}s</span>
              <span>Cost: ~${result.cost_estimate.toFixed(4)}</span>
            </div>
          </div>
          <div className="result-grid">
            {result.images.map((image, index) => (
              <div key={index} className="result-image">
                <img src={image.url} alt={`Generated ${index + 1}`} />
                <div className="image-overlay">
                  <span className="image-size">{image.width}x{image.height}</span>
                  <button
                    className="btn-download-image"
                    onClick={() => handleDownload(image, index)}
                  >
                    Download
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Setup Instructions */}
      <div className="imagegen-help">
        <details className="setup-instructions">
          <summary>
            <h4>Setup Instructions</h4>
          </summary>

          <div className="setup-content">
            <ol>
              <li>
                Create an account at{' '}
                <a href="https://fal.ai" target="_blank" rel="noopener noreferrer">
                  fal.ai
                </a>
              </li>
              <li>
                Go to{' '}
                <a href="https://fal.ai/dashboard/keys" target="_blank" rel="noopener noreferrer">
                  API Keys
                </a>{' '}
                and create a new key
              </li>
              <li>
                Add to your <code>.env</code>:
                <pre className="code-block">FAL_API_KEY=your_api_key_here</pre>
              </li>
              <li>Restart the backend server</li>
            </ol>

            <div className="pricing-info">
              <h5>Pricing</h5>
              <ul>
                <li><strong>Flux 2 Klein:</strong> ~$0.009-0.014 per megapixel</li>
                <li><strong>Z-Image Turbo:</strong> ~$0.005 per megapixel</li>
              </ul>
              <p className="pricing-example">
                Example: 1024x1024 image (~1MP) costs ~$0.005-0.014
              </p>
            </div>
          </div>
        </details>
      </div>
    </section>
  )
}

export default ImageGenerator
