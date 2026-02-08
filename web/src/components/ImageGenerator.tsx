import { useCallback, useEffect, useState } from 'react'
import {
  GeneratedImage,
  ImageGenerationResult,
  ImageGenMode,
  ImageGenModel,
  ImageGenServerStatus,
  ImageGenStatus,
} from '../types'
import InpaintingCanvas from './InpaintingCanvas'
import './ImageGenerator.css'

const API_BASE = ''  // Same origin
const IMAGE_GEN_MODEL_KEY = 'stockpile_image_gen_model'

const MODEL_INFO = [
  { id: 'runware-flux-klein-4b' as ImageGenModel, name: 'Flux Klein 4B', price: '$0.0006/img', badge: 'Cheapest', badgeClass: 'budget' },
  { id: 'runware-z-image' as ImageGenModel, name: 'Z-Image Turbo', price: '$0.0006/img', badge: 'Fast', badgeClass: '' },
  { id: 'runware-flux-klein-9b' as ImageGenModel, name: 'Flux Klein 9B', price: '$0.0008/img', badge: 'Quality', badgeClass: 'quality' },
  { id: 'gemini-flash' as ImageGenModel, name: 'Gemini Flash', price: 'FREE', badge: 'Free - 500/day', badgeClass: 'free' },
  { id: 'nano-banana-pro' as ImageGenModel, name: 'Nano Banana Pro', price: '~$0.04/img', badge: 'Best Quality', badgeClass: 'premium' },
]

function ImageGenerator() {
  // Mode and model state
  const [mode, setMode] = useState<ImageGenMode>('generate')
  const [model, setModel] = useState<ImageGenModel>(() => {
    const saved = localStorage.getItem(IMAGE_GEN_MODEL_KEY)
    const validModels: ImageGenModel[] = ['runware-flux-klein-4b', 'runware-z-image', 'runware-flux-klein-9b', 'gemini-flash', 'nano-banana-pro']
    return validModels.includes(saved as ImageGenModel) ? (saved as ImageGenModel) : 'runware-flux-klein-4b'
  })

  // Server status
  const [serverStatus, setServerStatus] = useState<ImageGenServerStatus | null>(null)

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

  // Inpainting state
  const [inpaintImageUrl, setInpaintImageUrl] = useState<string | null>(null)

  // Save model preference
  useEffect(() => {
    localStorage.setItem(IMAGE_GEN_MODEL_KEY, model)
  }, [model])

  // Check server status on mount
  const checkStatus = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/image/status`)
      const data = await response.json()
      setServerStatus(data)
    } catch (e) {
      console.error('Failed to check image generation status:', e)
      const fallback: ImageGenServerStatus = {
        runware: { configured: false, available: false },
        gemini: { configured: false, available: false },
        runpod: { configured: false, available: false },
      }
      setServerStatus(fallback)
    }
  }, [])

  useEffect(() => {
    checkStatus()
  }, [checkStatus])

  // Handle image file selection for edit mode
  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setInputImage(file)
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

    setStatus('generating')
    setError(null)
    setResult(null)

    try {
      const response = await fetch(`${API_BASE}/api/image/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: prompt.trim(),
          model,
          width,
          height,
          num_images: numImages,
          seed: seed ? parseInt(seed, 10) : null,
          guidance_scale: guidanceScale,
        }),
      })

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

  // Edit image (upload-based)
  const handleEdit = async () => {
    if (!prompt.trim()) {
      setError('Please enter a prompt')
      return
    }

    if (!inputImage) {
      setError('Please select an image to edit')
      return
    }

    setStatus('generating')
    setError(null)
    setResult(null)

    try {
      const imageDataUrl = await fileToDataUrl(inputImage)

      const response = await fetch(`${API_BASE}/api/image/edit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: prompt.trim(),
          image_url: imageDataUrl,
          model,
          strength,
          guidance_scale: guidanceScale,
          seed: seed ? parseInt(seed, 10) : null,
        }),
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

  // Inpainting apply handler
  const handleInpaintApply = async (maskDataUrl: string, editPrompt: string) => {
    if (!inpaintImageUrl) return

    setStatus('generating')
    setError(null)

    try {
      const response = await fetch(`${API_BASE}/api/image/edit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: editPrompt,
          image_url: inpaintImageUrl,
          mask_image: maskDataUrl,
          model: 'nano-banana-pro',
          strength: 0.75,
        }),
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Inpainting failed')
      }

      const data: ImageGenerationResult = await response.json()
      setResult(data)
      setStatus('completed')
      setInpaintImageUrl(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Inpainting failed')
      setStatus('error')
    }
  }

  // Enter inpaint mode from a generated result
  const enterInpaintMode = (imageUrl: string) => {
    setInpaintImageUrl(imageUrl)
    setMode('inpaint')
    setError(null)
  }

  // Cancel inpainting
  const cancelInpaint = () => {
    setInpaintImageUrl(null)
    setMode('generate')
  }

  // Handle form submission
  const handleSubmit = () => {
    if (mode === 'generate') {
      handleGenerate()
    } else if (mode === 'edit') {
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

  // Preset dimensions
  const dimensionPresets = [
    { label: 'Square (1:1)', width: 1024, height: 1024 },
    { label: 'Landscape (16:9)', width: 1920, height: 1080 },
    { label: 'Portrait (9:16)', width: 1080, height: 1920 },
    { label: 'Standard (4:3)', width: 1024, height: 768 },
    { label: 'Portrait (3:4)', width: 768, height: 1024 },
    { label: 'Wide (21:9)', width: 2016, height: 864 },
    { label: 'Social (4:5)', width: 1080, height: 1350 },
  ]

  const isReadyToGenerate = serverStatus?.runware?.configured || serverStatus?.gemini?.configured || serverStatus?.runpod?.configured
  const isGenerating = status === 'generating'

  // If in inpainting mode, show the inpainting canvas
  if (mode === 'inpaint' && inpaintImageUrl) {
    return (
      <section className="imagegen-section">
        <div className="section-header">
          <div className="section-icon">&#x1F3A8;</div>
          <div>
            <h2>AI Image Generator</h2>
            <p className="section-description">
              Inpainting mode - draw on the image to select areas to edit
            </p>
          </div>
        </div>

        {error && (
          <div className="imagegen-error">
            <span className="error-icon">&#x26A0;</span>
            <span>{error}</span>
          </div>
        )}

        <InpaintingCanvas
          imageUrl={inpaintImageUrl}
          onApplyEdit={handleInpaintApply}
          onCancel={cancelInpaint}
          isGenerating={isGenerating}
        />

        {/* Show result after inpainting */}
        {result && (
          <div className="imagegen-result">
            <div className="result-header">
              <h3>Edited Image</h3>
              <div className="result-meta">
                <span>Model: {result.model}</span>
                <span>Time: {(result.generation_time_ms / 1000).toFixed(1)}s</span>
                <span>Cost: ~${result.cost_estimate.toFixed(4)}</span>
              </div>
            </div>
            <div className="result-grid">
              {result.images.map((image, index) => (
                <div key={index} className="result-image">
                  <img src={image.url} alt={`Edited ${index + 1}`} />
                  <div className="image-overlay">
                    <span className="image-size">{image.width}x{image.height}</span>
                    <div className="image-actions">
                      <button className="btn-edit-image" onClick={() => enterInpaintMode(image.url)}>
                        Edit
                      </button>
                      <button className="btn-download-image" onClick={() => handleDownload(image, index)}>
                        Download
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </section>
    )
  }

  return (
    <section className="imagegen-section">
      <div className="section-header">
        <div className="section-icon">&#x1F3A8;</div>
        <div>
          <h2>AI Image Generator</h2>
          <p className="section-description">
            Generate images using Runware, Gemini Flash, and Nano Banana Pro
          </p>
        </div>
      </div>

      {/* Server Status */}
      {(() => {
        const anyConfigured = serverStatus?.runware?.configured || serverStatus?.gemini?.configured || serverStatus?.runpod?.configured
        return (
          <div className="imagegen-status-panel">
            <h3>Status</h3>
            <div className="status-grid">
              <div className="status-item">
                <span className="status-label">Image API:</span>
                {anyConfigured ? (
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

            {!anyConfigured && (
              <p className="setup-hint">
                Set <code>RUNWARE_API_KEY</code> and/or <code>GEMINI_API_KEY</code> in your <code>.env</code> file.
              </p>
            )}
          </div>
        )
      })()}

      {/* Mode Selector */}
      <div className="imagegen-mode-selector">
        <h3>Mode</h3>
        <div className="mode-tabs">
          <button
            className={`mode-tab ${mode === 'generate' ? 'active' : ''}`}
            onClick={() => setMode('generate')}
          >
            Text to Image
          </button>
          <button
            className={`mode-tab ${mode === 'edit' ? 'active' : ''}`}
            onClick={() => setMode('edit')}
          >
            Image Editing
          </button>
        </div>
      </div>

      {/* Model Selector - Cards */}
      <div className="imagegen-model-selector">
        <h3>Model</h3>
        <div className="model-cards">
          {MODEL_INFO.map((m) => (
            <button
              key={m.id}
              className={`model-card ${model === m.id ? 'selected' : ''}`}
              onClick={() => setModel(m.id)}
            >
              <div className="model-card-header">
                <span className="model-name">{m.name}</span>
                {m.badge && (
                  <span className={`model-badge ${m.badgeClass}`}>{m.badge}</span>
                )}
              </div>
              <span className="model-price">{m.price}</span>
            </button>
          ))}
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
                  <div className="image-actions">
                    <button className="btn-edit-image" onClick={() => enterInpaintMode(image.url)}>
                      Edit
                    </button>
                    <button className="btn-download-image" onClick={() => handleDownload(image, index)}>
                      Download
                    </button>
                  </div>
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
                Get a Runware API key from{' '}
                <a href="https://runware.ai" target="_blank" rel="noopener noreferrer">
                  runware.ai
                </a>
              </li>
              <li>
                Get a Gemini API key from{' '}
                <a href="https://ai.google.dev" target="_blank" rel="noopener noreferrer">
                  ai.google.dev
                </a>{' '}
                (free tier: 500 images/day)
              </li>
              <li>
                Add to your <code>.env</code>:
                <pre className="code-block">{`RUNWARE_API_KEY=your_runware_key\nGEMINI_API_KEY=your_gemini_key`}</pre>
              </li>
              <li>Restart the backend server</li>
            </ol>

            <div className="pricing-info">
              <h5>Pricing</h5>
              <ul>
                <li><strong>Flux Klein 4B:</strong> $0.0006/image (cheapest)</li>
                <li><strong>Z-Image Turbo:</strong> $0.0006/image (fast)</li>
                <li><strong>Flux Klein 9B:</strong> $0.0008/image (quality)</li>
                <li><strong>Gemini Flash:</strong> FREE - 500 images/day</li>
                <li><strong>Nano Banana Pro:</strong> ~$0.04/image (best quality)</li>
              </ul>
              <p className="pricing-example">
                Most images cost less than $0.001 - Gemini Flash is completely free
              </p>
            </div>
          </div>
        </details>
      </div>
    </section>
  )
}

export default ImageGenerator
