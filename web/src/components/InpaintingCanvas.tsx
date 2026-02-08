import { useCallback, useEffect, useRef, useState } from 'react'
import './InpaintingCanvas.css'

interface InpaintingCanvasProps {
  imageUrl: string
  onApplyEdit: (maskDataUrl: string, prompt: string) => void
  onCancel: () => void
  isGenerating: boolean
}

function InpaintingCanvas({ imageUrl, onApplyEdit, onCancel, isGenerating }: InpaintingCanvasProps) {
  const imageCanvasRef = useRef<HTMLCanvasElement>(null)
  const maskCanvasRef = useRef<HTMLCanvasElement>(null)
  const [isDrawing, setIsDrawing] = useState(false)
  const [brushSize, setBrushSize] = useState(30)
  const [tool, setTool] = useState<'brush' | 'eraser'>('brush')
  const [editPrompt, setEditPrompt] = useState('')
  const [canvasSize, setCanvasSize] = useState({ width: 512, height: 512 })

  // Load image and set up canvas
  useEffect(() => {
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => {
      const maxSize = 512
      let w = img.width
      let h = img.height
      if (w > maxSize || h > maxSize) {
        const scale = maxSize / Math.max(w, h)
        w = Math.round(w * scale)
        h = Math.round(h * scale)
      }
      setCanvasSize({ width: w, height: h })

      const imageCanvas = imageCanvasRef.current
      if (imageCanvas) {
        imageCanvas.width = w
        imageCanvas.height = h
        const ctx = imageCanvas.getContext('2d')
        if (ctx) {
          ctx.drawImage(img, 0, 0, w, h)
        }
      }

      const maskCanvas = maskCanvasRef.current
      if (maskCanvas) {
        maskCanvas.width = w
        maskCanvas.height = h
        const ctx = maskCanvas.getContext('2d')
        if (ctx) {
          ctx.clearRect(0, 0, w, h)
        }
      }
    }
    img.src = imageUrl
  }, [imageUrl])

  const getPosition = (e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    const canvas = maskCanvasRef.current
    if (!canvas) return { x: 0, y: 0 }
    const rect = canvas.getBoundingClientRect()
    const scaleX = canvas.width / rect.width
    const scaleY = canvas.height / rect.height

    if ('touches' in e) {
      const touch = e.touches[0]
      return {
        x: (touch.clientX - rect.left) * scaleX,
        y: (touch.clientY - rect.top) * scaleY,
      }
    }
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    }
  }

  const draw = useCallback((x: number, y: number) => {
    const canvas = maskCanvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    ctx.beginPath()
    ctx.arc(x, y, brushSize / 2, 0, Math.PI * 2)

    if (tool === 'brush') {
      ctx.fillStyle = 'rgba(255, 50, 80, 0.5)'
      ctx.fill()
    } else {
      ctx.globalCompositeOperation = 'destination-out'
      ctx.fill()
      ctx.globalCompositeOperation = 'source-over'
    }
  }, [brushSize, tool])

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (isGenerating) return
    setIsDrawing(true)
    const pos = getPosition(e)
    draw(pos.x, pos.y)
  }

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing || isGenerating) return
    const pos = getPosition(e)
    draw(pos.x, pos.y)
  }

  const handleMouseUp = () => setIsDrawing(false)

  const handleTouchStart = (e: React.TouchEvent<HTMLCanvasElement>) => {
    if (isGenerating) return
    e.preventDefault()
    setIsDrawing(true)
    const pos = getPosition(e)
    draw(pos.x, pos.y)
  }

  const handleTouchMove = (e: React.TouchEvent<HTMLCanvasElement>) => {
    if (!isDrawing || isGenerating) return
    e.preventDefault()
    const pos = getPosition(e)
    draw(pos.x, pos.y)
  }

  const handleTouchEnd = () => setIsDrawing(false)

  const clearMask = () => {
    const canvas = maskCanvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (ctx) {
      ctx.clearRect(0, 0, canvas.width, canvas.height)
    }
  }

  const exportMask = (): string => {
    const maskCanvas = maskCanvasRef.current
    if (!maskCanvas) return ''

    const exportCanvas = document.createElement('canvas')
    exportCanvas.width = maskCanvas.width
    exportCanvas.height = maskCanvas.height
    const ctx = exportCanvas.getContext('2d')
    if (!ctx) return ''

    ctx.fillStyle = 'black'
    ctx.fillRect(0, 0, exportCanvas.width, exportCanvas.height)

    const maskCtx = maskCanvas.getContext('2d')
    if (!maskCtx) return ''
    const maskData = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height)

    const exportData = ctx.getImageData(0, 0, exportCanvas.width, exportCanvas.height)
    for (let i = 0; i < maskData.data.length; i += 4) {
      if (maskData.data[i + 3] > 0) {
        exportData.data[i] = 255
        exportData.data[i + 1] = 255
        exportData.data[i + 2] = 255
        exportData.data[i + 3] = 255
      }
    }
    ctx.putImageData(exportData, 0, 0)

    return exportCanvas.toDataURL('image/png')
  }

  const handleApply = () => {
    if (!editPrompt.trim()) return
    const maskDataUrl = exportMask()
    onApplyEdit(maskDataUrl, editPrompt.trim())
  }

  return (
    <div className="inpainting-container">
      <div className="inpainting-header">
        <h3>Edit Image</h3>
        <p className="inpainting-hint">Draw on the areas you want to change, then describe the replacement.</p>
      </div>

      <div className="inpainting-toolbar">
        <div className="tool-group">
          <button
            className={`tool-btn ${tool === 'brush' ? 'active' : ''}`}
            onClick={() => setTool('brush')}
            disabled={isGenerating}
            title="Brush - paint areas to edit"
          >
            Brush
          </button>
          <button
            className={`tool-btn ${tool === 'eraser' ? 'active' : ''}`}
            onClick={() => setTool('eraser')}
            disabled={isGenerating}
            title="Eraser - remove painted areas"
          >
            Eraser
          </button>
        </div>

        <div className="brush-size-group">
          <label>Size: {brushSize}px</label>
          <input
            type="range"
            min="5"
            max="100"
            value={brushSize}
            onChange={(e) => setBrushSize(parseInt(e.target.value))}
            disabled={isGenerating}
          />
        </div>

        <button
          className="tool-btn clear-btn"
          onClick={clearMask}
          disabled={isGenerating}
        >
          Clear All
        </button>
      </div>

      <div className="inpainting-canvas-wrapper" style={{ width: canvasSize.width, height: canvasSize.height }}>
        <canvas
          ref={imageCanvasRef}
          className="inpainting-image-canvas"
        />
        <canvas
          ref={maskCanvasRef}
          className="inpainting-mask-canvas"
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onTouchStart={handleTouchStart}
          onTouchMove={handleTouchMove}
          onTouchEnd={handleTouchEnd}
          style={{ cursor: isGenerating ? 'not-allowed' : 'crosshair' }}
        />
      </div>

      <div className="inpainting-prompt-section">
        <label htmlFor="inpaint-prompt">What should replace the selected area?</label>
        <textarea
          id="inpaint-prompt"
          value={editPrompt}
          onChange={(e) => setEditPrompt(e.target.value)}
          placeholder="Describe what should appear in the painted area..."
          rows={3}
          disabled={isGenerating}
        />
      </div>

      <div className="inpainting-actions">
        <button
          className="btn-cancel"
          onClick={onCancel}
          disabled={isGenerating}
        >
          Cancel
        </button>
        <button
          className="btn-apply-edit"
          onClick={handleApply}
          disabled={isGenerating || !editPrompt.trim()}
        >
          {isGenerating ? (
            <>
              <span className="loading-spinner-sm"></span>
              Applying Edit...
            </>
          ) : (
            'Apply Edit'
          )}
        </button>
      </div>
    </div>
  )
}

export default InpaintingCanvas
