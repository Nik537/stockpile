import { JobStatus } from '../types'
import './ProgressBar.css'

interface ProgressBarProps {
  progress: number
  status: JobStatus
}

function ProgressBar({ progress, status }: ProgressBarProps) {
  const getProgressColor = () => {
    switch (status) {
      case 'completed':
        return '#10b981'
      case 'failed':
        return '#ef4444'
      case 'processing':
        return '#667eea'
      default:
        return '#9ca3af'
    }
  }

  return (
    <div className="progress-bar-container">
      <div className="progress-bar-track">
        <div
          className="progress-bar-fill"
          style={{
            width: `${progress}%`,
            backgroundColor: getProgressColor(),
          }}
        />
      </div>
      <span className="progress-percentage">{progress}%</span>
    </div>
  )
}

export default ProgressBar
