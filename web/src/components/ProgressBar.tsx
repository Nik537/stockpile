import { JobStatus } from '../types'
import './ProgressBar.css'

interface ProgressBarProps {
  progress: number
  status: JobStatus
}

function ProgressBar({ progress, status }: ProgressBarProps) {
  const getProgressClass = () => {
    switch (status) {
      case 'completed':
        return 'status-completed'
      case 'failed':
        return 'status-failed'
      case 'processing':
        return 'status-processing'
      default:
        return 'status-queued'
    }
  }

  const getPercentageClass = () => {
    switch (status) {
      case 'completed':
        return 'completed'
      case 'failed':
        return 'failed'
      default:
        return ''
    }
  }

  return (
    <div className="progress-bar-container">
      <div className="progress-bar-track">
        <div
          className={`progress-bar-fill ${getProgressClass()}`}
          style={{ width: `${progress}%` }}
        />
      </div>
      <span className={`progress-percentage ${getPercentageClass()}`}>
        {progress}%
      </span>
    </div>
  )
}

export default ProgressBar
