import { useEffect, useState } from 'react'
import { Job, StatusUpdate } from '../types'
import ProgressBar from './ProgressBar'
import './JobCard.css'

interface JobCardProps {
  job: Job
  onDeleted: (jobId: string) => void
}

function JobCard({ job: initialJob, onDeleted }: JobCardProps) {
  const [job, setJob] = useState<Job>(initialJob)
  const [ws, setWs] = useState<WebSocket | null>(null)

  useEffect(() => {
    // Connect to WebSocket for real-time updates
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsUrl = `${protocol}//${window.location.host}/ws/status/${job.id}`
    const websocket = new WebSocket(wsUrl)

    websocket.onopen = () => {
      console.log(`WebSocket connected for job ${job.id}`)
    }

    websocket.onmessage = (event) => {
      const update: StatusUpdate = JSON.parse(event.data)
      setJob((prevJob) => ({
        ...prevJob,
        status: update.status,
        progress: update.progress,
        error: update.error,
      }))
    }

    websocket.onerror = (error) => {
      console.error('WebSocket error:', error)
    }

    websocket.onclose = () => {
      console.log(`WebSocket disconnected for job ${job.id}`)
    }

    setWs(websocket)

    // Cleanup on unmount
    return () => {
      websocket.close()
    }
  }, [job.id])

  const handleDelete = async () => {
    if (!confirm(`Delete job for ${job.video_filename}?`)) {
      return
    }

    try {
      const response = await fetch(`/api/jobs/${job.id}`, {
        method: 'DELETE',
      })

      if (!response.ok) {
        throw new Error('Failed to delete job')
      }

      onDeleted(job.id)
    } catch (error) {
      console.error('Delete failed:', error)
      alert('Failed to delete job. Please try again.')
    }
  }

  const handleDownload = () => {
    window.location.href = `/api/jobs/${job.id}/download`
  }

  const getStatusBadge = () => {
    const badges = {
      queued: { label: 'Queued', className: 'status-queued' },
      processing: { label: 'Processing', className: 'status-processing' },
      completed: { label: 'Completed', className: 'status-completed' },
      failed: { label: 'Failed', className: 'status-failed' },
    }
    return badges[job.status] || badges.queued
  }

  const statusBadge = getStatusBadge()

  return (
    <div className={`job-card status-${job.status}`}>
      <div className="job-header">
        <div className="job-info">
          <h3>{job.video_filename}</h3>
          <span className={`status-badge ${statusBadge.className}`}>{statusBadge.label}</span>
        </div>
        <div className="job-actions">
          {job.status === 'completed' && (
            <button onClick={handleDownload} className="btn-download">
              ⬇️ Download
            </button>
          )}
          <button onClick={handleDelete} className="btn-delete">
            🗑️ Delete
          </button>
        </div>
      </div>

      <div className="job-body">
        <ProgressBar progress={job.progress.percent} status={job.status} />
        <p className="progress-message">{job.progress.message}</p>

        {job.error && (
          <div className="error-message">
            <strong>Error:</strong> {job.error}
          </div>
        )}

        <div className="job-meta">
          <span>Created: {new Date(job.created_at).toLocaleString()}</span>
          <span>Updated: {new Date(job.updated_at).toLocaleString()}</span>
        </div>
      </div>
    </div>
  )
}

export default JobCard
