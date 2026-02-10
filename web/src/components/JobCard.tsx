import { useEffect, useState } from 'react'
import { Job, StatusUpdate } from '../types'
import ProgressBar from './ProgressBar'
import './JobCard.css'
import { API_BASE, getWsUrl } from '../config'

interface JobCardProps {
  job: Job
  onDeleted: (jobId: string) => void
}

function JobCard({ job: initialJob, onDeleted }: JobCardProps) {
  const [job, setJob] = useState<Job>(initialJob)
  const [_ws, setWs] = useState<WebSocket | null>(null)

  useEffect(() => {
    // Connect to WebSocket for real-time updates
    const wsUrl = getWsUrl(`/ws/status/${job.id}`)
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
      const response = await fetch(`${API_BASE}/api/jobs/${job.id}`, {
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
    window.location.href = `${API_BASE}/api/jobs/${job.id}/download`
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

  const formatDate = (dateString: string) => {
    const date = new Date(dateString)
    return new Intl.DateTimeFormat('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    }).format(date)
  }

  return (
    <div className={`job-card status-${job.status}`}>
      <div className="job-header">
        <div className="job-info">
          <h3>{job.video_filename}</h3>
          <span className={`status-badge ${statusBadge.className}`}>
            <span className="status-dot"></span>
            {statusBadge.label}
          </span>
        </div>
        <div className="job-actions">
          {job.status === 'completed' && (
            <button onClick={handleDownload} className="btn-download">
              <span className="btn-icon">&#x2B07;</span>
              Download
            </button>
          )}
          <button onClick={handleDelete} className="btn-delete">
            <span className="btn-icon">&#x1F5D1;</span>
            Delete
          </button>
        </div>
      </div>

      <div className="job-body">
        <ProgressBar progress={job.progress.percent} status={job.status} />
        <p className="progress-message">{job.progress.message}</p>

        {job.error && (
          <div className="error-message">
            <span className="error-icon">&#x26A0;</span>
            <span className="error-text">
              <strong>Error:</strong> {job.error}
            </span>
          </div>
        )}

        <div className="job-meta">
          <span className="job-meta-item">
            <span className="meta-icon">&#x1F4C5;</span>
            Created: {formatDate(job.created_at)}
          </span>
          <span className="job-meta-item">
            <span className="meta-icon">&#x1F504;</span>
            Updated: {formatDate(job.updated_at)}
          </span>
        </div>
      </div>
    </div>
  )
}

export default JobCard
