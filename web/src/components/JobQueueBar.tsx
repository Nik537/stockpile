import { useState, useEffect } from 'react'
import { useJobQueueStore } from '../stores'
import { BackgroundJobType } from '../types'
import './JobQueueBar.css'

const TYPE_ICONS: Record<BackgroundJobType, string> = {
  tts: '\uD83C\uDFA4',
  image: '\uD83C\uDFA8',
  'image-edit': '\u270F\uFE0F',
  music: '\uD83C\uDFB5',
  storyboard: '\uD83D\uDCF7',
  video: '\uD83C\uDFAC',
}

function formatElapsed(createdAt: string, completedAt?: string): string {
  const start = new Date(createdAt).getTime()
  const end = completedAt ? new Date(completedAt).getTime() : Date.now()
  const seconds = Math.floor((end - start) / 1000)
  if (seconds < 60) return `${seconds}s`
  const minutes = Math.floor(seconds / 60)
  const remainingSeconds = seconds % 60
  return `${minutes}m ${remainingSeconds}s`
}

export default function JobQueueBar() {
  const { jobs, clearCompleted, removeJob } = useJobQueueStore()
  const [expanded, setExpanded] = useState(false)
  const [, setTick] = useState(0)

  const activeCount = jobs.filter((j) => j.status === 'processing').length
  const hasCompleted = jobs.some((j) => j.status !== 'processing')

  // Tick every second to update elapsed times for active jobs
  useEffect(() => {
    if (activeCount === 0) return
    const interval = setInterval(() => setTick((t) => t + 1), 1000)
    return () => clearInterval(interval)
  }, [activeCount])

  if (jobs.length === 0) return null

  return (
    <div className={`job-queue-bar ${expanded ? 'expanded' : ''} ${jobs.length === 0 ? 'hidden' : ''}`}>
      {/* Expandable job list */}
      <div className="job-queue-list">
        <div className="job-queue-list-inner">
          {jobs.map((job) => (
            <div key={job.id} className="job-queue-item">
              <span className="job-queue-item-icon">{TYPE_ICONS[job.type]}</span>
              <span className="job-queue-item-label" title={job.label}>
                {job.label}
              </span>
              <span className={`job-queue-item-status ${job.status}`}>
                {job.status === 'processing' && (
                  <span className="job-queue-spinner" style={{ width: 10, height: 10, borderWidth: 1.5 }} />
                )}
                {job.status === 'completed' && '\u2713'}
                {job.status === 'failed' && '\u2717'}
                {job.status}
              </span>
              <span className="job-queue-item-time">
                {formatElapsed(job.createdAt, job.completedAt)}
              </span>
              {job.status !== 'processing' && (
                <button
                  className="job-queue-item-remove"
                  onClick={(e) => {
                    e.stopPropagation()
                    removeJob(job.id)
                  }}
                  title="Remove"
                >
                  \u2715
                </button>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Summary bar */}
      <div className="job-queue-summary" onClick={() => setExpanded(!expanded)}>
        <div className="job-queue-summary-left">
          {activeCount > 0 ? (
            <span className="job-queue-spinner" />
          ) : (
            <span className="job-queue-check">{'\u2713'}</span>
          )}
          <span className="job-queue-label">
            {activeCount > 0
              ? `${activeCount} active job${activeCount !== 1 ? 's' : ''}`
              : `All ${jobs.length} job${jobs.length !== 1 ? 's' : ''} complete`}
          </span>
        </div>
        <div className="job-queue-summary-right">
          {hasCompleted && (
            <button
              className="job-queue-clear-btn"
              onClick={(e) => {
                e.stopPropagation()
                clearCompleted()
              }}
            >
              Clear completed
            </button>
          )}
          <span className="job-queue-expand-icon">{'\u25B2'}</span>
        </div>
      </div>
    </div>
  )
}
