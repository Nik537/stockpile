import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useVideoStore, useJobQueueStore } from '../stores'
import { VideoJobCost } from '../types'
import VoiceLibrary from './VoiceLibrary'
import './VideoAgent.css'
import { API_BASE } from '../config'

/** Client-side cost estimation matching backend logic. */
function estimateCost(targetDuration: number): VideoJobCost {
  const scenes = targetDuration * 6
  const tts = Math.round(targetDuration * 0.045 * 10000) / 10000
  const images = Math.round(scenes * 0.0006 * 10000) / 10000
  const music = 0.03
  const broll = 0
  const director = 0.02
  const total = Math.round((tts + images + music + broll + director) * 10000) / 10000
  return { tts, images, music, broll, director, total }
}

const STYLE_OPTIONS = [
  { value: 'documentary', label: 'Documentary' },
  { value: 'motivational', label: 'Motivational' },
  { value: 'educational', label: 'Educational' },
  { value: 'hormozi', label: 'Hormozi' },
]

const SUBTITLE_OPTIONS = [
  { value: 'hormozi', label: 'Hormozi (Bold)' },
  { value: 'documentary', label: 'Documentary (Clean)' },
  { value: 'minimal', label: 'Minimal' },
]

const SPEED_OPTIONS = [0.5, 0.75, 1, 1.25, 1.5, 2] as const

const STAGE_LABELS: Record<string, string> = {
  queued: 'Queued',
  script_generation: 'Generating Script',
  narration: 'Recording Narration',
  word_timestamps: 'Aligning Timestamps',
  asset_acquisition: 'Acquiring Assets',
  subtitle_generation: 'Generating Subtitles',
  director_review: 'Director Reviewing Draft',
  video_composition: 'Composing Video',
}

function VideoAgent() {
  // Form state
  const [topic, setTopic] = useState('')
  const [style, setStyle] = useState('documentary')
  const [targetDuration, setTargetDuration] = useState(8)
  const [subtitleStyle, setSubtitleStyle] = useState('hormozi')
  const [voiceId, setVoiceId] = useState<string | null>(null)

  // Script preview expand state per job
  const [expandedScripts, setExpandedScripts] = useState<Set<string>>(new Set())

  // Cost estimate
  const [costExpanded, setCostExpanded] = useState(false)
  const costEstimate = useMemo(() => estimateCost(targetDuration), [targetDuration])

  // Video player speed state per job
  const [playbackSpeeds, setPlaybackSpeeds] = useState<Record<string, number>>({})
  const videoRefs = useRef<Record<string, HTMLVideoElement | null>>({})

  // Store
  const { jobs, submitting, error, fetchJobs, startProduction, deleteJob, connectWebSocket } =
    useVideoStore()
  const { addJob, connectWebSocket: connectBgWs } = useJobQueueStore()

  useEffect(() => {
    fetchJobs()
  }, [fetchJobs])

  // Reconnect WebSockets for any active jobs on mount
  useEffect(() => {
    for (const job of jobs) {
      if (job.status === 'processing' || job.status === 'queued') {
        connectWebSocket(job.id)
      }
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const handleSubmit = useCallback(async () => {
    if (!topic.trim()) return

    const jobId = await startProduction({
      topic: topic.trim(),
      style,
      target_duration: targetDuration,
      subtitle_style: subtitleStyle,
      voice_id: voiceId,
    })

    if (jobId) {
      connectWebSocket(jobId)
      addJob({
        id: jobId,
        type: 'video',
        status: 'processing',
        label: `Video: ${topic.trim().slice(0, 40)}`,
        createdAt: new Date().toISOString(),
      })
      connectBgWs(jobId, 'video')
    }
  }, [topic, style, targetDuration, subtitleStyle, voiceId, startProduction, connectWebSocket, addJob, connectBgWs])

  const handleDownload = (jobId: string) => {
    window.open(`${API_BASE}/api/video/jobs/${jobId}/download`, '_blank')
  }

  const toggleScript = (jobId: string) => {
    setExpandedScripts((prev) => {
      const next = new Set(prev)
      if (next.has(jobId)) {
        next.delete(jobId)
      } else {
        next.add(jobId)
      }
      return next
    })
  }

  const setVideoRef = useCallback((jobId: string, el: HTMLVideoElement | null) => {
    videoRefs.current[jobId] = el
    if (el) {
      el.preservesPitch = true
      el.playbackRate = playbackSpeeds[jobId] ?? 1
    }
  }, [playbackSpeeds])

  const setSpeed = useCallback((jobId: string, speed: number) => {
    setPlaybackSpeeds((prev) => ({ ...prev, [jobId]: speed }))
    const video = videoRefs.current[jobId]
    if (video) {
      video.playbackRate = speed
    }
  }, [])

  const activeJobs = jobs.filter((j) => j.status === 'processing' || j.status === 'queued')
  const completedJobs = jobs.filter((j) => j.status === 'completed')
  const failedJobs = jobs.filter((j) => j.status === 'failed')

  return (
    <section className="video-agent-section">
      {/* Header */}
      <div className="section-header">
        <div className="section-icon">&#x1F3AC;</div>
        <div>
          <h2>Video Agent</h2>
          <p className="section-description">
            AI-powered short-form video production from topic to final render
          </p>
        </div>
      </div>

      {/* Form */}
      <div className="video-form">
        {/* Topic */}
        <div className="form-group">
          <label htmlFor="video-topic">Topic</label>
          <textarea
            id="video-topic"
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
            placeholder="Enter a topic for your video, e.g. '5 morning habits of successful entrepreneurs'"
            rows={3}
            disabled={submitting}
          />
        </div>

        {/* Options Row */}
        <div className="video-options-row">
          {/* Style */}
          <div className="form-group">
            <label htmlFor="video-style">Style</label>
            <select
              id="video-style"
              value={style}
              onChange={(e) => setStyle(e.target.value)}
              disabled={submitting}
            >
              {STYLE_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>

          {/* Duration */}
          <div className="form-group video-duration-control">
            <label htmlFor="video-duration">
              Duration
              <span className="video-duration-value">{targetDuration} min</span>
            </label>
            <input
              id="video-duration"
              type="range"
              min="1"
              max="30"
              step="1"
              value={targetDuration}
              onChange={(e) => setTargetDuration(parseInt(e.target.value))}
              disabled={submitting}
            />
            <div className="video-duration-bounds">
              <span>1 min</span>
              <span>30 min</span>
            </div>
          </div>

          {/* Subtitle Style */}
          <div className="form-group">
            <label htmlFor="video-subtitle-style">Subtitles</label>
            <select
              id="video-subtitle-style"
              value={subtitleStyle}
              onChange={(e) => setSubtitleStyle(e.target.value)}
              disabled={submitting}
            >
              {SUBTITLE_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Cost Estimate Panel */}
        <div className="video-cost-estimate">
          <button
            className="video-cost-toggle"
            onClick={() => setCostExpanded(!costExpanded)}
          >
            <span className="video-cost-summary">
              Est. cost: ~${costEstimate.total < 0.01 ? '< 0.01' : costEstimate.total.toFixed(2)}
            </span>
            <span className={`video-cost-chevron${costExpanded ? ' expanded' : ''}`}>
              &#x25B6;
            </span>
          </button>
          {costExpanded && (
            <div className="video-cost-breakdown">
              <div className="video-cost-row">
                <span>Narration (TTS)</span>
                <span>${costEstimate.tts.toFixed(4)}</span>
              </div>
              <div className="video-cost-row">
                <span>Images (~{targetDuration * 6} scenes)</span>
                <span>${costEstimate.images.toFixed(4)}</span>
              </div>
              <div className="video-cost-row">
                <span>Music (1 track)</span>
                <span>${costEstimate.music.toFixed(4)}</span>
              </div>
              <div className="video-cost-row">
                <span>B-roll (free)</span>
                <span>$0.00</span>
              </div>
              <div className="video-cost-row">
                <span>Director review (~2 loops)</span>
                <span>${costEstimate.director.toFixed(4)}</span>
              </div>
              <div className="video-cost-row video-cost-total">
                <span>Total</span>
                <span>${costEstimate.total.toFixed(4)}</span>
              </div>
            </div>
          )}
        </div>

        {/* Voice Selection */}
        <VoiceLibrary
          selectedVoiceId={voiceId}
          onSelectVoice={setVoiceId}
          disabled={submitting}
        />

        {/* Error */}
        {error && (
          <div className="video-error">
            <span className="error-icon">&#x26A0;</span>
            <span>{error}</span>
          </div>
        )}

        {/* Submit Button */}
        <button
          onClick={handleSubmit}
          disabled={submitting || !topic.trim()}
          className="btn-generate"
        >
          {submitting ? (
            <>
              <span className="loading-spinner-sm"></span>
              Submitting...
            </>
          ) : (
            <>
              <span className="btn-icon">&#x1F3AC;</span>
              Produce Video
            </>
          )}
        </button>
      </div>

      {/* Active Jobs */}
      {activeJobs.length > 0 && (
        <div className="video-jobs-section">
          <h3>Active Jobs ({activeJobs.length})</h3>
          {activeJobs.map((job) => (
            <div key={job.id} className={`video-job-card ${job.status}`}>
              <div className="video-job-header">
                <div>
                  <h4 className="video-job-title">{job.topic}</h4>
                  <div className="video-job-meta">
                    {job.style} &middot; {new Date(job.created_at).toLocaleTimeString()}
                  </div>
                </div>
                <span className={`video-job-status-badge ${job.status}`}>
                  {job.status}
                </span>
              </div>

              {/* Progress Bar */}
              <div className="video-progress-bar-track">
                <div
                  className="video-progress-bar-fill"
                  style={{ width: `${job.progress.percent}%` }}
                />
              </div>
              <div className="video-progress-info">
                <span className="video-progress-stage">
                  {STAGE_LABELS[job.progress.stage] ?? job.progress.stage}
                </span>
                <span>{job.progress.percent}%</span>
              </div>

              {/* Running Cost */}
              {job.cost && (
                <div className="video-job-cost-inline">
                  Est. cost: ~${job.cost.total.toFixed(2)}
                </div>
              )}

              {/* Script Preview */}
              {job.script && (
                <>
                  <button
                    className="video-script-toggle"
                    onClick={() => toggleScript(job.id)}
                  >
                    {expandedScripts.has(job.id) ? 'Hide Script' : 'Show Script'}
                  </button>
                  {expandedScripts.has(job.id) && (
                    <div className="video-script-preview">
                      <div className="script-title">{job.script.title}</div>
                      <div className="script-hook">{job.script.hook_voiceover}</div>
                      {job.script.scenes.map((scene) => (
                        <div key={scene.id} className="script-scene">
                          <div className="scene-label">Scene {scene.id}</div>
                          <div className="scene-voiceover">{scene.voiceover}</div>
                          <div className="scene-visual">
                            {scene.visual_type} &middot; {scene.visual_keywords.join(', ')}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Completed Jobs */}
      {completedJobs.length > 0 && (
        <div className="video-jobs-section">
          <h3>Completed ({completedJobs.length})</h3>
          {completedJobs.map((job) => (
            <div key={job.id} className="video-job-card completed">
              <div className="video-job-header">
                <div>
                  <h4 className="video-job-title">{job.topic}</h4>
                  <div className="video-job-meta">
                    {job.style} &middot; {new Date(job.created_at).toLocaleTimeString()}
                  </div>
                </div>
                <span className="video-job-status-badge completed">completed</span>
              </div>

              {/* Inline Video Player */}
              <div className="video-player-container">
                <video
                  ref={(el) => setVideoRef(job.id, el)}
                  className="video-player"
                  controls
                  playsInline
                  preload="metadata"
                  src={`${API_BASE}/api/video/jobs/${job.id}/stream`}
                />
                <div className="video-speed-controls">
                  {SPEED_OPTIONS.map((speed) => (
                    <button
                      key={speed}
                      className={`video-speed-btn${(playbackSpeeds[job.id] ?? 1) === speed ? ' active' : ''}`}
                      onClick={() => setSpeed(job.id, speed)}
                    >
                      {speed}x
                    </button>
                  ))}
                </div>
              </div>

              {/* Script Preview */}
              {job.script && (
                <>
                  <button
                    className="video-script-toggle"
                    onClick={() => toggleScript(job.id)}
                  >
                    {expandedScripts.has(job.id) ? 'Hide Script' : 'Show Script'}
                  </button>
                  {expandedScripts.has(job.id) && (
                    <div className="video-script-preview">
                      <div className="script-title">{job.script.title}</div>
                      <div className="script-hook">{job.script.hook_voiceover}</div>
                      {job.script.scenes.map((scene) => (
                        <div key={scene.id} className="script-scene">
                          <div className="scene-label">Scene {scene.id}</div>
                          <div className="scene-voiceover">{scene.voiceover}</div>
                          <div className="scene-visual">
                            {scene.visual_type} &middot; {scene.visual_keywords.join(', ')}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </>
              )}

              {/* Final Cost */}
              {job.cost && (
                <div className="video-job-cost-inline completed-cost">
                  Final cost: ~${job.cost.total.toFixed(2)}
                </div>
              )}

              <div className="video-job-actions">
                <button
                  className="btn-video-download"
                  onClick={() => handleDownload(job.id)}
                >
                  <span>&#x2B07;</span>
                  Download
                </button>
                <button
                  className="btn-video-delete"
                  onClick={() => deleteJob(job.id)}
                >
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Failed Jobs */}
      {failedJobs.length > 0 && (
        <div className="video-jobs-section">
          <h3>Failed ({failedJobs.length})</h3>
          {failedJobs.map((job) => (
            <div key={job.id} className="video-job-card failed">
              <div className="video-job-header">
                <div>
                  <h4 className="video-job-title">{job.topic}</h4>
                  <div className="video-job-meta">
                    {job.style} &middot; {new Date(job.created_at).toLocaleTimeString()}
                  </div>
                </div>
                <span className="video-job-status-badge failed">failed</span>
              </div>
              {job.error && <div className="video-job-error">{job.error}</div>}
              <div className="video-job-actions">
                <button
                  className="btn-video-delete"
                  onClick={() => deleteJob(job.id)}
                >
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </section>
  )
}

export default VideoAgent
