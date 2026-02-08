import { useJobStore } from '../stores'
import JobCard from './JobCard'
import './JobList.css'

function JobList() {
  const { jobs, deleteJob } = useJobStore()

  if (jobs.length === 0) {
    return (
      <div className="empty-state">
        <div className="empty-state-icon">&#x1F4F9;</div>
        <h3 className="empty-state-title">No processing jobs yet</h3>
        <p className="empty-state-description">
          Upload a video above to start generating AI-curated B-roll footage for your content.
        </p>
        <div className="empty-state-hint">
          <span className="hint-arrow">&#x2191;</span>
          <span>Drag & drop or click to upload</span>
        </div>
      </div>
    )
  }

  return (
    <div className="job-list">
      {jobs.map((job) => (
        <JobCard key={job.id} job={job} onDeleted={deleteJob} />
      ))}
    </div>
  )
}

export default JobList
