import { Job } from '../types'
import JobCard from './JobCard'
import './JobList.css'

interface JobListProps {
  jobs: Job[]
  onJobDeleted: (jobId: string) => void
}

function JobList({ jobs, onJobDeleted }: JobListProps) {
  if (jobs.length === 0) {
    return (
      <div className="empty-state">
        <p>No jobs yet. Upload a video to get started!</p>
      </div>
    )
  }

  return (
    <div className="job-list">
      {jobs.map((job) => (
        <JobCard key={job.id} job={job} onDeleted={onJobDeleted} />
      ))}
    </div>
  )
}

export default JobList
