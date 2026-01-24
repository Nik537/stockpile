import { useEffect, useState } from 'react'
import './App.css'
import JobList from './components/JobList'
import UploadForm from './components/UploadForm'
import { Job } from './types'

function App() {
  const [jobs, setJobs] = useState<Job[]>([])
  const [loading, setLoading] = useState(true)

  const fetchJobs = async () => {
    try {
      const response = await fetch('/api/jobs')
      const data = await response.json()
      setJobs(data.jobs)
    } catch (error) {
      console.error('Failed to fetch jobs:', error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchJobs()
    // Refresh jobs every 5 seconds
    const interval = setInterval(fetchJobs, 5000)
    return () => clearInterval(interval)
  }, [])

  const handleJobCreated = (jobId: string) => {
    // Immediately refresh jobs list
    fetchJobs()
  }

  const handleJobDeleted = (jobId: string) => {
    setJobs(jobs.filter(job => job.id !== jobId))
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>🎬 Stockpile</h1>
        <p>AI-Powered B-roll Automation</p>
      </header>

      <main className="app-main">
        <section className="upload-section">
          <h2>Upload Video</h2>
          <UploadForm onJobCreated={handleJobCreated} />
        </section>

        <section className="jobs-section">
          <h2>Processing Jobs</h2>
          {loading ? (
            <p>Loading jobs...</p>
          ) : (
            <JobList jobs={jobs} onJobDeleted={handleJobDeleted} />
          )}
        </section>
      </main>

      <footer className="app-footer">
        <p>Stockpile v1.0.0 - Transform your videos with AI-curated B-roll footage</p>
      </footer>
    </div>
  )
}

export default App
