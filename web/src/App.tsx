import { useEffect, useState } from 'react'
import './App.css'
import JobList from './components/JobList'
import UploadForm from './components/UploadForm'
import OutlierFinder from './components/OutlierFinder'
import TTSGenerator from './components/TTSGenerator'
import { Job } from './types'

type ActiveTab = 'broll' | 'outliers' | 'tts'

function App() {
  const [jobs, setJobs] = useState<Job[]>([])
  const [loading, setLoading] = useState(true)
  const [activeTab, setActiveTab] = useState<ActiveTab>('broll')

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

  const handleJobCreated = (_jobId: string) => {
    // Immediately refresh jobs list
    fetchJobs()
  }

  const handleJobDeleted = (jobId: string) => {
    setJobs(jobs.filter(job => job.id !== jobId))
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="app-header-content">
          <div className="logo">
            <span className="logo-icon">&#x1F3AC;</span>
            <h1>Stockpile</h1>
          </div>
          <p className="tagline">AI-Powered B-roll Automation</p>

          {/* Tab Navigation */}
          <nav className="tab-nav">
            <button
              className={`tab-btn ${activeTab === 'broll' ? 'active' : ''}`}
              onClick={() => setActiveTab('broll')}
            >
              <span className="tab-icon">&#x1F4E4;</span>
              B-Roll Processor
            </button>
            <button
              className={`tab-btn ${activeTab === 'outliers' ? 'active' : ''}`}
              onClick={() => setActiveTab('outliers')}
            >
              <span className="tab-icon">&#x1F4CA;</span>
              Outlier Finder
            </button>
            <button
              className={`tab-btn ${activeTab === 'tts' ? 'active' : ''}`}
              onClick={() => setActiveTab('tts')}
            >
              <span className="tab-icon">&#x1F3A4;</span>
              TTS Generator
            </button>
          </nav>
        </div>
      </header>

      <main className="app-main">
        {/* Both tabs stay mounted - hidden via CSS to preserve state */}
        <div style={{ display: activeTab === 'broll' ? 'block' : 'none' }}>
          <section className="upload-section">
            <div className="section-header">
              <div className="section-icon">&#x1F4E4;</div>
              <div>
                <h2>Upload Video</h2>
                <p className="section-description">Drop your video to generate curated B-roll footage</p>
              </div>
            </div>
            <UploadForm onJobCreated={handleJobCreated} />
          </section>

          <section className="jobs-section">
            <div className="section-header">
              <div className="section-icon">&#x1F4CB;</div>
              <div>
                <h2>Processing Jobs</h2>
                <p className="section-description">Track your video processing in real-time</p>
              </div>
            </div>
            {loading ? (
              <div className="loading-state">
                <div className="loading-spinner"></div>
                <span className="loading-text">Loading jobs...</span>
              </div>
            ) : (
              <JobList jobs={jobs} onJobDeleted={handleJobDeleted} />
            )}
          </section>
        </div>

        <div style={{ display: activeTab === 'outliers' ? 'block' : 'none' }}>
          <OutlierFinder />
        </div>

        <div style={{ display: activeTab === 'tts' ? 'block' : 'none' }}>
          <TTSGenerator />
        </div>
      </main>

      <footer className="app-footer">
        <div className="footer-content">
          <span className="version-badge">v1.0.0</span>
          <p>Transform your videos with AI-curated B-roll footage</p>
        </div>
      </footer>
    </div>
  )
}

export default App
