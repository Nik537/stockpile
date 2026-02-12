import { useEffect, useState } from 'react'
import './App.css'
import JobList from './components/JobList'
import UploadForm from './components/UploadForm'
import OutlierFinder from './components/OutlierFinder'
import TTSGenerator from './components/TTSGenerator'
import ImageGenerator from './components/ImageGenerator'
import BulkImageGenerator from './components/BulkImageGenerator'
import MusicGenerator from './components/MusicGenerator'
import DatasetGenerator from './components/DatasetGenerator'
import VideoAgent from './components/VideoAgent'
import JobQueueBar from './components/JobQueueBar'
import { useJobStore, useJobQueueStore } from './stores'

type ActiveTab = 'broll' | 'outliers' | 'tts' | 'imagegen' | 'bulkimage' | 'music' | 'dataset' | 'video'

function App() {
  const { loading, fetchJobs } = useJobStore()
  const bgJobs = useJobQueueStore((s) => s.jobs)
  const [activeTab, setActiveTab] = useState<ActiveTab>('broll')

  const ttsJobCount = bgJobs.filter((j) => j.type === 'tts' && j.status === 'processing').length
  const imageJobCount = bgJobs.filter((j) => (j.type === 'image' || j.type === 'image-edit') && j.status === 'processing').length
  const musicJobCount = bgJobs.filter((j) => j.type === 'music' && j.status === 'processing').length
  const videoJobCount = bgJobs.filter((j) => j.type === 'video' && j.status === 'processing').length

  useEffect(() => {
    fetchJobs()
    // Refresh jobs every 5 seconds
    const interval = setInterval(fetchJobs, 5000)
    return () => clearInterval(interval)
  }, [fetchJobs])

  return (
    <div className="app">
      <header className="app-header">
        <div className="app-header-content">
          <div className="logo">
            <span className="logo-icon">&#x1F3AC;</span>
            <h1>Social Media Multi Tool</h1>
          </div>
          <p className="tagline">AI-Powered Content Creation Suite</p>

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
              {ttsJobCount > 0 && <span className="tab-job-badge">{ttsJobCount}</span>}
            </button>
            <button
              className={`tab-btn ${activeTab === 'imagegen' ? 'active' : ''}`}
              onClick={() => setActiveTab('imagegen')}
            >
              <span className="tab-icon">&#x1F3A8;</span>
              Image Generator
              {imageJobCount > 0 && <span className="tab-job-badge">{imageJobCount}</span>}
            </button>
            <button
              className={`tab-btn ${activeTab === 'bulkimage' ? 'active' : ''}`}
              onClick={() => setActiveTab('bulkimage')}
            >
              <span className="tab-icon">&#x1F5BC;</span>
              Bulk Images
            </button>
            <button
              className={`tab-btn ${activeTab === 'music' ? 'active' : ''}`}
              onClick={() => setActiveTab('music')}
            >
              <span className="tab-icon">&#x1F3B5;</span>
              Music Generator
              {musicJobCount > 0 && <span className="tab-job-badge">{musicJobCount}</span>}
            </button>
            <button
              className={`tab-btn ${activeTab === 'dataset' ? 'active' : ''}`}
              onClick={() => setActiveTab('dataset')}
            >
              <span className="tab-icon">&#x1F34C;</span>
              Dataset Gen
            </button>
            <button
              className={`tab-btn ${activeTab === 'video' ? 'active' : ''}`}
              onClick={() => setActiveTab('video')}
            >
              <span className="tab-icon">&#x1F3AC;</span>
              Video Agent
              {videoJobCount > 0 && <span className="tab-job-badge">{videoJobCount}</span>}
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
            <UploadForm onJobCreated={fetchJobs} />
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
              <JobList />
            )}
          </section>
        </div>

        <div style={{ display: activeTab === 'outliers' ? 'block' : 'none' }}>
          <OutlierFinder />
        </div>

        <div style={{ display: activeTab === 'tts' ? 'block' : 'none' }}>
          <TTSGenerator />
        </div>

        <div style={{ display: activeTab === 'imagegen' ? 'block' : 'none' }}>
          <ImageGenerator />
        </div>

        <div style={{ display: activeTab === 'bulkimage' ? 'block' : 'none' }}>
          <BulkImageGenerator />
        </div>

        <div style={{ display: activeTab === 'music' ? 'block' : 'none' }}>
          <MusicGenerator />
        </div>

        <div style={{ display: activeTab === 'dataset' ? 'block' : 'none' }}>
          <DatasetGenerator />
        </div>

        <div style={{ display: activeTab === 'video' ? 'block' : 'none' }}>
          <VideoAgent />
        </div>
      </main>

      <JobQueueBar />

      <footer className="app-footer">
        <div className="footer-content">
          <span className="version-badge">v1.0.0</span>
          <p>AI-Powered Content Creation Suite</p>
        </div>
      </footer>
    </div>
  )
}

export default App
