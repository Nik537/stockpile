import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  OutlierVideo,
  OutlierSearchParams,
  OutlierSearchStatus,
  OutlierWSMessage,
  OutlierSortField,
  SortDirection,
  OutlierFilters,
} from '../types'
import OutlierCard from './OutlierCard'
import './OutlierFinder.css'

const API_BASE = ''  // Same origin

function OutlierFinder() {
  // Search form state
  const [topic, setTopic] = useState('')
  const [maxChannels, setMaxChannels] = useState(10)
  const [minScore, setMinScore] = useState(3.0)
  const [days, setDays] = useState<number | null>(null)
  const [includeShorts, setIncludeShorts] = useState(false)
  const [minViews, setMinViews] = useState(5000)
  const [excludeIndian, setExcludeIndian] = useState(true)
  const [channelSize, setChannelSize] = useState<string>('any')

  // Channel size presets (subscriber counts)
  const channelSizePresets: Record<string, { min: number | null; max: number | null; label: string }> = {
    any: { min: null, max: null, label: 'Any size' },
    tiny: { min: null, max: 1000, label: 'Tiny (<1K)' },
    micro: { min: 1000, max: 10000, label: 'Micro (1K-10K)' },
    small: { min: 10000, max: 100000, label: 'Small (10K-100K)' },
    medium: { min: 100000, max: 1000000, label: 'Medium (100K-1M)' },
    large: { min: 1000000, max: null, label: 'Large (1M+)' },
  }

  // Search state
  const [searchId, setSearchId] = useState<string | null>(null)
  const [status, setStatus] = useState<OutlierSearchStatus | null>(null)
  const [channelsAnalyzed, setChannelsAnalyzed] = useState(0)
  const [totalChannels, setTotalChannels] = useState(0)
  const [videosScanned, setVideosScanned] = useState(0)
  const [outliers, setOutliers] = useState<OutlierVideo[]>([])
  const [error, setError] = useState<string | null>(null)
  const [isSearching, setIsSearching] = useState(false)

  // Sorting and filtering state
  const [sortField, setSortField] = useState<OutlierSortField>('composite_score')
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc')
  const [filters, setFilters] = useState<OutlierFilters>({
    minEngagementRate: null,
    minVelocity: null,
    tiers: {
      exceptional: true,
      strong: true,
      solid: true,
    },
    redditOnly: false,
  })

  const wsRef = useRef<WebSocket | null>(null)

  // Sort outliers based on current sort field
  const sortOutliers = useCallback((items: OutlierVideo[]): OutlierVideo[] => {
    return [...items].sort((a, b) => {
      let aVal: number | string | null = null
      let bVal: number | string | null = null

      switch (sortField) {
        case 'composite_score':
          aVal = a.composite_score ?? a.outlier_score
          bVal = b.composite_score ?? b.outlier_score
          break
        case 'outlier_score':
          aVal = a.outlier_score
          bVal = b.outlier_score
          break
        case 'engagement_rate':
          aVal = a.engagement_rate ?? 0
          bVal = b.engagement_rate ?? 0
          break
        case 'velocity_score':
          aVal = a.velocity_score ?? 0
          bVal = b.velocity_score ?? 0
          break
        case 'view_count':
          aVal = a.view_count
          bVal = b.view_count
          break
        case 'upload_date':
          aVal = a.upload_date
          bVal = b.upload_date
          break
      }

      if (aVal == null && bVal == null) return 0
      if (aVal == null) return 1
      if (bVal == null) return -1

      const comparison = aVal < bVal ? -1 : aVal > bVal ? 1 : 0
      return sortDirection === 'asc' ? comparison : -comparison
    })
  }, [sortField, sortDirection])

  // Filter and sort outliers
  const filteredOutliers = useMemo(() => {
    let result = outliers.filter((o) => {
      // Tier filter
      if (!filters.tiers[o.outlier_tier as keyof typeof filters.tiers]) {
        return false
      }

      // Engagement rate filter
      if (
        filters.minEngagementRate != null &&
        (o.engagement_rate == null || o.engagement_rate < filters.minEngagementRate)
      ) {
        return false
      }

      // Velocity filter
      if (
        filters.minVelocity != null &&
        (o.views_per_day == null || o.views_per_day < filters.minVelocity)
      ) {
        return false
      }

      // Reddit only filter
      if (filters.redditOnly && !o.found_on_reddit) {
        return false
      }

      return true
    })

    return sortOutliers(result)
  }, [outliers, filters, sortOutliers])

  // Group filtered outliers by tier
  const groupedOutliers = useMemo(() => ({
    exceptional: filteredOutliers.filter((o) => o.outlier_tier === 'exceptional'),
    strong: filteredOutliers.filter((o) => o.outlier_tier === 'strong'),
    solid: filteredOutliers.filter((o) => o.outlier_tier === 'solid'),
  }), [filteredOutliers])

  // Handle WebSocket messages
  const handleWSMessage = useCallback((message: OutlierWSMessage) => {
    switch (message.type) {
      case 'status':
        setStatus(message.status || null)
        setChannelsAnalyzed(message.channels_analyzed || 0)
        setTotalChannels(message.total_channels || 0)
        setVideosScanned(message.videos_scanned || 0)
        if (message.outliers) {
          setOutliers(message.outliers)
        }
        if (message.error) {
          setError(message.error)
        }
        break

      case 'progress':
        setChannelsAnalyzed(message.channels_analyzed || 0)
        setTotalChannels(message.total_channels || 0)
        setVideosScanned(message.videos_scanned || 0)
        break

      case 'outlier':
        if (message.outlier) {
          setOutliers((prev) => [...prev, message.outlier!])
        }
        break

      case 'complete':
        setStatus('completed')
        setIsSearching(false)
        break

      case 'error':
        setStatus('failed')
        setError(message.message || 'Search failed')
        setIsSearching(false)
        break
    }
  }, [])

  // Connect to WebSocket when we have a search ID
  useEffect(() => {
    if (!searchId) return

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsUrl = `${protocol}//${window.location.host}/ws/outliers/${searchId}`
    const ws = new WebSocket(wsUrl)

    ws.onopen = () => {
      console.log(`WebSocket connected for outlier search ${searchId}`)
    }

    ws.onmessage = (event) => {
      try {
        const message: OutlierWSMessage = JSON.parse(event.data)
        handleWSMessage(message)
      } catch (e) {
        console.error('Failed to parse WebSocket message:', e)
      }
    }

    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
      setError('WebSocket connection error')
    }

    ws.onclose = () => {
      console.log(`WebSocket disconnected for outlier search ${searchId}`)
    }

    wsRef.current = ws

    return () => {
      ws.close()
      wsRef.current = null
    }
  }, [searchId, handleWSMessage])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!topic.trim()) {
      setError('Please enter a topic')
      return
    }

    // Reset state
    setError(null)
    setOutliers([])
    setChannelsAnalyzed(0)
    setTotalChannels(0)
    setVideosScanned(0)
    setIsSearching(true)

    const sizePreset = channelSizePresets[channelSize]
    const params: OutlierSearchParams = {
      topic: topic.trim(),
      max_channels: maxChannels,
      min_score: minScore,
      days: days,
      include_shorts: includeShorts,
      min_subs: sizePreset?.min ?? null,
      max_subs: sizePreset?.max ?? null,
      min_views: minViews,
      exclude_indian: excludeIndian,
    }

    try {
      const response = await fetch(`${API_BASE}/api/outliers/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(params),
      })

      if (!response.ok) {
        const data = await response.json()
        throw new Error(data.detail || 'Failed to start search')
      }

      const data = await response.json()
      setSearchId(data.search_id)
      setStatus('searching')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start search')
      setIsSearching(false)
    }
  }

  // Export handlers
  const handleExport = async (format: 'csv' | 'json') => {
    if (!searchId) return

    try {
      const response = await fetch(
        `${API_BASE}/api/outliers/${searchId}/export?format=${format}`
      )
      if (!response.ok) {
        throw new Error('Export failed')
      }

      const blob = await response.blob()
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `outliers_${topic.replace(/\s+/g, '_')}.${format}`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    } catch (err) {
      setError(`Export failed: ${err instanceof Error ? err.message : 'Unknown error'}`)
    }
  }

  // Toggle tier visibility
  const toggleTier = (tier: 'exceptional' | 'strong' | 'solid') => {
    setFilters((prev) => ({
      ...prev,
      tiers: {
        ...prev.tiers,
        [tier]: !prev.tiers[tier],
      },
    }))
  }

  const progressPercent =
    totalChannels > 0 ? Math.round((channelsAnalyzed / totalChannels) * 100) : 0

  return (
    <section className="outlier-section">
      <div className="section-header">
        <div className="section-icon">&#x1F4CA;</div>
        <div>
          <h2>Find Viral YouTube Videos</h2>
          <p className="section-description">
            Discover videos that significantly outperform their channel's average
          </p>
        </div>
      </div>

      {/* Search Form */}
      <form className="outlier-form" onSubmit={handleSubmit}>
        <div className="form-row">
          <div className="form-group form-group-topic">
            <label htmlFor="topic">Topic</label>
            <input
              id="topic"
              type="text"
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              placeholder="e.g., tech reviews, cooking tutorials, fitness"
              disabled={isSearching}
            />
          </div>
        </div>

        <div className="form-row">
          <div className="form-group">
            <label htmlFor="maxChannels">Channels</label>
            <select
              id="maxChannels"
              value={maxChannels}
              onChange={(e) => setMaxChannels(Number(e.target.value))}
              disabled={isSearching}
            >
              <option value={5}>5 channels</option>
              <option value={10}>10 channels</option>
              <option value={20}>20 channels</option>
              <option value={30}>30 channels</option>
              <option value={50}>50 channels</option>
              <option value={75}>75 channels</option>
              <option value={100}>100 channels</option>
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="minScore">Min Score</label>
            <select
              id="minScore"
              value={minScore}
              onChange={(e) => setMinScore(Number(e.target.value))}
              disabled={isSearching}
            >
              <option value={2}>2x (low bar)</option>
              <option value={3}>3x (solid)</option>
              <option value={5}>5x (strong)</option>
              <option value={10}>10x (exceptional)</option>
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="days">Time Period</label>
            <select
              id="days"
              value={days ?? ''}
              onChange={(e) => setDays(e.target.value ? Number(e.target.value) : null)}
              disabled={isSearching}
            >
              <option value="">All time</option>
              <option value={7}>Last 7 days</option>
              <option value={30}>Last 30 days</option>
              <option value={90}>Last 90 days</option>
              <option value={365}>Last year</option>
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="channelSize">Channel Size</label>
            <select
              id="channelSize"
              value={channelSize}
              onChange={(e) => setChannelSize(e.target.value)}
              disabled={isSearching}
            >
              {Object.entries(channelSizePresets).map(([key, preset]) => (
                <option key={key} value={key}>
                  {preset.label}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="form-row">
          <div className="form-group">
            <label htmlFor="minViews">Min Views</label>
            <select
              id="minViews"
              value={minViews}
              onChange={(e) => setMinViews(Number(e.target.value))}
              disabled={isSearching}
            >
              <option value={0}>No minimum</option>
              <option value={1000}>1K+</option>
              <option value={5000}>5K+</option>
              <option value={10000}>10K+</option>
              <option value={50000}>50K+</option>
              <option value={100000}>100K+</option>
            </select>
          </div>
        </div>

        <div className="form-row form-row-options">
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={includeShorts}
              onChange={(e) => setIncludeShorts(e.target.checked)}
              disabled={isSearching}
            />
            <span>Include YouTube Shorts</span>
          </label>

          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={excludeIndian}
              onChange={(e) => setExcludeIndian(e.target.checked)}
              disabled={isSearching}
            />
            <span>Exclude Indian videos</span>
          </label>

          <button
            type="submit"
            className="btn-search"
            disabled={isSearching || !topic.trim()}
          >
            {isSearching ? (
              <>
                <span className="loading-spinner-sm"></span>
                Searching...
              </>
            ) : (
              <>
                <span className="btn-icon">&#x1F50D;</span>
                Search
              </>
            )}
          </button>
        </div>
      </form>

      {/* Error message */}
      {error && (
        <div className="outlier-error">
          <span className="error-icon">&#x26A0;</span>
          <span>{error}</span>
        </div>
      )}

      {/* Progress bar */}
      {isSearching && (
        <div className="outlier-progress">
          <div className="progress-header">
            <span>
              Analyzing {channelsAnalyzed}/{totalChannels || '?'} channels...
            </span>
            <span>{videosScanned} videos scanned</span>
          </div>
          <div className="progress-bar-container">
            <div
              className="progress-bar-fill"
              style={{ width: `${progressPercent}%` }}
            />
          </div>
        </div>
      )}

      {/* Results */}
      {outliers.length > 0 && (
        <div className="outlier-results">
          {/* Results Toolbar */}
          <div className="results-toolbar">
            <div className="results-summary">
              <span className="results-count">
                {filteredOutliers.length} outlier{filteredOutliers.length !== 1 ? 's' : ''}
                {filteredOutliers.length !== outliers.length &&
                  ` (${outliers.length} total)`}
              </span>
            </div>

            <div className="results-controls">
              {/* Sort dropdown */}
              <div className="sort-control">
                <label htmlFor="sort">Sort:</label>
                <select
                  id="sort"
                  value={sortField}
                  onChange={(e) => setSortField(e.target.value as OutlierSortField)}
                >
                  <option value="composite_score">Score</option>
                  <option value="outlier_score">Ratio</option>
                  <option value="engagement_rate">Engagement</option>
                  <option value="velocity_score">Velocity</option>
                  <option value="view_count">Views</option>
                  <option value="upload_date">Date</option>
                </select>
                <button
                  className="sort-direction-btn"
                  onClick={() =>
                    setSortDirection((d) => (d === 'asc' ? 'desc' : 'asc'))
                  }
                  title={sortDirection === 'asc' ? 'Ascending' : 'Descending'}
                >
                  {sortDirection === 'asc' ? '↑' : '↓'}
                </button>
              </div>

              {/* Export buttons */}
              {searchId && status === 'completed' && (
                <div className="export-controls">
                  <button
                    className="btn-export"
                    onClick={() => handleExport('csv')}
                    title="Export as CSV"
                  >
                    CSV
                  </button>
                  <button
                    className="btn-export"
                    onClick={() => handleExport('json')}
                    title="Export as JSON"
                  >
                    JSON
                  </button>
                </div>
              )}
            </div>
          </div>

          {/* Filters row */}
          <div className="filters-toolbar">
            <div className="filter-group">
              <label>Min Engagement:</label>
              <input
                type="number"
                min="0"
                max="100"
                step="0.5"
                placeholder="%"
                value={filters.minEngagementRate ?? ''}
                onChange={(e) =>
                  setFilters((f) => ({
                    ...f,
                    minEngagementRate: e.target.value ? Number(e.target.value) : null,
                  }))
                }
              />
            </div>

            <div className="filter-group">
              <label>Min Velocity:</label>
              <input
                type="number"
                min="0"
                step="1000"
                placeholder="/day"
                value={filters.minVelocity ?? ''}
                onChange={(e) =>
                  setFilters((f) => ({
                    ...f,
                    minVelocity: e.target.value ? Number(e.target.value) : null,
                  }))
                }
              />
            </div>

            <div className="tier-toggles">
              <button
                className={`tier-toggle ${filters.tiers.exceptional ? 'active' : ''}`}
                onClick={() => toggleTier('exceptional')}
              >
                Exceptional
              </button>
              <button
                className={`tier-toggle ${filters.tiers.strong ? 'active' : ''}`}
                onClick={() => toggleTier('strong')}
              >
                Strong
              </button>
              <button
                className={`tier-toggle ${filters.tiers.solid ? 'active' : ''}`}
                onClick={() => toggleTier('solid')}
              >
                Solid
              </button>
            </div>

            <label className="checkbox-label reddit-toggle">
              <input
                type="checkbox"
                checked={filters.redditOnly}
                onChange={(e) =>
                  setFilters((f) => ({ ...f, redditOnly: e.target.checked }))
                }
              />
              <span>Reddit only</span>
            </label>
          </div>

          <div className="results-header">
            <h3>{status === 'completed' ? 'Search Complete' : 'Found So Far'}</h3>
          </div>

          {/* Exceptional tier */}
          {groupedOutliers.exceptional.length > 0 && (
            <div className="outlier-tier-group tier-exceptional">
              <div className="tier-header">
                <span className="tier-badge tier-badge-exceptional">EXCEPTIONAL</span>
                <span className="tier-description">
                  10x+ average views ({groupedOutliers.exceptional.length})
                </span>
              </div>
              <div className="outlier-grid">
                {groupedOutliers.exceptional.map((outlier) => (
                  <OutlierCard key={outlier.video_id} outlier={outlier} />
                ))}
              </div>
            </div>
          )}

          {/* Strong tier */}
          {groupedOutliers.strong.length > 0 && (
            <div className="outlier-tier-group tier-strong">
              <div className="tier-header">
                <span className="tier-badge tier-badge-strong">STRONG</span>
                <span className="tier-description">
                  5-10x average views ({groupedOutliers.strong.length})
                </span>
              </div>
              <div className="outlier-grid">
                {groupedOutliers.strong.map((outlier) => (
                  <OutlierCard key={outlier.video_id} outlier={outlier} />
                ))}
              </div>
            </div>
          )}

          {/* Solid tier */}
          {groupedOutliers.solid.length > 0 && (
            <div className="outlier-tier-group tier-solid">
              <div className="tier-header">
                <span className="tier-badge tier-badge-solid">SOLID</span>
                <span className="tier-description">
                  3-5x average views ({groupedOutliers.solid.length})
                </span>
              </div>
              <div className="outlier-grid">
                {groupedOutliers.solid.map((outlier) => (
                  <OutlierCard key={outlier.video_id} outlier={outlier} />
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Empty state */}
      {status === 'completed' && outliers.length === 0 && (
        <div className="outlier-empty">
          <span className="empty-icon">&#x1F50D;</span>
          <h3>No Outliers Found</h3>
          <p>Try adjusting your search parameters or searching for a different topic.</p>
        </div>
      )}

      {/* Filtered empty state */}
      {outliers.length > 0 && filteredOutliers.length === 0 && (
        <div className="outlier-empty">
          <span className="empty-icon">&#x1F50D;</span>
          <h3>No Matching Results</h3>
          <p>Try adjusting your filters to see more results.</p>
        </div>
      )}
    </section>
  )
}

export default OutlierFinder
