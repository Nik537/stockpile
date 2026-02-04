import { useState } from 'react'
import { OutlierVideo } from '../types'
import './OutlierCard.css'

interface OutlierCardProps {
  outlier: OutlierVideo
}

function OutlierCard({ outlier }: OutlierCardProps) {
  const [expanded, setExpanded] = useState(false)

  const formatViewCount = (count: number | null | undefined): string => {
    if (count == null) return '-'
    if (count >= 1_000_000) {
      return `${(count / 1_000_000).toFixed(1)}M`
    } else if (count >= 1_000) {
      return `${(count / 1_000).toFixed(1)}K`
    }
    return count.toString()
  }

  const formatDate = (dateStr: string): string => {
    if (!dateStr || dateStr.length !== 8) return dateStr

    // Parse YYYYMMDD format
    const year = dateStr.slice(0, 4)
    const month = dateStr.slice(4, 6)
    const day = dateStr.slice(6, 8)

    const date = new Date(`${year}-${month}-${day}`)
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    })
  }

  const handleClick = () => {
    window.open(outlier.url, '_blank', 'noopener,noreferrer')
  }

  const handleExpandClick = (e: React.MouseEvent) => {
    e.stopPropagation()
    setExpanded(!expanded)
  }

  const tierBadgeClass = `score-badge score-badge-${outlier.outlier_tier}`

  // Calculate the best display score (prefer composite if available)
  const displayScore = outlier.composite_score ?? outlier.outlier_score

  // Determine if engagement rate is good (> 3% is considered good)
  const hasGoodEngagement =
    outlier.engagement_rate != null && outlier.engagement_rate >= 3

  return (
    <div className="outlier-card" onClick={handleClick}>
      <div className="outlier-thumbnail">
        <img
          src={outlier.thumbnail_url}
          alt={outlier.title}
          loading="lazy"
        />
        <div className={tierBadgeClass}>{displayScore.toFixed(1)}x</div>

        {/* Reddit badge */}
        {outlier.found_on_reddit && (
          <div className="reddit-badge" title={`Reddit: r/${outlier.reddit_subreddit || 'unknown'}`}>
            🔥 {outlier.reddit_score != null ? formatViewCount(outlier.reddit_score) : ''}
          </div>
        )}

        {/* Trending badge */}
        {outlier.is_trending && (
          <div className="trending-badge" title="Trending: Growing faster than usual">
            📈
          </div>
        )}
      </div>

      <div className="outlier-content">
        <h4 className="outlier-title" title={outlier.title}>
          {outlier.title}
        </h4>

        <div className="outlier-meta">
          <span className="outlier-channel" title={outlier.channel_name}>
            @{outlier.channel_name}
          </span>
          <span className="outlier-separator">&#x2022;</span>
          <span className="outlier-views">
            {formatViewCount(outlier.view_count)} views
          </span>
        </div>

        {/* Main stats row */}
        <div className="outlier-stats">
          <div className="stat">
            <span className="stat-label">Ratio</span>
            <span className="stat-value">{outlier.outlier_score.toFixed(1)}x</span>
          </div>

          {outlier.engagement_rate != null && (
            <div className={`stat ${hasGoodEngagement ? 'stat-highlight' : ''}`}>
              <span className="stat-label">Engagement</span>
              <span className="stat-value">{outlier.engagement_rate.toFixed(1)}%</span>
            </div>
          )}

          {outlier.views_per_day != null && (
            <div className="stat">
              <span className="stat-label">Velocity</span>
              <span className="stat-value">
                {formatViewCount(outlier.views_per_day)}/day
              </span>
            </div>
          )}
        </div>

        {/* Expandable details section */}
        <button
          className="expand-button"
          onClick={handleExpandClick}
          aria-expanded={expanded}
        >
          {expanded ? 'Less ▲' : 'More ▼'}
        </button>

        {expanded && (
          <div className="outlier-details">
            <div className="details-grid">
              <div className="detail-item">
                <span className="detail-label">Channel avg</span>
                <span className="detail-value">
                  {formatViewCount(outlier.channel_average_views)}
                </span>
              </div>
              <div className="detail-item">
                <span className="detail-label">Uploaded</span>
                <span className="detail-value">
                  {formatDate(outlier.upload_date)}
                </span>
              </div>
              {outlier.days_since_upload != null && (
                <div className="detail-item">
                  <span className="detail-label">Days old</span>
                  <span className="detail-value">{outlier.days_since_upload}</span>
                </div>
              )}
              {outlier.like_count != null && (
                <div className="detail-item">
                  <span className="detail-label">Likes</span>
                  <span className="detail-value">
                    {formatViewCount(outlier.like_count)}
                  </span>
                </div>
              )}
              {outlier.comment_count != null && (
                <div className="detail-item">
                  <span className="detail-label">Comments</span>
                  <span className="detail-value">
                    {formatViewCount(outlier.comment_count)}
                  </span>
                </div>
              )}
              {outlier.velocity_score != null && (
                <div className="detail-item">
                  <span className="detail-label">Velocity score</span>
                  <span className="detail-value">
                    {outlier.velocity_score.toFixed(1)}x
                  </span>
                </div>
              )}
              {outlier.statistical_score != null && (
                <div className="detail-item">
                  <span className="detail-label">Statistical score</span>
                  <span className="detail-value">
                    {outlier.statistical_score.toFixed(1)}
                  </span>
                </div>
              )}
              {outlier.momentum_score != null && (
                <div className="detail-item">
                  <span className="detail-label">Momentum</span>
                  <span className="detail-value">
                    {outlier.momentum_score.toFixed(1)}x
                  </span>
                </div>
              )}
              {outlier.found_on_reddit && outlier.reddit_subreddit && (
                <div className="detail-item detail-item-full">
                  <span className="detail-label">Reddit</span>
                  <span className="detail-value">
                    r/{outlier.reddit_subreddit}
                    {outlier.reddit_score != null && ` (${formatViewCount(outlier.reddit_score)} pts)`}
                  </span>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default OutlierCard
