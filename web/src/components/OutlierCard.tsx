import { OutlierVideo } from '../types'
import './OutlierCard.css'

interface OutlierCardProps {
  outlier: OutlierVideo
}

function OutlierCard({ outlier }: OutlierCardProps) {
  const formatViewCount = (count: number): string => {
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

  const tierBadgeClass = `score-badge score-badge-${outlier.outlier_tier}`

  return (
    <div className="outlier-card" onClick={handleClick}>
      <div className="outlier-thumbnail">
        <img
          src={outlier.thumbnail_url}
          alt={outlier.title}
          loading="lazy"
        />
        <div className={tierBadgeClass}>
          {outlier.outlier_score.toFixed(1)}x
        </div>
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

        <div className="outlier-stats">
          <div className="stat">
            <span className="stat-label">Channel avg</span>
            <span className="stat-value">
              {formatViewCount(outlier.channel_average_views)}
            </span>
          </div>
          <div className="stat">
            <span className="stat-label">Uploaded</span>
            <span className="stat-value">
              {formatDate(outlier.upload_date)}
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default OutlierCard
