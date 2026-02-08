"""SQLite-based persistent job storage for Stockpile API.

Replaces in-memory dict storage so jobs survive server restarts.
Uses aiosqlite for async database operations.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)

# Default database path
DEFAULT_DB_PATH = ".stockpile/jobs.db"


class JobStore:
    """Async SQLite job storage.

    Provides persistent storage for processing jobs with async CRUD operations.
    WebSocket connections remain in-memory (they're ephemeral by nature).
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        """Initialize job store with database path.

        Args:
            db_path: Path to SQLite database file. Parent directory
                     will be created if it doesn't exist.
        """
        self.db_path = Path(db_path)
        self.db: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        """Initialize database connection and create schema.

        Creates the jobs table if it doesn't exist and enables
        WAL mode for better concurrent read performance.
        """
        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.db = await aiosqlite.connect(str(self.db_path))
        self.db.row_factory = aiosqlite.Row

        # Enable WAL mode for better concurrent read performance
        await self.db.execute("PRAGMA journal_mode=WAL")

        # Create jobs table
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL DEFAULT 'broll',
                status TEXT NOT NULL DEFAULT 'queued',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                data JSON,
                error TEXT
            )
        """)

        # Create index on type and status for faster filtering
        await self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_jobs_type_status
            ON jobs (type, status)
        """)

        # Create index on created_at for faster sorting
        await self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_jobs_created_at
            ON jobs (created_at DESC)
        """)

        await self.db.commit()
        logger.info(f"Job store connected: {self.db_path}")

    async def close(self) -> None:
        """Close database connection."""
        if self.db:
            await self.db.close()
            self.db = None
            logger.info("Job store connection closed")

    async def create_job(
        self,
        job_id: str,
        job_type: str = "broll",
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a new job.

        Args:
            job_id: Unique job identifier
            job_type: Job type (broll, outlier, bulk_image)
            data: Job data (video_filename, preferences, progress, output_dir, etc.)

        Returns:
            Created job as dict

        Raises:
            RuntimeError: If database is not connected
        """
        if self.db is None:
            raise RuntimeError("Database not connected. Call connect() first.")

        now = datetime.now().isoformat()
        data = data or {}

        # Ensure default progress is set
        if "progress" not in data:
            data["progress"] = {"stage": "queued", "percent": 0, "message": "Job queued"}

        await self.db.execute(
            "INSERT INTO jobs (id, type, status, created_at, updated_at, data) VALUES (?, ?, ?, ?, ?, ?)",
            (job_id, job_type, "queued", now, now, json.dumps(data)),
        )
        await self.db.commit()

        logger.info(f"Created job {job_id} of type {job_type}")

        return {
            "id": job_id,
            "type": job_type,
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "data": data,
            "error": None,
            # Flatten commonly accessed fields for backward compatibility
            "video_filename": data.get("video_filename", ""),
            "preferences": data.get("preferences", {}),
            "progress": data.get("progress", {"stage": "queued", "percent": 0, "message": "Job queued"}),
            "output_dir": data.get("output_dir"),
        }

    async def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Get a job by ID.

        Args:
            job_id: Job identifier

        Returns:
            Job dict or None if not found

        Raises:
            RuntimeError: If database is not connected
        """
        if self.db is None:
            raise RuntimeError("Database not connected. Call connect() first.")

        async with self.db.execute(
            "SELECT * FROM jobs WHERE id = ?", (job_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return self._row_to_dict(row)

    async def update_job(
        self,
        job_id: str,
        status: str | None = None,
        data: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> dict[str, Any] | None:
        """Update a job.

        Args:
            job_id: Job identifier
            status: New status (optional)
            data: Data fields to merge (optional)
            error: Error message (optional)

        Returns:
            Updated job dict or None if not found

        Raises:
            RuntimeError: If database is not connected
        """
        if self.db is None:
            raise RuntimeError("Database not connected. Call connect() first.")

        # Get current job
        current = await self.get_job(job_id)
        if current is None:
            return None

        now = datetime.now().isoformat()
        new_status = status or current["status"]
        new_error = error if error is not None else current["error"]

        # Merge data
        current_data = current.get("data", {})
        if data:
            current_data.update(data)

        await self.db.execute(
            "UPDATE jobs SET status = ?, updated_at = ?, data = ?, error = ? WHERE id = ?",
            (new_status, now, json.dumps(current_data), new_error, job_id),
        )
        await self.db.commit()

        logger.debug(f"Updated job {job_id}: status={new_status}")

        return await self.get_job(job_id)

    async def list_jobs(
        self,
        job_type: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List jobs with optional filters.

        Args:
            job_type: Filter by type (optional)
            status: Filter by status (optional)
            limit: Max results (default 100)

        Returns:
            List of job dicts, ordered by created_at desc

        Raises:
            RuntimeError: If database is not connected
        """
        if self.db is None:
            raise RuntimeError("Database not connected. Call connect() first.")

        query = "SELECT * FROM jobs WHERE 1=1"
        params: list[Any] = []

        if job_type:
            query += " AND type = ?"
            params.append(job_type)
        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        async with self.db.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [self._row_to_dict(row) for row in rows]

    async def delete_job(self, job_id: str) -> bool:
        """Delete a job.

        Args:
            job_id: Job identifier

        Returns:
            True if deleted, False if not found

        Raises:
            RuntimeError: If database is not connected
        """
        if self.db is None:
            raise RuntimeError("Database not connected. Call connect() first.")

        async with self.db.execute(
            "DELETE FROM jobs WHERE id = ? RETURNING id", (job_id,)
        ) as cursor:
            row = await cursor.fetchone()
            await self.db.commit()

            if row is not None:
                logger.info(f"Deleted job {job_id}")
                return True
            return False

    async def cleanup_old_jobs(self, days: int = 7) -> int:
        """Delete jobs older than specified days.

        Only deletes jobs with 'completed' or 'failed' status.
        Active or queued jobs are preserved regardless of age.

        Args:
            days: Delete jobs older than this many days

        Returns:
            Number of deleted jobs

        Raises:
            RuntimeError: If database is not connected
        """
        if self.db is None:
            raise RuntimeError("Database not connected. Call connect() first.")

        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        # Get count before delete
        async with self.db.execute(
            "SELECT COUNT(*) FROM jobs WHERE created_at < ? AND status IN ('completed', 'failed')",
            (cutoff,),
        ) as cursor:
            row = await cursor.fetchone()
            count = row[0] if row else 0

        if count > 0:
            await self.db.execute(
                "DELETE FROM jobs WHERE created_at < ? AND status IN ('completed', 'failed')",
                (cutoff,),
            )
            await self.db.commit()
            logger.info(f"Cleaned up {count} old jobs (older than {days} days)")

        return count

    async def get_job_count(
        self,
        job_type: str | None = None,
        status: str | None = None,
    ) -> int:
        """Get count of jobs matching filters.

        Args:
            job_type: Filter by type (optional)
            status: Filter by status (optional)

        Returns:
            Number of matching jobs

        Raises:
            RuntimeError: If database is not connected
        """
        if self.db is None:
            raise RuntimeError("Database not connected. Call connect() first.")

        query = "SELECT COUNT(*) FROM jobs WHERE 1=1"
        params: list[Any] = []

        if job_type:
            query += " AND type = ?"
            params.append(job_type)
        if status:
            query += " AND status = ?"
            params.append(status)

        async with self.db.execute(query, params) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else 0

    def _row_to_dict(self, row: aiosqlite.Row) -> dict[str, Any]:
        """Convert a database row to a dict.

        Parses the JSON data column and flattens commonly accessed
        fields for backward compatibility with the old in-memory storage.

        Args:
            row: Database row to convert

        Returns:
            Dictionary representation of the job
        """
        result: dict[str, Any] = {
            "id": row["id"],
            "type": row["type"],
            "status": row["status"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "error": row["error"],
        }

        # Parse and merge data JSON
        data_str = row["data"]
        if data_str:
            try:
                data = json.loads(data_str)
                result["data"] = data
                # Flatten commonly accessed fields for backward compatibility
                result["video_filename"] = data.get("video_filename", "")
                result["preferences"] = data.get("preferences", {})
                result["progress"] = data.get("progress", {"stage": "queued", "percent": 0, "message": "Job queued"})
                result["output_dir"] = data.get("output_dir")
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON data for job {row['id']}")
                result["data"] = {}
                result["video_filename"] = ""
                result["preferences"] = {}
                result["progress"] = {"stage": "queued", "percent": 0, "message": "Job queued"}
                result["output_dir"] = None
        else:
            result["data"] = {}
            result["video_filename"] = ""
            result["preferences"] = {}
            result["progress"] = {"stage": "queued", "percent": 0, "message": "Job queued"}
            result["output_dir"] = None

        return result


# Module-level singleton
_job_store: JobStore | None = None


async def get_job_store() -> JobStore:
    """Get or create the global JobStore singleton.

    Creates the database connection if it doesn't exist.

    Returns:
        The global JobStore instance

    Example:
        job_store = await get_job_store()
        job = await job_store.create_job("123", "broll", {"video_filename": "test.mp4"})
    """
    global _job_store
    if _job_store is None:
        _job_store = JobStore()
        await _job_store.connect()
    return _job_store


async def close_job_store() -> None:
    """Close the global JobStore connection.

    Call this during application shutdown to properly close the database.
    """
    global _job_store
    if _job_store is not None:
        await _job_store.close()
        _job_store = None
