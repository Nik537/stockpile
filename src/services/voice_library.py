"""Voice Library Service - Persistent storage for TTS voice references.

Supports two storage backends:
  1. Turso (libSQL) cloud database - used when TURSO_DATABASE_URL is configured.
     Audio bytes are stored as BLOBs. Persists across ephemeral deployments (Render, etc.).
  2. Local filesystem (~/.stockpile/voices/) - fallback when Turso is not available.
"""

import json
import logging
import os
import struct
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

VOICES_DIR = Path.home() / ".stockpile" / "voices"
METADATA_FILE = VOICES_DIR / "metadata.json"
AUDIO_DIR = VOICES_DIR / "audio"

# Try to import libsql for Turso support
try:
    import libsql_experimental as libsql

    LIBSQL_AVAILABLE = True
except ImportError:
    LIBSQL_AVAILABLE = False
    logger.debug("libsql not available, using local filesystem only for voices")


@dataclass
class Voice:
    """A voice reference for TTS cloning."""

    id: str
    name: str
    is_preset: bool
    audio_path: str
    created_at: str
    duration_seconds: float
    is_favorite: bool = False


class VoiceLibrary:
    """Persistent voice library for TTS voice references.

    Automatically uses Turso when configured, falls back to local filesystem.
    """

    def __init__(self, storage_dir: str | None = None):
        turso_url = os.environ.get("TURSO_DATABASE_URL")
        turso_token = os.environ.get("TURSO_AUTH_TOKEN")

        self.use_db = bool(LIBSQL_AVAILABLE and turso_url and turso_token)

        if self.use_db:
            self._init_db(turso_url, turso_token)
            logger.info("VoiceLibrary using Turso cloud database")
        else:
            self._conn = None
            self.storage_dir = Path(storage_dir) if storage_dir else VOICES_DIR
            self.metadata_file = self.storage_dir / "metadata.json"
            self.audio_dir = self.storage_dir / "audio"
            self._ensure_dirs()
            self._remove_presets()
            self._migrate_favorites()
            logger.info("VoiceLibrary using local filesystem")

    # -------------------------------------------------------------------------
    # Turso database backend
    # -------------------------------------------------------------------------

    def _init_db(self, turso_url: str, turso_token: str) -> None:
        """Initialize Turso database connection and create table."""
        # Local replica for faster reads
        db_dir = VOICES_DIR
        db_dir.mkdir(parents=True, exist_ok=True)
        db_path = str(db_dir / "voices_replica.db")

        try:
            self._conn = libsql.connect(
                db_path,
                sync_url=turso_url,
                auth_token=turso_token,
            )
            self._conn.sync()
        except Exception as e:
            logger.warning(f"Turso connection failed: {e}, falling back to filesystem")
            self.use_db = False
            self.storage_dir = VOICES_DIR
            self.metadata_file = self.storage_dir / "metadata.json"
            self.audio_dir = self.storage_dir / "audio"
            self._ensure_dirs()
            self._remove_presets()
            self._migrate_favorites()
            return

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS voices (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                is_preset INTEGER DEFAULT 0,
                audio_data BLOB,
                audio_format TEXT DEFAULT '.wav',
                created_at TEXT NOT NULL,
                duration_seconds REAL DEFAULT 0.0,
                is_favorite INTEGER DEFAULT 0
            )
        """)
        self._conn.commit()
        self._conn.sync()

    def _db_list_voices(self) -> list[Voice]:
        rows = self._conn.execute(
            "SELECT id, name, is_preset, audio_format, created_at, duration_seconds, is_favorite "
            "FROM voices ORDER BY is_favorite DESC, LOWER(name) ASC"
        ).fetchall()
        return [
            Voice(
                id=r[0],
                name=r[1],
                is_preset=bool(r[2]),
                audio_path=f"db:{r[0]}{r[3]}",
                created_at=r[4],
                duration_seconds=r[5],
                is_favorite=bool(r[6]),
            )
            for r in rows
        ]

    def _db_get_voice(self, voice_id: str) -> Voice | None:
        row = self._conn.execute(
            "SELECT id, name, is_preset, audio_format, created_at, duration_seconds, is_favorite "
            "FROM voices WHERE id = ?",
            (voice_id,),
        ).fetchone()
        if not row:
            return None
        return Voice(
            id=row[0],
            name=row[1],
            is_preset=bool(row[2]),
            audio_path=f"db:{row[0]}{row[3]}",
            created_at=row[4],
            duration_seconds=row[5],
            is_favorite=bool(row[6]),
        )

    def _db_save_voice(self, name: str, audio_bytes: bytes, filename: str) -> Voice:
        voice_id = str(uuid.uuid4())
        ext = Path(filename).suffix.lower() or ".wav"
        duration = self._calculate_duration(audio_bytes, ext)

        self._conn.execute(
            "INSERT INTO voices (id, name, is_preset, audio_data, audio_format, created_at, duration_seconds, is_favorite) "
            "VALUES (?, ?, 0, ?, ?, ?, ?, 0)",
            (voice_id, name, audio_bytes, ext, datetime.now().isoformat(), round(duration, 1)),
        )
        self._conn.commit()
        self._conn.sync()

        logger.info(f"Saved voice '{name}' to Turso (id={voice_id}, {len(audio_bytes)} bytes)")
        return Voice(
            id=voice_id,
            name=name,
            is_preset=False,
            audio_path=f"db:{voice_id}{ext}",
            created_at=datetime.now().isoformat(),
            duration_seconds=round(duration, 1),
        )

    def _db_toggle_favorite(self, voice_id: str) -> bool | None:
        row = self._conn.execute(
            "SELECT is_favorite FROM voices WHERE id = ?", (voice_id,)
        ).fetchone()
        if not row:
            return None
        new_val = 0 if row[0] else 1
        self._conn.execute(
            "UPDATE voices SET is_favorite = ? WHERE id = ?", (new_val, voice_id)
        )
        self._conn.commit()
        self._conn.sync()
        return bool(new_val)

    def _db_delete_voice(self, voice_id: str) -> bool:
        row = self._conn.execute(
            "SELECT id FROM voices WHERE id = ?", (voice_id,)
        ).fetchone()
        if not row:
            return False
        self._conn.execute("DELETE FROM voices WHERE id = ?", (voice_id,))
        self._conn.commit()
        self._conn.sync()
        logger.info(f"Deleted voice {voice_id} from Turso")
        return True

    def get_audio_bytes(self, voice_id: str) -> tuple[bytes | None, str | None]:
        """Get raw audio bytes and format for a voice (Turso backend only).

        Returns:
            (audio_bytes, audio_format) or (None, None) if not found.
        """
        if not self.use_db:
            # Filesystem fallback: read from file
            path = self._fs_get_audio_path(voice_id)
            if path and path.exists():
                return path.read_bytes(), path.suffix.lower()
            return None, None

        row = self._conn.execute(
            "SELECT audio_data, audio_format FROM voices WHERE id = ?",
            (voice_id,),
        ).fetchone()
        if row and row[0]:
            return row[0], row[1]
        return None, None

    def _db_get_audio_path(self, voice_id: str) -> Path | None:
        """Write voice audio to a temp file and return its path (for TTS generation)."""
        audio_bytes, audio_format = self.get_audio_bytes(voice_id)
        if not audio_bytes:
            return None

        ext = audio_format or ".wav"
        tmp = tempfile.NamedTemporaryFile(
            suffix=ext, prefix=f"voice_{voice_id}_", delete=False
        )
        tmp.write(audio_bytes)
        tmp.close()
        return Path(tmp.name)

    # -------------------------------------------------------------------------
    # Local filesystem backend
    # -------------------------------------------------------------------------

    def _ensure_dirs(self):
        """Create storage directories if they don't exist."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)

    def _load_metadata(self) -> list[dict]:
        """Load voice metadata from JSON file."""
        try:
            if self.metadata_file.exists():
                return json.loads(self.metadata_file.read_text())
        except Exception as e:
            logger.warning(f"Failed to load voice metadata: {e}")
        return []

    def _save_metadata(self, voices: list[dict]) -> None:
        """Save voice metadata to JSON file."""
        try:
            self.metadata_file.write_text(json.dumps(voices, indent=2))
        except Exception as e:
            logger.error(f"Failed to save voice metadata: {e}")

    def _remove_presets(self) -> None:
        """Remove legacy preset voice entries (they have no audio)."""
        voices = self._load_metadata()
        filtered = [v for v in voices if not v.get("is_preset")]
        if len(filtered) < len(voices):
            self._save_metadata(filtered)
            logger.info("Removed legacy preset voices")

    def _migrate_favorites(self) -> None:
        """Add is_favorite field to voices that don't have it."""
        voices = self._load_metadata()
        changed = False
        for v in voices:
            if "is_favorite" not in v:
                v["is_favorite"] = False
                changed = True
        if changed:
            self._save_metadata(voices)

    def _fs_list_voices(self) -> list[Voice]:
        voices_data = self._load_metadata()
        voices = [Voice(**v) for v in voices_data]
        voices.sort(key=lambda v: (not v.is_favorite, v.name.lower()))
        return voices

    def _fs_get_voice(self, voice_id: str) -> Voice | None:
        voices_data = self._load_metadata()
        for v in voices_data:
            if v["id"] == voice_id:
                return Voice(**v)
        return None

    def _fs_save_voice(self, name: str, audio_bytes: bytes, filename: str) -> Voice:
        voice_id = str(uuid.uuid4())
        ext = Path(filename).suffix.lower() or ".wav"
        audio_filename = f"{voice_id}{ext}"
        audio_path = self.audio_dir / audio_filename

        audio_path.write_bytes(audio_bytes)
        logger.info(f"Saved voice audio: {audio_path} ({len(audio_bytes)} bytes)")

        duration = self._calculate_duration(audio_bytes, ext)

        voice = Voice(
            id=voice_id,
            name=name,
            is_preset=False,
            audio_path=str(audio_path),
            created_at=datetime.now().isoformat(),
            duration_seconds=round(duration, 1),
        )

        voices = self._load_metadata()
        voices.append(asdict(voice))
        self._save_metadata(voices)

        logger.info(f"Saved custom voice '{name}' (id={voice_id})")
        return voice

    def _fs_toggle_favorite(self, voice_id: str) -> bool | None:
        voices = self._load_metadata()
        for v in voices:
            if v["id"] == voice_id:
                v["is_favorite"] = not v.get("is_favorite", False)
                self._save_metadata(voices)
                return v["is_favorite"]
        return None

    def _fs_delete_voice(self, voice_id: str) -> bool:
        voices = self._load_metadata()
        for i, v in enumerate(voices):
            if v["id"] == voice_id:
                audio_path = Path(v.get("audio_path", ""))
                if audio_path.exists():
                    try:
                        audio_path.unlink()
                        logger.info(f"Deleted audio file: {audio_path}")
                    except Exception as e:
                        logger.warning(f"Failed to delete audio file: {e}")
                voices.pop(i)
                self._save_metadata(voices)
                logger.info(f"Deleted voice: {voice_id}")
                return True
        return False

    def _fs_get_audio_path(self, voice_id: str) -> Path | None:
        voice = self._fs_get_voice(voice_id)
        if voice and voice.audio_path:
            path = Path(voice.audio_path)
            if path.exists():
                return path
        return None

    # -------------------------------------------------------------------------
    # Public interface (dispatches to the active backend)
    # -------------------------------------------------------------------------

    def list_voices(self) -> list[Voice]:
        """List all voices, favorites first."""
        if self.use_db:
            return self._db_list_voices()
        return self._fs_list_voices()

    def get_voice(self, voice_id: str) -> Voice | None:
        """Get a single voice by ID."""
        if self.use_db:
            return self._db_get_voice(voice_id)
        return self._fs_get_voice(voice_id)

    def save_voice(self, name: str, audio_bytes: bytes, filename: str) -> Voice:
        """Save a new custom voice."""
        if self.use_db:
            return self._db_save_voice(name, audio_bytes, filename)
        return self._fs_save_voice(name, audio_bytes, filename)

    def toggle_favorite(self, voice_id: str) -> bool | None:
        """Toggle favorite status. Returns new value or None if not found."""
        if self.use_db:
            return self._db_toggle_favorite(voice_id)
        return self._fs_toggle_favorite(voice_id)

    def delete_voice(self, voice_id: str) -> bool:
        """Delete a voice. Returns True if deleted."""
        if self.use_db:
            return self._db_delete_voice(voice_id)
        return self._fs_delete_voice(voice_id)

    def get_audio_path(self, voice_id: str) -> Path | None:
        """Get the path to a voice's audio file.

        For DB backend, writes audio to a temp file and returns that path.
        Caller should not delete the file (TTS generation needs it during the job).
        """
        if self.use_db:
            return self._db_get_audio_path(voice_id)
        return self._fs_get_audio_path(voice_id)

    @staticmethod
    def _calculate_duration(audio_bytes: bytes, ext: str) -> float:
        """Calculate audio duration from file bytes."""
        if ext in (".wav", ".x-wav") and len(audio_bytes) >= 44:
            try:
                sample_rate = struct.unpack_from("<I", audio_bytes, 24)[0]
                num_channels = struct.unpack_from("<H", audio_bytes, 22)[0]
                bits_per_sample = struct.unpack_from("<H", audio_bytes, 34)[0]
                if sample_rate > 0 and num_channels > 0 and bits_per_sample > 0:
                    bytes_per_second = sample_rate * num_channels * (bits_per_sample // 8)
                    data_size = len(audio_bytes) - 44
                    return round(data_size / bytes_per_second, 1)
            except Exception:
                pass
            return round((len(audio_bytes) - 44) / 48_000, 1)
        return round(len(audio_bytes) / 16_000, 1)
