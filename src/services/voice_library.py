"""Voice Library Service - Persistent storage for TTS voice references."""

import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

VOICES_DIR = Path.home() / ".stockpile" / "voices"
METADATA_FILE = VOICES_DIR / "metadata.json"
AUDIO_DIR = VOICES_DIR / "audio"


@dataclass
class Voice:
    """A voice reference for TTS cloning."""

    id: str
    name: str
    is_preset: bool
    audio_path: str
    created_at: str
    duration_seconds: float


class VoiceLibrary:
    """Persistent voice library for TTS voice references."""

    def __init__(self, storage_dir: str | None = None):
        self.storage_dir = Path(storage_dir) if storage_dir else VOICES_DIR
        self.metadata_file = self.storage_dir / "metadata.json"
        self.audio_dir = self.storage_dir / "audio"
        self._ensure_dirs()
        self._ensure_presets()

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

    def _ensure_presets(self) -> None:
        """Create preset voice entries if they don't exist."""
        voices = self._load_metadata()
        preset_ids = {v["id"] for v in voices if v.get("is_preset")}

        presets = [
            {"name": "Default", "id": "preset-default"},
            {"name": "Deep Male", "id": "preset-deep-male"},
            {"name": "Warm Female", "id": "preset-warm-female"},
            {"name": "Narrator", "id": "preset-narrator"},
        ]

        added = False
        for preset in presets:
            if preset["id"] not in preset_ids:
                voice = {
                    "id": preset["id"],
                    "name": preset["name"],
                    "is_preset": True,
                    "audio_path": "",
                    "created_at": datetime.now().isoformat(),
                    "duration_seconds": 0.0,
                }
                voices.append(voice)
                added = True
                logger.info(f"Created preset voice: {preset['name']}")

        if added:
            self._save_metadata(voices)

    def list_voices(self) -> list[Voice]:
        """List all voices (presets + custom)."""
        voices_data = self._load_metadata()
        return [Voice(**v) for v in voices_data]

    def get_voice(self, voice_id: str) -> Voice | None:
        """Get a single voice by ID."""
        voices_data = self._load_metadata()
        for v in voices_data:
            if v["id"] == voice_id:
                return Voice(**v)
        return None

    def save_voice(self, name: str, audio_bytes: bytes, filename: str) -> Voice:
        """Save a new custom voice.

        Args:
            name: Display name for the voice
            audio_bytes: Raw audio file bytes
            filename: Original filename (for extension detection)

        Returns:
            The created Voice object
        """
        voice_id = str(uuid.uuid4())

        # Determine file extension
        ext = Path(filename).suffix.lower() or ".wav"
        audio_filename = f"{voice_id}{ext}"
        audio_path = self.audio_dir / audio_filename

        # Save audio file
        audio_path.write_bytes(audio_bytes)
        logger.info(f"Saved voice audio: {audio_path} ({len(audio_bytes)} bytes)")

        # Calculate duration (approximate from file size)
        # WAV: ~176KB per second at 44.1kHz 16-bit mono
        # MP3: ~16KB per second at 128kbps
        if ext in (".wav", ".x-wav"):
            duration = len(audio_bytes) / 176_000
        else:
            duration = len(audio_bytes) / 16_000

        voice = Voice(
            id=voice_id,
            name=name,
            is_preset=False,
            audio_path=str(audio_path),
            created_at=datetime.now().isoformat(),
            duration_seconds=round(duration, 1),
        )

        # Add to metadata
        voices = self._load_metadata()
        voices.append(asdict(voice))
        self._save_metadata(voices)

        logger.info(f"Saved custom voice '{name}' (id={voice_id})")
        return voice

    def delete_voice(self, voice_id: str) -> bool:
        """Delete a custom voice (presets cannot be deleted).

        Returns:
            True if deleted, False if not found or is preset
        """
        voices = self._load_metadata()

        for i, v in enumerate(voices):
            if v["id"] == voice_id:
                if v.get("is_preset"):
                    logger.warning(f"Cannot delete preset voice: {voice_id}")
                    return False

                # Delete audio file
                audio_path = Path(v.get("audio_path", ""))
                if audio_path.exists():
                    try:
                        audio_path.unlink()
                        logger.info(f"Deleted audio file: {audio_path}")
                    except Exception as e:
                        logger.warning(f"Failed to delete audio file: {e}")

                # Remove from metadata
                voices.pop(i)
                self._save_metadata(voices)
                logger.info(f"Deleted voice: {voice_id}")
                return True

        return False

    def get_audio_path(self, voice_id: str) -> Path | None:
        """Get the path to a voice's audio file."""
        voice = self.get_voice(voice_id)
        if voice and voice.audio_path:
            path = Path(voice.audio_path)
            if path.exists():
                return path
        return None
