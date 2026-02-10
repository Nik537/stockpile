"""Fetch 20+ diverse voice clips from YouTube for the TTS voice library.

Downloads audio-only clips using yt-dlp, converts to WAV 24kHz mono
(optimal for Chatterbox/Qwen3-TTS), and registers them via VoiceLibrary.

Usage:
    cd stockpile && source .venv/bin/activate && python scripts/fetch_voices.py
"""

import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Add stockpile src to path so we can import VoiceLibrary
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yt_dlp

from services.voice_library import VoiceLibrary

# Voice queries: each search targets a specific voice type.
# skip_seconds skips past intros/music to find clean solo speech.
VOICE_QUERIES = [
    # Male podcast hosts
    {"query": "solo podcast intro male speaking", "name": "Male Podcast Host 1", "skip": 60},
    {"query": "male podcaster monologue advice", "name": "Male Podcast Host 2", "skip": 45},
    {"query": "solo male podcast motivation talk", "name": "Male Podcast Host 3", "skip": 90},
    # Female podcast hosts
    {"query": "female podcast solo speaking intro", "name": "Female Podcast Host 1", "skip": 60},
    {"query": "woman podcaster monologue advice", "name": "Female Podcast Host 2", "skip": 45},
    {"query": "solo female podcast motivation talk", "name": "Female Podcast Host 3", "skip": 90},
    # Calm / meditation voices
    {"query": "guided meditation male voice calm", "name": "Calm Male Narrator", "skip": 30},
    {"query": "guided meditation female voice relaxing", "name": "Calm Female Narrator", "skip": 30},
    # Documentary narration
    {"query": "documentary narration solo voice", "name": "Documentary Narrator 1", "skip": 45},
    {"query": "nature documentary narration male", "name": "Documentary Narrator 2", "skip": 60},
    # Audiobook reading
    {"query": "audiobook reading male solo chapter", "name": "Male Audiobook Reader", "skip": 60},
    {"query": "audiobook reading female solo chapter", "name": "Female Audiobook Reader", "skip": 60},
    # Tech / energetic presenters
    {"query": "tech review solo male speaking", "name": "Tech Reviewer Male", "skip": 45},
    {"query": "tech review solo female speaking", "name": "Tech Reviewer Female", "skip": 45},
    # History / educational
    {"query": "history lecture solo professor speaking", "name": "History Lecturer 1", "skip": 90},
    {"query": "educational lecture solo explaining", "name": "Educational Speaker", "skip": 60},
    # Storytelling
    {"query": "storytelling podcast solo narrator", "name": "Storyteller 1", "skip": 45},
    {"query": "story narration solo male deep voice", "name": "Storyteller 2", "skip": 60},
    # Backup / diverse
    {"query": "motivational speech solo speaker", "name": "Motivational Speaker", "skip": 60},
    {"query": "news anchor solo speaking practice", "name": "News Anchor Style", "skip": 30},
    # Extra backups in case some fail
    {"query": "voiceover demo reel male", "name": "Voiceover Male", "skip": 30},
    {"query": "voiceover demo reel female", "name": "Voiceover Female", "skip": 30},
    {"query": "interview solo speaking Ted talk", "name": "TED Speaker", "skip": 60},
    {"query": "cooking show host solo speaking", "name": "Cooking Host", "skip": 45},
    {"query": "travel vlog solo narration", "name": "Travel Narrator", "skip": 60},
]

CLIP_DURATION = 15  # seconds
TARGET_VOICES = 20


def search_youtube(query: str) -> dict | None:
    """Search YouTube for a single video matching the query.

    Returns video info dict or None if search fails.
    """
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
        "default_search": "ytsearch1",
        "noplaylist": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"ytsearch1:{query}", download=False)
            if info and "entries" in info:
                entries = list(info["entries"])
                if entries:
                    return entries[0]
            elif info and info.get("id"):
                return info
    except Exception as e:
        print(f"  Search failed for '{query}': {e}")

    return None


def download_audio_clip(
    video_url: str, skip_seconds: int, duration: int, output_path: Path
) -> bool:
    """Download an audio-only clip from a YouTube video.

    Uses yt-dlp download_ranges to grab only the needed segment,
    then converts to WAV 24kHz mono via ffmpeg.

    Returns True if successful.
    """
    temp_dir = tempfile.mkdtemp()
    temp_audio = Path(temp_dir) / "audio.%(ext)s"

    start = skip_seconds
    end = skip_seconds + duration

    def download_ranges(info_dict, ydl):
        return [{"start_time": start, "end_time": end}]

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(temp_audio),
        "download_ranges": download_ranges,
        "force_keyframes_at_cuts": True,
        "quiet": True,
        "no_warnings": True,
        "no_progress": True,
        "retries": 3,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        # Find the downloaded file (could be .wav, .opus, .webm, etc.)
        temp_files = list(Path(temp_dir).glob("audio.*"))
        if not temp_files:
            print("  No audio file produced by yt-dlp")
            return False

        source_file = temp_files[0]

        # Convert to WAV 24kHz 16-bit mono (optimal for Chatterbox)
        result = subprocess.run(
            [
                "ffmpeg",
                "-i",
                str(source_file),
                "-ar",
                "24000",
                "-ac",
                "1",
                "-acodec",
                "pcm_s16le",
                "-y",
                "-loglevel",
                "error",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Clean up temp files
        for f in Path(temp_dir).glob("*"):
            f.unlink()
        Path(temp_dir).rmdir()

        if result.returncode != 0:
            print(f"  ffmpeg conversion failed: {result.stderr}")
            return False

        if not output_path.exists() or output_path.stat().st_size < 1000:
            print("  Output file too small or missing")
            return False

        return True

    except Exception as e:
        print(f"  Download/conversion failed: {e}")
        # Clean up temp dir
        for f in Path(temp_dir).glob("*"):
            try:
                f.unlink()
            except Exception:
                pass
        try:
            Path(temp_dir).rmdir()
        except Exception:
            pass
        return False


def main():
    print("=" * 60)
    print("Voice Library Fetcher - Downloading 20+ YouTube voice clips")
    print("=" * 60)

    library = VoiceLibrary()
    existing = library.list_voices()
    existing_names = {v.name for v in existing}
    print(f"Existing voices: {len(existing)} ({len([v for v in existing if v.is_preset])} presets)")

    temp_dir = Path(tempfile.mkdtemp())
    added = 0
    failed = 0
    skipped = 0

    for i, voice_spec in enumerate(VOICE_QUERIES, 1):
        if added >= TARGET_VOICES:
            print(f"\nReached target of {TARGET_VOICES} voices, stopping.")
            break

        query = voice_spec["query"]
        name = voice_spec["name"]
        skip = voice_spec["skip"]

        print(f"\n[{i}/{len(VOICE_QUERIES)}] {name}")
        print(f"  Searching: {query}")

        if name in existing_names:
            print(f"  Skipped: voice '{name}' already exists")
            skipped += 1
            continue

        # Search for video
        video_info = search_youtube(query)
        if not video_info:
            print("  Failed: no search results")
            failed += 1
            time.sleep(2)
            continue

        video_url = video_info.get("webpage_url") or video_info.get("url", "")
        video_title = video_info.get("title", "Unknown")
        video_duration = video_info.get("duration", 0)

        print(f"  Found: {video_title[:60]}... ({video_duration}s)")

        # Ensure skip doesn't exceed video duration
        if video_duration and skip + CLIP_DURATION > video_duration:
            skip = max(0, video_duration - CLIP_DURATION - 5)
            print(f"  Adjusted skip to {skip}s (video too short)")

        # Download audio clip
        wav_path = temp_dir / f"{name.replace(' ', '_').lower()}.wav"
        print(f"  Downloading {CLIP_DURATION}s clip from {skip}s...")

        success = download_audio_clip(video_url, skip, CLIP_DURATION, wav_path)
        if not success:
            print("  Failed: download/conversion error")
            failed += 1
            time.sleep(2)
            continue

        # Calculate actual duration from file size (WAV 24kHz 16-bit mono = 48000 bytes/sec)
        file_size = wav_path.stat().st_size
        actual_duration = round((file_size - 44) / 48000, 1)

        # Register in voice library
        wav_bytes = wav_path.read_bytes()
        voice = library.save_voice(name, wav_bytes, "clip.wav")
        added += 1

        print(f"  Added voice: {name} ({actual_duration}s) from {video_title[:50]}")

        # Clean up temp wav
        wav_path.unlink(missing_ok=True)

        # Rate limit between downloads
        time.sleep(2)

    # Clean up temp dir
    try:
        temp_dir.rmdir()
    except Exception:
        pass

    # Summary
    print("\n" + "=" * 60)
    print(f"DONE: Added {added}/{TARGET_VOICES} voices ({failed} failed, {skipped} skipped)")
    total = len(library.list_voices())
    print(f"Total voices in library: {total} ({len([v for v in library.list_voices() if v.is_preset])} presets + {total - len([v for v in library.list_voices() if v.is_preset])} custom)")
    print("=" * 60)

    if added < TARGET_VOICES:
        print(f"\nWARNING: Only added {added} voices. Run again or add more queries.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
