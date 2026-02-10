"""Fetch clean voice clips (no background music/audio) from YouTube.

Targets sources known for clean solo speech: university lectures,
conference talks, raw podcast recordings, interviews, court hearings.

Usage:
    cd stockpile && source .venv/bin/activate && python scripts/fetch_voices_clean.py
"""

import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yt_dlp

from services.voice_library import VoiceLibrary

# Curated queries targeting sources with NO background music.
# University lectures, raw podcasts, conference talks, interviews, depositions.
VOICE_QUERIES = [
    # University lectures (always clean audio, no music)
    {"query": "MIT OpenCourseWare lecture professor speaking", "name": "MIT Professor", "skip": 120},
    {"query": "Yale open course lecture history professor", "name": "Yale Lecturer", "skip": 90},
    {"query": "Stanford lecture computer science professor", "name": "Stanford CS Professor", "skip": 120},
    {"query": "Harvard lecture philosophy professor speaking", "name": "Harvard Professor", "skip": 90},
    # Conference talks / presentations (clean podium audio)
    {"query": "Google tech talk presentation speaker", "name": "Tech Talk Speaker", "skip": 120},
    {"query": "TED talk no music speaking solo stage", "name": "TED Speaker", "skip": 90},
    {"query": "PyCon conference talk python developer", "name": "PyCon Speaker", "skip": 60},
    # Raw podcast / interview (no production music)
    {"query": "Lex Fridman solo monologue intro speaking", "name": "Interview Host Male", "skip": 180},
    {"query": "NPR interview solo host speaking segment", "name": "NPR Host", "skip": 120},
    {"query": "C-SPAN congressional hearing speaking", "name": "Congressional Speaker", "skip": 300},
    # Audiobook / reading (professional studio, no effects)
    {"query": "LibriVox public domain audiobook reading chapter", "name": "LibriVox Reader Male", "skip": 60},
    {"query": "LibriVox female reader public domain book", "name": "LibriVox Reader Female", "skip": 60},
    # Educational / explainer (clean voiceover)
    {"query": "Khan Academy style math explanation voice", "name": "Math Explainer", "skip": 30},
    {"query": "language learning pronunciation teacher speaking", "name": "Language Teacher", "skip": 45},
    # Professional / formal speech (podium mic, no music)
    {"query": "courtroom lawyer opening statement speaking", "name": "Courtroom Speaker", "skip": 60},
    {"query": "book author reading own work live event", "name": "Author Reading", "skip": 90},
    # Diverse voices
    {"query": "BBC radio 4 presenter speaking solo segment", "name": "BBC Presenter", "skip": 120},
    {"query": "Australian professor university lecture speaking", "name": "Australian Lecturer", "skip": 90},
    {"query": "Indian English professor university lecture", "name": "Indian English Speaker", "skip": 90},
    {"query": "Japanese English speaker conference presentation", "name": "Japanese English Speaker", "skip": 120},
    # Backups
    {"query": "physics lecture professor blackboard explanation", "name": "Physics Professor", "skip": 120},
    {"query": "medical lecture doctor explaining procedure", "name": "Medical Lecturer", "skip": 90},
    {"query": "law school lecture professor speaking classroom", "name": "Law Professor", "skip": 120},
    {"query": "debate tournament solo speaker argument", "name": "Debate Speaker", "skip": 30},
    {"query": "chess commentary solo grandmaster explaining", "name": "Chess Commentator", "skip": 60},
]

CLIP_DURATION = 15
TARGET_VOICES = 20


def search_youtube(query: str) -> dict | None:
    """Search YouTube for a single video matching the query."""
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
        print(f"  Search failed: {e}")
    return None


def download_audio_clip(video_url: str, skip_seconds: int, duration: int, output_path: Path) -> bool:
    """Download audio clip and convert to WAV 24kHz mono."""
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
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        temp_files = list(Path(temp_dir).glob("audio.*"))
        if not temp_files:
            print("  No audio file produced")
            return False

        result = subprocess.run(
            ["ffmpeg", "-i", str(temp_files[0]),
             "-ar", "24000", "-ac", "1", "-acodec", "pcm_s16le",
             "-y", "-loglevel", "error", str(output_path)],
            capture_output=True, text=True, timeout=30,
        )

        for f in Path(temp_dir).glob("*"):
            f.unlink()
        Path(temp_dir).rmdir()

        if result.returncode != 0:
            print(f"  ffmpeg failed: {result.stderr}")
            return False

        if not output_path.exists() or output_path.stat().st_size < 1000:
            print("  Output too small or missing")
            return False

        return True

    except Exception as e:
        print(f"  Download failed: {e}")
        for f in Path(temp_dir).glob("*"):
            try: f.unlink()
            except: pass
        try: Path(temp_dir).rmdir()
        except: pass
        return False


def main():
    print("=" * 60)
    print("Clean Voice Fetcher - No background music/audio")
    print("=" * 60)

    library = VoiceLibrary()
    existing = library.list_voices()
    existing_names = {v.name for v in existing}
    print(f"Existing voices: {len(existing)}")

    temp_dir = Path(tempfile.mkdtemp())
    added = 0
    failed = 0
    skipped = 0

    for i, spec in enumerate(VOICE_QUERIES, 1):
        if added >= TARGET_VOICES:
            print(f"\nReached target of {TARGET_VOICES} voices.")
            break

        query = spec["query"]
        name = spec["name"]
        skip = spec["skip"]

        print(f"\n[{i}/{len(VOICE_QUERIES)}] {name}")
        print(f"  Query: {query}")

        if name in existing_names:
            print(f"  Skipped: already exists")
            skipped += 1
            continue

        video_info = search_youtube(query)
        if not video_info:
            print("  Failed: no results")
            failed += 1
            time.sleep(2)
            continue

        video_url = video_info.get("webpage_url") or video_info.get("url", "")
        video_title = video_info.get("title", "Unknown")
        video_duration = video_info.get("duration", 0)

        print(f"  Found: {video_title[:60]}... ({video_duration}s)")

        if video_duration and skip + CLIP_DURATION > video_duration:
            skip = max(0, video_duration - CLIP_DURATION - 5)
            print(f"  Adjusted skip to {skip}s")

        wav_path = temp_dir / f"{name.replace(' ', '_').lower()}.wav"
        print(f"  Downloading {CLIP_DURATION}s from {skip}s...")

        if not download_audio_clip(video_url, skip, CLIP_DURATION, wav_path):
            print("  Failed: download error")
            failed += 1
            time.sleep(2)
            continue

        file_size = wav_path.stat().st_size
        actual_duration = round((file_size - 44) / 48000, 1)

        voice = library.save_voice(name, wav_path.read_bytes(), "clip.wav")
        added += 1
        print(f"  Added: {name} ({actual_duration}s)")

        wav_path.unlink(missing_ok=True)
        time.sleep(2)

    try: temp_dir.rmdir()
    except: pass

    print("\n" + "=" * 60)
    print(f"DONE: Added {added}/{TARGET_VOICES} clean voices ({failed} failed, {skipped} skipped)")
    total = library.list_voices()
    custom = [v for v in total if not v.is_preset]
    print(f"Total: {len(total)} voices ({len(total) - len(custom)} presets + {len(custom)} custom)")
    print("=" * 60)

    return 0 if added >= TARGET_VOICES else 1


if __name__ == "__main__":
    sys.exit(main())
