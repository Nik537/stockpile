"""Fetch American voice clips from Freesound.org for the TTS voice library.

Downloads HQ preview MP3s from Freesound API, converts to WAV 24kHz mono
(optimal for Chatterbox/Qwen3-TTS), and registers via VoiceLibrary.

Usage:
    cd stockpile && source .venv/bin/activate && python scripts/fetch_voices_freesound.py

    # Fetch only male voices:
    python scripts/fetch_voices_freesound.py --male-only

    # Fetch only female voices:
    python scripts/fetch_voices_freesound.py --female-only
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

# Add stockpile src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from services.voice_library import VoiceLibrary

# Load .env
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

FREESOUND_API_KEY = os.getenv("FREESOUND_API_KEY", "")
FREESOUND_SEARCH_URL = "https://freesound.org/apiv2/search/text/"
PAGE_SIZE = 15  # max results per query page

# Search queries grouped by category
SEARCH_QUERIES = {
    "male": [
        {"query": "american male voice speech", "filter": "duration:[10 TO 30] tag:voice", "category": "American Male"},
        {"query": "male speaking english american", "filter": "duration:[10 TO 30] tag:speech", "category": "American Male"},
        {"query": "man talking american accent", "filter": "duration:[10 TO 30]", "category": "American Male"},
        {"query": "male narration english", "filter": "duration:[10 TO 30]", "category": "Male Narrator"},
        {"query": "male voiceover english", "filter": "duration:[10 TO 30]", "category": "Male Voiceover"},
        {"query": "english speech recording male", "filter": "duration:[10 TO 30] tag:voice", "category": "American Male"},
    ],
    "female": [
        {"query": "american female voice speech", "filter": "duration:[10 TO 30] tag:voice", "category": "American Female"},
        {"query": "female speaking english american", "filter": "duration:[10 TO 30] tag:speech", "category": "American Female"},
        {"query": "woman talking american accent", "filter": "duration:[10 TO 30]", "category": "American Female"},
        {"query": "female narration english", "filter": "duration:[10 TO 30]", "category": "Female Narrator"},
        {"query": "female voiceover english", "filter": "duration:[10 TO 30]", "category": "Female Voiceover"},
        {"query": "english speech recording female", "filter": "duration:[10 TO 30] tag:voice", "category": "American Female"},
    ],
}

# Existing Freesound IDs already in the library (from voice names)
EXISTING_FREESOUND_IDS = {6209, 8088, 7190, 730, 2007, 3168}

# Tags that suggest non-speech content (skip these)
SKIP_TAGS = {"music", "song", "singing", "beat", "loop", "ambient", "noise", "effect", "sfx", "drum"}


def search_freesound(query: str, filter_str: str) -> list[dict]:
    """Search Freesound API and return results."""
    params = {
        "query": query,
        "filter": filter_str,
        "fields": "id,name,tags,duration,previews,description",
        "page_size": PAGE_SIZE,
        "sort": "score",
        "token": FREESOUND_API_KEY,
    }
    try:
        resp = requests.get(FREESOUND_SEARCH_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", [])
    except Exception as e:
        print(f"  Search error: {e}")
        return []


def is_speech_clip(result: dict) -> bool:
    """Check if a Freesound result is likely clean speech (not music/sfx)."""
    tags = set(t.lower() for t in result.get("tags", []))
    # Must have at least one speech-related tag
    speech_tags = {"voice", "speech", "speaking", "talk", "talking", "narration",
                   "voiceover", "spoken", "human", "male", "female", "man", "woman"}
    if not tags & speech_tags:
        return False
    # Must not have music/sfx tags
    if tags & SKIP_TAGS:
        return False
    # Duration sanity check (10-30s)
    duration = result.get("duration", 0)
    if duration < 10 or duration > 35:
        return False
    return True


def download_and_convert(preview_url: str, output_wav: Path) -> bool:
    """Download MP3 preview and convert to WAV 24kHz mono."""
    try:
        resp = requests.get(preview_url, timeout=30)
        resp.raise_for_status()
        if len(resp.content) < 5000:
            print("  Download too small, skipping")
            return False

        # Write MP3 to temp file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(resp.content)
            mp3_path = tmp.name

        # Convert to WAV 24kHz mono via ffmpeg
        result = subprocess.run(
            ["ffmpeg", "-i", mp3_path, "-ar", "24000", "-ac", "1",
             "-acodec", "pcm_s16le", "-y", "-loglevel", "error", str(output_wav)],
            capture_output=True, text=True, timeout=30,
        )
        os.unlink(mp3_path)

        if result.returncode != 0:
            print(f"  ffmpeg failed: {result.stderr}")
            return False

        if not output_wav.exists() or output_wav.stat().st_size < 2000:
            print("  Converted WAV too small")
            return False

        return True

    except Exception as e:
        print(f"  Download/convert error: {e}")
        return False


def get_existing_freesound_ids(library: VoiceLibrary) -> set[int]:
    """Extract Freesound IDs from existing voice names like 'American Male - Deep (8088)'."""
    ids = set(EXISTING_FREESOUND_IDS)
    for voice in library.list_voices():
        # Look for (NNNNN) pattern at end of name
        name = voice.name
        if "(" in name and name.endswith(")"):
            try:
                fs_id = int(name.rsplit("(", 1)[1].rstrip(")"))
                ids.add(fs_id)
            except ValueError:
                pass
    return ids


def make_voice_name(result: dict, category: str) -> str:
    """Create a descriptive voice name from Freesound result."""
    fs_id = result["id"]
    tags = [t.lower() for t in result.get("tags", [])]

    # Try to find a descriptive adjective from tags
    descriptors = []
    tone_tags = {"deep", "warm", "clear", "bright", "smooth", "soft", "crisp",
                 "calm", "rich", "gentle", "strong", "bold", "baritone", "alto",
                 "tenor", "soprano", "husky", "raspy", "professional", "natural"}
    for t in tags:
        if t in tone_tags:
            descriptors.append(t.title())

    if descriptors:
        descriptor = descriptors[0]
    else:
        # Fallback: use duration as a differentiator
        dur = result.get("duration", 0)
        if dur < 15:
            descriptor = "Short"
        elif dur < 22:
            descriptor = "Medium"
        else:
            descriptor = "Long"

    return f"{category} - {descriptor} ({fs_id})"


def fetch_voices(gender: str, target: int, library: VoiceLibrary, existing_ids: set[int]) -> int:
    """Fetch voices for a specific gender. Returns count of voices added."""
    queries = SEARCH_QUERIES[gender]
    added = 0
    seen_ids = set(existing_ids)
    temp_dir = Path(tempfile.mkdtemp())

    for q_idx, query_spec in enumerate(queries, 1):
        if added >= target:
            break

        query = query_spec["query"]
        filter_str = query_spec["filter"]
        category = query_spec["category"]

        print(f"\n  [{q_idx}/{len(queries)}] Searching: {query}")
        results = search_freesound(query, filter_str)
        print(f"    Found {len(results)} results")

        for result in results:
            if added >= target:
                break

            fs_id = result["id"]
            if fs_id in seen_ids:
                continue

            if not is_speech_clip(result):
                continue

            name = result.get("name", "Unknown")
            duration = result.get("duration", 0)
            preview_url = result.get("previews", {}).get("preview-hq-mp3")
            if not preview_url:
                continue

            print(f"    Trying: {name} ({duration:.1f}s) [ID: {fs_id}]")

            # Download and convert
            wav_path = temp_dir / f"freesound_{fs_id}.wav"
            if not download_and_convert(preview_url, wav_path):
                continue

            # Create voice name and save
            voice_name = make_voice_name(result, category)
            wav_bytes = wav_path.read_bytes()
            voice = library.save_voice(voice_name, wav_bytes, "clip.wav")
            wav_path.unlink(missing_ok=True)

            seen_ids.add(fs_id)
            added += 1
            print(f"    ✓ Added: {voice_name} ({duration:.1f}s)")

            # Be polite to the API
            time.sleep(0.5)

    # Cleanup
    try:
        for f in temp_dir.glob("*"):
            f.unlink()
        temp_dir.rmdir()
    except Exception:
        pass

    return added


def main():
    parser = argparse.ArgumentParser(description="Fetch American voices from Freesound.org")
    parser.add_argument("--male-only", action="store_true", help="Fetch only male voices")
    parser.add_argument("--female-only", action="store_true", help="Fetch only female voices")
    parser.add_argument("--target", type=int, default=10, help="Target voices per gender (default: 10)")
    args = parser.parse_args()

    if not FREESOUND_API_KEY:
        print("ERROR: FREESOUND_API_KEY not set in .env")
        print("Get one at: https://freesound.org/apiv2/apply")
        return 1

    print("=" * 60)
    print("Freesound Voice Fetcher - American Voices")
    print("=" * 60)

    library = VoiceLibrary()
    existing = library.list_voices()
    existing_ids = get_existing_freesound_ids(library)
    print(f"Existing voices: {len(existing)} ({len(existing_ids)} known Freesound IDs)")

    total_added = 0

    if not args.female_only:
        print(f"\n{'='*40}")
        print(f"MALE VOICES (target: {args.target})")
        print(f"{'='*40}")
        male_added = fetch_voices("male", args.target, library, existing_ids)
        total_added += male_added
        # Update existing_ids so female fetch doesn't duplicate
        existing_ids = get_existing_freesound_ids(library)
        print(f"\nMale voices added: {male_added}/{args.target}")

    if not args.male_only:
        print(f"\n{'='*40}")
        print(f"FEMALE VOICES (target: {args.target})")
        print(f"{'='*40}")
        female_added = fetch_voices("female", args.target, library, existing_ids)
        total_added += female_added
        print(f"\nFemale voices added: {female_added}/{args.target}")

    # Final summary
    all_voices = library.list_voices()
    american_voices = [v for v in all_voices if "American" in v.name or "Male" in v.name or "Female" in v.name]

    print(f"\n{'='*60}")
    print(f"DONE: Added {total_added} new voices")
    print(f"Total voices in library: {len(all_voices)}")
    print(f"American/gendered voices: {len(american_voices)}")
    print("=" * 60)

    return 0 if total_added > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
