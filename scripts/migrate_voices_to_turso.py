#!/usr/bin/env python3
"""Migrate local voice library to Turso cloud database.

Reads voices from ~/.stockpile/voices/metadata.json and audio files,
then inserts them into the Turso voices table.

Usage:
    # From the stockpile directory with venv activated:
    python scripts/migrate_voices_to_turso.py

    # Or with explicit env vars:
    TURSO_DATABASE_URL=libsql://... TURSO_AUTH_TOKEN=... python scripts/migrate_voices_to_turso.py
"""

import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Load .env if present
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import libsql_experimental as libsql

VOICES_DIR = Path.home() / ".stockpile" / "voices"
METADATA_FILE = VOICES_DIR / "metadata.json"


def main():
    turso_url = os.environ.get("TURSO_DATABASE_URL")
    turso_token = os.environ.get("TURSO_AUTH_TOKEN")

    if not turso_url or not turso_token:
        print("ERROR: TURSO_DATABASE_URL and TURSO_AUTH_TOKEN must be set")
        sys.exit(1)

    if not METADATA_FILE.exists():
        print(f"ERROR: No metadata file found at {METADATA_FILE}")
        sys.exit(1)

    # Connect to Turso
    db_path = str(VOICES_DIR / "voices_replica.db")
    conn = libsql.connect(db_path, sync_url=turso_url, auth_token=turso_token)
    conn.sync()

    # Create table
    conn.execute("""
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
    conn.commit()
    conn.sync()

    # Load local metadata
    voices = json.loads(METADATA_FILE.read_text())
    print(f"Found {len(voices)} voices in local metadata")

    # Check existing voices in DB
    existing = set()
    for row in conn.execute("SELECT id FROM voices").fetchall():
        existing.add(row[0])
    print(f"Found {len(existing)} voices already in Turso")

    migrated = 0
    skipped = 0
    errors = 0

    for v in voices:
        voice_id = v["id"]
        if voice_id in existing:
            print(f"  SKIP  {v['name']} (already in DB)")
            skipped += 1
            continue

        audio_path = Path(v.get("audio_path", ""))
        if not audio_path.exists():
            print(f"  ERROR {v['name']} - audio file not found: {audio_path}")
            errors += 1
            continue

        audio_bytes = audio_path.read_bytes()
        ext = audio_path.suffix.lower() or ".wav"

        try:
            conn.execute(
                "INSERT INTO voices (id, name, is_preset, audio_data, audio_format, created_at, duration_seconds, is_favorite) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    voice_id,
                    v["name"],
                    1 if v.get("is_preset") else 0,
                    audio_bytes,
                    ext,
                    v.get("created_at", ""),
                    v.get("duration_seconds", 0.0),
                    1 if v.get("is_favorite") else 0,
                ),
            )
            conn.commit()
            conn.sync()
            migrated += 1
            size_kb = len(audio_bytes) / 1024
            print(f"  OK    {v['name']} ({size_kb:.0f} KB)")
        except Exception as e:
            print(f"  ERROR {v['name']}: {e}")
            errors += 1

    print(f"\nDone! Migrated: {migrated}, Skipped: {skipped}, Errors: {errors}")


if __name__ == "__main__":
    main()
