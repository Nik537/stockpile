"""Unit tests for TTSService chunking and audio format handling."""

import io
import wave

import pytest
from services.tts_service import TTSService, TTSServiceError


def _make_wav_bytes(frame_count: int, framerate: int = 16000) -> bytes:
    """Create a simple silent mono WAV payload."""
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(framerate)
        wav_file.writeframes(b"\x00\x00" * frame_count)
    return output.getvalue()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_split_text_chunks_preserves_text():
    """Long text should be chunked and reconstruct to the same normalized text."""
    service = TTSService("http://example.com")
    try:
        text = (
            "This is a very long sentence for testing chunk behavior. "
            "It should be split into multiple chunks without dropping words. "
            "The resulting chunks should still preserve order and readability."
        )

        chunks = service._split_text_chunks(text, max_chars=60)

        assert len(chunks) > 1
        assert all(len(chunk) <= 60 for chunk in chunks)
        assert " ".join(chunks) == " ".join(text.split())
    finally:
        await service.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_merge_audio_chunks_wav():
    """WAV chunks should merge into one valid WAV stream."""
    service = TTSService("http://example.com")
    try:
        chunk1 = _make_wav_bytes(4000)
        chunk2 = _make_wav_bytes(6000)

        merged = service._merge_audio_chunks([chunk1, chunk2])

        assert service.detect_audio_format(merged) == "wav"
        with wave.open(io.BytesIO(merged), "rb") as wav_file:
            assert wav_file.getnframes() == 10000
            assert wav_file.getframerate() == 16000
    finally:
        await service.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_merge_audio_chunks_mixed_format_raises():
    """Mixed audio formats should fail fast when merging chunks."""
    service = TTSService("http://example.com")
    try:
        wav_chunk = _make_wav_bytes(1000)
        fake_mp3_chunk = b"ID3" + b"\x00" * 100

        with pytest.raises(TTSServiceError, match="mixed audio formats"):
            service._merge_audio_chunks([wav_chunk, fake_mp3_chunk])
    finally:
        await service.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_public_audio_chunks_and_merges(monkeypatch):
    """Public generation should chunk long text and merge downloaded audio."""
    service = TTSService("http://example.com")
    try:
        requested_chunks: list[str] = []
        split_chunks = ["chunk one", "chunk two", "chunk three"]
        wav_chunks = {
            "https://audio.local/1.wav": _make_wav_bytes(2000),
            "https://audio.local/2.wav": _make_wav_bytes(3000),
            "https://audio.local/3.wav": _make_wav_bytes(4000),
        }

        async def fake_generate_public(text: str) -> tuple[str, float]:
            requested_chunks.append(text)
            index = len(requested_chunks)
            return f"https://audio.local/{index}.wav", 0.25

        async def fake_download_audio(url: str) -> bytes:
            return wav_chunks[url]

        monkeypatch.setattr(service, "generate_public", fake_generate_public)
        monkeypatch.setattr(service, "download_audio", fake_download_audio)
        monkeypatch.setattr(service, "_split_text_chunks", lambda _text: split_chunks)

        merged_audio, total_cost = await service.generate_public_audio("very long text")

        assert len(requested_chunks) == 3
        assert total_cost == pytest.approx(0.75)
        assert service.detect_audio_format(merged_audio) == "wav"
        with wave.open(io.BytesIO(merged_audio), "rb") as wav_file:
            assert wav_file.getnframes() == 9000
    finally:
        await service.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_runpod_timeout_recovery_splits_and_recovers(monkeypatch):
    """Timed-out RunPod chunks should split and recover automatically."""
    service = TTSService("http://example.com")
    try:
        long_chunk = "a" * 240
        split_chunks = [long_chunk[:120], long_chunk[120:]]
        call_lengths: list[int] = []

        async def fake_generate_single(
            text: str,
            voice_ref_path: str | None = None,
            exaggeration: float = 0.5,
            cfg_weight: float = 0.5,
            temperature: float = 0.8,
        ) -> bytes:
            call_lengths.append(len(text))
            if len(text) > 120:
                raise TTSServiceError("RunPod execution failed: job timed out after 1 retries")
            return _make_wav_bytes(1500)

        monkeypatch.setattr(
            service,
            "_generate_runpod_single",
            fake_generate_single,
        )
        monkeypatch.setattr(
            service,
            "_split_text_chunks",
            lambda text, max_chars=450: split_chunks if len(text) > 120 else [text],
        )

        audio_bytes = await service._generate_runpod_chunk_with_recovery(long_chunk)

        assert call_lengths == [240, 120, 120]
        assert service.detect_audio_format(audio_bytes) == "wav"
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
            assert wav_file.getnframes() == 3000
    finally:
        await service.close()
