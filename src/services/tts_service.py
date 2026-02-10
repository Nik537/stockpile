"""TTS Service - HTTP client for TTS generation via RunPod or Colab server."""

import asyncio
import base64
import io
import json
import logging
import os
import re
import wave
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

# Settings file for URL persistence
SETTINGS_FILE = Path.home() / ".stockpile" / "tts_settings.json"

# RunPod configuration from environment
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID", "")
RUNPOD_QWEN3_ENDPOINT_ID = os.getenv("RUNPOD_QWEN3_ENDPOINT_ID", "")
RUNPOD_CHATTERBOX_EXT_ENDPOINT_ID = os.getenv("RUNPOD_CHATTERBOX_EXT_ENDPOINT_ID", "")

# RunPod public TTS endpoint
RUNPOD_PUBLIC_TTS_ENDPOINT = "chatterbox-turbo"
RUNPOD_API_BASE = "https://api.runpod.ai/v2"
TTS_CHUNK_MAX_CHARS = 450
RUNPOD_CHUNK_MAX_CHARS = 260
RUNPOD_TIMEOUT_RECOVERY_MIN_CHARS = 120
RUNPOD_TIMEOUT_RECOVERY_MAX_DEPTH = 3


class TTSServiceError(Exception):
    """Error from TTS service."""

    pass


class TTSService:
    """HTTP client for TTS generation via RunPod Serverless or Colab server."""

    def __init__(self, server_url: str = ""):
        """Initialize TTS service.

        Args:
            server_url: URL of the Chatterbox-TTS-Server (e.g., https://xxx.ngrok.io)
                        Only used for 'colab' mode.
        """
        # Load from settings file if not provided
        if not server_url:
            settings = self._load_settings()
            server_url = settings.get("server_url", "")
        self.server_url = server_url.rstrip("/") if server_url else ""
        # Long timeout for TTS generation (can take minutes for long text)
        self.client = httpx.AsyncClient(timeout=600.0)

        # RunPod configuration
        self.runpod_api_key = RUNPOD_API_KEY
        self.runpod_endpoint_id = RUNPOD_ENDPOINT_ID
        self.runpod_qwen3_endpoint_id = RUNPOD_QWEN3_ENDPOINT_ID
        self.runpod_chatterbox_ext_endpoint_id = RUNPOD_CHATTERBOX_EXT_ENDPOINT_ID

    def _load_settings(self) -> dict:
        """Load TTS settings from file."""
        try:
            if SETTINGS_FILE.exists():
                return json.loads(SETTINGS_FILE.read_text())
        except Exception as e:
            logger.warning(f"Failed to load TTS settings: {e}")
        return {}

    def _save_settings(self) -> None:
        """Persist TTS settings to file."""
        try:
            SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            settings = {"server_url": self.server_url}
            SETTINGS_FILE.write_text(json.dumps(settings))
            logger.debug(f"TTS settings saved to {SETTINGS_FILE}")
        except Exception as e:
            logger.warning(f"Failed to save TTS settings: {e}")

    def set_server_url(self, url: str) -> None:
        """Update the server URL and persist to settings file."""
        self.server_url = url.rstrip("/") if url else ""
        self._save_settings()
        logger.info(f"TTS server URL set to: {self.server_url}")

    @staticmethod
    def detect_audio_format(audio_bytes: bytes) -> str:
        """Detect audio format from magic bytes."""
        if len(audio_bytes) >= 12 and audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE":
            return "wav"
        if audio_bytes[:3] == b"ID3" or (
            len(audio_bytes) >= 2
            and audio_bytes[0] == 0xFF
            and (audio_bytes[1] & 0xE0) == 0xE0
        ):
            return "mp3"
        if audio_bytes[:4] == b"OggS":
            return "ogg"
        return "bin"

    @staticmethod
    def media_type_for_audio_format(audio_format: str) -> str:
        """Map internal audio format to HTTP content-type."""
        return {
            "wav": "audio/wav",
            "mp3": "audio/mpeg",
            "ogg": "audio/ogg",
        }.get(audio_format, "application/octet-stream")

    @staticmethod
    def file_extension_for_audio_format(audio_format: str) -> str:
        """Map internal audio format to file extension."""
        return {
            "wav": "wav",
            "mp3": "mp3",
            "ogg": "ogg",
        }.get(audio_format, "bin")

    def _split_text_chunks(self, text: str, max_chars: int = TTS_CHUNK_MAX_CHARS) -> list[str]:
        """Split long text into sentence-aware chunks."""
        normalized = " ".join(text.strip().split())
        if not normalized:
            return []

        if len(normalized) <= max_chars:
            return [normalized]

        sentence_parts = re.split(r"(?<=[.!?])\s+", normalized)
        if len(sentence_parts) == 1:
            sentence_parts = [normalized]

        chunks: list[str] = []
        current = ""

        def append_part(part: str) -> None:
            nonlocal current
            if not current:
                current = part
                return
            candidate = f"{current} {part}"
            if len(candidate) <= max_chars:
                current = candidate
                return
            chunks.append(current)
            current = part

        for sentence in sentence_parts:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) <= max_chars:
                append_part(sentence)
                continue

            # Hard wrap overlong sentences by word boundaries.
            words = sentence.split()
            current_word_chunk = ""
            for word in words:
                if not current_word_chunk:
                    current_word_chunk = word
                    continue
                candidate = f"{current_word_chunk} {word}"
                if len(candidate) <= max_chars:
                    current_word_chunk = candidate
                else:
                    append_part(current_word_chunk)
                    current_word_chunk = word

            if current_word_chunk:
                append_part(current_word_chunk)

        if current:
            chunks.append(current)

        return chunks

    def _merge_wav_chunks(self, audio_chunks: list[bytes]) -> bytes:
        """Merge WAV byte chunks into one valid WAV file."""
        frames: list[bytes] = []
        expected_params: tuple[int, int, int, str] | None = None

        for chunk in audio_chunks:
            with wave.open(io.BytesIO(chunk), "rb") as wav_file:
                current_params = (
                    wav_file.getnchannels(),
                    wav_file.getsampwidth(),
                    wav_file.getframerate(),
                    wav_file.getcomptype(),
                )
                if expected_params is None:
                    expected_params = current_params
                elif current_params != expected_params:
                    raise TTSServiceError(
                        "RunPod returned WAV chunks with incompatible audio parameters"
                    )
                frames.append(wav_file.readframes(wav_file.getnframes()))

        if expected_params is None:
            raise TTSServiceError("No WAV chunks to merge")

        output = io.BytesIO()
        with wave.open(output, "wb") as wav_out:
            wav_out.setnchannels(expected_params[0])
            wav_out.setsampwidth(expected_params[1])
            wav_out.setframerate(expected_params[2])
            wav_out.writeframes(b"".join(frames))

        return output.getvalue()

    def _merge_audio_chunks(self, audio_chunks: list[bytes]) -> bytes:
        """Merge chunked TTS audio into a single file."""
        if not audio_chunks:
            raise TTSServiceError("No audio chunks returned from TTS generation")
        if len(audio_chunks) == 1:
            return audio_chunks[0]

        formats = {self.detect_audio_format(chunk) for chunk in audio_chunks}
        if len(formats) != 1:
            raise TTSServiceError("TTS chunks returned mixed audio formats")

        audio_format = next(iter(formats))
        if audio_format == "wav":
            return self._merge_wav_chunks(audio_chunks)
        if audio_format == "mp3":
            # MP3 frame streams can be concatenated for sequential playback.
            return b"".join(audio_chunks)

        raise TTSServiceError(f"Unsupported chunk merge format: {audio_format}")

    @staticmethod
    def _is_timeout_error(error: TTSServiceError) -> bool:
        """Check whether a TTS error indicates timeout behavior."""
        error_text = str(error).lower()
        return "timed out" in error_text or "timeout" in error_text

    async def _generate_runpod_chunk_with_recovery(
        self,
        text: str,
        voice_ref_path: str | None = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        depth: int = 0,
    ) -> bytes:
        """Generate one chunk with timeout-aware recursive fallback splitting."""
        try:
            return await self._generate_runpod_single(
                text=text,
                voice_ref_path=voice_ref_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
            )
        except TTSServiceError as error:
            can_retry_with_split = (
                self._is_timeout_error(error)
                and depth < RUNPOD_TIMEOUT_RECOVERY_MAX_DEPTH
                and len(text) > RUNPOD_TIMEOUT_RECOVERY_MIN_CHARS
            )
            if not can_retry_with_split:
                raise

            split_size = max(RUNPOD_TIMEOUT_RECOVERY_MIN_CHARS, len(text) // 2)
            sub_chunks = self._split_text_chunks(text, max_chars=split_size)
            if len(sub_chunks) <= 1:
                raise

            logger.warning(
                f"RunPod timeout on chunk ({len(text)} chars). "
                f"Retrying with {len(sub_chunks)} smaller chunks (depth={depth + 1})."
            )

            sub_audio_chunks: list[bytes] = []
            for sub_chunk in sub_chunks:
                sub_audio_chunks.append(
                    await self._generate_runpod_chunk_with_recovery(
                        text=sub_chunk,
                        voice_ref_path=voice_ref_path,
                        exaggeration=exaggeration,
                        cfg_weight=cfg_weight,
                        temperature=temperature,
                        depth=depth + 1,
                    )
                )

            return self._merge_audio_chunks(sub_audio_chunks)

    async def check_health(self) -> dict:
        """Check if the TTS server is reachable.

        Returns:
            Health status dict with 'connected' boolean, 'server_url', and optional 'error'.
            Always includes 'server_url' (even if None) so frontend can restore it.
        """
        if not self.server_url:
            return {"connected": False, "server_url": None, "error": "No server URL configured"}

        # Base response always includes server_url
        base_response = {"server_url": self.server_url}

        try:
            # Try the health/initial-data endpoint
            response = await self.client.get(
                f"{self.server_url}/api/ui/initial-data",
                timeout=10.0,
            )
            if response.status_code == 200:
                return {"connected": True, **base_response}

            # Fallback: try the docs endpoint
            response = await self.client.get(
                f"{self.server_url}/docs",
                timeout=10.0,
            )
            if response.status_code == 200:
                return {"connected": True, **base_response}

            return {
                "connected": False,
                "error": f"Server returned status {response.status_code}",
                **base_response,
            }
        except httpx.TimeoutException:
            return {"connected": False, "error": "Connection timed out", **base_response}
        except httpx.ConnectError as e:
            return {"connected": False, "error": f"Connection failed: {e}", **base_response}
        except Exception as e:
            return {"connected": False, "error": str(e), **base_response}

    async def generate(
        self,
        text: str,
        voice_ref_path: str | None = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        output_format: str = "mp3",
    ) -> bytes:
        """Generate audio from text using the TTS server.

        Args:
            text: Text to convert to speech
            voice_ref_path: Optional path to voice reference audio file
            exaggeration: Voice exaggeration level (0.0-1.0)
            cfg_weight: CFG weight for generation (0.0-1.0)
            temperature: Generation temperature (0.0-1.0)
            output_format: Output audio format (mp3, wav)

        Returns:
            Audio bytes

        Raises:
            TTSServiceError: If generation fails
        """
        if not self.server_url:
            raise TTSServiceError("No TTS server URL configured")

        payload: dict = {
            "text": text,
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
            "temperature": temperature,
            "output_format": output_format,
        }

        # Add voice reference if provided
        if voice_ref_path:
            voice_path = Path(voice_ref_path)
            if not voice_path.exists():
                raise TTSServiceError(f"Voice reference file not found: {voice_ref_path}")

            with open(voice_path, "rb") as f:
                audio_bytes = f.read()
                payload["voice_reference"] = base64.b64encode(audio_bytes).decode()
                logger.info(f"Using voice reference: {voice_path.name}")

        logger.info(
            f"Generating TTS for {len(text)} characters "
            f"(exag={exaggeration}, cfg={cfg_weight}, temp={temperature})"
        )

        try:
            response = await self.client.post(
                f"{self.server_url}/tts",
                json=payload,
            )
            response.raise_for_status()

            audio_bytes = response.content
            logger.info(f"TTS generation complete: {len(audio_bytes)} bytes")
            return audio_bytes

        except httpx.TimeoutException:
            raise TTSServiceError(
                "TTS generation timed out. The text may be too long or the server is overloaded."
            )
        except httpx.HTTPStatusError as e:
            error_detail = ""
            try:
                error_data = e.response.json()
                error_detail = error_data.get("detail", str(e))
            except Exception:
                error_detail = e.response.text or str(e)
            raise TTSServiceError(f"TTS server error: {error_detail}")
        except Exception as e:
            raise TTSServiceError(f"TTS generation failed: {e}")

    def is_runpod_configured(self) -> bool:
        """Check if RunPod credentials are configured."""
        return bool(self.runpod_api_key and self.runpod_endpoint_id)

    async def check_runpod_health(self) -> dict:
        """Check if RunPod endpoint is configured and accessible.

        Returns:
            Health status dict with 'configured', 'available', and optional 'error'.
        """
        if not self.is_runpod_configured():
            return {
                "configured": False,
                "available": False,
                "error": "RunPod API key or endpoint ID not configured",
            }

        # RunPod serverless endpoints are always "available" when configured
        # The actual availability is determined at request time (cold start)
        return {
            "configured": True,
            "available": True,
            "endpoint_id": self.runpod_endpoint_id,
        }

    async def generate_runpod(
        self,
        text: str,
        voice_ref_path: str | None = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
    ) -> bytes:
        """Generate audio from text using RunPod Serverless.

        Args:
            text: Text to convert to speech
            voice_ref_path: Optional path to voice reference audio file
            exaggeration: Voice exaggeration level (0.0-1.0)
            cfg_weight: CFG weight for generation (0.0-1.0)
            temperature: Generation temperature (0.0-1.0)

        Returns:
            Audio bytes (format depends on backend output)

        Raises:
            TTSServiceError: If generation fails
        """
        if not self.is_runpod_configured():
            raise TTSServiceError(
                "RunPod not configured. Set RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID in .env"
            )

        chunks = self._split_text_chunks(text, max_chars=RUNPOD_CHUNK_MAX_CHARS)
        if not chunks:
            raise TTSServiceError("Text is empty after normalization")

        logger.info(
            f"Generating TTS via RunPod for {len(text)} characters across "
            f"{len(chunks)} chunk(s)"
        )

        audio_chunks: list[bytes] = []
        for index, chunk in enumerate(chunks, start=1):
            logger.info(
                f"RunPod TTS chunk {index}/{len(chunks)} ({len(chunk)} characters)"
            )
            audio_chunks.append(
                await self._generate_runpod_chunk_with_recovery(
                    text=chunk,
                    voice_ref_path=voice_ref_path,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=temperature,
                )
            )

        return self._merge_audio_chunks(audio_chunks)

    async def _poll_runpod_job(self, endpoint_id: str, payload: dict, max_wait: int = 600) -> dict:
        """Submit a RunPod job and poll until completion.

        Args:
            endpoint_id: RunPod serverless endpoint ID.
            payload: Full request payload (must include "input" key).
            max_wait: Maximum seconds to wait for completion.

        Returns:
            The "output" dict from the completed job.

        Raises:
            TTSServiceError: On timeout, failure, or API errors.
        """
        base_url = f"https://api.runpod.ai/v2/{endpoint_id}"
        headers = {
            "Authorization": f"Bearer {self.runpod_api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = await self.client.post(
                f"{base_url}/run", headers=headers, json=payload
            )
            response.raise_for_status()
            run_data = response.json()
            job_id = run_data.get("id")

            if not job_id:
                raise TTSServiceError("RunPod did not return a job ID")

            logger.info(f"RunPod job submitted: {job_id} (endpoint: {endpoint_id})")

            poll_url = f"{base_url}/status/{job_id}"
            elapsed = 0

            while elapsed < max_wait:
                # Progressive backoff: 2s for first 60s, 5s after that
                poll_interval = 2 if elapsed < 60 else 5
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval

                status_response = await self.client.get(poll_url, headers=headers)
                status_response.raise_for_status()
                result = status_response.json()

                job_status = result.get("status")
                logger.debug(
                    f"RunPod job {job_id} status: {job_status} ({elapsed}s elapsed)"
                )

                if job_status == "COMPLETED":
                    output = result.get("output", {})
                    if "error" in output:
                        raise TTSServiceError(
                            f"TTS generation error: {output['error']}"
                        )
                    return output

                if job_status == "FAILED":
                    error_msg = result.get("error", "Unknown error")
                    raise TTSServiceError(f"RunPod execution failed: {error_msg}")

                if job_status in ("IN_QUEUE", "IN_PROGRESS"):
                    continue

                logger.warning(f"Unexpected RunPod status: {job_status}")
                continue

            raise TTSServiceError(
                f"RunPod job timed out after {max_wait}s. "
                "The worker may still be cold-starting. Try again shortly."
            )

        except httpx.TimeoutException:
            raise TTSServiceError(
                "RunPod request timed out. The endpoint may be experiencing a cold start. "
                "Try again in a few seconds."
            )
        except httpx.HTTPStatusError as e:
            error_detail = ""
            try:
                error_data = e.response.json()
                error_detail = error_data.get("error", {}).get("message", str(e))
            except Exception:
                error_detail = e.response.text or str(e)
            raise TTSServiceError(f"RunPod API error: {error_detail}")
        except TTSServiceError:
            raise
        except Exception as e:
            raise TTSServiceError(f"RunPod TTS generation failed: {e}")

    async def _generate_runpod_single(
        self,
        text: str,
        voice_ref_path: str | None = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
    ) -> bytes:
        """Generate one TTS chunk using RunPod serverless."""
        logger.info(
            f"RunPod single chunk: exaggeration={exaggeration}, "
            f"cfg_weight={cfg_weight}, temperature={temperature}"
        )
        payload: dict = {
            "input": {
                "text": text,
                "exaggeration": exaggeration,
                "cfg_weight": cfg_weight,
                "temperature": temperature,
            }
        }

        # Add voice reference if provided
        if voice_ref_path:
            voice_path = Path(voice_ref_path)
            if not voice_path.exists():
                raise TTSServiceError(f"Voice reference file not found: {voice_ref_path}")

            with open(voice_path, "rb") as f:
                audio_bytes = f.read()
                payload["input"]["voice_reference"] = base64.b64encode(audio_bytes).decode()
                logger.info(f"Using voice reference: {voice_path.name}")

        output = await self._poll_runpod_job(self.runpod_endpoint_id, payload)

        if "audio_base64" not in output:
            raise TTSServiceError("No audio returned from RunPod")

        audio_bytes = base64.b64decode(output["audio_base64"])
        logger.info(f"RunPod TTS generation complete: {len(audio_bytes)} bytes")
        return audio_bytes

    def is_public_endpoint_configured(self) -> bool:
        """Check if RunPod API key is configured for public endpoint."""
        return bool(self.runpod_api_key)

    async def check_public_health(self) -> dict:
        """Check if public TTS endpoint is configured.

        Returns:
            Health status dict with 'configured' and 'available'.
        """
        if not self.is_public_endpoint_configured():
            return {
                "configured": False,
                "available": False,
                "error": "RUNPOD_API_KEY not configured for public endpoint",
            }

        return {
            "configured": True,
            "available": True,
            "endpoint": RUNPOD_PUBLIC_TTS_ENDPOINT,
        }

    async def generate_public(
        self,
        text: str,
    ) -> tuple[str, float]:
        """Generate audio using RunPod's public Chatterbox Turbo endpoint.

        This uses the pre-deployed public endpoint - no custom Docker image needed.
        Note: Does NOT support voice cloning (use generate_runpod for that).

        Args:
            text: Text to convert to speech

        Returns:
            Tuple of (audio_url, cost)

        Raises:
            TTSServiceError: If generation fails
        """
        if not self.is_public_endpoint_configured():
            raise TTSServiceError(
                "RUNPOD_API_KEY not configured. Set it in your .env file."
            )

        url = f"{RUNPOD_API_BASE}/{RUNPOD_PUBLIC_TTS_ENDPOINT}/runsync"
        headers = {
            "Authorization": f"Bearer {self.runpod_api_key}",
            "Content-Type": "application/json",
        }

        # Public endpoint only accepts 'prompt' - no voice/format params
        payload = {
            "input": {
                "prompt": text,
            }
        }

        logger.info(
            f"Generating TTS via public endpoint for {len(text)} characters"
        )

        try:
            response = await self.client.post(url, headers=headers, json=payload)
            response.raise_for_status()

            result = response.json()

            # Check for RunPod execution errors
            if result.get("status") == "FAILED":
                error_msg = result.get("error", "Unknown error")
                raise TTSServiceError(f"Public TTS endpoint failed: {error_msg}")

            # Extract output - public endpoint returns audio_url
            output = result.get("output", {})

            if "error" in output:
                raise TTSServiceError(f"TTS generation error: {output['error']}")

            audio_url = output.get("audio_url") or output.get("result")
            if not audio_url:
                raise TTSServiceError("No audio URL returned from public endpoint")

            cost = output.get("cost", 0.0)

            logger.info(f"Public TTS generation complete: {audio_url} (cost: ${cost:.4f})")
            return audio_url, cost

        except httpx.TimeoutException:
            raise TTSServiceError(
                "Public TTS request timed out. Try again in a few seconds."
            )
        except httpx.HTTPStatusError as e:
            error_detail = ""
            try:
                error_data = e.response.json()
                error_detail = error_data.get("error", {}).get("message", str(e))
            except Exception:
                error_detail = e.response.text or str(e)
            raise TTSServiceError(f"Public TTS API error: {error_detail}")
        except TTSServiceError:
            raise
        except Exception as e:
            raise TTSServiceError(f"Public TTS generation failed: {e}")

    async def generate_public_audio(self, text: str) -> tuple[bytes, float]:
        """Generate and download public endpoint audio with chunking for long text."""
        chunks = self._split_text_chunks(text)
        if not chunks:
            raise TTSServiceError("Text is empty after normalization")

        logger.info(
            f"Generating public TTS for {len(text)} characters across {len(chunks)} chunk(s)"
        )

        total_cost = 0.0
        audio_chunks: list[bytes] = []

        for index, chunk in enumerate(chunks, start=1):
            logger.info(
                f"Public TTS chunk {index}/{len(chunks)} ({len(chunk)} characters)"
            )
            audio_url, cost = await self.generate_public(chunk)
            total_cost += cost
            audio_chunks.append(await self.download_audio(audio_url))

        merged_audio = self._merge_audio_chunks(audio_chunks)
        return merged_audio, total_cost

    # =========================================================================
    # Qwen3-TTS
    # =========================================================================

    def is_qwen3_configured(self) -> bool:
        """Check if Qwen3-TTS endpoint is configured."""
        return bool(self.runpod_api_key and self.runpod_qwen3_endpoint_id)

    async def check_qwen3_health(self) -> dict:
        """Check if Qwen3-TTS endpoint is configured."""
        if not self.is_qwen3_configured():
            return {
                "configured": False,
                "available": False,
                "error": "RUNPOD_QWEN3_ENDPOINT_ID not configured",
            }
        return {
            "configured": True,
            "available": True,
            "endpoint_id": self.runpod_qwen3_endpoint_id,
        }

    async def generate_qwen3(
        self,
        text: str,
        voice_ref_path: str | None = None,
        voice_reference_transcript: str | None = None,
        speaker_name: str | None = None,
        instruction: str | None = None,
        language: str = "auto",
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> bytes:
        """Generate audio using Qwen3-TTS via RunPod serverless.

        Args:
            text: Text to convert to speech.
            voice_ref_path: Optional path to 3-second voice reference audio.
            voice_reference_transcript: Transcript of the voice reference audio.
            speaker_name: Preset speaker name (e.g. "Chelsie", "Ethan").
            instruction: Style instruction (e.g. "Speak with excitement").
            language: Language code or "auto".
            temperature: Sampling temperature.
            top_p: Nucleus sampling top-p.

        Returns:
            Audio bytes (WAV).
        """
        if not self.is_qwen3_configured():
            raise TTSServiceError(
                "Qwen3-TTS not configured. Set RUNPOD_API_KEY and RUNPOD_QWEN3_ENDPOINT_ID in .env"
            )

        input_data: dict = {
            "text": text,
            "language": language,
            "temperature": temperature,
            "top_p": top_p,
        }

        if speaker_name:
            input_data["speaker_name"] = speaker_name
        if instruction:
            input_data["instruction"] = instruction

        # Voice reference for cloning
        if voice_ref_path:
            voice_path = Path(voice_ref_path)
            if not voice_path.exists():
                raise TTSServiceError(f"Voice reference file not found: {voice_ref_path}")
            with open(voice_path, "rb") as f:
                audio_bytes = f.read()
                input_data["voice_reference"] = base64.b64encode(audio_bytes).decode()
                logger.info(f"Qwen3: Using voice reference: {voice_path.name}")
            if voice_reference_transcript:
                input_data["voice_reference_transcript"] = voice_reference_transcript

        logger.info(
            f"Generating TTS via Qwen3 for {len(text)} characters "
            f"(lang={language}, speaker={speaker_name})"
        )

        payload = {"input": input_data}
        output = await self._poll_runpod_job(self.runpod_qwen3_endpoint_id, payload)

        if "audio_base64" not in output:
            raise TTSServiceError("No audio returned from Qwen3-TTS")

        audio_bytes = base64.b64decode(output["audio_base64"])
        logger.info(f"Qwen3 TTS generation complete: {len(audio_bytes)} bytes")
        return audio_bytes

    # =========================================================================
    # Chatterbox Extended
    # =========================================================================

    def is_chatterbox_ext_configured(self) -> bool:
        """Check if Chatterbox Extended endpoint is configured."""
        return bool(self.runpod_api_key and self.runpod_chatterbox_ext_endpoint_id)

    async def check_chatterbox_ext_health(self) -> dict:
        """Check if Chatterbox Extended endpoint is configured."""
        if not self.is_chatterbox_ext_configured():
            return {
                "configured": False,
                "available": False,
                "error": "RUNPOD_CHATTERBOX_EXT_ENDPOINT_ID not configured",
            }
        return {
            "configured": True,
            "available": True,
            "endpoint_id": self.runpod_chatterbox_ext_endpoint_id,
        }

    async def generate_chatterbox_extended(
        self,
        text: str,
        voice_ref_path: str | None = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        num_candidates: int = 1,
        enable_denoising: bool = False,
        enable_whisper_validation: bool = False,
    ) -> bytes:
        """Generate audio using Chatterbox Extended via RunPod serverless.

        Args:
            text: Text to convert to speech.
            voice_ref_path: Optional voice reference audio for cloning.
            exaggeration: Voice expressiveness (0.0-1.0).
            cfg_weight: CFG guidance weight (0.0-1.0).
            temperature: Sampling temperature (0.0-1.0).
            num_candidates: Number of candidates to generate (best selected).
            enable_denoising: Apply denoising post-processing.
            enable_whisper_validation: Validate output with Whisper.

        Returns:
            Audio bytes.
        """
        if not self.is_chatterbox_ext_configured():
            raise TTSServiceError(
                "Chatterbox Extended not configured. "
                "Set RUNPOD_API_KEY and RUNPOD_CHATTERBOX_EXT_ENDPOINT_ID in .env"
            )

        input_data: dict = {
            "text": text,
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
            "temperature": temperature,
            "num_candidates": num_candidates,
            "enable_denoising": enable_denoising,
            "enable_whisper_validation": enable_whisper_validation,
        }

        if voice_ref_path:
            voice_path = Path(voice_ref_path)
            if not voice_path.exists():
                raise TTSServiceError(f"Voice reference file not found: {voice_ref_path}")
            with open(voice_path, "rb") as f:
                audio_bytes = f.read()
                input_data["voice_reference"] = base64.b64encode(audio_bytes).decode()
                logger.info(f"Chatterbox Ext: Using voice reference: {voice_path.name}")

        logger.info(
            f"Generating TTS via Chatterbox Extended for {len(text)} characters "
            f"(candidates={num_candidates}, denoise={enable_denoising}, "
            f"whisper={enable_whisper_validation})"
        )

        payload = {"input": input_data}
        output = await self._poll_runpod_job(
            self.runpod_chatterbox_ext_endpoint_id, payload
        )

        if "audio_base64" not in output:
            raise TTSServiceError("No audio returned from Chatterbox Extended")

        audio_bytes = base64.b64decode(output["audio_base64"])
        logger.info(f"Chatterbox Extended TTS complete: {len(audio_bytes)} bytes")
        return audio_bytes

    async def download_audio(self, audio_url: str) -> bytes:
        """Download audio from a URL.

        Args:
            audio_url: URL to download audio from

        Returns:
            Audio bytes

        Raises:
            TTSServiceError: If download fails
        """
        try:
            response = await self.client.get(audio_url)
            response.raise_for_status()
            return response.content
        except Exception as e:
            raise TTSServiceError(f"Failed to download audio: {e}")

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()
