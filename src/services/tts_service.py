"""TTS Service - HTTP client for TTS generation via RunPod or Colab server."""

import base64
import json
import logging
import os
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

# Settings file for URL persistence
SETTINGS_FILE = Path.home() / ".stockpile" / "tts_settings.json"

# RunPod configuration from environment
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID", "")

# RunPod public TTS endpoint
RUNPOD_PUBLIC_TTS_ENDPOINT = "chatterbox-turbo"
RUNPOD_API_BASE = "https://api.runpod.ai/v2"


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
            Audio bytes (MP3 format)

        Raises:
            TTSServiceError: If generation fails
        """
        if not self.is_runpod_configured():
            raise TTSServiceError(
                "RunPod not configured. Set RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID in .env"
            )

        url = f"https://api.runpod.ai/v2/{self.runpod_endpoint_id}/runsync"
        headers = {
            "Authorization": f"Bearer {self.runpod_api_key}",
            "Content-Type": "application/json",
        }

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

        logger.info(
            f"Generating TTS via RunPod for {len(text)} characters "
            f"(exag={exaggeration}, cfg={cfg_weight}, temp={temperature})"
        )

        try:
            response = await self.client.post(url, headers=headers, json=payload)
            response.raise_for_status()

            result = response.json()

            # Check for RunPod execution errors
            if result.get("status") == "FAILED":
                error_msg = result.get("error", "Unknown error")
                raise TTSServiceError(f"RunPod execution failed: {error_msg}")

            # Extract output
            output = result.get("output", {})

            if "error" in output:
                raise TTSServiceError(f"TTS generation error: {output['error']}")

            if "audio_base64" not in output:
                raise TTSServiceError("No audio returned from RunPod")

            # Decode base64 audio
            audio_bytes = base64.b64decode(output["audio_base64"])
            logger.info(f"RunPod TTS generation complete: {len(audio_bytes)} bytes")
            return audio_bytes

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
        voice: str = "Lucy",
        output_format: str = "wav",
    ) -> tuple[str, float]:
        """Generate audio using RunPod's public Chatterbox Turbo endpoint.

        This uses the pre-deployed public endpoint - no custom Docker image needed.
        Note: Does NOT support voice cloning (use generate_runpod for that).

        Args:
            text: Text to convert to speech
            voice: Voice to use (default: "Lucy")
            output_format: Output format (wav or mp3)

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

        # Public endpoint uses 'prompt' not 'text'
        payload = {
            "input": {
                "prompt": text,
                "voice": voice,
                "format": output_format,
            }
        }

        logger.info(
            f"Generating TTS via public endpoint for {len(text)} characters "
            f"(voice={voice})"
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
