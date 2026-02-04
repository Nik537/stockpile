"""
RunPod Serverless Handler for Chatterbox TTS

This handler runs on RunPod's serverless infrastructure and provides
TTS generation using the Chatterbox model.

Deployment:
1. Build: docker build --platform linux/amd64 -t username/chatterbox-tts-runpod:latest .
2. Push: docker push username/chatterbox-tts-runpod:latest
3. Deploy on RunPod: https://runpod.io/console/serverless
"""

import runpod
import torch
import io
import base64
import threading
import time
from chatterbox.tts import ChatterboxTTS
import torchaudio

# Load model once at cold start (stays in memory for warm starts)
print("Loading ChatterboxTTS model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ChatterboxTTS.from_pretrained(device=device)
print(f"Model loaded on {device}!")


def generate_tts_threaded(model, text, audio_prompt, exaggeration, cfg_weight, temperature):
    """
    Run TTS generation in a separate thread to avoid blocking heartbeats.
    Returns a dict with 'wav' or 'error' key.
    """
    result = {"wav": None, "error": None, "done": False}

    def _generate():
        try:
            if audio_prompt is not None:
                result["wav"] = model.generate(
                    text,
                    audio_prompt=audio_prompt,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=temperature
                )
            else:
                result["wav"] = model.generate(
                    text,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=temperature
                )
        except Exception as e:
            result["error"] = str(e)
        finally:
            result["done"] = True

    thread = threading.Thread(target=_generate)
    thread.start()
    return thread, result


def handler(job):
    """
    RunPod serverless handler for TTS generation.

    Input:
        text (str): Text to convert to speech
        exaggeration (float): Voice expressiveness (0.0-1.0, default 0.5)
        cfg_weight (float): Text adherence (0.0-1.0, default 0.5)
        temperature (float): Variation (0.0-1.0, default 0.8)
        voice_reference (str, optional): Base64-encoded audio for voice cloning

    Output:
        audio_base64 (str): Base64-encoded MP3 audio
        sample_rate (int): Audio sample rate
        format (str): Audio format ("mp3")
    """
    job_input = job["input"]

    # Extract parameters with defaults
    text = job_input.get("text", "")
    exaggeration = float(job_input.get("exaggeration", 0.5))
    cfg_weight = float(job_input.get("cfg_weight", 0.5))
    temperature = float(job_input.get("temperature", 0.8))
    voice_reference_b64 = job_input.get("voice_reference")

    # Validate input
    if not text:
        return {"error": "No text provided"}

    if not text.strip():
        return {"error": "Text cannot be empty"}

    try:
        # Handle voice reference if provided
        audio_prompt = None
        if voice_reference_b64:
            voice_bytes = base64.b64decode(voice_reference_b64)
            voice_buffer = io.BytesIO(voice_bytes)
            audio_prompt, _ = torchaudio.load(voice_buffer)

        # Generate audio in background thread to avoid blocking heartbeats
        print(f"Generating TTS for text: {text[:100]}...")
        thread, result = generate_tts_threaded(
            model, text, audio_prompt, exaggeration, cfg_weight, temperature
        )

        # Send progress updates while generation runs (keeps heartbeat alive)
        progress = 0
        while not result["done"]:
            progress = min(progress + 5, 95)
            try:
                runpod.serverless.progress_update(job, progress)
            except Exception:
                pass  # Progress update is optional, don't fail if it errors
            time.sleep(3)  # Update every 3 seconds

        # Wait for thread to fully complete
        thread.join(timeout=300)  # 5 minute max wait

        if result["error"]:
            print(f"Error generating TTS: {result['error']}")
            return {"error": result["error"]}

        wav = result["wav"]
        if wav is None:
            return {"error": "Generation failed - no audio produced"}

        # Convert to MP3 bytes
        buffer = io.BytesIO()
        torchaudio.save(buffer, wav, model.sr, format="mp3")
        buffer.seek(0)

        # Return base64 encoded audio
        audio_base64 = base64.b64encode(buffer.read()).decode("utf-8")

        print(f"Generated {len(audio_base64)} bytes of audio")

        return {
            "audio_base64": audio_base64,
            "sample_rate": model.sr,
            "format": "mp3"
        }

    except Exception as e:
        print(f"Error generating TTS: {e}")
        return {"error": str(e)}


# Start the serverless handler
runpod.serverless.start({"handler": handler})
