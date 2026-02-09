"""
RunPod Serverless Handler for Qwen3-TTS

This handler runs on RunPod's serverless infrastructure and provides
TTS generation using the Qwen3-TTS model with voice cloning and custom voice support.

Deployment:
1. Build: docker build --platform linux/amd64 -t username/qwen3-tts-runpod:latest .
2. Push: docker push username/qwen3-tts-runpod:latest
3. Deploy on RunPod: https://runpod.io/console/serverless
"""

import runpod
import torch
import numpy as np
import io
import base64
import threading
import time
import soundfile as sf

from qwen_tts import Qwen3TTSModel

# Load model once at cold start (stays in memory for warm starts)
print("Loading Qwen3-TTS model...")
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Try FlashAttention2 first (faster, less VRAM), fall back to eager if not installed
attn_impl = "eager"
if torch.cuda.is_available():
    try:
        import importlib
        importlib.import_module("flash_attn")
        attn_impl = "flash_attention_2"
        print("Using FlashAttention2")
    except ImportError:
        print("FlashAttention2 not available, using eager attention")

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map=device,
    dtype=torch.bfloat16,
    attn_implementation=attn_impl,
)
print(f"Qwen3-TTS model loaded on {device}!")

# Cache: sample rate from model (discovered on first generation)
_cached_sr = None

# Maximum characters per chunk for long-form generation.
# Prevents quality degradation on very long inputs.
MAX_CHARS_PER_CHUNK = 500

# Map ISO 639-1 codes to full language names expected by Qwen3-TTS
LANGUAGE_CODE_MAP = {
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "auto": "auto",
}


def split_text_into_chunks(text, max_chars=MAX_CHARS_PER_CHUNK):
    """
    Split long text into paragraph-aware chunks for stable long-form generation.
    Splits on paragraph breaks first, then sentence boundaries if paragraphs are too long.
    """
    # Split on double newlines (paragraphs)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    for para in paragraphs:
        if len(para) <= max_chars:
            chunks.append(para)
        else:
            # Split long paragraphs on sentence boundaries
            sentences = []
            current = ""
            for char in para:
                current += char
                if char in ".!?" and len(current) > 20:
                    sentences.append(current.strip())
                    current = ""
            if current.strip():
                sentences.append(current.strip())

            # Group sentences into chunks under max_chars
            current_chunk = ""
            for sentence in sentences:
                if current_chunk and len(current_chunk) + len(sentence) + 1 > max_chars:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    current_chunk = (current_chunk + " " + sentence).strip()
            if current_chunk.strip():
                chunks.append(current_chunk.strip())

    return chunks if chunks else [text]


def generate_voice_clone_threaded(text_chunks, language, ref_audio, ref_text, temperature, top_p):
    """
    Run voice clone generation in a separate thread to avoid blocking heartbeats.
    Generates each chunk separately and concatenates for long-form stability.
    """
    result = {"wavs": None, "sr": None, "error": None, "done": False, "progress_chunks": 0, "total_chunks": len(text_chunks)}

    def _generate():
        global _cached_sr
        try:
            # Build reusable voice clone prompt once
            voice_clone_prompt = model.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text,
            )

            all_wavs = []
            for i, chunk in enumerate(text_chunks):
                wavs, sr = model.generate_voice_clone(
                    text=chunk,
                    language=language,
                    voice_clone_prompt=voice_clone_prompt,
                    temperature=temperature,
                    top_p=top_p,
                )
                all_wavs.append(wavs[0])
                _cached_sr = sr
                result["progress_chunks"] = i + 1

            # Concatenate all chunks
            result["wavs"] = np.concatenate(all_wavs) if len(all_wavs) > 1 else all_wavs[0]
            result["sr"] = _cached_sr
        except Exception as e:
            result["error"] = str(e)
        finally:
            result["done"] = True

    thread = threading.Thread(target=_generate)
    thread.start()
    return thread, result


def generate_custom_voice_threaded(text_chunks, language, speaker, instruct, temperature, top_p):
    """
    Run custom voice generation in a separate thread to avoid blocking heartbeats.
    Generates each chunk separately and concatenates for long-form stability.
    """
    result = {"wavs": None, "sr": None, "error": None, "done": False, "progress_chunks": 0, "total_chunks": len(text_chunks)}

    def _generate():
        global _cached_sr
        try:
            all_wavs = []
            for i, chunk in enumerate(text_chunks):
                wavs, sr = model.generate_custom_voice(
                    text=chunk,
                    language=language,
                    speaker=speaker,
                    instruct=instruct or "",
                    temperature=temperature,
                    top_p=top_p,
                )
                all_wavs.append(wavs[0])
                _cached_sr = sr
                result["progress_chunks"] = i + 1

            # Concatenate all chunks
            result["wavs"] = np.concatenate(all_wavs) if len(all_wavs) > 1 else all_wavs[0]
            result["sr"] = _cached_sr
        except Exception as e:
            result["error"] = str(e)
        finally:
            result["done"] = True

    thread = threading.Thread(target=_generate)
    thread.start()
    return thread, result


def handler(job):
    """
    RunPod serverless handler for Qwen3-TTS generation.

    Input:
        text (str): Text to convert to speech (required)
        language (str): Language code - "English", "Chinese", etc. (default "English")
        voice_reference (str, optional): Base64-encoded reference audio for voice cloning
        voice_reference_transcript (str, optional): Transcript of the reference audio
        speaker_name (str, optional): Preset speaker name for custom voice mode
            Supported: Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee
        instruction (str, optional): Voice instruction for custom voice mode
        temperature (float): Sampling temperature (default 0.7)
        top_p (float): Nucleus sampling parameter (default 0.95)

    Output:
        audio_base64 (str): Base64-encoded MP3 audio
        sample_rate (int): Audio sample rate
        format (str): Audio format ("mp3")

    Modes:
        1. Voice Cloning: Provide voice_reference + voice_reference_transcript
        2. Custom Voice: Provide speaker_name (+ optional instruction)
        3. Default: Uses "Vivian" speaker with no instruction
    """
    job_input = job["input"]

    # Extract parameters with defaults
    text = job_input.get("text", "")
    raw_language = job_input.get("language", "English")
    # Accept both ISO codes ("en") and full names ("English")
    language = LANGUAGE_CODE_MAP.get(raw_language, raw_language)
    voice_reference_b64 = job_input.get("voice_reference")
    voice_reference_transcript = job_input.get("voice_reference_transcript")
    speaker_name = job_input.get("speaker_name")
    instruction = job_input.get("instruction", "")
    temperature = float(job_input.get("temperature", 0.7))
    top_p = float(job_input.get("top_p", 0.95))

    # Validate input
    if not text:
        return {"error": "No text provided"}

    if not text.strip():
        return {"error": "Text cannot be empty"}

    # Split long text into chunks for stable long-form generation
    text_chunks = split_text_into_chunks(text.strip())
    print(f"Split text into {len(text_chunks)} chunk(s) for generation")

    try:
        # Determine generation mode
        if voice_reference_b64:
            # Voice Cloning mode
            if not voice_reference_transcript:
                return {"error": "voice_reference_transcript is required when using voice_reference for cloning"}

            # Decode reference audio - pass as base64 string directly (Qwen3-TTS supports it)
            # Or decode to numpy array for reliability
            voice_bytes = base64.b64decode(voice_reference_b64)
            voice_buffer = io.BytesIO(voice_bytes)
            ref_audio_data, ref_sr = sf.read(voice_buffer)
            ref_audio = (ref_audio_data, ref_sr)

            print(f"Voice cloning mode: ref audio {len(ref_audio_data)} samples at {ref_sr}Hz")
            print(f"Generating TTS for text ({len(text)} chars): {text[:100]}...")

            thread, result = generate_voice_clone_threaded(
                text_chunks, language, ref_audio, voice_reference_transcript,
                temperature, top_p
            )
        else:
            # Custom Voice mode
            speaker = speaker_name or "Vivian"
            print(f"Custom voice mode: speaker={speaker}, instruction={instruction[:50] if instruction else 'none'}")
            print(f"Generating TTS for text ({len(text)} chars): {text[:100]}...")

            thread, result = generate_custom_voice_threaded(
                text_chunks, language, speaker, instruction,
                temperature, top_p
            )

        # Send progress updates while generation runs (keeps heartbeat alive)
        while not result["done"]:
            chunks_done = result["progress_chunks"]
            total_chunks = result["total_chunks"]
            if total_chunks > 1:
                progress = int((chunks_done / total_chunks) * 90) + 5
            else:
                progress = min(50, 95)  # Simple progress for single chunk
            try:
                runpod.serverless.progress_update(job, min(progress, 95))
            except Exception:
                pass  # Progress update is optional, don't fail if it errors
            time.sleep(3)  # Update every 3 seconds

        # Wait for thread to fully complete
        thread.join(timeout=600)  # 10 minute max wait for long-form

        if result["error"]:
            print(f"Error generating TTS: {result['error']}")
            return {"error": result["error"]}

        wav = result["wavs"]
        sr = result["sr"]
        if wav is None:
            return {"error": "Generation failed - no audio produced"}

        # Convert numpy array to MP3 bytes via WAV buffer then ffmpeg
        # soundfile writes WAV, then we use torchaudio or pydub for MP3
        # Simplest approach: write WAV to buffer, return as MP3 via torchaudio
        import torchaudio

        wav_tensor = torch.from_numpy(wav).unsqueeze(0).float()
        buffer = io.BytesIO()
        torchaudio.save(buffer, wav_tensor, sr, format="mp3")
        buffer.seek(0)

        # Return base64 encoded audio
        audio_base64 = base64.b64encode(buffer.read()).decode("utf-8")

        print(f"Generated {len(audio_base64)} bytes of base64 audio ({len(text_chunks)} chunks, sr={sr})")

        return {
            "audio_base64": audio_base64,
            "sample_rate": sr,
            "format": "mp3"
        }

    except Exception as e:
        print(f"Error generating TTS: {e}")
        return {"error": str(e)}


# Start the serverless handler
runpod.serverless.start({"handler": handler})
