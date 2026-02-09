"""
RunPod Serverless Handler for Chatterbox TTS Extended v2

Features beyond base Chatterbox:
- Multi-candidate generation with Whisper validation (pick best match)
- pyrnnoise denoising for artifact removal
- Long-form text chunking with automatic concatenation
- Voice conversion via ChatterboxVC

Deployment:
1. Build: docker build --platform linux/amd64 -t username/chatterbox-extended-runpod:latest .
2. Push: docker push username/chatterbox-extended-runpod:latest
3. Deploy on RunPod: https://runpod.io/console/serverless
"""

HANDLER_VERSION = "v5-debug"

import runpod
import torch
import torchaudio
import io
import os
import re
import gc
import base64
import threading
import time
import tempfile
import difflib
import string
import subprocess
import inspect

import numpy as np
import soundfile as sf
import nltk

# Download sentence tokenizer data
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import sent_tokenize

# --- pyrnnoise (optional) ---
try:
    import pyrnnoise
    _PYRNNOISE_AVAILABLE = True
except Exception:
    _PYRNNOISE_AVAILABLE = False

# --- Device selection ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load TTS model at cold start ---
print("Loading ChatterboxTTS model...")
from chatterbox.src.chatterbox.tts import ChatterboxTTS

MODEL = ChatterboxTTS.from_pretrained(DEVICE)
if hasattr(MODEL, "eval"):
    MODEL.eval()
SAMPLE_RATE = MODEL.sr
print(f"TTS model loaded on {DEVICE}, sample_rate={SAMPLE_RATE}")

# --- Whisper model (lazy-loaded on first use) ---
WHISPER_MODEL = None
WHISPER_LOCK = threading.Lock()


def get_whisper_model():
    """Lazy-load faster-whisper model on first validation request."""
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        with WHISPER_LOCK:
            if WHISPER_MODEL is None:
                print("Loading faster-whisper model (base)...")
                from faster_whisper import WhisperModel as FasterWhisperModel
                WHISPER_MODEL = FasterWhisperModel(
                    "base",
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    compute_type="float16" if torch.cuda.is_available() else "int8",
                )
                print("Whisper model loaded.")
    return WHISPER_MODEL


# ---------------------------------------------------------------------------
# Text chunking utilities
# ---------------------------------------------------------------------------

def split_long_sentence(sentence, max_len=300, seps=None):
    """Recursively split a long sentence at natural break points."""
    if seps is None:
        seps = [";", ":", " - ", ",", " "]
    if len(sentence) <= max_len:
        return [sentence]
    for sep in seps:
        parts = sentence.split(sep)
        if len(parts) > 1:
            mid = len(parts) // 2
            left = sep.join(parts[:mid]).strip()
            right = sep.join(parts[mid:]).strip()
            return split_long_sentence(left, max_len, seps) + \
                   split_long_sentence(right, max_len, seps)
    # No separator found - hard split at max_len
    return [sentence[:max_len].strip(), sentence[max_len:].strip()]


def chunk_text(text, max_chars=300):
    """Split text into sentence-based chunks of roughly max_chars each."""
    sentences = sent_tokenize(text)
    chunks = []
    current = ""
    for sent in sentences:
        if len(sent) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(split_long_sentence(sent, max_chars))
        elif len(current) + len(sent) + 1 > max_chars:
            if current:
                chunks.append(current.strip())
            current = sent
        else:
            current = (current + " " + sent).strip() if current else sent
    if current:
        chunks.append(current.strip())
    return [c for c in chunks if c]


# ---------------------------------------------------------------------------
# Denoising (pyrnnoise)
# ---------------------------------------------------------------------------

def _convert_to_pcm48k_mono(input_path, output_path):
    """Convert any audio to 48kHz mono s16 WAV via ffmpeg."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-ar", "48000", "-ac", "1",
         "-sample_fmt", "s16", output_path],
        check=True, capture_output=True,
    )


def _run_pyrnnoise(input_wav, output_wav):
    """Apply pyrnnoise denoising to a 48kHz mono WAV."""
    if not _PYRNNOISE_AVAILABLE:
        return False
    try:
        audio, sr = sf.read(input_wav, dtype="int16")
        if audio.ndim > 1:
            audio = audio[:, 0]
        denoised = pyrnnoise.denoise(audio, sample_rate=sr)
        sf.write(output_wav, denoised, sr, subtype="PCM_16")
        return True
    except Exception as e:
        print(f"[DENOISE] pyrnnoise error: {e}")
        return False


def apply_denoising(wav_path):
    """Denoise a WAV file in-place using pyrnnoise."""
    if not _PYRNNOISE_AVAILABLE:
        print("[DENOISE] pyrnnoise not available, skipping")
        return False

    try:
        import librosa
        original_sr = librosa.get_samplerate(wav_path)
    except Exception:
        original_sr = None

    tmp_48k = wav_path + ".48k.wav"
    tmp_dn = wav_path + ".dn.wav"
    tmp_back = wav_path + ".dnr.wav"

    try:
        _convert_to_pcm48k_mono(wav_path, tmp_48k)
        ok = _run_pyrnnoise(tmp_48k, tmp_dn)
        if not ok:
            return False

        if original_sr and original_sr != 48000:
            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_dn, "-ar", str(original_sr),
                 "-ac", "1", tmp_back],
                check=True, capture_output=True,
            )
            os.replace(tmp_back, wav_path)
        else:
            os.replace(tmp_dn, wav_path)

        print(f"[DENOISE] Denoised: {wav_path}")
        return True
    except Exception as e:
        print(f"[DENOISE] Failed: {e}")
        return False
    finally:
        for p in [tmp_48k, tmp_dn, tmp_back]:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Whisper validation
# ---------------------------------------------------------------------------

def normalize_for_compare(text):
    """Normalize text for comparison: lowercase, strip punctuation."""
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text


def whisper_validate(wav_path, target_text, whisper_model):
    """Transcribe wav_path and return similarity score to target_text."""
    try:
        segments, _ = whisper_model.transcribe(wav_path)
        transcribed = "".join([seg.text for seg in segments]).strip()
        score = difflib.SequenceMatcher(
            None,
            normalize_for_compare(transcribed),
            normalize_for_compare(target_text),
        ).ratio()
        return score, transcribed
    except Exception as e:
        print(f"[WHISPER] Validation error: {e}")
        return 0.0, f"ERROR: {e}"


# ---------------------------------------------------------------------------
# Single chunk generation
# ---------------------------------------------------------------------------

def generate_chunk(model, text, audio_prompt_path, exaggeration, cfg_weight,
                   temperature, seed=None):
    """Generate audio for a single text chunk. Returns waveform tensor."""
    # Determine if model.generate supports generator kwarg
    supports_generator = False
    try:
        sig = inspect.signature(model.generate)
        supports_generator = "generator" in sig.parameters
    except Exception:
        pass

    kwargs = dict(
        exaggeration=min(exaggeration, 1.0),
        temperature=temperature,
        cfg_weight=cfg_weight,
    )

    if audio_prompt_path:
        kwargs["audio_prompt_path"] = audio_prompt_path

    on_cuda = torch.cuda.is_available() and str(getattr(model, "device", "cpu")) == "cuda"
    devices = [torch.cuda.current_device()] if on_cuda else []

    if seed is not None and supports_generator and not (str(getattr(model, "device", "")) == "mps"):
        gen_device = "cuda" if on_cuda else "cpu"
        gen = torch.Generator(device=gen_device)
        gen.manual_seed(int(seed) & 0xFFFFFFFFFFFFFFFF)
        kwargs["generator"] = gen
        wav = model.generate(text, **kwargs)
    elif seed is not None:
        with torch.random.fork_rng(devices=devices, enabled=True):
            torch.manual_seed(int(seed))
            if on_cuda:
                torch.cuda.manual_seed_all(int(seed))
            wav = model.generate(text, **kwargs)
    else:
        wav = model.generate(text, **kwargs)

    return wav


# ---------------------------------------------------------------------------
# Multi-candidate generation with optional Whisper validation
# ---------------------------------------------------------------------------

def generate_best_candidate(model, text, audio_prompt_path, exaggeration,
                            cfg_weight, temperature, num_candidates,
                            enable_whisper, tmpdir):
    """Generate num_candidates audio clips, return best one (by Whisper score
    if validation enabled, otherwise the first)."""
    best_wav = None
    best_score = -1.0
    best_transcribed = None

    whisper_model = get_whisper_model() if enable_whisper else None

    for cand_idx in range(num_candidates):
        seed = int.from_bytes(os.urandom(4), "big")
        wav = generate_chunk(model, text, audio_prompt_path, exaggeration,
                             cfg_weight, temperature, seed=seed)

        if not enable_whisper or num_candidates == 1:
            return wav, 1.0, None

        # Save to temp file for whisper
        tmp_path = os.path.join(tmpdir, f"cand_{cand_idx}.wav")
        torchaudio.save(tmp_path, wav, model.sr)
        score, transcribed = whisper_validate(tmp_path, text, whisper_model)
        print(f"  Candidate {cand_idx+1}/{num_candidates}: score={score:.3f}")

        if score > best_score:
            best_score = score
            best_wav = wav
            best_transcribed = transcribed

        # Perfect match - stop early
        if score >= 0.95:
            break

    return best_wav, best_score, best_transcribed


# ---------------------------------------------------------------------------
# Main generation pipeline
# ---------------------------------------------------------------------------

def generate_tts_extended(text, audio_prompt_path, exaggeration, cfg_weight,
                          temperature, num_candidates, enable_denoising,
                          enable_whisper, tmpdir):
    """Full TTS pipeline: chunk, generate, validate, denoise, concatenate.
    Returns dict with 'wav', 'scores', or 'error'."""
    result = {"wav": None, "scores": [], "error": None, "done": False}

    try:
        model = MODEL

        # Chunk text for long-form
        chunks = chunk_text(text, max_chars=300)
        print(f"Text split into {len(chunks)} chunks")

        waveforms = []
        scores = []

        for i, chunk in enumerate(chunks):
            print(f"Generating chunk {i+1}/{len(chunks)}: "
                  f"{chunk[:60]}{'...' if len(chunk) > 60 else ''}")

            wav, score, _ = generate_best_candidate(
                model, chunk, audio_prompt_path, exaggeration, cfg_weight,
                temperature, num_candidates, enable_whisper, tmpdir,
            )
            waveforms.append(wav)
            scores.append(score)

            # Clear CUDA cache between chunks to prevent VRAM leak
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not waveforms:
            result["error"] = "No audio generated"
            return result

        # Concatenate all chunks
        full_audio = torch.cat(waveforms, dim=1)

        # Apply denoising to final concatenated audio
        if enable_denoising and _PYRNNOISE_AVAILABLE:
            concat_path = os.path.join(tmpdir, "concat.wav")
            torchaudio.save(concat_path, full_audio, model.sr)
            if apply_denoising(concat_path):
                full_audio, _ = torchaudio.load(concat_path)

        result["wav"] = full_audio
        result["scores"] = scores

    except Exception as e:
        result["error"] = str(e)
        import traceback
        traceback.print_exc()
    finally:
        result["done"] = True

    return result


# ---------------------------------------------------------------------------
# RunPod handler
# ---------------------------------------------------------------------------

def handler(job):
    """
    RunPod serverless handler for Chatterbox TTS Extended.

    Input:
        text (str): Text to convert to speech
        exaggeration (float): Voice expressiveness 0-2 (default 0.5)
        cfg_weight (float): Text adherence / pace (default 0.5)
        temperature (float): Variation (default 0.8)
        voice_reference (str, optional): Base64-encoded reference audio
        num_candidates (int): Candidates per chunk, best picked (default 1)
        enable_denoising (bool): Apply pyrnnoise denoising (default true)
        enable_whisper_validation (bool): Validate via Whisper (default false)

    Output:
        audio_base64 (str): Base64-encoded MP3 audio
        sample_rate (int): Audio sample rate
        format (str): "mp3"
        whisper_score (float, optional): Average Whisper similarity score
    """
    job_input = job["input"]
    print(f"[{HANDLER_VERSION}] Handler invoked, input keys: {list(job_input.keys())}")

    # Diagnostic ping mode - return immediately to verify handler works
    if job_input.get("ping"):
        return {"pong": True, "version": HANDLER_VERSION, "device": str(DEVICE), "sample_rate": SAMPLE_RATE}

    # Debug mode: return immediately with diagnostics
    if job_input.get("debug"):
        import sys
        return {
            "version": HANDLER_VERSION,
            "device": str(DEVICE),
            "sample_rate": SAMPLE_RATE,
            "python": sys.version,
            "runpod_version": getattr(runpod, '__version__', 'unknown'),
            "pyrnnoise": _PYRNNOISE_AVAILABLE,
        }

    text = job_input.get("text", "")
    exaggeration = float(job_input.get("exaggeration", 0.5))
    cfg_weight = float(job_input.get("cfg_weight", 0.5))
    temperature = float(job_input.get("temperature", 0.8))
    voice_reference_b64 = job_input.get("voice_reference")
    num_candidates = int(job_input.get("num_candidates", 1))
    enable_denoising = bool(job_input.get("enable_denoising", True))
    enable_whisper = bool(job_input.get("enable_whisper_validation", False))

    if not text or not text.strip():
        return {"error": "No text provided"}

    # Create temp directory for intermediate files
    with tempfile.TemporaryDirectory(prefix="chatterbox_") as tmpdir:
        # Save voice reference to temp file if provided
        audio_prompt_path = None
        if voice_reference_b64:
            try:
                voice_bytes = base64.b64decode(voice_reference_b64)
                audio_prompt_path = os.path.join(tmpdir, "voice_ref.wav")
                with open(audio_prompt_path, "wb") as f:
                    f.write(voice_bytes)
            except Exception as e:
                return {"error": f"Invalid voice_reference: {e}"}

        # Run generation in background thread to avoid blocking heartbeats
        print(f"[{HANDLER_VERSION}] Starting generation: {len(text)} chars, "
              f"{num_candidates} candidates, "
              f"denoise={enable_denoising}, whisper={enable_whisper}")

        # Use a shared mutable container to avoid race condition with nonlocal
        shared = {"result": None, "done": False}

        def _run():
            r = generate_tts_extended(
                text, audio_prompt_path, exaggeration, cfg_weight,
                temperature, num_candidates, enable_denoising,
                enable_whisper, tmpdir,
            )
            shared["result"] = r
            shared["done"] = True
            print(f"[{HANDLER_VERSION}] Generation thread finished, error={r.get('error')}")

        thread = threading.Thread(target=_run)
        thread.start()

        # Send progress updates while generation runs
        progress = 0
        start_time = time.time()
        while not shared["done"]:
            elapsed = int(time.time() - start_time)
            progress = min(progress + 3, 95)
            try:
                runpod.serverless.progress_update(job, progress)
            except Exception:
                pass
            print(f"[{HANDLER_VERSION}] Heartbeat: {elapsed}s elapsed, progress={progress}")
            time.sleep(3)

        thread.join(timeout=600)  # 10 minute max for long-form
        result = shared["result"] or {"error": "Generation thread did not produce result"}
        print(f"[{HANDLER_VERSION}] Thread joined, elapsed={int(time.time() - start_time)}s")

        if result.get("error"):
            print(f"Error: {result['error']}")
            return {"error": result["error"]}

        wav = result.get("wav")
        if wav is None:
            return {"error": "Generation failed - no audio produced"}

        # Convert to MP3
        buffer = io.BytesIO()
        torchaudio.save(buffer, wav, SAMPLE_RATE, format="mp3")
        buffer.seek(0)
        audio_base64 = base64.b64encode(buffer.read()).decode("utf-8")

        print(f"Generated {len(audio_base64)} bytes of base64 audio")

        response = {
            "audio_base64": audio_base64,
            "sample_rate": SAMPLE_RATE,
            "format": "mp3",
        }

        scores = result.get("scores", [])
        if enable_whisper and scores:
            avg_score = sum(scores) / len(scores)
            response["whisper_score"] = round(avg_score, 4)

        return response


# Start the serverless handler with explicit timeout
print(f"[{HANDLER_VERSION}] RunPod SDK version: {runpod.__version__ if hasattr(runpod, '__version__') else 'unknown'}")
print(f"[{HANDLER_VERSION}] Starting serverless handler...")
runpod.serverless.start({
    "handler": handler,
    "execution_timeout": 600,  # 10 minutes
})
