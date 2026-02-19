"""
RunPod Serverless Handler for MOSS-TTSD (Multi-Speaker Dialogue TTS)

8B parameter model supporting up to 5 speakers, 20 languages,
and zero-shot voice cloning. Generates natural conversational audio
from dialogue scripts with [S1]-[S5] speaker tags.

Deployment:
1. Build: docker build --platform linux/amd64 -t username/moss-ttsd-runpod:latest .
2. Push: docker push username/moss-ttsd-runpod:latest
3. Deploy on RunPod: https://runpod.io/console/serverless
   - GPU: A5000 (24GB) minimum, A100 40GB+ recommended
"""

import runpod
import torch
import io
import base64
import threading
import time
import tempfile
import os

import numpy as np
import soundfile as sf

# --- Device selection ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Lazy model loading ---
# Models are loaded on first request instead of at startup. This allows
# the RunPod handler to register and send heartbeats while the (potentially
# large) model download and loading happens during the first job.
PROCESSOR = None
MODEL = None
_model_lock = threading.Lock()


def _ensure_model_loaded():
    """Load MOSS-TTSD model lazily on first request."""
    global PROCESSOR, MODEL
    if MODEL is not None:
        return

    with _model_lock:
        if MODEL is not None:
            return

        print("Loading MOSS-TTSD model and audio tokenizer...")

        # Pre-check critical imports before transformers dynamic loading
        try:
            import soundfile as _sf
            print(f"  soundfile {_sf.__version__} OK")
        except Exception as e:
            print(f"  soundfile FAILED: {e}")
        try:
            import torchaudio as _ta
            print(f"  torchaudio {_ta.__version__} OK")
        except Exception as e:
            print(f"  torchaudio FAILED: {e}")
        try:
            import transformers as _tfm
            print(f"  transformers {_tfm.__version__} OK")
        except Exception as e:
            print(f"  transformers FAILED: {e}")

        from transformers import AutoModel, AutoProcessor

        # Try FlashAttention2 first, fall back to SDPA
        attn_impl = "sdpa"
        if torch.cuda.is_available():
            try:
                import importlib
                importlib.import_module("flash_attn")
                attn_impl = "flash_attention_2"
                print("Using FlashAttention2")
            except ImportError:
                print("FlashAttention2 not available, using SDPA attention")

        try:
            PROCESSOR = AutoProcessor.from_pretrained(
                "OpenMOSS-Team/MOSS-TTSD-v1.0",
                trust_remote_code=True,
                codec_path="OpenMOSS-Team/MOSS-Audio-Tokenizer",
            )
            print("  Processor loaded OK")
        except Exception as proc_err:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Processor load failed: {proc_err}") from proc_err

        try:
            MODEL = AutoModel.from_pretrained(
                "OpenMOSS-Team/MOSS-TTSD-v1.0",
                trust_remote_code=True,
                attn_implementation=attn_impl,
                torch_dtype=torch.bfloat16,
            ).to(DEVICE)
            print("  Model loaded OK")
        except Exception as model_err:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Model load failed: {model_err}") from model_err

        print(f"MOSS-TTSD model loaded on {DEVICE}!")


def generate_audio_threaded(text, voice_references, language, temperature,
                            max_tokens, inference_mode, num_speakers, tmpdir):
    """
    Run MOSS-TTSD generation in a separate thread to avoid blocking heartbeats.

    Args:
        text: Dialogue text with [S1]-[S5] speaker tags.
        voice_references: Dict mapping speaker tags to base64 audio, or None.
        language: Language code.
        temperature: Sampling temperature.
        max_tokens: Max generation tokens.
        inference_mode: generation/voice_clone/continuation/voice_clone_and_continuation
        num_speakers: Number of speakers (1-5).
        tmpdir: Temp directory for intermediate files.
    """
    result = {"audio": None, "sr": None, "error": None, "done": False}

    def _generate():
        try:
            _ensure_model_loaded()

            # Prepare voice reference audio codes if provided
            reference_audio_codes = None
            prompt_audio_codes = None

            if voice_references and inference_mode in ("voice_clone", "voice_clone_and_continuation"):
                # Decode and encode all voice references
                all_ref_codes = []
                for speaker_tag, audio_b64 in voice_references.items():
                    audio_bytes = base64.b64decode(audio_b64)
                    ref_path = os.path.join(tmpdir, f"ref_{speaker_tag}.wav")
                    with open(ref_path, "wb") as f:
                        f.write(audio_bytes)
                    codes = PROCESSOR.encode_audio(ref_path)
                    all_ref_codes.append(codes)
                    print(f"Encoded voice reference for {speaker_tag}")

                # Combine references
                if all_ref_codes:
                    reference_audio_codes = all_ref_codes[0]
                    # For continuation modes, use the first reference as prompt too
                    if inference_mode == "voice_clone_and_continuation":
                        prompt_audio_codes = all_ref_codes[0]

            elif voice_references and inference_mode == "continuation":
                # Use first reference as continuation prompt
                for speaker_tag, audio_b64 in voice_references.items():
                    audio_bytes = base64.b64decode(audio_b64)
                    ref_path = os.path.join(tmpdir, f"prompt_{speaker_tag}.wav")
                    with open(ref_path, "wb") as f:
                        f.write(audio_bytes)
                    prompt_audio_codes = PROCESSOR.encode_audio(ref_path)
                    print(f"Encoded prompt audio for {speaker_tag}")
                    break  # Only use first reference as prompt

            # Build conversation messages
            user_msg_kwargs = {"text": text}
            if reference_audio_codes is not None:
                user_msg_kwargs["reference"] = reference_audio_codes

            conversations = [[
                PROCESSOR.build_user_message(**user_msg_kwargs),
            ]]

            # Add assistant message for continuation modes
            if prompt_audio_codes is not None:
                conversations[0].append(
                    PROCESSOR.build_assistant_message(audio_codes_list=[prompt_audio_codes])
                )

            # Determine processor mode
            proc_mode = "continuation" if prompt_audio_codes is not None else "text_to_audio"

            # Process and generate
            batch = PROCESSOR(conversations, mode=proc_mode)
            outputs = MODEL.generate(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
            )

            # Decode outputs to audio
            messages = PROCESSOR.decode(outputs)

            # Extract audio from decoded messages
            # The decode method returns messages with audio data
            audio_data = None
            sample_rate = 32000  # MOSS-TTSD native sample rate

            if messages and len(messages) > 0:
                for msg_group in messages:
                    if isinstance(msg_group, list):
                        for msg in msg_group:
                            if hasattr(msg, "audio") and msg.audio is not None:
                                audio_data = msg.audio
                                break
                            if isinstance(msg, dict) and "audio" in msg:
                                audio_data = msg["audio"]
                                break
                    elif hasattr(msg_group, "audio") and msg_group.audio is not None:
                        audio_data = msg_group.audio
                        break

            # Fallback: try to get audio directly from the output
            if audio_data is None:
                # Some versions return audio directly
                for msg_group in messages:
                    if isinstance(msg_group, (np.ndarray, torch.Tensor)):
                        audio_data = msg_group
                        break
                    if isinstance(msg_group, list):
                        for item in msg_group:
                            if isinstance(item, (np.ndarray, torch.Tensor)):
                                audio_data = item
                                break
                            if isinstance(item, dict):
                                for v in item.values():
                                    if isinstance(v, (np.ndarray, torch.Tensor)):
                                        audio_data = v
                                        break

            if audio_data is None:
                # Last resort: save all outputs and check
                output_path = os.path.join(tmpdir, "output.wav")
                try:
                    PROCESSOR.save_audio(messages, output_path)
                    audio_data, sample_rate = sf.read(output_path)
                except Exception as save_err:
                    result["error"] = f"Could not extract audio from model output: {save_err}"
                    return

            # Convert to numpy if tensor
            if isinstance(audio_data, torch.Tensor):
                audio_data = audio_data.cpu().numpy()

            if isinstance(audio_data, np.ndarray) and audio_data.size > 0:
                result["audio"] = audio_data
                result["sr"] = sample_rate
            else:
                result["error"] = "Generation produced empty audio"

        except Exception as e:
            import traceback
            traceback.print_exc()
            result["error"] = str(e)
        finally:
            result["done"] = True

    thread = threading.Thread(target=_generate)
    thread.start()
    return thread, result


def handler(job):
    """
    RunPod serverless handler for MOSS-TTSD.

    Input:
        text (str): Dialogue text with [S1]-[S5] speaker tags (required)
        language (str): Language code (default "en")
        temperature (float): Sampling temperature (default 0.9)
        max_tokens (int): Max generation tokens (default 2000)
        inference_mode (str): generation/voice_clone/continuation/voice_clone_and_continuation
        num_speakers (int): Number of speakers 1-5 (default 1)
        voice_references (dict, optional): {"S1": "<base64>", "S2": "<base64>", ...}

    Output:
        audio_base64 (str): Base64-encoded WAV audio
        sample_rate (int): Audio sample rate (32000)
        format (str): "wav"
    """
    job_input = job["input"]

    # Diagnostic ping (don't load model for ping - just report status)
    if job_input.get("ping"):
        return {"pong": True, "device": str(DEVICE), "model_loaded": MODEL is not None}

    text = job_input.get("text", "")
    language = job_input.get("language", "en")
    temperature = float(job_input.get("temperature", 0.9))
    max_tokens = int(job_input.get("max_tokens", 2000))
    inference_mode = job_input.get("inference_mode", "generation")
    num_speakers = int(job_input.get("num_speakers", 1))
    voice_references = job_input.get("voice_references")

    if not text or not text.strip():
        return {"error": "No text provided"}

    valid_modes = ("generation", "voice_clone", "continuation", "voice_clone_and_continuation")
    if inference_mode not in valid_modes:
        return {"error": f"Invalid inference_mode. Use one of: {', '.join(valid_modes)}"}

    if num_speakers < 1 or num_speakers > 5:
        return {"error": "num_speakers must be between 1 and 5"}

    with tempfile.TemporaryDirectory(prefix="moss_ttsd_") as tmpdir:
        print(f"Starting MOSS-TTSD generation: {len(text)} chars, "
              f"mode={inference_mode}, speakers={num_speakers}, lang={language}")

        thread, result = generate_audio_threaded(
            text=text.strip(),
            voice_references=voice_references,
            language=language,
            temperature=temperature,
            max_tokens=max_tokens,
            inference_mode=inference_mode,
            num_speakers=num_speakers,
            tmpdir=tmpdir,
        )

        # Send progress updates while generation runs
        start_time = time.time()
        progress = 5
        while not result["done"]:
            elapsed = int(time.time() - start_time)
            progress = min(progress + 2, 95)
            try:
                runpod.serverless.progress_update(job, progress)
            except Exception:
                pass
            time.sleep(3)

        thread.join(timeout=600)

        if result["error"]:
            print(f"Error: {result['error']}")
            return {"error": result["error"]}

        audio = result["audio"]
        sr = result["sr"] or 32000

        if audio is None:
            return {"error": "Generation failed - no audio produced"}

        # Ensure float32 for soundfile
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Normalize if needed
        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            audio = audio / max_val

        # Write to WAV buffer
        buffer = io.BytesIO()
        sf.write(buffer, audio, sr, format="WAV")
        buffer.seek(0)

        audio_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        print(f"Generated {len(audio_base64)} bytes of base64 audio (sr={sr})")

        return {
            "audio_base64": audio_base64,
            "sample_rate": sr,
            "format": "wav",
        }


# Start the serverless handler
runpod.serverless.start({
    "handler": handler,
    "execution_timeout": 600,
})
