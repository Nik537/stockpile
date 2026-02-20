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
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# --- Lazy model loading ---
# Models are loaded on first request instead of at startup. This allows
# the RunPod handler to register and send heartbeats while the (potentially
# large) model download and loading happens during the first job.
PROCESSOR = None
MODEL = None
_model_lock = threading.Lock()
_model_load_error = None  # Stores load error for returning via job result


def _ensure_model_loaded():
    """Load MOSS-TTSD model lazily on first request. Returns error string or None."""
    global PROCESSOR, MODEL, _model_load_error
    if MODEL is not None:
        return None
    if _model_load_error is not None:
        return _model_load_error

    with _model_lock:
        if MODEL is not None:
            return None
        if _model_load_error is not None:
            return _model_load_error

        print("Loading MOSS-TTSD model and audio tokenizer...")
        import traceback

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
            # NOTE: audio_tokenizer stays on CPU to save VRAM (model needs ~16GB in bf16)
            print(f"  Processor loaded OK (sampling_rate={PROCESSOR.model_config.sampling_rate})")
        except Exception:
            _model_load_error = f"Processor load failed:\n{traceback.format_exc()}"
            print(_model_load_error)
            return _model_load_error

        try:
            MODEL = AutoModel.from_pretrained(
                "OpenMOSS-Team/MOSS-TTSD-v1.0",
                trust_remote_code=True,
                attn_implementation=attn_impl,
                torch_dtype=DTYPE,
            ).to(DEVICE)
            MODEL.eval()
            print("  Model loaded OK")
        except Exception:
            _model_load_error = f"Model load failed:\n{traceback.format_exc()}"
            print(_model_load_error)
            return _model_load_error

        print(f"MOSS-TTSD model loaded on {DEVICE}!")
        return None


def generate_audio_threaded(text, voice_references, language, temperature,
                            max_tokens, inference_mode, num_speakers, tmpdir):
    """
    Run MOSS-TTSD generation in a separate thread to avoid blocking heartbeats.
    """
    result = {"audio": None, "sr": None, "error": None, "done": False}

    def _generate():
        try:
            import torchaudio

            load_err = _ensure_model_loaded()
            if load_err:
                result["error"] = load_err
                return

            sample_rate = int(PROCESSOR.model_config.sampling_rate)

            # Prepare voice reference audio codes if provided
            reference_audio_codes = None
            prompt_audio_codes = None

            if voice_references and inference_mode in ("voice_clone", "voice_clone_and_continuation"):
                # Decode and encode all voice references using correct API
                ref_wavs = []
                for speaker_tag, audio_b64 in voice_references.items():
                    audio_bytes = base64.b64decode(audio_b64)
                    ref_path = os.path.join(tmpdir, f"ref_{speaker_tag}.wav")
                    with open(ref_path, "wb") as f:
                        f.write(audio_bytes)
                    wav, sr = torchaudio.load(ref_path)
                    # Convert to mono
                    if wav.shape[0] > 1:
                        wav = wav.mean(dim=0, keepdim=True)
                    # Resample to model's sampling rate
                    if sr != sample_rate:
                        wav = torchaudio.functional.resample(wav, sr, sample_rate)
                    ref_wavs.append(wav)
                    print(f"Loaded voice reference for {speaker_tag}: {wav.shape}")

                if ref_wavs:
                    reference_audio_codes = PROCESSOR.encode_audios_from_wav(
                        ref_wavs, sampling_rate=sample_rate
                    )
                    print(f"Encoded {len(ref_wavs)} voice references")

                    # For continuation modes, concatenate all refs as prompt
                    if inference_mode == "voice_clone_and_continuation":
                        concat_wav = torch.cat(ref_wavs, dim=-1)
                        prompt_audio_codes = PROCESSOR.encode_audios_from_wav(
                            [concat_wav], sampling_rate=sample_rate
                        )[0]

            elif voice_references and inference_mode == "continuation":
                # Use first reference as continuation prompt
                for speaker_tag, audio_b64 in voice_references.items():
                    audio_bytes = base64.b64decode(audio_b64)
                    ref_path = os.path.join(tmpdir, f"prompt_{speaker_tag}.wav")
                    with open(ref_path, "wb") as f:
                        f.write(audio_bytes)
                    wav, sr = torchaudio.load(ref_path)
                    if wav.shape[0] > 1:
                        wav = wav.mean(dim=0, keepdim=True)
                    if sr != sample_rate:
                        wav = torchaudio.functional.resample(wav, sr, sample_rate)
                    prompt_audio_codes = PROCESSOR.encode_audios_from_wav(
                        [wav], sampling_rate=sample_rate
                    )[0]
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

            # Determine processor mode (only "generation" and "continuation" are valid)
            proc_mode = "continuation" if prompt_audio_codes is not None else "generation"

            # Process and generate
            batch = PROCESSOR(conversations, mode=proc_mode)
            print(f"Batch prepared (mode={proc_mode}), generating...")

            with torch.no_grad():
                outputs = MODEL.generate(
                    input_ids=batch["input_ids"].to(DEVICE),
                    attention_mask=batch["attention_mask"].to(DEVICE),
                    max_new_tokens=max_tokens,
                    audio_temperature=temperature,
                    audio_top_p=0.9,
                    audio_top_k=50,
                    audio_repetition_penalty=1.1,
                )

            # Handle list or tensor outputs
            if isinstance(outputs, list):
                print(f"Generation complete (list of {len(outputs)} items), decoding...")
                outputs_cpu = outputs  # Already CPU-friendly
            else:
                print(f"Generation complete (shape: {outputs.shape}), decoding...")
                outputs_cpu = outputs.cpu()

            # Decode outputs to audio
            audio_data = None
            try:
                decoded_messages = list(PROCESSOR.decode(outputs_cpu))
            except Exception as decode_err:
                import traceback as tb
                result["error"] = f"Decode failed: {decode_err}\n{tb.format_exc()}"
                return
            print(f"Decoded {len(decoded_messages)} message(s)")

            for idx, message in enumerate(decoded_messages):
                print(f"  msg[{idx}]: type={type(message).__name__}")

                # Extract audio tensor from message
                audio_tensor = None
                if hasattr(message, "audio_codes_list") and message.audio_codes_list:
                    audio_tensor = message.audio_codes_list[0]
                    print(f"  audio_codes_list[0]: shape={audio_tensor.shape}, dtype={audio_tensor.dtype}")
                elif hasattr(message, "audio") and message.audio is not None:
                    audio_tensor = message.audio
                    print(f"  .audio: type={type(audio_tensor).__name__}")

                if audio_tensor is not None and isinstance(audio_tensor, torch.Tensor):
                    # Convert tensor to numpy for soundfile
                    wav = audio_tensor.cpu().float().numpy()
                    if wav.ndim == 2:
                        wav = wav.squeeze(0)  # Remove batch dim if present
                    audio_data = wav
                    print(f"  Audio extracted: {audio_data.shape}, sr={sample_rate}")
                    break

            if audio_data is not None and isinstance(audio_data, np.ndarray) and audio_data.size > 0:
                result["audio"] = audio_data
                result["sr"] = sample_rate
            else:
                debug = f"No audio in {len(decoded_messages)} messages. "
                for idx, m in enumerate(decoded_messages):
                    debug += f"msg[{idx}]: type={type(m).__name__}, "
                    if hasattr(m, "audio_codes_list"):
                        debug += f"acl={bool(m.audio_codes_list)}, "
                result["error"] = debug[:500]

        except Exception as e:
            import traceback
            tb_str = traceback.format_exc()
            print(tb_str)
            msg = str(e) or repr(e) or f"Unknown error: {type(e).__name__}"
            result["error"] = f"{msg}\n\nTraceback:\n{tb_str}"[:1500]
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
        sample_rate (int): Audio sample rate
        format (str): "wav"
    """
    job_input = job["input"]

    # Diagnostic ping (don't load model for ping - just report status)
    if job_input.get("ping"):
        return {"pong": True, "device": str(DEVICE), "model_loaded": MODEL is not None}

    # Deep diagnostic - test each import/load step
    if job_input.get("diagnose"):
        import traceback
        results = {"device": str(DEVICE), "steps": []}
        try:
            import transformers
            results["steps"].append(f"transformers={transformers.__version__} OK")
        except Exception as e:
            results["steps"].append(f"transformers FAIL: {e}")
            return results
        try:
            results["steps"].append(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
        except Exception as e:
            results["steps"].append(f"torch FAIL: {e}")
        try:
            import torchaudio
            results["steps"].append(f"torchaudio={torchaudio.__version__} OK")
        except Exception as e:
            results["steps"].append(f"torchaudio FAIL: {traceback.format_exc()}")
        try:
            import soundfile
            results["steps"].append(f"soundfile={soundfile.__version__} OK")
        except Exception as e:
            results["steps"].append(f"soundfile FAIL: {traceback.format_exc()}")
        try:
            from transformers import AutoProcessor
            proc = AutoProcessor.from_pretrained(
                "OpenMOSS-Team/MOSS-TTSD-v1.0",
                trust_remote_code=True,
                codec_path="OpenMOSS-Team/MOSS-Audio-Tokenizer",
            )
            results["steps"].append(f"AutoProcessor loaded: {type(proc).__name__}")
        except Exception as e:
            results["steps"].append(f"AutoProcessor FAIL: {traceback.format_exc()}")
        return results

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

        if result["error"] is not None:
            print(f"Error: {result['error']}")
            return {"error": result["error"] or "Unknown thread error"}

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
