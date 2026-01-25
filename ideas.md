# Stockpile Feature Ideas

## Latest AI Models (January 2026)

| Provider | Latest Model | Notes |
|----------|--------------|-------|
| Google | **Gemini 3 Pro/Flash** | NOT 2.5 - Gemini 3 released Nov 2025 |
| OpenAI | **GPT-5.2** | GPT-5.2-Codex for coding |
| Anthropic | **Claude Opus 4.5** | Best for agentic workflows |
| Google Image | **Nano Banana Pro** | Gemini 3 Pro Image - 4K support |
| Stability AI | **SD 3.5 Large** | No SD 4 exists yet |
| Black Forest | **FLUX.2 Pro** | Released Nov 2025, 4MP output |
| Video | **Sora 2, Runway Gen-4.5** | Sora 2 released Sept 2025 |

---

## Feature Ideas

### 1. Script Writer

#### GitHub Repos (Verified Jan 2026)
| Repo | Stars | Last Update | Why Keep |
|------|-------|-------------|----------|
| [harry0703/MoneyPrinterTurbo](https://github.com/harry0703/MoneyPrinterTurbo) | **49,154** | Dec 2025 | Market leader, GPT-4/Gemini/DeepSeek/Qwen |
| [pollinations/pollinations](https://github.com/pollinations/pollinations) | 3,831 | Jan 2026 | Most active, GPT-5/Claude/Gemini/FLUX |
| [HKUDS/ViMax](https://github.com/HKUDS/ViMax) | 2,069 | Dec 2025 | Best script match - Agentic Director/Screenwriter/Producer |
| [szczyglis-dev/py-gpt](https://github.com/szczyglis-dev/py-gpt) | 1,565 | Jan 2026 | Latest AI tech - GPT-5, Gemini 3, Claude 4.5, o4 |

---

### 2. Title and Thumbnail Maker

#### GitHub Repos (Verified Jan 2026)
| Repo | Stars | Last Update | Why Keep |
|------|-------|-------------|----------|
| [filipstrand/mflux](https://github.com/filipstrand/mflux) | 1,790 | Jan 2026 | Most active, native MLX for Apple Silicon, FLUX.2 support |
| [minimaxir/gemimg](https://github.com/minimaxir/gemimg) | 337 | Dec 2025 | Gemini 2.5 Flash/Nano Banana Pro, lightweight wrapper |
| [hoodini/nano-banana-ui](https://github.com/hoodini/nano-banana-ui) | 21 | Dec 2025 | Full Gemini image gen UI for thumbnails |

### 3. B-Roll Photos

#### GitHub Repos (Verified Jan 2026)
| Repo | Stars | Last Update | Why Keep |
|------|-------|-------------|----------|
| [CharlesPikachu/imagedl](https://github.com/CharlesPikachu/imagedl) | Active | Jan 2026 | **16 sources** including Pexels, Pixabay, Unsplash, Google, Bing |

**Dropped:** python-unsplash (20 months stale/abandoned)

---

### 4. B-Roll Videos from Other Sources than YouTube

#### GitHub Repos (Verified Jan 2026)
| Repo | Stars | Last Update | Why Keep |
|------|-------|-------------|----------|
| [iwatkot/pypexel](https://github.com/iwatkot/pypexel) | Active | Sept 2025 | Async Pexels API with httpx, Pydantic, Python 3.11/3.12, VIDEO SUPPORT |

**Dropped:** Faceless-short (misaligned - video generator, not B-roll source)

---

### 5. HeyGen-like Avatar and/or TTS

#### Avatar Generation (Verified Jan 2026)
| Repo | Stars | Last Update | Why Keep |
|------|-------|-------------|----------|
| [KwaiVGI/LivePortrait](https://github.com/KwaiVGI/LivePortrait) | **17,690** | Active | Real-time portrait animation, 12.8ms on RTX 4090 |
| [duixcom/Duix-Avatar](https://github.com/duixcom/Duix-Avatar) | **12,204** | Active | FREE HeyGen alternative (= HeyGem.ai), fully offline, 8 languages |
| [antgroup/ditto-talkinghead](https://github.com/antgroup/ditto-talkinghead) | 673 | Active | Motion-Space Diffusion + TensorRT, real-time (ACM MM 2025) |


#### TTS (Verified Jan 2026)
| Repo | Stars | Last Update | Why Keep |
|------|-------|-------------|----------|
| [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) | **54,000** | Active | GPT + Diffusion, 5-sec voice clone, RTF 0.014, MIT license |
| [fishaudio/fish-speech](https://github.com/fishaudio/fish-speech) | **24,700** | Jan 2026 | 4B Dual-AR Transformer, #1 TTS-Arena2, 13+ languages |
| [resemble-ai/chatterbox](https://github.com/resemble-ai/chatterbox) | **21,900** | Active | 0.5B Llama backbone, 23 languages, watermarking, MIT |


#### Lip Sync (Verified Jan 2026)
| Repo | Stars | Last Update | Why Keep |
|------|-------|-------------|----------|
| [TMElyralab/MuseTalk](https://github.com/TMElyralab/MuseTalk) | Active | Apr 2025 | Real-time 30+ FPS, multilingual audio, training code open |


---

### 6. The Workflow

1. Talk with AI to generate title and thumbnail combo ideas
2. Generate script
3. Generate b-roll photos and videos, generate video (can adjust speed)
4. AI inserts b-roll into video
5. User edits b-roll in video (already in timeline, just drag)
6. User exports video

#### AI Video Editing/B-Roll (Verified Jan 2026)
| Repo | Stars | Last Update | Why Keep |
|------|-------|-------------|----------|
| [HKUDS/VideoRAG](https://github.com/HKUDS/VideoRAG) | 2,555 | Jan 2026 | Claude + ImageBind, graph-driven video knowledge (KDD 2026) |
| [HKUDS/ViMax](https://github.com/HKUDS/ViMax) | 2,069 | Dec 2025 | Gemini 2.5 Flash + Veo + Nanobana - Idea2Video, Script2Video |


---

## Key Takeaways (Updated Jan 2026)

**Best Multi-Model Approach:** HKUDS projects (ViMax, VideoRAG) use Claude for routing, GPT-4o for editing, Gemini for multimodal

**Best Free HeyGen Alternative:** Duix-Avatar (= HeyGem.ai) - 12K stars, fully offline, 8 languages

**Best TTS Stack:**
1. GPT-SoVITS (54K stars) - Best voice cloning, fastest RTF
2. fish-speech (24.7K stars) - Best multilingual, most active
3. chatterbox (21.9K stars) - Best Llama-based option

**Best for Stockpile Integration:**
- pypexel for async Pexels video/photo
- imagedl for multi-source aggregation
- ViMax patterns for Gemini integration
- mflux for local FLUX image generation on Apple Silicon

---
