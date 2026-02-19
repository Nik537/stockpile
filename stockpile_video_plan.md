https://www.youtube.com/watch?v=jndWxpCzO5g, https://www.youtube.com/watch?v=OYg_A6J_e0g, https://www.youtube.com/watch?v=AoE_MTbQ45U,   https://www.youtube.com/watch?v=3YKELuJxs8s - if i have the script, tts, and broll finder, how could ai make a video like this autonomasly. watch the videos first and make a plan i can give to claude code. i would also maybe implement gemeni into this workflow since it understands videos and can maybe give feedback or something. - i gave gemini this prompt and it came up with this plan:


# Autonomous AI Video Production Agent: Implementation Plan

## Project Goal
Create a fully autonomous agent capable of producing partial or complete high-quality "faceless" YouTube videos (style: educational/documentary/motivational) from a simple topic prompt.

## Core Philosophy
**"The Director-Editor Loop"**: Instead of a linear script-to-video pipeline, we implement a feedback loop where Gemini acts as the "Director", reviewing the "Editor's" (script code) work and requesting changes before final render.

## Tech Stack
*   **Language**: Python 3.11+
*   **Orchestration**: `stockpile` (existing project features)
*   **LLM / Vision**: Google Gemini 3 Pro Preview
*   **Video Engine**: `moviepy` for composition, `ffmpeg` for raw processing.
*   **Audio**: Chatterbox Extended TTS
*   **broll**: stockpile
*   **Subtitles**: idk
*   **Images**: stockpile

---

## Architecture Modules

### 1. The Screenwriter (Text Generation)
*   **Input**: Topic (e.g., "The History of Coffee").
*   **Role**: Generates a structured script with visual cues.
*   **Output JSON**:
    ```json
    {
      "title": "How Coffee Took Over the World",
      "scenes": [
        {
          "id": 1,
          "duration_est": 5,
          "voiceover": "It started in Ethiopia...",
          "visual_keywords": ["ethiopian mountains", "coffee beans red", "goat herder"],
          "visual_style": "cinematic drone shot"
        }
      ]
    }
    ```

### 2. The Asset Hunter (Resource Retrieval)
*   **Role**: Fetches raw media based on visual keywords.
*   **Tools**:
    *   **Pexels Video Search**: Async IO fetching of 10-15s clips.
    *   **Fallback**: If Pexels fails, generate image using Gemini/Imagen.
*   **Logic**:
    *   Query Pexels with "ethiopian mountains".
    *   Download top 3 matches (low res for draft).
    *   Store metadata (duration, resolution).

### 3. The Voice Artist (TTS)
*   **Role**: Generates audio for each scene.
*   **Tool**: ElevenLabs (API) for premium feel, or OpenAI TTS.
*   **Output**: `.mp3` files per scene or one master audio file with timestamp metadata.

### 4. The Editor (Composition)
*   **Role**: Assembles the timeline programmatically using `moviepy`.
*   **Features**:
    *   **Ken Burns Effect**: Slowly zooming in on static images.
    *   **J-Cuts/L-Cuts**: Audio overlapping video transitions (essential for pro feel).
    *   **Kinetic Typography**: `TextClip` appearing word-by-word (using Whisper timestamps).
    *   **Transitions**: Cross-dissolves or whip-pans between clips.

### 5. The Director (Gemini Feedback Loop)
*   **This is the "Secret Sauce".**
*   **Step A**: Editor renders a low-res "Draft Cut" (480p, no effects).
*   **Step B**: Upload Draft to Gemini 1.5 Pro.
*   **Step C**: Prompt Gemini:
    > "Watch this video draft. Compare it to the script.
    > 1. visual_relevance: Does the clip at 0:15 match the narration 'economic collapse'? (It currently shows a happy dog).
    > 2. pacing: Is the cut at 0:30 too abrupt?
    > Return a JSON list of `fix_requests`."
*   **Step D**: Code parses `fix_requests`. If "irrelevant clip", the Asset Hunter searches for "stock market crash" instead of "economic collapse" (Gemini suggests better keywords) and The Editor replaces the clip.

---

## Step-by-Step Implementation Guide for Claude Code

### Phase 1: Foundation Setup
1.  **Environment**: Ensure `moviepy`, `google-genai`, `pexels-api`, `requests` are installed.
2.  **API Keys**: Set up `.env` with `GEMINI_API_KEY`, `PEXELS_API_KEY`, `ELEVENLABS_API_KEY`.
3.  **Directory Structure**:
    ```text
    /stockpile
      /video_agent
        /assets        # specific downloads
        /output        # renders
        agent.py       # main orchestrator
        editor.py      # moviepy logic
        search.py      # pexels logic
    ```

### Phase 2: The "MVP" Pipeline
1.  Implement `generate_script(topic)` using Gemini to get the JSON structure.
2.  Implement `search_media(keywords)` to download 1 video from Pexels.
3.  Implement `generate_audio(text)` to get an MP3.
4.  Implement `assemble_video()`: simplest MoviePy concatenation `concatenate_videoclips`.

### Phase 3: The "Director" Integration
1.  Add the `review_draft(video_path, script)` function using Gemini 1.5 Flash (for speed) or Pro (for quality).
2.  Create the `refine_timeline()` logic to swap out assets based on feedback ID.

### Phase 4: Polish (Kinetic Typography)
1.  Use `faster-whisper` on the generated audio to get word-level timestamps.
2.  Create a `SubtitleClip` generator in MoviePy that creates a distinct `TextClip` for each word/phrase, engaging the user (Alex Hormozi style captions).

---

## Example Prompt to Start Claude
*"I want to build a `video_agent` module within `stockpile`. Start by creating `stockpile/video_agent/script_gen.py`. It should use `google-genai` to take a topic and return a JSON object with scenes, narration, and visual search terms. Here is the JSON schema we need..."*

Use agent teams to look through the code see what we already have, search the web (reddit, x, google, linkedin, github), see what can be improved in the plan