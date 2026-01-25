import { NextResponse } from "next/server";
import type { TTSConfig, GeneratedAudio, TTSGenerationRequest } from "@/types/workflow";

// Generate mock waveform data
function generateWaveformData(duration: number): number[] {
  const samplesPerSecond = 10;
  const totalSamples = Math.floor(duration * samplesPerSecond);
  const data: number[] = [];

  for (let i = 0; i < totalSamples; i++) {
    const base = 0.3 + Math.random() * 0.4;
    const variation = Math.sin(i * 0.15) * 0.2;
    const noise = (Math.random() - 0.5) * 0.15;
    data.push(Math.max(0.1, Math.min(1, base + variation + noise)));
  }

  return data;
}

// Estimate duration based on text length and speed
function estimateDuration(text: string, speed: number): number {
  // Average speaking rate: ~150 words per minute at 1x speed
  const words = text.split(/\s+/).length;
  const baseMinutes = words / 150;
  const baseSeconds = baseMinutes * 60;
  // Adjust for speed
  return baseSeconds / speed;
}

export async function POST(request: Request) {
  try {
    const body: TTSGenerationRequest = await request.json();
    const { text, voiceId, config } = body;

    // Validate request
    if (!text || text.trim().length === 0) {
      return NextResponse.json(
        { error: "Text is required" },
        { status: 400 }
      );
    }

    if (!voiceId) {
      return NextResponse.json(
        { error: "Voice ID is required" },
        { status: 400 }
      );
    }

    // Validate config
    if (config.speed < 0.5 || config.speed > 2.0) {
      return NextResponse.json(
        { error: "Speed must be between 0.5 and 2.0" },
        { status: 400 }
      );
    }

    if (config.pitch < -12 || config.pitch > 12) {
      return NextResponse.json(
        { error: "Pitch must be between -12 and 12" },
        { status: 400 }
      );
    }

    // In production, call Fish-Speech or GPT-SoVITS API
    // const ttsResponse = await generateWithFishSpeech(text, voiceId, config);
    // OR
    // const ttsResponse = await generateWithGPTSoVITS(text, voiceId, config);

    // Simulate processing delay
    await new Promise((resolve) => setTimeout(resolve, 1500));

    // Estimate duration
    const duration = estimateDuration(text, config.speed);

    // Generate mock audio response
    const generatedAudio: GeneratedAudio = {
      id: `audio-${Date.now()}`,
      url: `/api/avatar-tts/audio/${Date.now()}.mp3`, // Mock URL
      duration,
      waveformData: generateWaveformData(duration),
      config,
    };

    return NextResponse.json({
      audio: generatedAudio,
      message: "Audio generated successfully",
      processingTime: 1.5,
    });
  } catch (error) {
    console.error("Failed to generate audio:", error);
    return NextResponse.json(
      { error: "Failed to generate audio" },
      { status: 500 }
    );
  }
}

// Fish-Speech API integration placeholder
async function generateWithFishSpeech(
  text: string,
  voiceId: string,
  config: TTSConfig
): Promise<ArrayBuffer> {
  const apiUrl = process.env.FISH_SPEECH_API_URL;
  if (!apiUrl) {
    throw new Error("FISH_SPEECH_API_URL not configured");
  }

  const response = await fetch(`${apiUrl}/v1/tts`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${process.env.FISH_SPEECH_API_KEY}`,
    },
    body: JSON.stringify({
      text,
      reference_id: voiceId,
      speed: config.speed,
      pitch: config.pitch,
      emotion: config.emotion,
      format: "mp3",
    }),
  });

  if (!response.ok) {
    throw new Error(`Fish-Speech API error: ${response.statusText}`);
  }

  return response.arrayBuffer();
}

// GPT-SoVITS API integration placeholder
async function generateWithGPTSoVITS(
  text: string,
  voiceId: string,
  config: TTSConfig
): Promise<ArrayBuffer> {
  const apiUrl = process.env.GPT_SOVITS_API_URL;
  if (!apiUrl) {
    throw new Error("GPT_SOVITS_API_URL not configured");
  }

  const response = await fetch(`${apiUrl}/tts`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      text,
      refer_wav_path: voiceId,
      prompt_text: "",
      prompt_language: "en",
      text_language: "en",
      speed: config.speed,
    }),
  });

  if (!response.ok) {
    throw new Error(`GPT-SoVITS API error: ${response.statusText}`);
  }

  return response.arrayBuffer();
}

// GET endpoint to retrieve generated audio
export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const audioId = searchParams.get("id");

  if (!audioId) {
    return NextResponse.json(
      { error: "Audio ID is required" },
      { status: 400 }
    );
  }

  // In production, fetch the audio file from storage
  // const audioBuffer = await fetchAudioFromStorage(audioId);

  // For now, return a placeholder response
  return NextResponse.json({
    message: "Audio retrieval not implemented in mock mode",
    audioId,
  });
}
