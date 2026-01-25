import { NextResponse } from "next/server";
import type { Voice } from "@/types/workflow";

// Mock voice data - in production, this would come from Fish-Speech or GPT-SoVITS API
const MOCK_VOICES: Voice[] = [
  {
    id: "v1",
    name: "Alex - Natural",
    language: "English (US)",
    gender: "male",
    previewUrl: "/audio/previews/alex.mp3",
    style: "natural",
  },
  {
    id: "v2",
    name: "Sarah - Professional",
    language: "English (US)",
    gender: "female",
    previewUrl: "/audio/previews/sarah.mp3",
    style: "professional",
  },
  {
    id: "v3",
    name: "James - British",
    language: "English (UK)",
    gender: "male",
    previewUrl: "/audio/previews/james.mp3",
    style: "professional",
  },
  {
    id: "v4",
    name: "Emma - Casual",
    language: "English (US)",
    gender: "female",
    previewUrl: "/audio/previews/emma.mp3",
    style: "casual",
  },
  {
    id: "v5",
    name: "Marcus - Dramatic",
    language: "English (US)",
    gender: "male",
    previewUrl: "/audio/previews/marcus.mp3",
    style: "dramatic",
  },
  {
    id: "v6",
    name: "Luna - Natural",
    language: "English (US)",
    gender: "female",
    previewUrl: "/audio/previews/luna.mp3",
    style: "natural",
  },
  {
    id: "v7",
    name: "David - Casual",
    language: "English (AU)",
    gender: "male",
    previewUrl: "/audio/previews/david.mp3",
    style: "casual",
  },
  {
    id: "v8",
    name: "Sophie - Dramatic",
    language: "English (UK)",
    gender: "female",
    previewUrl: "/audio/previews/sophie.mp3",
    style: "dramatic",
  },
  {
    id: "v9",
    name: "Kai - Neutral",
    language: "English (US)",
    gender: "neutral",
    previewUrl: "/audio/previews/kai.mp3",
    style: "natural",
  },
  {
    id: "v10",
    name: "Jordan - Professional",
    language: "English (US)",
    gender: "neutral",
    previewUrl: "/audio/previews/jordan.mp3",
    style: "professional",
  },
];

export async function GET() {
  try {
    // In production, fetch from Fish-Speech or GPT-SoVITS API
    // const response = await fetch(`${process.env.FISH_SPEECH_API_URL}/voices`);
    // const voices = await response.json();

    // For now, return mock data
    return NextResponse.json({
      voices: MOCK_VOICES,
      total: MOCK_VOICES.length,
    });
  } catch (error) {
    console.error("Failed to fetch voices:", error);
    return NextResponse.json(
      { error: "Failed to fetch voices" },
      { status: 500 }
    );
  }
}

// POST - Upload custom voice sample for cloning
export async function POST(request: Request) {
  try {
    const formData = await request.formData();
    const audioFile = formData.get("audio") as File;
    const name = formData.get("name") as string;

    if (!audioFile || !name) {
      return NextResponse.json(
        { error: "Audio file and name are required" },
        { status: 400 }
      );
    }

    // Validate file type
    const allowedTypes = ["audio/wav", "audio/mp3", "audio/mpeg", "audio/m4a"];
    if (!allowedTypes.includes(audioFile.type)) {
      return NextResponse.json(
        { error: "Invalid audio format. Supported: WAV, MP3, M4A" },
        { status: 400 }
      );
    }

    // Validate file size (max 50MB)
    if (audioFile.size > 50 * 1024 * 1024) {
      return NextResponse.json(
        { error: "File size must be less than 50MB" },
        { status: 400 }
      );
    }

    // In production, send to Fish-Speech or GPT-SoVITS for voice cloning
    // const response = await fetch(`${process.env.FISH_SPEECH_API_URL}/clone`, {
    //   method: "POST",
    //   body: formData,
    // });

    // Mock response for now
    const newVoice: Voice = {
      id: `custom-${Date.now()}`,
      name: name,
      language: "Custom",
      gender: "neutral",
      previewUrl: URL.createObjectURL(audioFile),
      style: "natural",
    };

    return NextResponse.json({
      voice: newVoice,
      message: "Voice created successfully",
    });
  } catch (error) {
    console.error("Failed to create custom voice:", error);
    return NextResponse.json(
      { error: "Failed to create custom voice" },
      { status: 500 }
    );
  }
}
