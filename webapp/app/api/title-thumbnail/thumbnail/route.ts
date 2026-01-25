import { NextRequest, NextResponse } from "next/server";
import type {
  ThumbnailGenerationRequest,
  ThumbnailGenerationResponse,
  GeneratedThumbnail,
  ThumbnailStyle,
  ColorScheme,
} from "@/types/workflow";
import { generateId } from "@/lib/utils";

// Thumbnail style prompt modifiers for AI image generation
const STYLE_PROMPTS: Record<ThumbnailStyle, string> = {
  cinematic:
    "cinematic composition, dramatic lighting, shallow depth of field, movie poster style, high contrast, professional color grading",
  minimal:
    "clean minimalist design, lots of white space, simple geometric shapes, modern sans-serif typography, flat design elements",
  bold:
    "bold vibrant colors, large impactful text, high energy, dynamic composition, eye-catching contrast, maximum visual impact",
  viral:
    "attention-grabbing, shocked expression, bright saturated colors, arrow pointing, circle highlight, youtube thumbnail style, clickbait aesthetic",
  professional:
    "corporate clean look, muted professional colors, business aesthetic, trustworthy appearance, subtle gradients, elegant typography",
};

const COLOR_SCHEME_MODIFIERS: Record<ColorScheme, string> = {
  dark: "dark moody background, deep shadows, neon accents, dark theme",
  light:
    "bright clean background, soft shadows, pastel accents, light airy feeling",
  vibrant:
    "highly saturated colors, rainbow gradient, colorful energetic, pop art influence",
};

function generatePrompt(
  title: string,
  topic: string,
  style: ThumbnailStyle,
  colorScheme: ColorScheme
): string {
  const stylePrompt = STYLE_PROMPTS[style];
  const colorPrompt = COLOR_SCHEME_MODIFIERS[colorScheme];

  return `YouTube thumbnail for "${title}" about ${topic}. ${stylePrompt}. ${colorPrompt}. 16:9 aspect ratio, high resolution, visually striking.`;
}

// Mock placeholder images for demonstration
// In production, these would be URLs from FLUX/DALL-E/Midjourney
function getMockThumbnailUrl(style: ThumbnailStyle, index: number): string {
  const styleColors: Record<ThumbnailStyle, string[]> = {
    cinematic: ["1a1a2e", "16213e", "0f3460", "e94560"],
    minimal: ["f8f9fa", "e9ecef", "dee2e6", "ced4da"],
    bold: ["ff6b6b", "feca57", "48dbfb", "ff9ff3"],
    viral: ["ff0000", "ffff00", "00ff00", "ff00ff"],
    professional: ["2d3436", "636e72", "b2bec3", "dfe6e9"],
  };

  const colors = styleColors[style];
  const bgColor = colors[index % colors.length];
  const textColor = style === "minimal" ? "333333" : "ffffff";

  // Using a placeholder service - in production this would be a real generated image
  return `https://placehold.co/1280x720/${bgColor}/${textColor}?text=Thumbnail+${index + 1}`;
}

function generateMockThumbnails(
  title: string,
  topic: string,
  config: ThumbnailGenerationRequest["config"],
  count: number = 4
): GeneratedThumbnail[] {
  const thumbnails: GeneratedThumbnail[] = [];

  for (let i = 0; i < count; i++) {
    const prompt = generatePrompt(title, topic, config.style, config.colorScheme);

    thumbnails.push({
      id: generateId(),
      url: getMockThumbnailUrl(config.style, i),
      prompt,
      config: {
        style: config.style,
        text: config.text,
        colorScheme: config.colorScheme,
      },
    });
  }

  return thumbnails;
}

export async function POST(request: NextRequest) {
  try {
    const body = (await request.json()) as ThumbnailGenerationRequest;

    if (!body.title || body.title.trim().length === 0) {
      return NextResponse.json(
        { error: "Title is required" },
        { status: 400 }
      );
    }

    if (!body.topic || body.topic.trim().length === 0) {
      return NextResponse.json(
        { error: "Topic is required" },
        { status: 400 }
      );
    }

    if (!body.config) {
      return NextResponse.json(
        { error: "Thumbnail configuration is required" },
        { status: 400 }
      );
    }

    const validStyles: ThumbnailStyle[] = [
      "cinematic",
      "minimal",
      "bold",
      "viral",
      "professional",
    ];
    if (!validStyles.includes(body.config.style)) {
      return NextResponse.json(
        { error: "Invalid thumbnail style" },
        { status: 400 }
      );
    }

    const validColorSchemes: ColorScheme[] = ["dark", "light", "vibrant"];
    if (!validColorSchemes.includes(body.config.colorScheme)) {
      return NextResponse.json(
        { error: "Invalid color scheme" },
        { status: 400 }
      );
    }

    const count = body.count ?? 4;
    if (count < 1 || count > 10) {
      return NextResponse.json(
        { error: "Count must be between 1 and 10" },
        { status: 400 }
      );
    }

    // Simulate AI processing delay (image generation typically takes longer)
    await new Promise((resolve) => setTimeout(resolve, 2000));

    // Generate mock thumbnails
    // In production, this would call FLUX/DALL-E/Midjourney API
    const thumbnails = generateMockThumbnails(
      body.title,
      body.topic,
      body.config,
      count
    );

    const response: ThumbnailGenerationResponse = {
      thumbnails,
      generatedAt: new Date().toISOString(),
    };

    return NextResponse.json(response);
  } catch (error) {
    console.error("Thumbnail generation error:", error);
    return NextResponse.json(
      { error: "Failed to generate thumbnails" },
      { status: 500 }
    );
  }
}
