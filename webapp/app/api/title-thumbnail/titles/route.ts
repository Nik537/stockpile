import { NextRequest, NextResponse } from "next/server";
import type {
  TitleGenerationRequest,
  TitleGenerationResponse,
  TitleSuggestion,
  TitleStyle,
} from "@/types/workflow";
import { generateId } from "@/lib/utils";

// Title style configurations with templates
const TITLE_STYLES: Record<TitleStyle, { templates: string[]; scoreBoost: number }> = {
  hook: {
    templates: [
      "The Shocking Truth About {topic}",
      "What Nobody Tells You About {topic}",
      "Why {topic} Changed Everything",
      "The {topic} Secret That Experts Hide",
    ],
    scoreBoost: 5,
  },
  curiosity: {
    templates: [
      "I Tried {topic} For 30 Days - Here's What Happened",
      "What Happens When You {topic}?",
      "The Unexpected Side of {topic}",
      "Why Is Everyone Talking About {topic}?",
    ],
    scoreBoost: 3,
  },
  howto: {
    templates: [
      "How To {topic} (Complete Guide)",
      "The Ultimate {topic} Tutorial",
      "{topic} For Beginners: Step by Step",
      "Master {topic} in Just 10 Minutes",
    ],
    scoreBoost: 0,
  },
  listicle: {
    templates: [
      "10 {topic} Tips That Actually Work",
      "5 Reasons Why {topic} Matters",
      "7 {topic} Mistakes You're Making",
      "Top 8 {topic} Hacks Nobody Knows",
    ],
    scoreBoost: 2,
  },
  story: {
    templates: [
      "My {topic} Journey: A Story",
      "How {topic} Changed My Life",
      "The Day I Discovered {topic}",
      "From Zero to Hero: My {topic} Story",
    ],
    scoreBoost: 1,
  },
};

function generateMockTitles(
  topic: string,
  count: number = 8
): TitleSuggestion[] {
  const suggestions: TitleSuggestion[] = [];
  const styles = Object.keys(TITLE_STYLES) as TitleStyle[];

  // Distribute titles across styles
  for (let i = 0; i < count; i++) {
    const style = styles[i % styles.length];
    const styleConfig = TITLE_STYLES[style];
    const template =
      styleConfig.templates[Math.floor(Math.random() * styleConfig.templates.length)];

    // Format topic for title (capitalize first letter of each word)
    const formattedTopic = topic
      .split(" ")
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(" ");

    const title = template.replace("{topic}", formattedTopic);

    // Calculate a mock score based on style and randomness
    const baseScore = 70 + Math.floor(Math.random() * 20);
    const score = Math.min(100, baseScore + styleConfig.scoreBoost);

    suggestions.push({
      id: generateId(),
      title,
      score,
      style,
    });
  }

  // Sort by score descending
  return suggestions.sort((a, b) => b.score - a.score);
}

export async function POST(request: NextRequest) {
  try {
    const body = (await request.json()) as TitleGenerationRequest;

    if (!body.topic || body.topic.trim().length === 0) {
      return NextResponse.json(
        { error: "Topic is required" },
        { status: 400 }
      );
    }

    const count = body.count ?? 8;
    if (count < 1 || count > 20) {
      return NextResponse.json(
        { error: "Count must be between 1 and 20" },
        { status: 400 }
      );
    }

    // Simulate AI processing delay
    await new Promise((resolve) => setTimeout(resolve, 1000));

    // Generate mock titles
    // In production, this would call FLUX/Gemini/OpenAI API
    const suggestions = generateMockTitles(body.topic, count);

    const response: TitleGenerationResponse = {
      suggestions,
      generatedAt: new Date().toISOString(),
    };

    return NextResponse.json(response);
  } catch (error) {
    console.error("Title generation error:", error);
    return NextResponse.json(
      { error: "Failed to generate titles" },
      { status: 500 }
    );
  }
}
