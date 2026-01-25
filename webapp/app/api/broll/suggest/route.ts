import { NextRequest, NextResponse } from "next/server";
import type { AISuggestion, ScriptSection, AIBRollSuggestionsResponse } from "@/types/broll";

// Mock AI analysis of script to generate B-roll suggestions
function analyzeScriptForBRoll(script: string): AIBRollSuggestionsResponse {
  // Split script into sections (mock logic - in production, use actual AI)
  const sentences = script.split(/[.!?]+/).filter((s) => s.trim().length > 0);
  const avgSentenceDuration = 4; // seconds per sentence (rough estimate)

  const suggestions: AISuggestion[] = [];
  const scriptSections: ScriptSection[] = [];

  let currentTime = 0;

  for (let i = 0; i < sentences.length; i++) {
    const sentence = sentences[i].trim();
    const sectionDuration = avgSentenceDuration;
    const endTime = currentTime + sectionDuration;

    // Create script section
    const section: ScriptSection = {
      id: `section-${i}`,
      text: sentence,
      startTime: currentTime,
      endTime: endTime,
    };

    // Analyze for B-roll opportunity (mock AI logic)
    const suggestion = generateBRollSuggestion(sentence, currentTime, i);
    if (suggestion) {
      suggestions.push(suggestion);
      section.suggestedBRoll = suggestion;
    }

    scriptSections.push(section);
    currentTime = endTime;
  }

  return {
    suggestions,
    scriptSections,
  };
}

function generateBRollSuggestion(
  sentence: string,
  timestamp: number,
  index: number
): AISuggestion | null {
  // Mock keyword extraction and B-roll suggestion generation
  const keywords = extractKeywords(sentence);

  if (keywords.length === 0) {
    return null;
  }

  // Generate search term from keywords
  const searchTerm = generateSearchTerm(keywords);

  // Format timestamp
  const minutes = Math.floor(timestamp / 60);
  const seconds = Math.floor(timestamp % 60);
  const formattedTimestamp = `${minutes}m${seconds.toString().padStart(2, "0")}s`;

  // Calculate confidence based on keyword quality
  const confidence = Math.min(0.95, 0.6 + keywords.length * 0.1);

  return {
    id: `suggestion-${index}`,
    timestamp: formattedTimestamp,
    searchTerm,
    description: generateDescription(keywords, sentence),
    scriptContext: sentence,
    confidence,
  };
}

function extractKeywords(sentence: string): string[] {
  // Mock keyword extraction - in production, use NLP or AI
  const stopWords = new Set([
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "up",
    "about", "into", "over", "after", "and", "but", "or", "if", "because",
    "as", "until", "while", "that", "this", "these", "those", "it", "its",
    "we", "you", "they", "them", "their", "our", "your", "i", "me", "my",
  ]);

  const words = sentence
    .toLowerCase()
    .replace(/[^\w\s]/g, "")
    .split(/\s+/)
    .filter((word) => word.length > 3 && !stopWords.has(word));

  // Return unique keywords
  return [...new Set(words)].slice(0, 4);
}

function generateSearchTerm(keywords: string[]): string {
  // B-roll specific modifiers
  const modifiers = [
    "cinematic",
    "aerial",
    "closeup",
    "slow motion",
    "timelapse",
    "establishing shot",
    "4k",
  ];

  // Occasionally add a modifier for variety
  const modifier = Math.random() > 0.5 ? modifiers[Math.floor(Math.random() * modifiers.length)] : "";

  const term = keywords.slice(0, 3).join(" ");
  return modifier ? `${term} ${modifier}` : term;
}

function generateDescription(keywords: string[], sentence: string): string {
  const descriptions = [
    `Visual footage to accompany discussion of ${keywords.join(", ")}`,
    `B-roll showing ${keywords[0]} in action`,
    `Supporting visuals for "${sentence.substring(0, 50)}..."`,
    `Stock footage featuring ${keywords.join(" and ")}`,
    `Dynamic shots related to ${keywords[0]}`,
  ];

  return descriptions[Math.floor(Math.random() * descriptions.length)];
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { script, projectId } = body;

    if (!script || typeof script !== "string" || script.trim().length === 0) {
      return NextResponse.json(
        { error: "Script content is required" },
        { status: 400 }
      );
    }

    // Simulate AI processing time
    await new Promise((resolve) => setTimeout(resolve, 1500 + Math.random() * 1000));

    // Analyze script and generate suggestions
    const response = analyzeScriptForBRoll(script.trim());

    return NextResponse.json(response);
  } catch (error) {
    console.error("AI suggestion error:", error);
    return NextResponse.json(
      { error: "Failed to generate B-roll suggestions" },
      { status: 500 }
    );
  }
}
