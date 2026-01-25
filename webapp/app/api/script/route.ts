import { NextRequest, NextResponse } from "next/server";
import type {
  Script,
  ScriptWriterSection,
  ChatMessage,
  ScriptGenerationRequest,
  ScriptGenerationResponse,
} from "@/types/workflow";

// Generate a unique ID
function generateId(): string {
  return Math.random().toString(36).substring(2, 15);
}

// Mock AI script generation - structured for future Claude/GPT/Gemini integration
async function generateScriptWithAI(
  messages: ChatMessage[],
  currentScript?: Script
): Promise<{ script: Script; message: string }> {
  // Extract the conversation context
  const userMessages = messages.filter((m) => m.role === "user");
  const lastUserMessage = userMessages[userMessages.length - 1]?.content || "";

  // Simulate AI processing delay
  await new Promise((resolve) => setTimeout(resolve, 1500));

  // Determine if this is a new script or a refinement
  const isRefinement = currentScript && currentScript.sections.length > 0;

  if (isRefinement && currentScript) {
    // Mock refinement response
    return {
      script: {
        ...currentScript,
        sections: currentScript.sections.map((section) => ({
          ...section,
          content: section.content + " [Refined based on your feedback]",
        })),
      },
      message: `I've refined the script based on your request: "${lastUserMessage}". The changes have been applied to maintain flow while incorporating your feedback.`,
    };
  }

  // Generate new script based on conversation
  const topic = extractTopic(messages);
  const sections = generateMockSections(topic);

  const script: Script = {
    id: generateId(),
    title: `Video Script: ${topic}`,
    sections,
    totalDuration: calculateTotalDuration(sections),
  };

  return {
    script,
    message: `I've created a structured script for your video about "${topic}". The script is organized into ${sections.length} sections with timestamps and B-roll suggestions. Feel free to ask me to refine any section or add more details!`,
  };
}

// Extract topic from conversation
function extractTopic(messages: ChatMessage[]): string {
  const userMessages = messages.filter((m) => m.role === "user");
  const firstMessage = userMessages[0]?.content || "your topic";

  // Simple extraction - in real implementation, use AI
  const words = firstMessage.split(" ").slice(0, 5).join(" ");
  return words.length > 30 ? words.substring(0, 30) + "..." : words;
}

// Generate mock script sections
function generateMockSections(topic: string): ScriptWriterSection[] {
  return [
    {
      id: generateId(),
      timestamp: "0:00",
      content: `[HOOK] Hey everyone! Today we're diving into ${topic}. By the end of this video, you'll have a complete understanding of everything you need to know.`,
      brollSuggestions: [
        "Dynamic intro animation",
        "Quick cuts of relevant imagery",
        "Text overlay with topic title",
      ],
    },
    {
      id: generateId(),
      timestamp: "0:30",
      content: `[INTRODUCTION] Before we get started, let me give you some context. ${topic} has become increasingly important because of [reason]. Let's break this down step by step.`,
      brollSuggestions: [
        "Background context visuals",
        "Statistics or data graphics",
        "Related industry footage",
      ],
    },
    {
      id: generateId(),
      timestamp: "1:30",
      content: `[MAIN POINT 1] The first thing you need to understand about ${topic} is the fundamentals. Here's what makes this so crucial...`,
      brollSuggestions: [
        "Diagram or illustration",
        "Screen recording demonstration",
        "Expert interview clip",
      ],
    },
    {
      id: generateId(),
      timestamp: "3:00",
      content: `[MAIN POINT 2] Now let's look at the practical application. When you apply ${topic} in real situations, you'll notice...`,
      brollSuggestions: [
        "Step-by-step tutorial footage",
        "Before/after comparison",
        "Real-world examples",
      ],
    },
    {
      id: generateId(),
      timestamp: "4:30",
      content: `[MAIN POINT 3] One common mistake people make with ${topic} is... Here's how to avoid that pitfall and get better results.`,
      brollSuggestions: [
        "Common mistakes montage",
        "Correct approach demonstration",
        "Tips and tricks overlay",
      ],
    },
    {
      id: generateId(),
      timestamp: "6:00",
      content: `[CONCLUSION] To wrap up, remember these key takeaways about ${topic}: [Point 1], [Point 2], and [Point 3]. If you found this helpful, don't forget to like and subscribe!`,
      brollSuggestions: [
        "Summary graphics",
        "Call-to-action overlay",
        "Subscribe button animation",
      ],
    },
    {
      id: generateId(),
      timestamp: "6:45",
      content: `[OUTRO] Thanks for watching! Drop a comment below with your questions about ${topic}, and I'll see you in the next video.`,
      brollSuggestions: [
        "End screen with related videos",
        "Social media handles",
        "Outro music and graphics",
      ],
    },
  ];
}

// Calculate total duration from sections
function calculateTotalDuration(sections: ScriptWriterSection[]): string {
  if (sections.length === 0) return "0:00";

  const lastSection = sections[sections.length - 1];
  const [mins, secs] = lastSection.timestamp.split(":").map(Number);

  // Add estimated 30 seconds for the last section
  const totalSeconds = mins * 60 + secs + 30;
  const totalMins = Math.floor(totalSeconds / 60);
  const remainingSecs = totalSeconds % 60;

  return `${totalMins}:${remainingSecs.toString().padStart(2, "0")}`;
}

// Chat endpoint - for iterating on script ideas
export async function POST(request: NextRequest) {
  try {
    const body = (await request.json()) as ScriptGenerationRequest;
    const { messages, currentScript } = body;

    if (!messages || messages.length === 0) {
      return NextResponse.json(
        { error: "Messages are required" },
        { status: 400 }
      );
    }

    const result = await generateScriptWithAI(messages, currentScript);

    const response: ScriptGenerationResponse = {
      script: result.script,
      message: result.message,
    };

    return NextResponse.json(response);
  } catch (error) {
    console.error("Script generation error:", error);
    return NextResponse.json(
      { error: "Failed to generate script" },
      { status: 500 }
    );
  }
}

// GET endpoint - for retrieving a saved script (placeholder)
export async function GET() {
  return NextResponse.json({
    message: "Script API is active. Use POST to generate scripts.",
  });
}
