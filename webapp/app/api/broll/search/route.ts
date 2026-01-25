import { NextRequest, NextResponse } from "next/server";
import type {
  BRollItem,
  MediaSource,
  MediaType,
  BRollSearchResponse,
} from "@/types/broll";

// Mock data generators for different sources
function generateMockResults(
  query: string,
  source: MediaSource,
  type: MediaType | "all",
  count: number = 10
): BRollItem[] {
  const results: BRollItem[] = [];

  for (let i = 0; i < count; i++) {
    const isVideo = type === "video" || (type === "all" && Math.random() > 0.5);
    const id = `${source}-${query.replace(/\s+/g, "-")}-${i}`;

    results.push({
      id,
      type: isVideo ? "video" : "photo",
      source,
      url: `https://example.com/${source}/${id}`,
      thumbnailUrl: getPlaceholderThumbnail(source, isVideo, i),
      title: generateTitle(query, source, i),
      duration: isVideo ? Math.floor(Math.random() * 60) + 5 : undefined,
      dimensions: {
        width: 1920,
        height: 1080,
      },
      relevanceScore: Math.random() * 0.5 + 0.5, // 0.5 - 1.0
      photographer: isVideo ? undefined : `Photographer ${i + 1}`,
      videographer: isVideo ? `Videographer ${i + 1}` : undefined,
      tags: generateTags(query),
      previewUrl: isVideo
        ? `https://example.com/${source}/${id}/preview.mp4`
        : undefined,
      downloadUrl: `https://example.com/${source}/${id}/download`,
      license: "Free to use",
    });
  }

  return results;
}

function getPlaceholderThumbnail(
  source: MediaSource,
  isVideo: boolean,
  index: number
): string {
  // Use picsum.photos for realistic placeholder images
  const width = 640;
  const height = 360;
  const seed = `${source}-${index}`;
  return `https://picsum.photos/seed/${seed}/${width}/${height}`;
}

function generateTitle(query: string, source: MediaSource, index: number): string {
  const adjectives = [
    "Beautiful",
    "Stunning",
    "Professional",
    "Cinematic",
    "Modern",
    "Dynamic",
    "Elegant",
    "Dramatic",
  ];
  const adjective = adjectives[index % adjectives.length];
  return `${adjective} ${query} - ${source.charAt(0).toUpperCase() + source.slice(1)} #${index + 1}`;
}

function generateTags(query: string): string[] {
  const baseTags = query.toLowerCase().split(" ");
  const additionalTags = ["stock", "footage", "broll", "hd", "4k"];
  return [...baseTags, ...additionalTags.slice(0, 3)];
}

// Simulate API call to each source
async function searchSource(
  source: MediaSource,
  query: string,
  type: MediaType | "all",
  page: number,
  perPage: number
): Promise<{ results: BRollItem[]; total: number; error?: string }> {
  // Simulate network delay
  await new Promise((resolve) =>
    setTimeout(resolve, Math.random() * 500 + 200)
  );

  // Simulate occasional errors for realism
  if (Math.random() < 0.05) {
    return {
      results: [],
      total: 0,
      error: `Failed to fetch from ${source}: Rate limit exceeded`,
    };
  }

  const totalResults = Math.floor(Math.random() * 100) + 20;
  const results = generateMockResults(query, source, type, perPage);

  return {
    results,
    total: totalResults,
  };
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const {
      query,
      sources = ["pexels", "pixabay", "unsplash", "youtube"],
      type = "all",
      page = 1,
      perPage = 15,
    } = body;

    if (!query || typeof query !== "string" || query.trim().length === 0) {
      return NextResponse.json(
        { error: "Query is required" },
        { status: 400 }
      );
    }

    // Search all sources in parallel
    const sourcePromises = (sources as MediaSource[]).map(async (source) => {
      const result = await searchSource(source, query.trim(), type, page, perPage);
      return {
        source,
        ...result,
      };
    });

    const sourceResults = await Promise.all(sourcePromises);

    // Combine all results
    const allResults: BRollItem[] = [];
    const sourceSummary: BRollSearchResponse["sources"] = [];

    for (const result of sourceResults) {
      sourceSummary.push({
        source: result.source,
        count: result.results.length,
        error: result.error,
      });

      if (result.results.length > 0) {
        allResults.push(...result.results);
      }
    }

    // Sort by relevance score
    allResults.sort((a, b) => (b.relevanceScore || 0) - (a.relevanceScore || 0));

    const response: BRollSearchResponse = {
      results: allResults,
      total: allResults.length,
      page,
      perPage,
      sources: sourceSummary,
    };

    return NextResponse.json(response);
  } catch (error) {
    console.error("B-Roll search error:", error);
    return NextResponse.json(
      { error: "Failed to search for B-roll media" },
      { status: 500 }
    );
  }
}

export async function GET(request: NextRequest) {
  // Support GET requests with query params for simple searches
  const searchParams = request.nextUrl.searchParams;
  const query = searchParams.get("q") || searchParams.get("query");

  if (!query) {
    return NextResponse.json(
      { error: "Query parameter 'q' or 'query' is required" },
      { status: 400 }
    );
  }

  const sources = searchParams.get("sources")?.split(",") || [
    "pexels",
    "pixabay",
    "unsplash",
    "youtube",
  ];
  const type = (searchParams.get("type") as MediaType | "all") || "all";
  const page = parseInt(searchParams.get("page") || "1", 10);
  const perPage = parseInt(searchParams.get("perPage") || "15", 10);

  // Reuse POST logic
  const mockRequest = {
    json: async () => ({ query, sources, type, page, perPage }),
  } as NextRequest;

  return POST(mockRequest);
}
