const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

export async function apiRequest<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<ApiResponse<T>> {
  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...options.headers,
      },
    });

    const data = await response.json();

    if (!response.ok) {
      return {
        success: false,
        error: data.message || `HTTP error ${response.status}`,
      };
    }

    return {
      success: true,
      data,
    };
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : "Unknown error occurred",
    };
  }
}

// Script API
export const scriptApi = {
  generate: (prompt: string) =>
    apiRequest<{ script: string }>("/api/script/generate", {
      method: "POST",
      body: JSON.stringify({ prompt }),
    }),
  refine: (script: string, instructions: string) =>
    apiRequest<{ script: string }>("/api/script/refine", {
      method: "POST",
      body: JSON.stringify({ script, instructions }),
    }),
};

// Title & Thumbnail API
export const titleThumbnailApi = {
  generateTitles: (script: string) =>
    apiRequest<{ titles: string[] }>("/api/title/generate", {
      method: "POST",
      body: JSON.stringify({ script }),
    }),
  generateThumbnail: (title: string, style: string) =>
    apiRequest<{ thumbnailUrl: string }>("/api/thumbnail/generate", {
      method: "POST",
      body: JSON.stringify({ title, style }),
    }),
};

// B-Roll API
export const brollApi = {
  search: (query: string, type: "photo" | "video") =>
    apiRequest<{ results: Array<{ url: string; thumbnail: string; title: string }> }>(
      "/api/broll/search",
      {
        method: "POST",
        body: JSON.stringify({ query, type }),
      }
    ),
  analyze: (videoUrl: string) =>
    apiRequest<{ clips: Array<{ start: number; end: number; score: number }> }>(
      "/api/broll/analyze",
      {
        method: "POST",
        body: JSON.stringify({ videoUrl }),
      }
    ),
};

// Avatar & TTS API
export const avatarApi = {
  listAvatars: () =>
    apiRequest<{ avatars: Array<{ id: string; name: string; thumbnail: string }> }>(
      "/api/avatar/list"
    ),
  listVoices: () =>
    apiRequest<{ voices: Array<{ id: string; name: string; preview: string }> }>(
      "/api/voice/list"
    ),
  generateSpeech: (text: string, voiceId: string) =>
    apiRequest<{ audioUrl: string }>("/api/tts/generate", {
      method: "POST",
      body: JSON.stringify({ text, voiceId }),
    }),
};

// Export API
export const exportApi = {
  render: (projectId: string, settings: Record<string, unknown>) =>
    apiRequest<{ jobId: string }>("/api/export/render", {
      method: "POST",
      body: JSON.stringify({ projectId, settings }),
    }),
  status: (jobId: string) =>
    apiRequest<{ status: string; progress: number; downloadUrl?: string }>(
      `/api/export/status/${jobId}`
    ),
};
