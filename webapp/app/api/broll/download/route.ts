import { NextRequest, NextResponse } from "next/server";
import type { BRollDownloadResponse } from "@/types/broll";

// In-memory store for download progress (in production, use Redis or database)
const downloadProgress = new Map<
  string,
  {
    status: "queued" | "downloading" | "complete" | "error";
    progress: number;
    error?: string;
    downloadUrl?: string;
    localPath?: string;
  }
>();

// Simulate download process
async function simulateDownload(downloadId: string): Promise<void> {
  // Start download
  downloadProgress.set(downloadId, {
    status: "downloading",
    progress: 0,
  });

  // Simulate progress updates
  for (let progress = 0; progress <= 100; progress += 10) {
    await new Promise((resolve) => setTimeout(resolve, 200));

    const current = downloadProgress.get(downloadId);
    if (!current || current.status === "error") break;

    downloadProgress.set(downloadId, {
      ...current,
      progress,
    });
  }

  // Random chance of error for realism
  if (Math.random() < 0.1) {
    const current = downloadProgress.get(downloadId);
    downloadProgress.set(downloadId, {
      ...current!,
      status: "error",
      error: "Download failed: Connection timeout",
    });
    return;
  }

  // Complete download
  const current = downloadProgress.get(downloadId);
  downloadProgress.set(downloadId, {
    ...current!,
    status: "complete",
    progress: 100,
    downloadUrl: `https://example.com/downloads/${downloadId}`,
    localPath: `/downloads/${downloadId}`,
  });
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { itemId, item, projectId } = body;

    if (!itemId || !item) {
      return NextResponse.json(
        { error: "itemId and item are required" },
        { status: 400 }
      );
    }

    // Generate a unique download ID
    const downloadId = `download-${itemId}-${Date.now()}`;

    // Initialize download
    downloadProgress.set(downloadId, {
      status: "queued",
      progress: 0,
    });

    // Start download in background (don't await)
    simulateDownload(downloadId).catch(console.error);

    const response: BRollDownloadResponse = {
      id: downloadId,
      status: "queued",
    };

    return NextResponse.json(response);
  } catch (error) {
    console.error("Download initiation error:", error);
    return NextResponse.json(
      { error: "Failed to initiate download" },
      { status: 500 }
    );
  }
}

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const downloadId = searchParams.get("id");

  if (!downloadId) {
    return NextResponse.json(
      { error: "Download ID is required" },
      { status: 400 }
    );
  }

  const progress = downloadProgress.get(downloadId);

  if (!progress) {
    return NextResponse.json(
      { error: "Download not found" },
      { status: 404 }
    );
  }

  const response: BRollDownloadResponse = {
    id: downloadId,
    status: progress.status,
    downloadUrl: progress.downloadUrl,
    localPath: progress.localPath,
    error: progress.error,
  };

  return NextResponse.json(response);
}

export async function DELETE(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const downloadId = searchParams.get("id");

  if (!downloadId) {
    return NextResponse.json(
      { error: "Download ID is required" },
      { status: 400 }
    );
  }

  // Remove from tracking
  downloadProgress.delete(downloadId);

  return NextResponse.json({ success: true });
}
