"use client";

import { useState, useCallback, useEffect } from "react";
import {
  Download,
  Film,
  ImageIcon,
  Plus,
  X,
} from "lucide-react";
import { Header } from "@/components/layout/header";
import { EnhancedStepIndicator } from "@/components/workflow/enhanced-step-indicator";
import { WorkflowNav } from "@/components/workflow/workflow-nav";
import { Button } from "@/components/ui/button";
import {
  SearchBar,
  SourceFilter,
  MediaGrid,
  DownloadQueue,
  AISuggestions,
} from "@/components/workflow/broll";
import { useWorkflowStore, useBRollData, useScriptData } from "@/lib/workflow-store";
import type {
  BRollItem,
  BRollSearchFilters,
  BRollSearchResponse,
  DownloadQueueItem,
  AISuggestion,
} from "@/types/broll";
import { generateId } from "@/lib/utils";

export default function BRollPage() {
  // Workflow store
  const brollData = useBRollData();
  const scriptData = useScriptData();
  const setSelectedMedia = useWorkflowStore((state) => state.setSelectedMedia);
  const markStepComplete = useWorkflowStore((state) => state.markStepComplete);

  // Get script text from previous step if available
  const scriptText = scriptData.script?.sections.map(s => s.content).join("\n") || "";

  // Search state
  const [searchQuery, setSearchQuery] = useState("");
  const [isSearching, setIsSearching] = useState(false);
  const [searchResults, setSearchResults] = useState<BRollItem[]>([]);
  const [sourceResults, setSourceResults] = useState<
    BRollSearchResponse["sources"]
  >([]);
  const [searchError, setSearchError] = useState<string | undefined>();
  const [recentSearches, setRecentSearches] = useState<string[]>([
    "technology office",
    "city skyline",
    "nature landscape",
    "business meeting",
  ]);

  // Filter state
  const [filters, setFilters] = useState<BRollSearchFilters>({
    type: "all",
    sources: ["pexels", "pixabay", "unsplash", "youtube"],
    orientation: "all",
  });

  // Selection state - synced with workflow store
  const [selectedItems, setSelectedItems] = useState<BRollItem[]>(brollData.selectedMedia);

  // Download queue state
  const [downloadQueue, setDownloadQueue] = useState<DownloadQueueItem[]>([]);

  // AI suggestions state
  const [aiSuggestions, setAiSuggestions] = useState<AISuggestion[]>([]);
  const [isGeneratingAI, setIsGeneratingAI] = useState(false);

  // Sync local state with store on mount
  useEffect(() => {
    if (brollData.selectedMedia.length > 0) {
      setSelectedItems(brollData.selectedMedia);
    }
  }, []);

  // Sync selected items to workflow store
  useEffect(() => {
    setSelectedMedia(selectedItems);
    if (selectedItems.length > 0) {
      markStepComplete("broll");
    }
  }, [selectedItems, setSelectedMedia, markStepComplete]);

  // Search handler
  const handleSearch = useCallback(
    async (query: string) => {
      if (!query.trim()) return;

      setIsSearching(true);
      setSearchError(undefined);
      setSearchQuery(query);

      // Add to recent searches
      setRecentSearches((prev) => {
        const filtered = prev.filter((s) => s !== query);
        return [query, ...filtered].slice(0, 10);
      });

      try {
        const response = await fetch("/api/broll/search", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query,
            sources: filters.sources,
            type: filters.type,
            page: 1,
            perPage: 20,
          }),
        });

        if (!response.ok) {
          throw new Error("Search failed");
        }

        const data: BRollSearchResponse = await response.json();
        setSearchResults(data.results);
        setSourceResults(data.sources);
      } catch (error) {
        console.error("Search error:", error);
        setSearchError("Failed to search for media. Please try again.");
        setSearchResults([]);
      } finally {
        setIsSearching(false);
      }
    },
    [filters.sources, filters.type]
  );

  // AI suggestions handler
  const handleGenerateAISuggestions = useCallback(async () => {
    setIsGeneratingAI(true);

    try {
      const response = await fetch("/api/broll/suggest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ script: scriptText || "Default video script about technology and innovation." }),
      });

      if (!response.ok) {
        throw new Error("Failed to generate suggestions");
      }

      const data = await response.json();
      setAiSuggestions(data.suggestions);
    } catch (error) {
      console.error("AI suggestion error:", error);
    } finally {
      setIsGeneratingAI(false);
    }
  }, [scriptText]);

  // Search from AI suggestion
  const handleSearchSuggestion = useCallback(
    (suggestion: AISuggestion) => {
      handleSearch(suggestion.searchTerm);
    },
    [handleSearch]
  );

  // Selection handlers
  const toggleItemSelection = useCallback((item: BRollItem) => {
    setSelectedItems((prev) => {
      const isSelected = prev.some(
        (i) => i.id === item.id && i.source === item.source
      );
      if (isSelected) {
        return prev.filter(
          (i) => !(i.id === item.id && i.source === item.source)
        );
      }
      return [...prev, item];
    });
  }, []);

  const removeSelectedItem = useCallback((item: BRollItem) => {
    setSelectedItems((prev) =>
      prev.filter((i) => !(i.id === item.id && i.source === item.source))
    );
  }, []);

  // Download handlers
  const handleDownloadItem = useCallback(async (item: BRollItem) => {
    const queueItem: DownloadQueueItem = {
      id: generateId(),
      item,
      status: "pending",
      progress: 0,
    };

    setDownloadQueue((prev) => [...prev, queueItem]);

    try {
      const response = await fetch("/api/broll/download", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ itemId: item.id, item }),
      });

      if (!response.ok) {
        throw new Error("Download failed");
      }

      const data = await response.json();

      // Simulate progress updates
      const interval = setInterval(() => {
        setDownloadQueue((prev) => {
          const updated = prev.map((qi) => {
            if (qi.id === queueItem.id && qi.status === "downloading") {
              const newProgress = Math.min(qi.progress + 10, 100);
              if (newProgress >= 100) {
                clearInterval(interval);
                return { ...qi, status: "complete" as const, progress: 100 };
              }
              return { ...qi, progress: newProgress };
            }
            return qi;
          });
          return updated;
        });
      }, 300);

      // Start download
      setDownloadQueue((prev) =>
        prev.map((qi) =>
          qi.id === queueItem.id ? { ...qi, status: "downloading" } : qi
        )
      );
    } catch (error) {
      console.error("Download error:", error);
      setDownloadQueue((prev) =>
        prev.map((qi) =>
          qi.id === queueItem.id
            ? { ...qi, status: "error", error: "Download failed" }
            : qi
        )
      );
    }
  }, []);

  const handleRemoveFromQueue = useCallback((id: string) => {
    setDownloadQueue((prev) => prev.filter((item) => item.id !== id));
  }, []);

  const handleClearCompleted = useCallback(() => {
    setDownloadQueue((prev) => prev.filter((item) => item.status !== "complete"));
  }, []);

  // Download all selected
  const handleDownloadSelected = useCallback(() => {
    selectedItems.forEach((item) => {
      handleDownloadItem(item);
    });
  }, [selectedItems, handleDownloadItem]);

  return (
    <div className="flex flex-col min-h-screen">
      <Header
        title="B-Roll Media"
        subtitle="Search and curate photos and videos for your content"
      />

      <div className="flex-1 p-6">
        {/* Step Indicator */}
        <div className="mb-6">
          <EnhancedStepIndicator />
        </div>

        <div className="grid gap-6 lg:grid-cols-4">
          {/* Main Search Section */}
          <div className="lg:col-span-3 space-y-4">
            {/* Search Card */}
            <div className="rounded-xl border border-border bg-card p-6">
              {/* Search Bar */}
              <SearchBar
                onSearch={handleSearch}
                onAISuggest={handleGenerateAISuggestions}
                isSearching={isSearching}
                isGeneratingAI={isGeneratingAI}
                recentSearches={recentSearches}
                className="mb-4"
              />

              {/* Filters */}
              <SourceFilter
                filters={filters}
                onFiltersChange={setFilters}
                className="mb-6"
              />

              {/* Results Grid */}
              <MediaGrid
                items={searchResults}
                selectedItems={selectedItems}
                onSelectItem={toggleItemSelection}
                onDownloadItem={handleDownloadItem}
                isLoading={isSearching}
                error={searchError}
                emptyMessage={
                  searchQuery
                    ? "No results found. Try different keywords or sources."
                    : "Enter a search term to find B-roll media"
                }
                showRelevanceScores={true}
                isDraggable={true}
                sourceResults={sourceResults}
              />
            </div>
          </div>

          {/* Sidebar */}
          <div className="space-y-4">
            {/* Selected Items Card */}
            <div className="rounded-xl border border-border bg-card p-4">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold text-foreground">
                  Selected ({selectedItems.length})
                </h3>
                {selectedItems.length > 0 && (
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={handleDownloadSelected}
                  >
                    <Download className="h-4 w-4 mr-1" />
                    Download All
                  </Button>
                )}
              </div>

              {selectedItems.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-8 text-center">
                  <div className="mb-2 flex h-12 w-12 items-center justify-center rounded-full bg-muted">
                    <Plus className="h-6 w-6 text-muted-foreground" />
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Click on media items to add them to your selection
                  </p>
                </div>
              ) : (
                <div className="space-y-2 max-h-[300px] overflow-y-auto">
                  {selectedItems.map((item) => (
                    <div
                      key={`${item.source}-${item.id}`}
                      className="flex items-center justify-between rounded-lg border border-border bg-background p-2"
                    >
                      <div className="flex items-center gap-2 min-w-0">
                        <div className="h-10 w-14 flex-shrink-0 rounded bg-muted overflow-hidden">
                          {item.thumbnailUrl ? (
                            <img
                              src={item.thumbnailUrl}
                              alt={item.title}
                              className="h-full w-full object-cover"
                            />
                          ) : item.type === "video" ? (
                            <div className="flex h-full w-full items-center justify-center">
                              <Film className="h-4 w-4 text-muted-foreground" />
                            </div>
                          ) : (
                            <div className="flex h-full w-full items-center justify-center">
                              <ImageIcon className="h-4 w-4 text-muted-foreground" />
                            </div>
                          )}
                        </div>
                        <span className="text-sm text-foreground truncate">
                          {item.title}
                        </span>
                      </div>
                      <button
                        type="button"
                        onClick={() => removeSelectedItem(item)}
                        className="text-muted-foreground hover:text-destructive flex-shrink-0"
                      >
                        <X className="h-4 w-4" />
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* AI Suggestions */}
            <AISuggestions
              suggestions={aiSuggestions}
              onSearchSuggestion={handleSearchSuggestion}
              onGenerateSuggestions={handleGenerateAISuggestions}
              isGenerating={isGeneratingAI}
              hasScript={!!scriptText}
            />

            {/* Download Queue */}
            <DownloadQueue
              items={downloadQueue}
              onRemoveItem={handleRemoveFromQueue}
              onClearCompleted={handleClearCompleted}
            />

            {/* Navigation */}
            <WorkflowNav className="flex-col" />
          </div>
        </div>
      </div>
    </div>
  );
}
