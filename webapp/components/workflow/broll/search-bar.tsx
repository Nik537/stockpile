"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { Search, Loader2, Sparkles, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";

interface SearchBarProps {
  onSearch: (query: string) => void;
  onAISuggest?: () => void;
  isSearching?: boolean;
  isGeneratingAI?: boolean;
  placeholder?: string;
  className?: string;
  recentSearches?: string[];
}

export function SearchBar({
  onSearch,
  onAISuggest,
  isSearching = false,
  isGeneratingAI = false,
  placeholder = "Search for B-roll footage...",
  className,
  recentSearches = [],
}: SearchBarProps) {
  const [query, setQuery] = useState("");
  const [showSuggestions, setShowSuggestions] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const handleSearch = useCallback(() => {
    if (!query.trim() || isSearching) return;
    onSearch(query.trim());
    setShowSuggestions(false);
  }, [query, isSearching, onSearch]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      handleSearch();
    } else if (e.key === "Escape") {
      setShowSuggestions(false);
    }
  };

  const handleSuggestionClick = (suggestion: string) => {
    setQuery(suggestion);
    onSearch(suggestion);
    setShowSuggestions(false);
  };

  const clearQuery = () => {
    setQuery("");
    inputRef.current?.focus();
  };

  // Close suggestions when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setShowSuggestions(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  return (
    <div ref={containerRef} className={cn("relative", className)}>
      <div className="flex gap-2">
        {/* Search Input */}
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              if (e.target.value && recentSearches.length > 0) {
                setShowSuggestions(true);
              }
            }}
            onFocus={() => {
              if (query && recentSearches.length > 0) {
                setShowSuggestions(true);
              }
            }}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            className="pl-10 pr-10"
            disabled={isSearching}
          />
          {query && (
            <button
              type="button"
              onClick={clearQuery}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
            >
              <X className="h-4 w-4" />
            </button>
          )}
        </div>

        {/* Search Button */}
        <Button
          onClick={handleSearch}
          disabled={isSearching || !query.trim()}
          className="min-w-[100px]"
        >
          {isSearching ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" />
              <span className="ml-2">Searching</span>
            </>
          ) : (
            "Search"
          )}
        </Button>

        {/* AI Suggest Button */}
        {onAISuggest && (
          <Button
            variant="secondary"
            onClick={onAISuggest}
            disabled={isGeneratingAI}
            className="min-w-[140px]"
          >
            {isGeneratingAI ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                <span className="ml-2">Generating</span>
              </>
            ) : (
              <>
                <Sparkles className="h-4 w-4" />
                <span className="ml-2">AI Suggest</span>
              </>
            )}
          </Button>
        )}
      </div>

      {/* Recent Searches Dropdown */}
      {showSuggestions && recentSearches.length > 0 && (
        <div className="absolute top-full left-0 right-0 z-50 mt-1 rounded-lg border border-border bg-card shadow-lg">
          <div className="p-2">
            <p className="px-2 py-1 text-xs font-medium text-muted-foreground">
              Recent Searches
            </p>
            {recentSearches
              .filter((s) => s.toLowerCase().includes(query.toLowerCase()))
              .slice(0, 5)
              .map((suggestion, index) => (
                <button
                  key={index}
                  type="button"
                  onClick={() => handleSuggestionClick(suggestion)}
                  className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-sm text-foreground hover:bg-muted transition-colors"
                >
                  <Search className="h-3 w-3 text-muted-foreground" />
                  {suggestion}
                </button>
              ))}
          </div>
        </div>
      )}
    </div>
  );
}
