"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { cn } from "@/lib/utils";

interface AudioWaveformProps {
  data: number[];
  progress?: number; // 0-100
  height?: number;
  barWidth?: number;
  barGap?: number;
  activeColor?: string;
  inactiveColor?: string;
  onClick?: (percentage: number) => void;
  isAnimating?: boolean;
}

export function AudioWaveform({
  data,
  progress = 0,
  height = 64,
  barWidth = 3,
  barGap = 2,
  activeColor = "hsl(var(--primary))",
  inactiveColor = "hsl(var(--muted-foreground) / 0.3)",
  onClick,
  isAnimating = false,
}: AudioWaveformProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number | null>(null);
  const [hoveredBar, setHoveredBar] = useState<number | null>(null);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx || data.length === 0) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();

    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    ctx.clearRect(0, 0, rect.width, rect.height);

    const totalBarWidth = barWidth + barGap;
    const barsCount = Math.floor(rect.width / totalBarWidth);
    const dataPointsPerBar = data.length / barsCount;

    for (let i = 0; i < barsCount; i++) {
      const dataIndex = Math.floor(i * dataPointsPerBar);
      const value = data[dataIndex] || 0;

      const barHeight = Math.max(value * (rect.height - 8), 4);
      const x = i * totalBarWidth;
      const y = (rect.height - barHeight) / 2;

      const barProgress = (i / barsCount) * 100;
      const isActive = barProgress <= progress;
      const isHovered = hoveredBar === i;

      // Animation effect
      let animationOffset = 0;
      if (isAnimating && isActive) {
        animationOffset = Math.sin((Date.now() / 200 + i * 0.5)) * 2;
      }

      ctx.fillStyle = isActive ? activeColor : inactiveColor;
      ctx.globalAlpha = isHovered ? 1 : isActive ? 0.9 : 0.5;

      ctx.beginPath();
      ctx.roundRect(
        x,
        y - animationOffset,
        barWidth,
        barHeight + animationOffset * 2,
        barWidth / 2
      );
      ctx.fill();
    }

    ctx.globalAlpha = 1;
  }, [data, progress, barWidth, barGap, activeColor, inactiveColor, hoveredBar, isAnimating]);

  useEffect(() => {
    draw();

    if (isAnimating) {
      const animate = () => {
        draw();
        animationRef.current = requestAnimationFrame(animate);
      };
      animationRef.current = requestAnimationFrame(animate);
    }

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [draw, isAnimating]);

  useEffect(() => {
    const handleResize = () => draw();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [draw]);

  const handleMouseMove = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const totalBarWidth = barWidth + barGap;
    const barIndex = Math.floor(x / totalBarWidth);
    setHoveredBar(barIndex);
  };

  const handleMouseLeave = () => {
    setHoveredBar(null);
  };

  const handleClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (!onClick || !canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const percentage = (x / rect.width) * 100;
    onClick(percentage);
  };

  return (
    <div
      ref={containerRef}
      className={cn(
        "relative w-full rounded-lg bg-muted/30 p-2",
        onClick && "cursor-pointer"
      )}
      style={{ height: `${height}px` }}
    >
      <canvas
        ref={canvasRef}
        className="h-full w-full"
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        onClick={handleClick}
      />
    </div>
  );
}

// Generate sample waveform data
export function generateWaveformData(length: number = 100): number[] {
  const data: number[] = [];
  for (let i = 0; i < length; i++) {
    // Create a more natural-looking waveform
    const base = 0.3 + Math.random() * 0.4;
    const variation = Math.sin(i * 0.1) * 0.2;
    const noise = (Math.random() - 0.5) * 0.1;
    data.push(Math.max(0.1, Math.min(1, base + variation + noise)));
  }
  return data;
}

// Compact waveform for small spaces
interface CompactWaveformProps {
  data: number[];
  progress?: number;
  isPlaying?: boolean;
}

export function CompactWaveform({
  data,
  progress = 0,
  isPlaying = false,
}: CompactWaveformProps) {
  return (
    <div className="flex h-8 items-end justify-center gap-0.5">
      {data.slice(0, 30).map((value, index) => {
        const barProgress = (index / 30) * 100;
        const isActive = barProgress <= progress;

        return (
          <div
            key={index}
            className={cn(
              "w-0.5 rounded-full transition-all",
              isActive ? "bg-primary" : "bg-muted-foreground/30",
              isPlaying && isActive && "animate-pulse"
            )}
            style={{
              height: `${Math.max(value * 100, 15)}%`,
            }}
          />
        );
      })}
    </div>
  );
}

// Animated equalizer-style waveform
interface EqualizerWaveformProps {
  isPlaying: boolean;
  barCount?: number;
}

export function EqualizerWaveform({
  isPlaying,
  barCount = 5,
}: EqualizerWaveformProps) {
  return (
    <div className="flex h-4 items-end justify-center gap-0.5">
      {Array.from({ length: barCount }).map((_, index) => (
        <div
          key={index}
          className={cn(
            "w-1 rounded-full bg-primary transition-all",
            isPlaying ? "animate-bounce" : "h-1"
          )}
          style={{
            animationDelay: `${index * 100}ms`,
            height: isPlaying ? undefined : "4px",
          }}
        />
      ))}
    </div>
  );
}
