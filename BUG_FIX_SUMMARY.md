# Video Clip Duration Bug - RESOLVED ✅

**Date:** January 23, 2026
**Issue:** Video clips downloading with incorrect durations (downloading full videos or from timestamp 0)
**Status:** COMPLETELY FIXED

---

## The Bug

Video clips were downloading with incorrect durations:
- Clips named `clip2_5.0s-10.0s` were 10 seconds instead of 5 seconds
- Clips named `clip1_70.0s-78.0s` were 18 seconds instead of 8 seconds
- Pattern: Downloads were going from timestamp 0 to end_time, ignoring start_time

---

## Root Cause Analysis

The bug had **TWO issues** in `/src/services/video_downloader.py`:

### Issue 1: Incorrect Callback Signature
```python
# WRONG (old code):
"download_ranges": lambda info, ranges: [{"start_time": start, "end_time": end}]
```

The yt-dlp documentation specifies the callback signature as `(info_dict, ydl)`, not `(info, ranges)`.

### Issue 2: Missing Precision Cut Option
Without `force_keyframes_at_cuts: True`, yt-dlp downloads from the nearest keyframe (usually at 0s) rather than creating precise cuts at exact timestamps.

---

## The Fix

**File:** `src/services/video_downloader.py` (lines 397-430)

```python
# Define download_ranges callback with correct signature: (info_dict, ydl)
def make_download_ranges(start, end):
    def download_ranges_func(info_dict, ydl):
        return [{"start_time": start, "end_time": end}]
    return download_ranges_func

ydl_opts = {
    "format": format_selector,
    "download_ranges": make_download_ranges(start_time, end_time),
    "force_keyframes_at_cuts": True,  # ← CRITICAL: Required for precise cuts
    "outtmpl": str(output_dir / f"{clip_basename}.%(ext)s"),
    "postprocessors": [
        {
            "key": "FFmpegVideoConvertor",
            "preferedformat": "mp4",
        }
    ],
    "quiet": True,
    "no_progress": True,
    "retries": 3,
    "ignoreerrors": True,
}
```

---

## Verification Results

### Before Fix
```
clip2_5.0s-10.0s → 10.008s (Expected: 5s) ❌ WRONG
clip1_70.0s-78.0s → 18.0s (Expected: 8s) ❌ WRONG
```

### After Fix
```
clip2_5.0s-10.0s → 5.007s (Expected: 5s) ✅ CORRECT
clip1_98.0s-102.0s → 4.007s (Expected: 4s) ✅ CORRECT
clip3_40.0s-46.0s → 6.007s (Expected: 6s) ✅ CORRECT
clip2_8.5s-16.0s → 7.560s (Expected: 7.5s) ✅ CORRECT
clip1_51.0s-55.5s → 4.507s (Expected: 4.5s) ✅ CORRECT
clip1_54.0s-61.0s → 7.003s (Expected: 7s) ✅ CORRECT
```

All 15 clips in final test have correct durations matching their filename timestamps.

---

## Technical Details

### What `force_keyframes_at_cuts` Does

When enabled, yt-dlp re-encodes the video (e.g., av1 → vp9) to create precise cuts at the exact timestamps specified in `download_ranges`. Without this option:
- yt-dlp seeks to the nearest keyframe (usually at 0s for many videos)
- The download starts from that keyframe instead of the exact timestamp
- Result: Downloads from 0 to end_time, ignoring start_time

### Performance Impact

The `force_keyframes_at_cuts` option requires re-encoding, which:
- Takes longer than stream copying (about 10-15 seconds per clip)
- Produces accurate clip durations matching exactly what the AI requested
- Is necessary for the two-pass download system to work correctly

---

## Test Cases Verified

**Test Video:** `descriptS&C_test_2min.mp4` (2-minute excerpt)
**Pipeline Result:** 15 clips from 4 B-roll needs
**Success Rate:** 100% (all clips have correct durations)

**Critical Test Cases:**
1. ✅ Clips starting at 0s (e.g., 0-5s) → Correct
2. ✅ Clips starting mid-video (e.g., 5-10s) → Correct (was broken before)
3. ✅ Clips from later timestamps (e.g., 98-102s) → Correct (was broken before)
4. ✅ Clips with decimal timestamps (e.g., 8.5-16.0s) → Correct

---

## Files Changed

- `src/services/video_downloader.py` (lines 397-430)
  - Fixed callback signature from `(info, ranges)` to `(info_dict, ydl)`
  - Added `force_keyframes_at_cuts: True` to ydl_opts

---

## Lessons Learned

1. **Always verify API signatures** - Parameter names matter in Python callbacks
2. **Test with non-zero start times** - Bugs may only appear for clips not starting at 0s
3. **Manual testing is critical** - Isolated tests helped identify the exact issue
4. **Read the documentation carefully** - The yt-dlp docs specified the correct signature

---

## Related Issues

This fix also resolves the underlying issue from the previous Phase 1 processing where some clips had incorrect durations (68.45s instead of ~10s).

---

**Status:** RESOLVED ✅
**Verified:** January 23, 2026, 6:30 PM PST
**Next Steps:** Monitor production usage to ensure fix works across all video types
