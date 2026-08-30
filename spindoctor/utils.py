"""Shared helpers used across the pipeline modules."""

import fnmatch
import glob
import os

VIDEO_PATTERNS = ("*.MOV", "*.mov")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def find_videos(path, patterns=VIDEO_PATTERNS):
    """
    List the videos to process, deduplicated and in deterministic order.

    `path` may be a folder (returns the videos inside) or a single video file
    (returns just that file).

    Deduplication matters because Windows filesystems are case-insensitive:
    glob("*.MOV") and glob("*.mov") return the same file, which would otherwise
    be processed twice and counted twice in the per-player statistics.

    Args:
        path: Folder to scan, or path to a single video.
        patterns: Glob patterns. The default covers the .MOV/.mov the inference
            pipeline expects; frame extraction passes a wider set (see
            training/extract_frames.py).

    Returns:
        Sorted list of paths. Empty if `path` does not exist, or is a file
        matching none of the patterns.
    """
    if os.path.isfile(path):
        # Case-insensitive on both sides: an explicitly named file is accepted
        # whether it ends .MOV or .mov, on any filesystem.
        name = os.path.basename(path).lower()
        if any(fnmatch.fnmatch(name, pattern.lower()) for pattern in patterns):
            return [path]
        return []

    matches = []
    for pattern in patterns:
        matches.extend(glob.glob(os.path.join(path, pattern)))
    return sorted(set(matches))
