"""Extract raw frames from videos, ready for annotation.

First step of the data-prep workflow: turns a folder of videos into JPEG images
at a regular interval, to be uploaded to Roboflow (or any other annotation
tool) and labelled by hand with the project's three classes.

The saved frames are RAW: no model is run, no box is drawn. That is the
difference from `spindoctor.detection.debug_all_frames`, which saves frames
already annotated with detections and is therefore unusable as an annotation
source.

`--input` takes either a folder (processes every video in it) or a single video
file; the type is detected automatically.

Usage (from the repo root):
    python -m training.extract_frames --input ./videos --output ./data/raw_frames
    python -m training.extract_frames --input ./videos/shot1.MOV
    python -m training.extract_frames --every 10 --max-per-video 40

Next step: label the frames with ball=1, dot=2, hand=3 (see the class contract
in training/README.md) and export to COCO format.
"""

import argparse
import os

import cv2
from tqdm import tqdm

from spindoctor.utils import ensure_dir, find_videos

# Wider than the inference pipeline's (.MOV only): source footage here can
# come from different phones and cameras.
VIDEO_PATTERNS = ("*.MOV", "*.mov", "*.mp4", "*.MP4", "*.avi", "*.AVI")

DEFAULT_INPUT = "./videos"
DEFAULT_OUTPUT = "./data/raw_frames"
DEFAULT_EVERY = 5        # 1 frame in 5: at 60fps that is 12 frames/second
DEFAULT_QUALITY = 95     # high, so no artefacts are baked in before labelling


def extract_frames_from_video(video_path, output_dir, every=DEFAULT_EVERY,
                              max_frames=None, quality=DEFAULT_QUALITY):
    """
    Extract frames from a single video and save them as JPEG.

    Files are named `{video_name}_{frame_index:05d}.jpg`, the index referring to
    the frame in the original video (0-based). Keeping the index in the name
    makes it possible to trace a labelled image back to the exact point of the
    clip it came from.

    Args:
        video_path: Path to the video.
        output_dir: Destination folder (must already exist).
        every: Save one frame every `every` frames read.
        max_frames: Max frames to save for this video (None = no limit).
        quality: JPEG quality, 0-100.

    Returns:
        Number of frames saved.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  WARNING: cannot open {video_path}, skipping it")
        return 0

    base = os.path.splitext(os.path.basename(video_path))[0]
    frame_idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % every == 0:
            filename = f"{base}_{frame_idx:05d}.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, quality])
            saved += 1
            if max_frames is not None and saved >= max_frames:
                break

        frame_idx += 1

    cap.release()
    return saved


def extract_all(input_path, output_dir, every=DEFAULT_EVERY, max_frames=None,
                quality=DEFAULT_QUALITY):
    """
    Extract frames from a single video, or from every video in a folder.

    Args:
        input_path: Folder of videos, or path to a single video.

    Returns:
        (videos_processed, total_frames)
    """
    videos = find_videos(input_path, patterns=VIDEO_PATTERNS)
    if not videos:
        if os.path.isfile(input_path):
            print(f"{input_path} is not a recognised video")
        else:
            print(f"No videos found in {input_path}")
        print(f"Accepted extensions: {', '.join(VIDEO_PATTERNS)}")
        return 0, 0

    ensure_dir(output_dir)
    if os.path.isfile(input_path):
        print(f"Single video: {os.path.basename(input_path)}")
    else:
        print(f"Found {len(videos)} videos in {input_path}")
    print(f"Extracting 1 frame in {every} into {output_dir}")

    total = 0
    for video_path in tqdm(videos, desc="Video"):
        n = extract_frames_from_video(video_path, output_dir, every=every,
                                      max_frames=max_frames, quality=quality)
        total += n
        tqdm.write(f"  {os.path.basename(video_path)}: {n} frames")

    return len(videos), total


def main():
    parser = argparse.ArgumentParser(
        description="Extract raw frames from videos, ready for annotation.")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT,
                        help=f"Folder of videos, or a single video file "
                             f"(default: {DEFAULT_INPUT})")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help=f"Destination folder for the frames (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--every", type=int, default=DEFAULT_EVERY,
                        help=f"Save one frame every N frames (default: {DEFAULT_EVERY})")
    parser.add_argument("--max-per-video", type=int, default=None,
                        help="Max frames per video (default: no limit)")
    parser.add_argument("--quality", type=int, default=DEFAULT_QUALITY,
                        help=f"JPEG quality 0-100 (default: {DEFAULT_QUALITY})")
    args = parser.parse_args()

    if args.every < 1:
        parser.error("--every must be >= 1")
    if not 0 <= args.quality <= 100:
        parser.error("--quality must be between 0 and 100")
    if not os.path.exists(args.input):
        parser.error(f"input path not found: {args.input}")

    n_videos, n_frames = extract_all(args.input, args.output, every=args.every,
                                     max_frames=args.max_per_video,
                                     quality=args.quality)

    print(f"\nDone: {n_frames} frames extracted from {n_videos} videos into {args.output}")
    if n_frames:
        print("Next step: label the frames with ball=1, dot=2, hand=3")
        print("and export to COCO format (see training/README.md).")


if __name__ == "__main__":
    main()
