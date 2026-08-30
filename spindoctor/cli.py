"""Command-line interface.

Three input modes:
  --video          a single clip
  --folder         a flat folder of clips
  --player-folder  a player/clip hierarchy, with an aggregated bullseye and a
                   CSV per player plus a global CSV

Plus two debug modes: --debug_frames annotates the release search, and
--debug_all_frames annotates every frame of the video. With --clean-viz the
--debug_all_frames output is saved with the boxes only, no labels.
"""

import argparse
import os

from tqdm import tqdm

from spindoctor.detection import debug_all_frames, load_model
from spindoctor.pipeline import process_video
from spindoctor.utils import ensure_dir, find_videos
from spindoctor.viz import make_aggregated_bullseye


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, help="Process a single video file")
    parser.add_argument("--folder", type=str, help="Process a flat folder of videos")
    parser.add_argument("--player-folder", type=str, help="Process hierarchical player folders")
    parser.add_argument("--output", type=str, default="./rigid_body_output")
    parser.add_argument("--debug_frames", action="store_true", help="Save all video frames during release detection (for debugging)")
    parser.add_argument("--debug_all_frames", action="store_true", help="Save ALL frames with detection boxes (full video debug mode)")
    parser.add_argument("--clean-viz", "--no-labels", dest="clean_viz", action="store_true",
                        help="With --debug_all_frames: draw only the boxes, no confidence labels and no info overlay (for GIFs/presentations)")
    args = parser.parse_args()

    ensure_dir(args.output)

    model, device = load_model()

    # Mode 1: Single video
    if args.video:
        # Full debug mode: Save all frames with detection boxes
        if args.debug_all_frames:
            print(f"\n=== FULL DEBUG MODE: Single video ===")
            debug_all_frames(args.video, model, device, args.output, clean_viz=args.clean_viz)
            print("Debug mode complete.")
            return

        print(f"\n=== Processing single video ===")
        r = process_video(args.video, model, device, args.output, save_debug_frames=args.debug_frames)
        if r:
            import csv
            with open(os.path.join(args.output, "results.csv"), 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=r.keys())
                writer.writeheader()
                writer.writerow(r)
        print("Done.")
        return

    # Mode 2: Flat folder (legacy)
    elif args.folder and not args.player_folder:
        print(f"\n=== Processing flat folder: {args.folder} ===")
        if not os.path.exists(args.folder):
            print("Folder not found.")
            return
        video_files = find_videos(args.folder)

        # Full debug mode: Process all videos in debug mode
        if args.debug_all_frames:
            print(f"\n=== FULL DEBUG MODE: Batch processing {len(video_files)} videos ===")
            for v in tqdm(video_files, desc="Debug processing"):
                debug_all_frames(v, model, device, args.output, clean_viz=args.clean_viz)
            print("Batch debug mode complete.")
            return

        results = []
        for v in tqdm(video_files):
            r = process_video(v, model, device, args.output, save_debug_frames=args.debug_frames)
            if r: results.append(r)

        if results:
            import csv
            with open(os.path.join(args.output, "results.csv"), 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
        print("Done.")
        return

    # Mode 3: Hierarchical player folders (NEW)
    elif args.player_folder:
        print(f"\n=== Processing hierarchical player folders: {args.player_folder} ===")
        if not os.path.exists(args.player_folder):
            print("Player folder not found.")
            return

        # Find all player subdirectories
        player_folders = [f for f in os.listdir(args.player_folder)
                         if os.path.isdir(os.path.join(args.player_folder, f))]

        if not player_folders:
            print("No player folders found.")
            return

        print(f"Found {len(player_folders)} players: {', '.join(player_folders)}")

        # Full debug mode: Process all videos in all player folders
        if args.debug_all_frames:
            print(f"\n=== FULL DEBUG MODE: Processing all player videos ===")
            for player_name in player_folders:
                player_path = os.path.join(args.player_folder, player_name)
                player_output = os.path.join(args.output, player_name)
                ensure_dir(player_output)

                video_files = find_videos(player_path)

                if video_files:
                    print(f"\n{player_name}: {len(video_files)} videos")
                    for video_path in tqdm(video_files, desc=f"  {player_name}"):
                        debug_all_frames(video_path, model, device, player_output, clean_viz=args.clean_viz)

            print("Batch debug mode complete.")
            return

        all_results = []  # Global results for all players

        for player_name in player_folders:
            print(f"\n{'='*60}")
            print(f"PLAYER: {player_name}")
            print('='*60)

            player_path = os.path.join(args.player_folder, player_name)
            player_output = os.path.join(args.output, player_name)
            ensure_dir(player_output)

            # Find all videos for this player
            video_files = find_videos(player_path)

            if not video_files:
                print(f"  No videos found for {player_name}")
                continue

            print(f"  Found {len(video_files)} videos")

            player_results = []

            for video_path in tqdm(video_files, desc=f"  {player_name}"):
                # Process each video, output goes to player_output folder
                r = process_video(video_path, model, device, player_output, save_debug_frames=args.debug_frames)
                if r:
                    r['player'] = player_name  # Add player name to result
                    player_results.append(r)
                    all_results.append(r)

            # Generate player-level aggregated bullseye
            if player_results:
                aggregated_path = os.path.join(player_output, "aggregated_bullseye.png")
                make_aggregated_bullseye(player_results, aggregated_path, player_name)
                print(f"\n  {player_name}: {len(player_results)} successful videos")
                print(f"  Aggregated bullseye: {aggregated_path}")

                # Save player-level CSV
                import csv
                player_csv = os.path.join(player_output, f"{player_name}_results.csv")
                with open(player_csv, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=player_results[0].keys())
                    writer.writeheader()
                    writer.writerows(player_results)
                print(f"  Results CSV: {player_csv}")

        # Save global results CSV
        if all_results:
            import csv
            global_csv = os.path.join(args.output, "all_players_results.csv")
            with open(global_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
                writer.writeheader()
                writer.writerows(all_results)
            print(f"\n{'='*60}")
            print(f"SUMMARY: Processed {len(all_results)} total videos across {len(player_folders)} players")
            print(f"Global results: {global_csv}")
            print('='*60)

        print("\nDone.")

    else:
        print("Please specify --video, --folder, or --player-folder")
        print("Examples:")
        print("  Single video:     --video path/to/video.MOV --output ./output")
        print("  Flat folder:      --folder ./videos --output ./output")
        print("  Player hierarchy: --player-folder ./gym_videos --output ./output")
        print("  Release debug:    --video shot1.MOV --output ./output --debug_frames")
        print("  Full debug mode:  --video shot1.MOV --output ./output --debug_all_frames")
        print("  Batch debug:      --folder ./problem_videos --output ./debug_output --debug_all_frames")
        print("  Clean boxes (GIF):--video shot1.MOV --output ./output --debug_all_frames --clean-viz")
