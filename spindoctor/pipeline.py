"""Orchestration: from a video to a spin result.

For each clip:
1. Find the release frame (`release`).
2. Take 4 consecutive frames from there and detect ball and dots in them
   (`detection` at full resolution, `dots` zoomed on the ball).
3. Chain the frame-to-frame matches into tracks - dots followed over time -
   using the previous rotation as an inertial prior (`matching`).
4. Pick the best window: 4 frames preferred, since a larger time delta means
   less relative error, falling back to 3 only when fewer than 3 dots survive
   end to end.
5. Solve Wahba between the window's first and last frame (`spin`), then render
   plots and overlays (`viz`).
"""

import os

import cv2
import numpy as np
from PIL import Image

from spindoctor.config import (
    CLASS_BALL, CLASS_HAND, CONF_BALL, CONF_DOT, CONF_HAND, ZOOM_FACTOR,
)
from spindoctor.dots import detect_dots_with_zoom
from spindoctor.geometry import get_ball_center, get_ball_radius
from spindoctor.matching import match_dots_rigid_body
from spindoctor.release import find_release_frame
from spindoctor.spin import calculate_spin_first_last
from spindoctor.utils import ensure_dir
from spindoctor.viz import make_bullseye, save_track_visualizations


def process_video(video_path, model, device, output_folder, save_debug_frames=False):
    video_name = os.path.basename(video_path)
    video_base = os.path.splitext(video_name)[0]
    video_output_folder = os.path.join(output_folder, video_base)
    ensure_dir(video_output_folder)

    print(f"\nProcessing: {video_name}")

    debug_folder = video_output_folder if save_debug_frames else None
    release_frame, fps = find_release_frame(video_path, model, device, debug_folder=debug_folder)
    if release_frame is None:
        print("  ERROR: Could not find release frame")
        return None

    # Release, +1, +2, +3: at 60fps this window is wide enough to cut the
    # relative error without leaving free flight.
    target_frames = [release_frame + i for i in range(4)]

    cap = cv2.VideoCapture(video_path)
    all_detections = {}  # frame_idx -> data dict
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        if frame_idx not in target_frames: continue

        # Pass 1: ball and hand, at full resolution.
        rgb_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detections = model.predict(rgb_image, threshold=0.2)

        ball_boxes = []
        hand_boxes = []

        if detections is not None:
            for i in range(len(detections)):
                cls = detections.class_id[i]
                conf = detections.confidence[i]
                box = detections.xyxy[i].tolist()
                if cls == CLASS_BALL and conf >= CONF_BALL: ball_boxes.append((box, conf))
                elif cls == CLASS_HAND and conf >= CONF_HAND: hand_boxes.append((box, conf))

        if not ball_boxes: continue
        ball_box = max(ball_boxes, key=lambda x:x[1])[0]
        bc = get_ball_center(ball_box)
        br = get_ball_radius(ball_box)

        # Pass 2: dots, on a zoomed crop of the ball.
        dots = detect_dots_with_zoom(frame, ball_box, model,
                                      debug_id=frame_idx,
                                      debug_folder=video_output_folder)
        print(f"  Frame {frame_idx}: Detected {len(dots)} dots with {ZOOM_FACTOR}x zoom")

        all_detections[frame_idx] = {
            'frame': frame.copy(), 'ball_center': bc, 'ball_radius': br,
            'dots': dots, 'ball_box': ball_box, 'hand_boxes': hand_boxes
        }

        if len(all_detections) == 4: break
    cap.release()

    if len(all_detections) < 4:
        print(f"  ERROR: Not enough frames ({len(all_detections)}/4)")
        return None

    # --- Tracking: chain the pairwise matches into tracks ---
    # Each match feeds its rotation forward as the next one's prior. A track is
    # a dict {frame_idx: dot_idx}; the goal is tracks spanning all 4 frames.
    active_tracks = []

    sorted_frames = sorted(all_detections.keys())

    # Seed one track per dot in the first frame.
    first_frame = sorted_frames[0]
    for i in range(len(all_detections[first_frame]['dots'])):
        active_tracks.append( {first_frame: i} )

    prev_R = None

    for i in range(len(sorted_frames) - 1):
        f_curr = sorted_frames[i]
        f_next = sorted_frames[i+1]

        data_curr = all_detections[f_curr]
        data_next = all_detections[f_next]

        matches, R, err = match_dots_rigid_body(
            data_curr['dots'], data_curr['ball_center'], data_curr['ball_radius'],
            data_next['dots'], data_next['ball_center'], data_next['ball_radius'],
            prior_R=prev_R
        )

        if R is not None:
            prev_R = R

        print(f"  Match {f_curr}->{f_next}: {len(matches)} pairs, Err={err:.3f}")

        curr_to_next = {c: n for c, n in matches}
        next_matched = set(n for c, n in matches)

        for track in active_tracks:
            if f_curr in track:
                dot_idx = track[f_curr]
                if dot_idx in curr_to_next:
                    track[f_next] = curr_to_next[dot_idx]

        # Unmatched dots start new tracks, but only if far enough from an
        # existing one: closer than that they are usually duplicate detections
        # or reflections, not new markers.
        MIN_DIST_FOR_NEW_DOT = data_next['ball_radius'] * 0.25

        for idx_next in range(len(data_next['dots'])):
            if idx_next not in next_matched:
                is_too_close = False
                new_dot_pos = np.array([data_next['dots'][idx_next][0],
                                       data_next['dots'][idx_next][1]])

                for i in range(len(data_next['dots'])):
                    if i != idx_next:
                        existing_dot_pos = np.array([data_next['dots'][i][0],
                                                     data_next['dots'][i][1]])
                        distance = np.linalg.norm(new_dot_pos - existing_dot_pos)
                        if distance < MIN_DIST_FOR_NEW_DOT:
                            is_too_close = True
                            break

                if not is_too_close and data_next['dots'][idx_next][2] >= CONF_DOT:
                    active_tracks.append({f_next: idx_next})

    # --- Window selection ---
    # A 4-frame window with >=3 dots wins outright: 3 dots over 4 frames beat
    # 6 dots over 3, because the wider time delta cuts the relative error more
    # than the extra dots do.
    print(f"  Total Tracks: {len(active_tracks)}")

    # Tracks present in both the first and the last frame.
    tracks_4frame = [t for t in active_tracks
                     if sorted_frames[0] in t and sorted_frames[-1] in t]

    final_tracks = []
    calculation_frames = []

    if len(tracks_4frame) >= 3:
        final_tracks = tracks_4frame
        calculation_frames = sorted_frames
        print(f"  ✓ SELECTED: 4-frame window {calculation_frames} with {len(tracks_4frame)} dots (high precision)")

    else:
        print(f"  ⚠ 4-frame window has only {len(tracks_4frame)} dots (<3). Searching 3-frame alternatives...")

        candidate_3frame = []

        for start_idx in range(len(sorted_frames) - 2):
            window_frames = sorted_frames[start_idx : start_idx + 3]
            first_f, last_f = window_frames[0], window_frames[-1]
            tracks_in_window = [t for t in active_tracks if first_f in t and last_f in t]

            candidate_3frame.append({
                'frames': window_frames,
                'tracks': tracks_in_window,
                'num_dots': len(tracks_in_window)
            })

        candidate_3frame.sort(key=lambda w: w['num_dots'], reverse=True)

        print(f"  3-frame window candidates:")
        for i, w in enumerate(candidate_3frame):
            print(f"    [{i}] {w['frames']}: {w['num_dots']} dots")

        best_3frame = None
        for window in candidate_3frame:
            if window['num_dots'] >= 3:
                best_3frame = window
                break

        if best_3frame:
            final_tracks = best_3frame['tracks']
            calculation_frames = best_3frame['frames']
            print(f"  ✓ SELECTED: 3-frame window {calculation_frames} with {best_3frame['num_dots']} dots (fallback)")
        else:
            # Last resort: accept 2 dots, flagged as low confidence.
            if len(tracks_4frame) >= 2:
                final_tracks = tracks_4frame
                calculation_frames = sorted_frames
                print(f"  ⚠ WARNING: Using 4-frame with only {len(tracks_4frame)} dots (low confidence)")
            elif candidate_3frame and candidate_3frame[0]['num_dots'] >= 2:
                best_3frame = candidate_3frame[0]
                final_tracks = best_3frame['tracks']
                calculation_frames = best_3frame['frames']
                print(f"  ⚠ WARNING: Using 3-frame with only {best_3frame['num_dots']} dots (low confidence)")
            else:
                print("  ✗ ERROR: No window with ≥2 tracked dots found")
                return None

    save_track_visualizations(sorted_frames, all_detections, final_tracks, video_output_folder)

    # calculation_frames, not sorted_frames: this is the window just selected.
    res = calculate_spin_first_last(final_tracks, all_detections, fps, calculation_frames)

    if not res['valid']:
        print("  ERROR: Spin calculation failed (not enough common dots between first and last frame).")
        return None

    make_bullseye([res['ez']], [res['ey']], [res['ex']],
                  os.path.join(video_output_folder, "bullseye.png"),
                  title=f"{video_name}",
                  num_tracked_dots=len(final_tracks),
                  spin_rate_hz=res['spin_rate_hz'])

    print(f"  Result (Frame {calculation_frames[0]}→{calculation_frames[-1]}): ex={res['ex']:.3f}, Hz={res['spin_rate_hz']:.2f}, Tracked dots: {len(final_tracks)}")

    return {
        'video': video_name,
        'release_frame': release_frame,
        'ex': res['ex'], 'ey': res['ey'], 'ez': res['ez'],
        'spin_rate_hz': res['spin_rate_hz'],
        'misalignment': res['theta_spin'],
        'quadrant': res['quadrant'],
        'rotation_deg': np.degrees(res['phi'])
    }
