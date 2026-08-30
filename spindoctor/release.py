"""Release-frame detection.

Spin must be measured just after the ball leaves the hand: before that, contact
distorts the rotation; later, air resistance does.

The primary criterion is the LAST hand-ball contact (bounding-box overlap > 5%).
The whole video is scanned rather than stopping at the first contact, because a
clip usually contains a reception and some dribbling before the actual shot. If
no contact is found, it falls back to the lowest point of the ball trajectory.
"""

import os

import cv2
import numpy as np
from PIL import Image

from spindoctor.config import (
    CLASS_BALL, CLASS_HAND, CONF_BALL, CONF_HAND, SKIP_INITIAL_FRAMES,
)
from spindoctor.utils import ensure_dir


def find_release_frame(video_path, model, device, debug_folder=None):
    """
    Find the release frame by scanning the whole video for the last hand-ball
    contact, falling back to the lowest point of the trajectory.

    Args:
        video_path: Path to video file
        model: Detection model
        device: Device for inference
        debug_folder: If set, saves annotated frames of the search

    Returns:
        (release_frame, fps); release_frame is None if nothing was found.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    last_contact_frame = None
    frame_idx = 0
    ball_positions = []

    if debug_folder:
        debug_path = os.path.join(debug_folder, "debug_release_frames")
        ensure_dir(debug_path)
        print(f"  [DEBUG] Saving release search frames to: {debug_path}")

    print(f"  Searching for release frame (Total video frames: {total_frames})...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx < SKIP_INITIAL_FRAMES:
            continue

        debug_frame = frame.copy() if debug_folder else None

        rgb_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # Low threshold so the debug view shows everything the model saw; the
        # real thresholds are applied during selection below.
        detections = model.predict(rgb_image, threshold=0.1)

        ball_box = None
        hand_boxes = []  # (box, conf) tuples

        if detections is not None:
            for i in range(len(detections)):
                cls_id = detections.class_id[i]
                conf = detections.confidence[i]
                box = detections.xyxy[i].tolist()

                # --- Debug view: every detection above 0.1 ---
                if debug_folder:
                    x1, y1, x2, y2 = [int(c) for c in box]
                    label = f"{conf:.2f}"
                    color = (128, 128, 128)

                    if cls_id == CLASS_BALL:
                        label = f"BALL {conf:.2f}"
                        color = (0, 255, 0)
                    elif cls_id == CLASS_HAND:
                        label = f"HAND {conf:.2f}"
                        color = (0, 0, 255)

                    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 1)
                    cv2.putText(debug_frame, label, (x1, y1-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                # --- Selection, at the real thresholds ---
                if cls_id == CLASS_BALL and conf >= CONF_BALL:
                    if ball_box is None or conf > ball_box[1]:
                        ball_box = (box, conf)
                elif cls_id == CLASS_HAND and conf >= CONF_HAND:
                    hand_boxes.append((box, conf))

        if ball_box:
            bx1, by1, bx2, by2 = ball_box[0]
            bc_y = (by1 + by2) / 2
            ball_positions.append((frame_idx, bc_y))

            # The selected ball, drawn thicker than the candidates.
            if debug_folder:
                x1, y1, x2, y2 = [int(c) for c in ball_box[0]]
                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(debug_frame, f"SELECTED BALL {ball_box[1]:.2f}",
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            is_touching = False
            for hbox, hconf in hand_boxes:
                hx1, hy1, hx2, hy2 = hbox

                x_left = max(bx1, hx1)
                y_top = max(by1, hy1)
                x_right = min(bx2, hx2)
                y_bottom = min(by2, hy2)

                if x_right > x_left and y_bottom > y_top:
                    intersection_area = (x_right - x_left) * (y_bottom - y_top)
                    ball_area = (bx2 - bx1) * (by2 - by1)
                    overlap_pct = intersection_area / ball_area

                    if debug_folder:
                        hx1_i, hy1_i, hx2_i, hy2_i = [int(c) for c in hbox]
                        # Yellow when touching, red when not.
                        h_color = (0, 255, 255) if overlap_pct > 0.05 else (0, 0, 255)
                        cv2.rectangle(debug_frame, (hx1_i, hy1_i), (hx2_i, hy2_i), h_color, 2)
                        cv2.putText(debug_frame, f"HAND IoU: {overlap_pct:.3f}",
                                   (hx1_i, hy2_i+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, h_color, 1)

                    if overlap_pct > 0.05:
                        is_touching = True

            if is_touching:
                # Overwritten on every contact, so it ends up holding the last.
                last_contact_frame = frame_idx
                if debug_folder:
                    cv2.putText(debug_frame, "CONTACT!", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        # No early exit when the ball goes missing: it often disappears during
        # dribbling and reappears for the actual shot, so the scan runs to the
        # end to be sure the last contact is the one that counts.

        if debug_folder:
            cv2.putText(debug_frame, f"Frame: {frame_idx}", (10, debug_frame.shape[0]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if last_contact_frame is not None:
                cv2.putText(debug_frame, f"Last Contact: {last_contact_frame}",
                           (10, debug_frame.shape[0]-35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if debug_folder:
            cv2.imwrite(os.path.join(debug_path, f"frame_{frame_idx:04d}.jpg"), debug_frame)

    cap.release()

    if last_contact_frame is not None:
        release_frame = last_contact_frame + 2
        print(f"  Release detected at frame {last_contact_frame} (scanned {frame_idx} frames). Start tracking at {release_frame}.")
        return release_frame, fps

    # Fallback: lowest point of the trajectory.
    if len(ball_positions) > 10:
        y_coords = [p[1] for p in ball_positions]
        # Max y, because image coordinates grow downward.
        min_y_idx = np.argmax(y_coords)
        release_frame = ball_positions[min_y_idx][0] + 3
        print(f"  Release fallback (vertical movement) at frame {release_frame}.")
        return release_frame, fps

    return None, fps
