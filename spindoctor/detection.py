"""RF-DETR model loading and the visual debug mode.

`load_model` centralises detector instantiation; `debug_all_frames` re-runs the
model over every frame at a low threshold (0.1) and saves the annotated frames,
to see where detection is failing on a problematic clip.
"""

import os

import cv2
from PIL import Image
from rfdetr import RFDETRMedium

from spindoctor.config import (
    CLASS_BALL, CLASS_DOT, CLASS_HAND, CONF_BALL, CONF_DOT, CONF_HAND,
    DEFAULT_DEVICE, DEFAULT_WEIGHTS,
)
from spindoctor.utils import ensure_dir


def load_model(weights=DEFAULT_WEIGHTS, device=DEFAULT_DEVICE):
    """
    Load the RF-DETR detector fine-tuned on the ball/dots/hands dataset.

    Args:
        weights: Checkpoint path (default: rf-detr/weights.pt)
        device: Inference device (default: cpu)

    Returns:
        (model, device)
    """
    print("Loading model...")
    model = RFDETRMedium(pretrain_weights=weights)
    model.device = device
    return model, device


def debug_all_frames(video_path, model, device, output_folder, clean_viz=False):
    """
    Save every frame with the detection boxes drawn on, for troubleshooting.

    Args:
        video_path: Path to video file
        model: Detection model
        device: Device for inference
        output_folder: Folder to save debug frames
        clean_viz: Draw only the rectangles, with no confidence labels and no
                   info overlay (for GIFs and presentations)
    """
    debug_path = os.path.join(output_folder, "debug_all_frames")
    ensure_dir(debug_path)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n[FULL DEBUG MODE]")
    print(f"  Video: {os.path.basename(video_path)}")
    print(f"  Total frames: {total_frames}, FPS: {fps}")
    print(f"  Saving all frames to: {debug_path}")

    frame_idx = 0
    stats = {'ball': 0, 'dots': 0, 'hands': 0}

    from tqdm import tqdm
    pbar = tqdm(total=total_frames, desc="  Processing frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        pbar.update(1)

        rgb_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # Low threshold: the point of this mode is to see everything.
        detections = model.predict(rgb_image, threshold=0.1)

        debug_frame = frame.copy()

        num_balls = 0
        num_dots = 0
        num_hands = 0

        if detections is not None:
            for i in range(len(detections)):
                cls_id = detections.class_id[i]
                conf = detections.confidence[i]
                box = detections.xyxy[i].tolist()

                x1, y1, x2, y2 = [int(c) for c in box]

                # Colour by class; dimmer and thinner below the threshold, so
                # near-misses are visible without being mistaken for keepers.
                if cls_id == CLASS_BALL:
                    color = (0, 255, 0) if conf >= CONF_BALL else (0, 200, 0)
                    label = f"BALL {conf:.2f}"
                    thickness = 3 if conf >= CONF_BALL else 1
                    num_balls += 1

                elif cls_id == CLASS_DOT:
                    color = (255, 255, 0) if conf >= CONF_DOT else (150, 150, 0)
                    label = f"DOT {conf:.2f}"
                    thickness = 2 if conf >= CONF_DOT else 1
                    num_dots += 1

                elif cls_id == CLASS_HAND:
                    color = (0, 0, 255) if conf >= CONF_HAND else (0, 0, 150)
                    label = f"HAND {conf:.2f}"
                    thickness = 2 if conf >= CONF_HAND else 1
                    num_hands += 1
                else:
                    color = (128, 128, 128)
                    label = f"? {conf:.2f}"
                    thickness = 1

                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, thickness)

                if not clean_viz:
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.rectangle(debug_frame, (x1, y1 - label_size[1] - 4),
                                (x1 + label_size[0], y1), color, -1)
                    cv2.putText(debug_frame, label, (x1, y1 - 2),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if num_balls > 0:
            stats['ball'] += 1
        if num_dots > 0:
            stats['dots'] += 1
        if num_hands > 0:
            stats['hands'] += 1

        if not clean_viz:
            cv2.rectangle(debug_frame, (0, 0), (300, 100), (0, 0, 0), -1)
            cv2.putText(debug_frame, f"Frame: {frame_idx}/{total_frames}", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(debug_frame, f"Ball: {num_balls}  Dots: {num_dots}  Hands: {num_hands}",
                       (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(debug_frame, f"Threshold: Ball={CONF_BALL} Dot={CONF_DOT}",
                       (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imwrite(os.path.join(debug_path, f"frame_{frame_idx:05d}.jpg"), debug_frame)

    cap.release()
    pbar.close()

    print(f"\n  Statistics:")
    print(f"    Frames with ball detected: {stats['ball']}/{total_frames} ({100*stats['ball']/total_frames:.1f}%)")
    print(f"    Frames with dots detected: {stats['dots']}/{total_frames} ({100*stats['dots']/total_frames:.1f}%)")
    print(f"    Frames with hands detected: {stats['hands']}/{total_frames} ({100*stats['hands']/total_frames:.1f}%)")
    print(f"  Debug frames saved to: {debug_path}\n")
