"""Detection of the markers on the ball surface.

`detect_dots_with_zoom` crops the ball region, magnifies it with bicubic
interpolation and re-runs the model on the crop. The padding leaves semantic
context around the ball (floor, background): without it the model sees a
full-screen orange texture and degrades. If the zoom finds nothing — typically
because of motion blur — it falls back to the unmagnified crop.
"""

import os

import cv2
import numpy as np
from PIL import Image

from spindoctor.config import CLASS_DOT, CONF_DOT, ZOOM_FACTOR, ZOOM_PADDING


def detect_dots_with_zoom(frame, ball_box, model, zoom_factor=ZOOM_FACTOR, padding_factor=ZOOM_PADDING, debug_id=None, debug_folder=None):
    """
    Detect dots by cropping and magnifying the ball region.

    Args:
        frame: Original frame (numpy array, BGR)
        ball_box: Ball bounding box [x1, y1, x2, y2] in original coordinates
        model: Detection model
        zoom_factor: Magnification (3.0 = 3x)
        padding_factor: Extra space around the ball (2.2 = 120% padding)
        debug_id: Frame id, to name the debug image
        debug_folder: Where to save the debug image

    Returns:
        List of (cx, cy, conf, box) in ORIGINAL frame coordinates.
    """
    h_frame, w_frame = frame.shape[:2]
    x1, y1, x2, y2 = ball_box

    ball_w = x2 - x1
    ball_h = y2 - y1

    pad_w = ball_w * (padding_factor - 1.0) / 2.0
    pad_h = ball_h * (padding_factor - 1.0) / 2.0

    # Clamped to the frame bounds.
    crop_x1 = int(max(0, x1 - pad_w))
    crop_y1 = int(max(0, y1 - pad_h))
    crop_x2 = int(min(w_frame, x2 + pad_w))
    crop_y2 = int(min(h_frame, y2 + pad_h))

    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

    if crop.size == 0:
        return []

    crop_h, crop_w = crop.shape[:2]
    zoom_w = int(crop_w * zoom_factor)
    zoom_h = int(crop_h * zoom_factor)

    # INTER_CUBIC: upscaling quality matters, the dots are only pixels wide.
    zoomed = cv2.resize(crop, (zoom_w, zoom_h), interpolation=cv2.INTER_CUBIC)

    # Save exactly what the model is about to see.
    if debug_id is not None and debug_folder is not None:
        debug_filename = os.path.join(debug_folder, f"debug_zoom_input_{debug_id}.jpg")
        cv2.imwrite(debug_filename, zoomed)
        print(f"    [DEBUG] Saved zoom input: {debug_filename} (size: {zoomed.shape[1]}x{zoomed.shape[0]})")

    zoomed_rgb = Image.fromarray(cv2.cvtColor(zoomed, cv2.COLOR_BGR2RGB))
    detections = model.predict(zoomed_rgb, threshold=0.2)

    dots = []
    if detections is not None:
        for i in range(len(detections)):
            cls_id = detections.class_id[i]
            conf = detections.confidence[i]
            box_zoom = detections.xyxy[i].tolist()

            if cls_id == CLASS_DOT and conf >= CONF_DOT:
                # Back to frame coordinates: unscale the zoom...
                bx1_crop = box_zoom[0] / zoom_factor
                by1_crop = box_zoom[1] / zoom_factor
                bx2_crop = box_zoom[2] / zoom_factor
                by2_crop = box_zoom[3] / zoom_factor

                # ...then offset by the crop's position in the frame.
                bx1_orig = bx1_crop + crop_x1
                by1_orig = by1_crop + crop_y1
                bx2_orig = bx2_crop + crop_x1
                by2_orig = by2_crop + crop_y1

                box_orig = [bx1_orig, by1_orig, bx2_orig, by2_orig]
                cx = int((bx1_orig + bx2_orig) / 2)
                cy = int((by1_orig + by2_orig) / 2)

                dots.append((cx, cy, conf, box_orig))

    # Fallback on the unmagnified crop: with motion blur, zooming amplifies the
    # smear and the model does better without it.
    if len(dots) == 0:
        crop_rgb = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        detections_fallback = model.predict(crop_rgb, threshold=0.2)

        if detections_fallback is not None:
            for i in range(len(detections_fallback)):
                cls_id = detections_fallback.class_id[i]
                conf = detections_fallback.confidence[i]
                box_crop = detections_fallback.xyxy[i].tolist()

                if cls_id == CLASS_DOT and conf >= CONF_DOT:
                    bx1_orig = box_crop[0] + crop_x1
                    by1_orig = box_crop[1] + crop_y1
                    bx2_orig = box_crop[2] + crop_x1
                    by2_orig = box_crop[3] + crop_y1

                    box_orig = [bx1_orig, by1_orig, bx2_orig, by2_orig]
                    cx = int((bx1_orig + bx2_orig) / 2)
                    cy = int((by1_orig + by2_orig) / 2)

                    dots.append((cx, cy, conf, box_orig))

    return dots
