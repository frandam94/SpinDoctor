"""Ball geometry: from pixel bounding box to a point on the unit sphere.

Two steps: `normalize_dot` maps a dot from pixels to ball-centred coordinates
in roughly [-1, 1], and `dot_to_unit_sphere` lifts those onto the visible
hemisphere (z >= 0) — the space where rigid matching and rotation estimation
happen.
"""

import numpy as np


def get_ball_center(box):
    x1, y1, x2, y2 = box
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


def get_ball_radius(box):
    x1, y1, x2, y2 = box
    return ((x2 - x1) + (y2 - y1)) / 4.0


def normalize_dot(x, y, cx, cy, R):
    """
    Convert pixel coordinates to ball-centered normalized coordinates (nx, ny).
    nx, ny are in approx range [-1, 1].
    """
    nx = (x - cx) / R
    ny = (y - cy) / R
    return nx, ny


def dot_to_unit_sphere(nx, ny):
    """
    Convert 2D normalized (nx, ny) to 3D point on unit sphere (u, v, z).
    Assumes visible hemisphere (z >= 0).
    """
    # Clamp so the sqrt below stays valid
    dist_sq = nx*nx + ny*ny
    if dist_sq > 1.0:
        scale = 1.0 / np.sqrt(dist_sq)
        nx *= scale
        ny *= scale
        dist_sq = 1.0

    z = np.sqrt(max(0.0, 1.0 - dist_sq))
    return np.array([nx, ny, z])
