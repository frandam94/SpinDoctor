"""Spin axis and rate estimation from the tracked dots.

`calculate_spin_first_last` solves Wahba's problem: given correspondences
between unit vectors at two instants, find the rotation that aligns them best
in a least-squares sense. It uses Davenport's q-method, building the 4x4 K
matrix and extracting its dominant eigenvector — the optimal quaternion — by
power iteration.

The rotation is measured between the FIRST and LAST frame of the window rather
than step by step: the same measurement noise spans a larger angle, so the
relative error is smaller.

Sign anchoring on the mean cross product keeps the axis pointing consistently
across shots. Without it the sign would be arbitrary and aggregated plots would
scatter meaninglessly.
"""

import numpy as np

from spindoctor.geometry import normalize_dot, dot_to_unit_sphere


def calculate_spin_first_last(track_list, all_detections, fps, frames_to_use):
    """
    Spin via Davenport's q-method, with sign anchoring so the axis stays
    consistent across shots.

    Args:
        track_list: List of track dicts {frame_idx: dot_idx, ...}
        all_detections: Dict of frame data
        fps: Frames per second
        frames_to_use: Sorted frame indices (e.g. [50, 51, 52])

    Returns:
        Dict of spin parameters, or {'valid': False}
    """
    first_frame_idx = frames_to_use[0]
    last_frame_idx = frames_to_use[-1]

    # Intervals, not frames: n frames span n-1 gaps.
    time_delta = (len(frames_to_use) - 1) / fps

    if time_delta <= 0:
        return {'valid': False}

    # Keep only dots the tracks follow all the way from first to last frame.
    vecs_start = []
    vecs_end = []

    for track in track_list:
        if first_frame_idx in track and last_frame_idx in track:
            idx_start = track[first_frame_idx]
            idx_end = track[last_frame_idx]

            d1 = all_detections[first_frame_idx]['dots'][idx_start]
            c1 = all_detections[first_frame_idx]['ball_center']
            r1 = all_detections[first_frame_idx]['ball_radius']

            d2 = all_detections[last_frame_idx]['dots'][idx_end]
            c2 = all_detections[last_frame_idx]['ball_center']
            r2 = all_detections[last_frame_idx]['ball_radius']

            nx1, ny1 = normalize_dot(d1[0], d1[1], c1[0], c1[1], r1)
            nx2, ny2 = normalize_dot(d2[0], d2[1], c2[0], c2[1], r2)

            vecs_start.append(dot_to_unit_sphere(nx1, ny1))
            vecs_end.append(dot_to_unit_sphere(nx2, ny2))

    if len(vecs_start) < 2:
        print(f"  WARNING: Only {len(vecs_start)} tracks span from frame {first_frame_idx} to {last_frame_idx}. Need at least 2.")
        return {'valid': False}

    # --- Davenport's q-method ---
    # Step 1: correlation matrix B = sum(v2 * v1^T), all dots weighted equally.
    B = np.zeros((3, 3), dtype=float)
    cross_sum = np.zeros(3)  # accumulated for sign anchoring

    for i in range(len(vecs_start)):
        v1 = vecs_start[i]
        v2 = vecs_end[i]

        B += np.outer(v2, v1)
        cross_sum += np.cross(v1, v2)

    # Step 2: the 4x4 symmetric K matrix, from B's trace, symmetric part and
    # antisymmetric vector.
    sigma = np.trace(B)
    S = B + B.T
    Z = np.array([
        B[1, 2] - B[2, 1],
        B[2, 0] - B[0, 2],
        B[0, 1] - B[1, 0]
    ])

    K = np.zeros((4, 4))
    K[0, 0] = sigma
    K[0, 1:4] = Z
    K[1:4, 0] = Z

    for r in range(3):
        for c in range(3):
            K[r+1, c+1] = S[r, c]
            if r == c:
                K[r+1, c+1] -= sigma

    # Step 3: dominant eigenvector of K, i.e. the optimal quaternion.
    q = np.array([1.0, 0.0, 0.0, 0.0])

    for iteration in range(50):
        q_new = K @ q
        mag = np.linalg.norm(q_new)
        if mag < 1e-12:
            return {'valid': False}
        q = q_new / mag

    # Step 4: the axis is the quaternion's vector part.
    spin_axis = np.array([q[1], q[2], q[3]])
    axis_mag = np.linalg.norm(spin_axis)

    if axis_mag < 1e-12:
        spin_axis = np.array([1.0, 0.0, 0.0])
    else:
        spin_axis = spin_axis / axis_mag

    # Step 5: anchor the sign on the mean cross product, so the axis points the
    # same way from shot to shot.
    cross_mag = np.linalg.norm(cross_sum)
    if cross_mag > 1e-8:
        dot_product = np.dot(spin_axis, cross_sum)
        if dot_product < 0:
            spin_axis = -spin_axis

    total_phi = 2.0 * np.arccos(np.clip(q[0], -1.0, 1.0))
    spin_rate_hz = total_phi / (2 * np.pi * time_delta)

    ex, ey, ez = spin_axis
    theta_spin = np.degrees(np.arccos(np.clip(ex, -1.0, 1.0)))

    if ey >= 0 and ez >= 0: quadrant = "I"
    elif ey >= 0 and ez < 0: quadrant = "II"
    elif ey < 0 and ez < 0: quadrant = "III"
    else: quadrant = "IV"

    return {
        'valid': True, 'spin_axis': spin_axis,
        'ex': ex, 'ey': ey, 'ez': ez, 'phi': total_phi,
        'theta_spin': theta_spin, 'spin_rate_hz': spin_rate_hz,
        'quadrant': quadrant
    }
