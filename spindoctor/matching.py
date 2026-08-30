"""Rigid-body dot matching between two consecutive frames.

The heart of the tracking: given the dots detected in two frames, find the
correct correspondence and the 3D rotation relating them, using four physical
constraints:

1. Geometric rigidity: pairwise distances on the sphere are rotation-invariant.
2. Kinematic limit: rotation between two frames at 60fps cannot exceed
   MAX_ROTATION_DEG, which rules out identity swaps.
3. Minimum displacement cost (Occam's razor): at equal inlier count, the
   hypothesis where points moved least wins.
4. Inertial prior: angular momentum is conserved, so the previous rotation
   predicts the current one and short-circuits the combinatorial search.

If STRICT matching fails, it retries with LOOSE parameters.
"""

import itertools

import numpy as np

from spindoctor.config import (
    DOWNWARD_EPS, MAX_STEP, MAX_ROTATION_DEG,
    INLIER_THRESH_STRICT, DIST_TOLERANCE_STRICT, MIN_INLIERS_STRICT,
    INLIER_THRESH_LOOSE, DIST_TOLERANCE_LOOSE, MIN_INLIERS_LOOSE,
)
from spindoctor.geometry import normalize_dot, dot_to_unit_sphere


def match_dots_rigid_body(dots_A, center_A, radius_A,
                          dots_B, center_B, radius_B,
                          prior_R=None):
    """
    Match dots between two frames, STRICT first and LOOSE as fallback.

    Args:
        prior_R: Optional 3x3 rotation from the previous frame pair.

    Returns:
        matches: list of (index_in_A, index_in_B)
        R: 3x3 rotation matrix
        error: residual error
    """
    # Stage 1: STRICT, high precision.
    matches, R, err = _match_dots_internal(
        dots_A, center_A, radius_A,
        dots_B, center_B, radius_B,
        prior_R=prior_R,
        inlier_thresh=INLIER_THRESH_STRICT,
        dist_tolerance=DIST_TOLERANCE_STRICT,
        min_inliers=MIN_INLIERS_STRICT
    )

    if len(matches) >= MIN_INLIERS_STRICT:
        return matches, R, err

    # Stage 2: LOOSE, for difficult frames.
    matches_loose, R_loose, err_loose = _match_dots_internal(
        dots_A, center_A, radius_A,
        dots_B, center_B, radius_B,
        prior_R=prior_R,
        inlier_thresh=INLIER_THRESH_LOOSE,
        dist_tolerance=DIST_TOLERANCE_LOOSE,
        min_inliers=MIN_INLIERS_LOOSE
    )

    if len(matches_loose) >= MIN_INLIERS_LOOSE:
        return matches_loose, R_loose, err_loose

    return [], None, float('inf')


def _match_dots_internal(dots_A, center_A, radius_A,
                         dots_B, center_B, radius_B,
                         prior_R=None,
                         inlier_thresh=0.08,
                         dist_tolerance=0.10,
                         min_inliers=3):
    """Matching with configurable thresholds; driven by match_dots_rigid_body."""
    # --- 1. Normalization ---
    ptsA = []
    for i, (x, y, conf, box) in enumerate(dots_A):
        nx, ny = normalize_dot(x, y, center_A[0], center_A[1], radius_A)
        # nx, ny are kept for the 2D displacement cost below.
        ptsA.append({'idx': i, 'nx': nx, 'ny': ny,
                     'vec3': dot_to_unit_sphere(nx, ny), 'conf': conf})

    ptsB = []
    for i, (x, y, conf, box) in enumerate(dots_B):
        nx, ny = normalize_dot(x, y, center_B[0], center_B[1], radius_B)
        ptsB.append({'idx': i, 'nx': nx, 'ny': ny,
                     'vec3': dot_to_unit_sphere(nx, ny), 'conf': conf})

    nA, nB = len(ptsA), len(ptsB)
    if nA < 2 or nB < 2:
        return [], None, float('inf')

    # --- 2. Fast track: inertial prior ---
    # Angular momentum is conserved, so the previous rotation predicts where
    # the points should land.
    if prior_R is not None:
        all_vecsA = np.array([p['vec3'] for p in ptsA])
        predicted_vecsA = (prior_R @ all_vecsA.T).T

        prior_inliers = []
        prior_error = 0
        used_B_prior = [False] * nB

        # Do the predicted positions match the actual ones in frame B?
        for i in range(nA):
            best_j = -1
            min_dist = float('inf')
            for j in range(nB):
                if not used_B_prior[j]:
                    d = np.linalg.norm(predicted_vecsA[i] - ptsB[j]['vec3'])
                    if d < min_dist:
                        min_dist = d
                        best_j = j

            # Relaxed threshold: 10% tolerance for jitter and acceleration.
            if best_j != -1 and min_dist < 0.10:
                prior_inliers.append((ptsA[i]['idx'], ptsB[best_j]['idx']))
                prior_error += min_dist
                used_B_prior[best_j] = True

        # Two consistent points are enough to confirm the prior's rotation, so
        # accept it and skip the expensive combinatorial search.
        if len(prior_inliers) >= 2:
            return prior_inliers, prior_R, prior_error

    # --- 3. Validity matrix (directional gating) ---
    is_valid = np.zeros((nA, nB), dtype=bool)
    for i in range(nA):
        for j in range(nB):
            # Backspin: a point should move downward.
            if (ptsB[j]['ny'] >= ptsA[i]['ny'] - DOWNWARD_EPS) and \
               (ptsB[j]['ny'] <= ptsA[i]['ny'] + MAX_STEP):
                is_valid[i, j] = True

    # --- 4. Distance matrices, for the rigidity check ---
    dist_mat_A = np.zeros((nA, nA))
    for i in range(nA):
        for j in range(i+1, nA):
            d = np.linalg.norm(ptsA[i]['vec3'] - ptsA[j]['vec3'])
            dist_mat_A[i, j] = dist_mat_A[j, i] = d

    dist_mat_B = np.zeros((nB, nB))
    for i in range(nB):
        for j in range(i+1, nB):
            d = np.linalg.norm(ptsB[i]['vec3'] - ptsB[j]['vec3'])
            dist_mat_B[i, j] = dist_mat_B[j, i] = d

    # Score is (inliers, -displacement): tuple ordering then maximizes inliers
    # and minimizes displacement in one comparison.
    best_score_tuple = (-1, float('inf'))
    best_pairs = []
    best_R = None
    best_error = float('inf')

    indices_A = list(range(nA))
    indices_B = list(range(nB))

    # --- 5. Hypotheses: triplets, or pairs when points are scarce ---
    use_triplets = (nA >= 3 and nB >= 3)
    iter_A = itertools.combinations(indices_A, 3 if use_triplets else 2)

    for subset_A_idxs in iter_A:
        permutations_B = itertools.permutations(indices_B, len(subset_A_idxs))

        for subset_B_idxs in permutations_B:
            # Rigidity check: dist(A1, A2) must match dist(B1, B2).
            consistent_geometry = True

            for i in range(len(subset_A_idxs)):
                for j in range(i + 1, len(subset_A_idxs)):
                    idx_a1, idx_a2 = subset_A_idxs[i], subset_A_idxs[j]
                    idx_b1, idx_b2 = subset_B_idxs[i], subset_B_idxs[j]

                    if not is_valid[idx_a1, idx_b1] or not is_valid[idx_a2, idx_b2]:
                        consistent_geometry = False
                        break

                    dist_A = dist_mat_A[idx_a1, idx_a2]
                    dist_B = dist_mat_B[idx_b1, idx_b2]

                    if abs(dist_A - dist_B) > dist_tolerance:
                        consistent_geometry = False
                        break

                if not consistent_geometry:
                    break

            if not consistent_geometry:
                continue

            # --- Rotation from the anchor points (Kabsch) ---
            vecsA = np.array([ptsA[i]['vec3'] for i in subset_A_idxs])
            vecsB = np.array([ptsB[i]['vec3'] for i in subset_B_idxs])

            H = vecsA.T @ vecsB
            U, _, Vt = np.linalg.svd(H)
            R_hyp = Vt.T @ U.T
            if np.linalg.det(R_hyp) < 0:
                Vt[-1, :] *= -1
                R_hyp = Vt.T @ U.T

            # --- Kinematic constraint: reject impossible rotations ---
            # Angle from the trace; clipped for arccos numerical safety.
            trace = np.clip(np.trace(R_hyp), -1.0, 3.0)
            theta_deg = np.degrees(np.arccos((trace - 1.0) / 2.0))

            # Too large a rotation means a jump or an identity swap, not motion.
            if theta_deg > MAX_ROTATION_DEG:
                continue

            # --- Consensus: inliers plus displacement cost ---
            current_inliers = []
            current_reproj_err = 0
            current_displacement_cost = 0

            # Project every point of A through the candidate rotation.
            all_vecsA = np.array([p['vec3'] for p in ptsA])
            rotated_A = (R_hyp @ all_vecsA.T).T

            used_B = [False] * nB

            # Greedy matching, counting inliers and accumulating costs.
            for i in range(nA):
                best_j = -1
                min_dist_3d = float('inf')

                for j in range(nB):
                    if not used_B[j] and is_valid[i, j]:
                        d = np.linalg.norm(rotated_A[i] - ptsB[j]['vec3'])
                        if d < min_dist_3d:
                            min_dist_3d = d
                            best_j = j

                if best_j != -1 and min_dist_3d < inlier_thresh:
                    current_inliers.append((ptsA[i]['idx'], ptsB[best_j]['idx']))
                    current_reproj_err += min_dist_3d
                    used_B[best_j] = True

                    # Displacement in the normalized plane, which penalizes
                    # jumps from one side of the ball to the other.
                    disp_2d = np.hypot(ptsA[i]['nx'] - ptsB[best_j]['nx'],
                                      ptsA[i]['ny'] - ptsB[best_j]['ny'])
                    current_displacement_cost += disp_2d

            n_inliers = len(current_inliers)

            # Negated cost so a plain tuple comparison maximizes inliers first,
            # then prefers the least movement — which is what avoids swaps.
            current_score_tuple = (n_inliers, -current_displacement_cost)

            if current_score_tuple > best_score_tuple:
                best_score_tuple = current_score_tuple
                best_pairs = current_inliers
                best_R = R_hyp
                best_error = current_reproj_err

    # Below the inlier floor the match is noise, not signal.
    if best_score_tuple[0] < min_inliers:
        return [], None, float('inf')

    return best_pairs, best_R, best_error
