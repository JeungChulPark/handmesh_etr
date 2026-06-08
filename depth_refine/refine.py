"""Depth-based 3D hand-joint refinement (no retraining).

Takes the frozen RGB model's root-relative metric joints + a monocular (inverse) depth
map and corrects the per-joint depth (Z). The monocular depth is aligned to the model's
OWN predicted depths (never to ground truth), so this is a fair inference-time refinement.

Pipeline per sample:
  1. project predicted absolute joints (J_rel + root) to 2D with the camera K
  2. sample the inverse-depth map at those 2D locations (robust window median)
  3. fit (1/Z_pred) ~= a*disp + b  -> convert monocular disparity to metric Z
  4. blend monocular Z with predicted Z (weight w), reject outliers
  5. optional bone-length constraint
  6. return refined root-relative joints (x,y kept, z corrected)
"""
import numpy as np

ROOT = 0  # wrist (FreiHAND center_idx = 0)

# 21-joint bone edges (MediaPipe topology, matches the model's joint order)
BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
FINGERTIPS = {4, 8, 12, 16, 20}


def cam2pixel(joints, K):
    """joints (N,3) camera-frame, K (3,3) -> (N,2) pixels."""
    z = np.where(np.abs(joints[:, 2]) < 1e-6, 1e-6, joints[:, 2])
    u = joints[:, 0] / z * K[0, 0] + K[0, 2]
    v = joints[:, 1] / z * K[1, 1] + K[1, 2]
    return np.stack([u, v], axis=1)


def sample_window_median(depth, pts, radius=2):
    """Robust sample of (H,W) map at (N,2) xy pixels via a (2r+1)^2 median."""
    H, W = depth.shape
    out = np.empty(len(pts), dtype=np.float32)
    for i, (x, y) in enumerate(pts):
        xi = int(round(np.clip(x, 0, W - 1)))
        yi = int(round(np.clip(y, 0, H - 1)))
        x0, x1 = max(0, xi - radius), min(W, xi + radius + 1)
        y0, y1 = max(0, yi - radius), min(H, yi + radius + 1)
        out[i] = np.median(depth[y0:y1, x0:x1])
    return out


def bone_length_constraint(J, ref_lengths, strength=0.5):
    """Pull each child onto its reference bone length along the current direction."""
    J = J.copy()
    for b, (p, c) in enumerate(BONES):
        d = J[c] - J[p]
        L = np.linalg.norm(d)
        if L < 1e-6 or ref_lengths[b] <= 0:
            continue
        corrected = J[p] + d * (ref_lengths[b] / L)
        J[c] = (1 - strength) * J[c] + strength * corrected
    return J


def refine_sample(J_rel, root, K, disp_map, w=0.5,
                  sample_radius=2, occ_thresh=0.03,
                  exclude_fingertips=True, bone_ref=None, bone_strength=0.0):
    """Refine one sample. Returns refined root-relative joints (21,3)."""
    J_rel = np.asarray(J_rel, dtype=np.float64)
    root = np.asarray(root, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)

    J_abs = J_rel + root[None, :]
    pts = cam2pixel(J_abs, K)
    disp = sample_window_median(disp_map, pts, radius=sample_radius).astype(np.float64)

    Z_pred = J_abs[:, 2]
    invZ_pred = 1.0 / np.where(Z_pred < 1e-6, 1e-6, Z_pred)

    # Which joints participate in the alignment fit (robust anchors only).
    fit_mask = np.isfinite(disp)
    if exclude_fingertips:
        for t in FINGERTIPS:
            fit_mask[t] = False
    if fit_mask.sum() < 4:
        return J_rel.astype(np.float32)  # not enough to align -> baseline

    # Fit 1/Z_pred ~= a*disp + b  (monocular disparity -> predictor metric scale)
    A = np.stack([disp[fit_mask], np.ones(fit_mask.sum())], axis=1)
    coef, *_ = np.linalg.lstsq(A, invZ_pred[fit_mask], rcond=None)
    a, b = coef
    invZ_mono = a * disp + b

    Z_ref = Z_pred.copy()
    eps = 1e-3
    for i in range(21):
        if exclude_fingertips and i in FINGERTIPS:
            continue
        if invZ_mono[i] <= eps or not np.isfinite(invZ_mono[i]):
            continue
        Z_mono = 1.0 / invZ_mono[i]
        if abs(Z_mono - Z_pred[i]) > occ_thresh:   # occlusion / outlier reject
            continue
        Z_ref[i] = w * Z_mono + (1 - w) * Z_pred[i]

    # Rebuild root-relative joints: keep model x,y, replace relative z.
    J_out = J_rel.copy()
    J_out[:, 2] = Z_ref - Z_ref[ROOT]

    if bone_ref is not None and bone_strength > 0:
        J_out = bone_length_constraint(J_out, bone_ref, strength=bone_strength)

    return J_out.astype(np.float32)


def monocular_rel_z(J_rel, root, K, disp_map, sample_radius=2, exclude_fingertips=True):
    """Diagnostic: monocular-derived root-relative Z per joint (aligned to predicted scale).

    Returns (z_rel_mono(21,), valid_mask(21,)). Lets us measure whether the monocular
    depth carries any independent signal about the true relative joint depth.
    """
    J_rel = np.asarray(J_rel, float); root = np.asarray(root, float); K = np.asarray(K, float)
    J_abs = J_rel + root[None, :]
    pts = cam2pixel(J_abs, K)
    disp = sample_window_median(disp_map, pts, radius=sample_radius).astype(float)
    Z_pred = J_abs[:, 2]
    invZ_pred = 1.0 / np.where(Z_pred < 1e-6, 1e-6, Z_pred)
    fit = np.isfinite(disp)
    if exclude_fingertips:
        for t in FINGERTIPS:
            fit[t] = False
    if fit.sum() < 4:
        return np.full(21, np.nan), np.zeros(21, bool)
    A = np.stack([disp[fit], np.ones(fit.sum())], axis=1)
    (a, b), *_ = np.linalg.lstsq(A, invZ_pred[fit], rcond=None)
    invZ_mono = a * disp + b
    valid = invZ_mono > 1e-3
    Z_mono = np.where(valid, 1.0 / np.where(valid, invZ_mono, 1.0), Z_pred)
    z_rel = Z_mono - Z_mono[ROOT]
    return z_rel, valid


def estimate_bone_lengths(J_gt_batch):
    """Mean bone lengths (meters) over a batch of GT joints (N,21,3) — used as reference."""
    lens = np.zeros(len(BONES))
    for b, (p, c) in enumerate(BONES):
        lens[b] = np.linalg.norm(J_gt_batch[:, c] - J_gt_batch[:, p], axis=1).mean()
    return lens
