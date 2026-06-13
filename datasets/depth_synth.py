# Copyright (c) 2026. Depth-Aware Hand Pose Initiative.
"""
Synthetic depth generation for RGB-D hand-pose retraining.

Background
----------
The `depth_refine/` study concluded that RGB-inferred monocular depth is a dead
end, while a *real* metric depth sensor has a large (but conditional) upper bound
(MPJPE 31.1 -> 14.2 mm, -54%). The surest path forward is to *train* an RGB-D
model on paired depth. HanCo (the training set) ships only multi-view mocap GT
(21 camera-space joints), not sensor depth, so here we synthesise a metric depth
channel directly from the GT geometry and corrupt it with a configurable sensor
noise model. That lets us train/validate the RGB-D architecture today and study
how much sensor quality is needed, before committing to a real paired-depth
capture.

Two geometry sources are supported by the same rasteriser:
  * MANO mesh  (verts [778,3] + faces [1538,3])  -- the proper "GT mesh render".
  * 21 joints  -> a coarse tapered-capsule hand mesh (fallback when MANO params
    are unavailable, as in the current HanCo loader).

All coordinates are camera-space metres (x right, y down, z forward) consistent
with the project's `cam2pixel` convention: u = X*fx/Z + cx, v = Y*fy/Z + cy.

This file has NO project dependencies (numpy only) and a self-test in __main__,
so the rasteriser can be sanity-checked without any dataset mounted:
    python -m datasets.depth_synth
"""

import numpy as np

# ----------------------------------------------------------------------------
# Hand skeleton (FreiHAND / MANO 21-keypoint order, shared by HanCo)
#   0: wrist
#   thumb  1-2-3-4   index 5-6-7-8   middle 9-10-11-12
#   ring   13-14-15-16            pinky 17-18-19-20
# ----------------------------------------------------------------------------
HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
]
# Palm bones get a fatter radius than fingers so the capsule mesh reads as a hand.
_PALM_BONES = {(0, 1), (0, 5), (0, 9), (0, 13), (0, 17)}


# ----------------------------------------------------------------------------
# Geometry -> mesh
# ----------------------------------------------------------------------------
def joints_to_capsule_mesh(joints, base_radius=0.009, palm_radius=0.014,
                           tip_scale=0.55, ring=6):
    """Build a tapered-tube ("capsule") triangle mesh skinned on a 21-joint hand.

    Args:
        joints: (21, 3) camera-space metres.
        base_radius: tube radius at the proximal end of a finger bone (m).
        palm_radius: radius for palm/metacarpal bones (m).
        tip_scale: distal-end radius = proximal radius * tip_scale (taper).
        ring: vertices around each tube cross-section (>=3).

    Returns:
        verts (V, 3) float32, faces (F, 3) int32.
    """
    joints = np.asarray(joints, dtype=np.float64)
    verts, faces = [], []
    for (a, b) in HAND_BONES:
        p0, p1 = joints[a], joints[b]
        axis = p1 - p0
        length = np.linalg.norm(axis)
        if length < 1e-6:
            continue
        zc = axis / length
        # orthonormal frame around the bone axis
        tmp = np.array([0.0, 0.0, 1.0]) if abs(zc[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
        xc = np.cross(tmp, zc)
        xc /= (np.linalg.norm(xc) + 1e-9)
        yc = np.cross(zc, xc)

        r_prox = palm_radius if (a, b) in _PALM_BONES else base_radius
        r_dist = r_prox * tip_scale
        base = len(verts)
        angles = 2.0 * np.pi * np.arange(ring) / ring
        for ang in angles:                                  # proximal ring
            d = np.cos(ang) * xc + np.sin(ang) * yc
            verts.append(p0 + d * r_prox)
        for ang in angles:                                  # distal ring
            d = np.cos(ang) * xc + np.sin(ang) * yc
            verts.append(p1 + d * r_dist)
        for i in range(ring):                               # side quads -> 2 tris
            j = (i + 1) % ring
            v00, v01 = base + i, base + j
            v10, v11 = base + ring + i, base + ring + j
            faces.append((v00, v10, v11))
            faces.append((v00, v11, v01))

    if not verts:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32)
    return np.asarray(verts, np.float32), np.asarray(faces, np.int32)


# ----------------------------------------------------------------------------
# Mesh -> depth (z-buffer rasteriser)
# ----------------------------------------------------------------------------
def rasterize_mesh_depth(verts, faces, K, height, width):
    """Render a metric depth map (metres) by z-buffering triangles.

    Background pixels are 0. Works for MANO meshes and capsule meshes alike.

    Args:
        verts: (V, 3) camera-space metres (z forward, >0 in front of camera).
        faces: (F, 3) int triangle indices.
        K: (3, 3) intrinsics for the *target* image resolution.
        height, width: output size.

    Returns:
        depth (height, width) float32, 0 = background.
    """
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    zbuf = np.full((height, width), np.inf, dtype=np.float64)
    if verts.shape[0] == 0 or faces.shape[0] == 0:
        return np.zeros((height, width), np.float32)

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    z = verts[:, 2]
    z_safe = np.where(z <= 1e-6, 1e-6, z)
    u = verts[:, 0] * fx / z_safe + cx
    v = verts[:, 1] * fy / z_safe + cy

    for f in faces:
        i0, i1, i2 = f
        if z[i0] <= 1e-6 or z[i1] <= 1e-6 or z[i2] <= 1e-6:
            continue  # any vertex behind the camera -> skip
        x0, y0 = u[i0], v[i0]
        x1, y1 = u[i1], v[i1]
        x2, y2 = u[i2], v[i2]

        minx = int(np.floor(min(x0, x1, x2)))
        maxx = int(np.ceil(max(x0, x1, x2)))
        miny = int(np.floor(min(y0, y1, y2)))
        maxy = int(np.ceil(max(y0, y1, y2)))
        minx = max(minx, 0); miny = max(miny, 0)
        maxx = min(maxx, width - 1); maxy = min(maxy, height - 1)
        if maxx < minx or maxy < miny:
            continue

        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-12:
            continue

        gx, gy = np.meshgrid(np.arange(minx, maxx + 1),
                             np.arange(miny, maxy + 1))
        a = ((y1 - y2) * (gx - x2) + (x2 - x1) * (gy - y2)) / denom
        b = ((y2 - y0) * (gx - x2) + (x0 - x2) * (gy - y2)) / denom
        c = 1.0 - a - b
        inside = (a >= 0) & (b >= 0) & (c >= 0)
        if not inside.any():
            continue
        zz = a * z[i0] + b * z[i1] + c * z[i2]  # linear z interp (small tris)
        gxi = gx[inside]; gyi = gy[inside]; zzi = zz[inside]
        cur = zbuf[gyi, gxi]
        upd = zzi < cur
        zbuf[gyi[upd], gxi[upd]] = zzi[upd]

    depth = np.where(np.isfinite(zbuf), zbuf, 0.0).astype(np.float32)
    return depth


def render_hand_depth(K, height, width, joints=None, verts=None, faces=None,
                      **capsule_kwargs):
    """Convenience: render depth from MANO (verts+faces) if given, else joints."""
    if verts is not None and faces is not None:
        return rasterize_mesh_depth(verts, faces, K, height, width)
    if joints is None:
        raise ValueError("render_hand_depth needs either (verts, faces) or joints")
    v, f = joints_to_capsule_mesh(joints, **capsule_kwargs)
    return rasterize_mesh_depth(v, f, K, height, width)


# ----------------------------------------------------------------------------
# Sensor noise model + normalisation
# ----------------------------------------------------------------------------
def add_sensor_noise(depth, sigma=0.005, dropout_p=0.05, quant=0.001,
                     n_holes=2, hole_frac=0.15, rng=None):
    """Corrupt a clean depth map to mimic a real metric depth sensor.

    Args:
        depth: (H, W) metres, 0 = background.
        sigma: per-pixel Gaussian depth noise std (m). ToF/structured-light is
            typically 3-10 mm at hand range.
        dropout_p: fraction of valid pixels randomly dropped to 0 (speckle holes).
        quant: depth quantisation step (m); 0 disables.
        n_holes: number of rectangular occlusion holes erased to 0.
        hole_frac: max hole side as a fraction of image size.
        rng: np.random.Generator (for reproducibility).

    Returns:
        corrupted depth (H, W) float32.
    """
    rng = rng if rng is not None else np.random.default_rng()
    out = depth.astype(np.float32).copy()
    valid = out > 0
    nval = int(valid.sum())
    if nval == 0:
        return out

    if sigma > 0:
        out[valid] = out[valid] + rng.normal(0.0, sigma, nval).astype(np.float32)
    if quant and quant > 0:
        out[valid] = np.round(out[valid] / quant) * quant
    if dropout_p > 0:
        drop = (rng.random(out.shape) < dropout_p) & valid
        out[drop] = 0.0
    h, w = out.shape
    for _ in range(int(n_holes)):
        if rng.random() < 0.5:  # erase holes only half the time
            continue
        hh = int(rng.integers(1, max(2, int(h * hole_frac))))
        ww = int(rng.integers(1, max(2, int(w * hole_frac))))
        y0 = int(rng.integers(0, max(1, h - hh)))
        x0 = int(rng.integers(0, max(1, w - ww)))
        out[y0:y0 + hh, x0:x0 + ww] = 0.0
    out[out < 0] = 0.0
    return out.astype(np.float32)


def simulate_lidar_depth(depth, downscale=4, edge_drop_p=0.5, edge_thresh=0.015,
                         sigma=0.008, dropout_p=0.03, quant=0.005,
                         n_holes=2, hole_frac=0.15, rng=None):
    """Corrupt a clean metric depth map to mimic an iPhone LiDAR (ARKit sceneDepth).

    The deploy target is iPhone Pro LiDAR: a direct-ToF sensor that, unlike the
    high-res stereo depth in the training sets (DexYCB/HO3D), is

      * low native resolution (~256x192) -> coarse / blocky, fine detail lost,
      * weak/sparse on thin geometry (fingers) and bleeds at depth edges,
      * dToF per-pixel noise + coarse quantisation.

    A model trained on sharp stereo/synthetic depth would otherwise never see
    that degradation. Applying this on the 256-crop makes the training depth
    *look like* LiDAR, bridging the stereo->LiDAR domain gap. Numpy-only (keeps
    this module dependency-free). 0 = invalid/background throughout.

    Args mirror add_sensor_noise, plus:
        downscale: block factor for the coarse LiDAR grid (>=1; ~4 over a 256 crop
            ~ the hand's effective LiDAR sampling).
        edge_drop_p: probability of zeroing a high-gradient (finger/edge) pixel.
        edge_thresh: |depth gradient| (m) above which a pixel counts as an edge.
    """
    rng = rng if rng is not None else np.random.default_rng()
    out = depth.astype(np.float32).copy()
    H, W = out.shape
    valid0 = out > 0
    if not valid0.any():
        return out

    # 1) dToF per-pixel noise + coarse quantisation (valid pixels only)
    if sigma > 0:
        out[valid0] += rng.normal(0.0, sigma, int(valid0.sum())).astype(np.float32)
    if quant and quant > 0:
        out[valid0] = np.round(out[valid0] / quant) * quant

    # 2) edge / thin-structure dropout (LiDAR misses fingers, bleeds at boundaries)
    if edge_drop_p > 0:
        gy = np.abs(np.diff(out, axis=0, prepend=out[:1, :]))
        gx = np.abs(np.diff(out, axis=1, prepend=out[:, :1]))
        edge = ((gx + gy) > edge_thresh) & valid0
        out[edge & (rng.random((H, W)) < edge_drop_p)] = 0.0

    # 3) low-res LiDAR sampling: block-nearest down- then up-sample (no cv2).
    d = max(1, int(downscale))
    if d > 1:
        small = out[::d, ::d]                                  # coarse grid sample
        up = np.repeat(np.repeat(small, d, axis=0), d, axis=1)[:H, :W]
        out = np.ascontiguousarray(up, dtype=np.float32)

    # 4) speckle dropout + rectangular low-confidence holes
    valid = out > 0
    if dropout_p > 0:
        out[(rng.random(out.shape) < dropout_p) & valid] = 0.0
    for _ in range(int(n_holes)):
        if rng.random() < 0.5:
            continue
        hh = int(rng.integers(1, max(2, int(H * hole_frac))))
        ww = int(rng.integers(1, max(2, int(W * hole_frac))))
        y0 = int(rng.integers(0, max(1, H - hh)))
        x0 = int(rng.integers(0, max(1, W - ww)))
        out[y0:y0 + hh, x0:x0 + ww] = 0.0
    out[out < 0] = 0.0
    return out.astype(np.float32)


def corrupt_depth(depth, dc, train, rng=None):
    """Dispatch depth corruption for a dataloader: LiDAR-sim if `dc['lidar_sim']`
    else the stereo/ToF sensor-noise model. `train` toggles random vs deterministic
    (eval) corruption. Keeps every loader's call site to one line and backward-
    compatible (no `lidar_sim` key -> original add_sensor_noise behaviour)."""
    if dc.get("lidar_sim"):
        if train:
            return simulate_lidar_depth(
                depth, downscale=dc.get("lidar_downscale", 4),
                edge_drop_p=dc.get("lidar_edge_drop", 0.5),
                edge_thresh=dc.get("lidar_edge_thresh", 0.015),
                sigma=dc["sigma"], dropout_p=dc["dropout_p"], quant=dc["quant"],
                n_holes=dc["n_holes"], hole_frac=dc["hole_frac"], rng=rng)
        # eval: deterministic LiDAR coarsening (no random noise/edge/holes) so the
        # held-out metric reflects the LiDAR resolution the model will deploy on.
        return simulate_lidar_depth(
            depth, downscale=dc.get("lidar_downscale", 4), edge_drop_p=0.0,
            sigma=0.0, dropout_p=0.0, quant=dc["quant"], n_holes=0, rng=rng)
    if train:
        return add_sensor_noise(depth, sigma=dc["sigma"], dropout_p=dc["dropout_p"],
                                quant=dc["quant"], n_holes=dc["n_holes"],
                                hole_frac=dc["hole_frac"], rng=rng)
    return add_sensor_noise(depth, sigma=0.0, dropout_p=0.0, quant=dc["quant"],
                            n_holes=0, rng=rng)


def normalize_depth(depth, scale=0.1):
    """Centre valid depth on its own median and scale to ~[-1, 1].

    Centring on the per-frame median (computed from the depth map itself) is
    *inference-safe*: a real sensor provides the same statistic at test time, so
    the network never sees the absolute camera distance and cannot overfit to it.
    Background stays exactly 0 (distinct from the centred hand).

    Args:
        depth: (H, W) metres, 0 = background.
        scale: metres mapped to 1.0 (hand depth span ~ +/- 10 cm).

    Returns:
        (H, W) float32 in [-1, 1], background = 0.
    """
    out = np.zeros_like(depth, dtype=np.float32)
    valid = depth > 0
    if not valid.any():
        return out
    center = float(np.median(depth[valid]))
    out[valid] = np.clip((depth[valid] - center) / scale, -1.0, 1.0)
    return out


# ----------------------------------------------------------------------------
# Self-test (no dataset required)
# ----------------------------------------------------------------------------
def _make_fake_hand():
    """A flat synthetic right hand ~45 cm from the camera, for smoke-testing."""
    joints = np.zeros((21, 3), np.float32)
    joints[0] = [0.00, 0.00, 0.45]                      # wrist
    finger_x = {1: -0.03, 5: -0.015, 9: 0.0, 13: 0.015, 17: 0.03}
    for base, x in finger_x.items():
        for k in range(4):
            j = base + k
            joints[j] = [x, -0.03 - 0.022 * k, 0.45 - 0.004 * k]
    return joints


if __name__ == "__main__":
    K = np.array([[480.0, 0.0, 112.0],
                  [0.0, 480.0, 112.0],
                  [0.0, 0.0, 1.0]], np.float32)
    H = W = 224
    joints = _make_fake_hand()

    v, f = joints_to_capsule_mesh(joints)
    print(f"[capsule] verts={v.shape} faces={f.shape}")
    depth = rasterize_mesh_depth(v, f, K, H, W)
    valid = depth > 0
    cov = 100.0 * valid.mean()
    print(f"[depth ] coverage={cov:.1f}%  z[min,med,max]="
          f"[{depth[valid].min():.3f},{np.median(depth[valid]):.3f},{depth[valid].max():.3f}] m")
    assert valid.sum() > 0, "rasteriser produced an empty depth map"
    assert 0.30 < np.median(depth[valid]) < 0.60, "depth out of expected range"

    rng = np.random.default_rng(0)
    noisy = add_sensor_noise(depth, sigma=0.005, dropout_p=0.05, rng=rng)
    norm = normalize_depth(noisy)
    print(f"[noisy ] valid_frac={100.0*(noisy>0).mean():.1f}%")
    print(f"[norm  ] range=[{norm.min():.2f},{norm.max():.2f}]  bg_is_zero={np.all(norm[~valid]==0)}")
    assert norm.min() >= -1.0 and norm.max() <= 1.0
    print("OK: depth_synth self-test passed.")
