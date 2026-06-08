# Depth-Aware Hand Pose Refinement (RGB → monocular depth → refine)

A **no-retraining** Python pipeline that augments the existing RGB-only Recon (MobRecon)
hand-joint model with **depth inferred from the same RGB image** (Depth Anything V2), then
uses that depth to correct the per-joint depth (Z). Evaluated on FreiHAND.

```
RGB image
   ├─→ MobRecon (frozen, pretrain/100.pt)      → root-relative 3D joints   (baseline)
   └─→ Depth Anything V2 (pretrained)           → monocular inverse-depth map
                          ↓
        refine.refine_sample: sample depth at joints → align to predicted scale
                              → blend per-joint Z → (optional) bone-length constraint
                          ↓
                   depth-refined 3D joints
```

No model is retrained; both networks use pretrained weights.

## Files

- `monocular_depth.py` — wraps Depth Anything V2 (HuggingFace `transformers`), RGB → inverse-depth map.
- `refine.py` — the refinement math: 2D projection, robust depth sampling, scale alignment
  (`1/Z_pred ≈ a·disp + b`, fit to the **predicted** depths — no GT leak), per-joint Z fusion,
  outlier/occlusion rejection, fingertip exclusion, optional bone-length constraint. Also
  `monocular_rel_z` for the diagnostic.
- `eval_refine.py` — baseline vs refined evaluation on FreiHAND across fusion weights `w`,
  reporting root-relative MPJPE, PA-MPJPE, Z-only error, and a correlation diagnostic.

## How to run

```bash
# from repo root
python -m depth_refine.eval_refine --n 500 --weights 0 0.1 0.25 0.5 1.0
python -m depth_refine.eval_refine --n 3960 --out depth_refine/results_full.json   # full eval set
```

Requirements (installed): `torch`, `transformers`, `timm`, `mediapipe`, `pycocotools`, `opencv`.
Depth Anything V2 weights download from HuggingFace on first run.

## Results (FreiHAND eval, **full 3960 samples**, ckpt `pretrain/100.pt`)

| config   | MPJPE (mm) | PA-MPJPE (mm) | Z-err (mm) | Δ MPJPE |
|----------|-----------:|--------------:|-----------:|--------:|
| baseline |      31.10 |         25.83 |      25.52 |    —    |
| w=0      |      31.11 |         25.83 |      25.51 |  +0.01  |
| w=0.25   |      32.67 |         26.62 |      27.26 |  +1.56  |
| w=0.5    |      34.53 |         27.69 |      29.32 |  +3.43  |
| w=1.0    |      38.97 |         30.38 |      34.13 |  +7.86  |

**Diagnostic — relative-Z vs GT (mean Pearson r over 3960 samples):**
- predicted-Z vs GT-Z: **r = +0.824**
- monocular-Z vs GT-Z: **r = +0.199**

(`w=0` reproduces the baseline exactly — sanity check passed. n=500 subset gave the same
trend: baseline 32.96 → w=1 40.61; r = 0.80 vs 0.20.)

## Conclusion (honest)

**Monocular depth inferred from the RGB does NOT improve hand-joint accuracy here — it
slightly harms it, monotonically with the fusion weight.** The diagnostic explains why:

- The RGB model already predicts relative joint depth well (**r=0.80** with GT).
- Depth Anything V2, sampled at the joints on a tight hand crop, barely tracks the true
  relative joint depth (**r=0.20**) — it resolves the hand-vs-background and coarse shape,
  not finger-level depth.
- Fusing the weaker signal (0.20) into the stronger one (0.80) can only add noise.

This is the **"circular depth" limitation**: a monocular depth net re-derives depth from the
*same RGB* the hand model already used — and does it worse at hand scale. So it adds little
independent information. (This was flagged as risk **R-1** in the project PRD.)

### When depth *does* help
Depth refinement pays off only when the depth is an **independent, higher-quality signal**
than the RGB model's own estimate — i.e. a real **depth sensor** (iPhone LiDAR / ToF). That
is exactly the on-device path implemented in `unity/DepthRefinement/` (it samples the phone's
LiDAR `sceneDepth`, not RGB-inferred depth). The refinement *math* is the same; what changes
is the **quality of the depth source**.

## Absolute root-depth experiment (`eval_root_depth.py`)

The RGB model predicts **no absolute scale** — so absolute placement is where a metric
monocular model could, in principle, add value the model lacks. We test whether monocular
depth recovers the **wrist's metric distance from the camera**, vs a no-depth constant guess.
(Calibration fit on a held-out split, applied to the test split; wrist 2D from GT.)

FreiHAND eval, n=2000 (calib 600 / test 1400), GT wrist Z = 0.71 ± 0.12 m:

| method | MAE (mm) | RMSE (mm) | corr r |
|--------|---------:|----------:|-------:|
| constant (no depth) | 95.7 | 114.4 | +0.00 |
| relative + affine calib | **93.8** | 114.0 | +0.22 |
| metric raw (Indoor) | 216.8 | 290.3 | +0.09 |
| metric + affine calib | 94.8 | 114.0 | +0.09 |

**Even for absolute depth, monocular barely helps:** the best variant beats a constant guess by
only ~2 mm of 96 mm, and correlation with true distance is weak (r≈0.22). The metric model is
wildly miscalibrated at hand range (raw MAE 217 mm) because it was trained on room-scale scenes.
A single image of a hand has fundamental **scale ambiguity** — the hand fills the frame at any
distance — which a generic monocular network cannot resolve.

## Depth-sensor upper bound (`eval_sensor_upperbound.py`)

How much would a **real** depth sensor help? The refinement corrects only Z, so the ceiling is:
keep the model's (x, y), replace each joint's Z with the sensor reading. A perfect sensor reads
the GT joint depth (exact upper bound); adding Gaussian noise maps sensor quality → accuracy.

FreiHAND eval, full 3960 samples:

| depth source | MPJPE (mm) | PA (mm) | Z-err (mm) | Δ MPJPE |
|--------------|-----------:|--------:|-----------:|--------:|
| baseline (RGB only) | 31.10 | 25.83 | 25.52 | — |
| **perfect sensor (σ=0)** | **14.17** | 14.15 | 0.00 | **−16.93 (−54%)** |
| sensor σ=2 mm | 14.56 | 14.37 | 2.14 | −16.54 |
| sensor σ=5 mm  (≈ iPhone LiDAR) | 16.03 | 15.34 | 5.33 | −15.07 |
| sensor σ=10 mm | 19.55 | 17.67 | 10.67 | −11.55 |
| sensor σ=20 mm (noisy ToF) | 28.27 | 23.17 | 21.45 | −2.84 |

**A real depth sensor would roughly halve MPJPE** (31 → 14 mm ideal; 16–20 mm at realistic
LiDAR/ToF noise of 5–10 mm). The benefit only collapses past ~20 mm noise. This is the headroom
monocular depth fails to capture.

*Idealized:* assumes the sensor reads the true joint depth at the correct pixel. A real sensor
reads the skin **surface** (~finger-radius above the joint) and fingertips are sub-pixel at
256×192 LiDAR — so the achievable gain sits between baseline and this ceiling.

## Realistic sensor gain — MANO surface rendering (`eval_surface_depth.py`)

The ideal bound assumed the sensor reads the true joint depth at the right pixel. A real sensor
reads the **skin surface** at the (imperfect) **predicted** 2D pixel, with **self-occlusion**.
We model this by ray-casting each joint pixel against the GT MANO mesh (z-buffer, 1538 faces).

FreiHAND eval, n=2000 (mesh-truth occlusion **34%**, surface-vs-joint offset **9.4 mm**):

| source | MPJPE (mm) | Z-err (mm) | Δ MPJPE |
|--------|-----------:|-----------:|--------:|
| baseline | 31.21 | 25.75 | — |
| ideal (joint Z, perfect 2D) | 14.05 | 0.00 | −17.16 |
| surface @ **GT** 2D (visible) | 29.03 | 21.93 | **−2.18** |
| surface @ **predicted** 2D (visible) | 35.83 | 30.71 | **+4.62** |
| surface @ pred 2D + 5 mm noise | 36.05 | 30.93 | +4.84 |

**The ideal −54% is NOT achievable in practice.** Three real effects collapse it:
1. **Surface-vs-joint offset (~9 mm):** the sensor reads skin, not the bone joint. Even with
   *perfect* 2D this caps the gain at ~−2 mm (not −17).
2. **2D sampling accuracy:** sampling at the *predicted* joint pixel — which this weak base model
   (31 mm) places ~cm off — reads the surface at the **wrong** spot and makes it **worse (+4.6 mm)**.
3. **Occlusion (~34% of joints):** a third of joints see an occluder, not themselves, and must
   fall back to the model's depth.

> **Big caveat — base-model confound:** the +4.6 mm at predicted-2D is dominated by this
> checkpoint's poor 2D (31 mm baseline vs ~7 mm for a good hand model). With an accurate base
> model, predicted 2D ≈ GT 2D, so the achievable regime is the **−2 mm** GT-2D row — and
> per-joint **surface-offset calibration** could recover more of the −17 mm headroom. The
> sensor's value is real but **conditional on a good 2D model + offset calibration + occlusion
> handling**, not the easy win the ideal bound implied.

## Overall finding

| depth source | relative | absolute | realistic MPJPE effect |
|---|---|---|---|
| **monocular (from RGB)** | ❌ harms (r 0.20 vs 0.82) | ❌ ~2 mm of 96 mm | 31 → 33–39 mm (worse) |
| **real sensor — ideal bound** | ✅ | ✅ | 31 → 14 mm (−54%, *unreachable*) |
| **real sensor — realistic (MANO surface)** | ➖ small/conditional | ✅ | 31 → ~29 mm (−2 mm) here; larger only with an accurate base model + offset calibration |

Two honest conclusions:
1. **RGB-inferred monocular depth is a dead end** — it can't add hand-scale depth the model didn't
   already have.
2. **A real depth sensor has large *theoretical* headroom (−54%) but modest *realistic* gain
   here** — limited by surface offset, the base model's 2D accuracy, and occlusion. It pays off
   only with an already-accurate pose model, surface-offset calibration, and good occlusion
   handling (the engineering in `unity/DepthRefinement/`), **or** by training an RGB-D model on
   real paired depth so the network learns to use depth directly rather than bolt it on.

## Caveats / scope

- **FreiHAND metric is root-relative.** Recovering **absolute** position (root depth) — where a
  *metric* monocular model could genuinely add value, since the RGB model predicts no absolute
  scale at all — is not measured by this protocol. A separate absolute-root-depth experiment
  would be the fair place to show monocular depth's upside.
- The checkpoint (`pretrain/100.pt`) is not FreiHAND-tuned, hence the high ~33 mm baseline; the
  refinement **Δ** (before/after, same samples) is unaffected by that offset.
- Uses the *relative* Depth Anything V2 Small. A hand-specialized or metric model, or sampling
  at predicted-vs-GT 2D, are variants worth trying — but the r=0.20 signal ceiling makes a large
  reversal unlikely.
