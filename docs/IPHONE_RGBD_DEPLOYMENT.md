# iPhone RGB-D Hand Pose — Deployment Status & Findings

_Last updated: 2026-06-30_

On-device RGB-D hand-pose deployment to iPhone Pro (LiDAR) via Unity, plus an
offline evaluation of the trained models on real iPhone captures.

Target sensor = iPhone Pro LiDAR (ToF, accuracy-first); ZED stereo was only an
intermediate test rig. See `docs/RGBD_DUAL_STREAM_DESIGN.md` for the model design.

---

## 1. Goal

Run the trained RGB-D hand-pose model on iPhone, extract the camera RGB + LiDAR
depth in Unity, render the 21 hand joints as spheres in AR, and capture real
RGB-D for offline evaluation / future fine-tuning.

---

## 2. Unity app (DepthRefinement)

**Stack:** Unity 6, `com.unity.ai.inference` 2.6.1 (Inference Engine / ex-Sentis),
`com.unity.xr.arfoundation` + `arkit` 6.x. Model imported as `ds_anchor_gate.onnx`.

### Components (`Assets/Runtime/Recon/`)

| Component | Role |
|---|---|
| `DualStreamHandProvider` | Runs the dual-stream model. RGB+depth crop → 4-ch input + depth_med + geometric root anchor → `keypoints_abs`. **Has a `SensorRotation` enum** (see §3). |
| `ARCameraImageProvider` | iOS camera RGB → 256² model input. |
| `CameraRgbDisplay` / `DepthImageVisualizer` | On-screen RGB / colorized-depth panels. |
| `CameraRgbInfoLabel` / `DepthInfoLabel` | Live text readout of image size/format/intrinsics/coverage (legacy Text **and** TMP auto-detected). |
| `HandSphereDriver` | Drives 21 existing sphere GameObjects from `AbsJoints`. |
| `HandBboxBridge` / `HandBboxOverlay` | 2D landmarks → model crop box (+ yellow debug overlay). |
| `HandVisionBridge` + `Plugins/iOS/HandPoseVision.swift` | **Apple Vision** (`VNDetectHumanHandPoseRequest`) native hand detector → feeds the bbox bridge. The missing detector that kept the crop centered. |
| `RgbdRecorder` (+ `RgbdRecorderStatusLabel`) | Saves RGB-D to disk (see §4). |

---

## 3. Orientation issue (sensor vs display) — root cause + fix

**Symptom:** the on-screen full-screen camera looked upright (landscape), but the
mini RGB/Depth panels — and therefore the model input — had the hand rotated ~90°.
Joints came out in the sensor frame and didn't match the displayed hand.

**Cause:** `XRCpuImage` (`TryAcquireLatestCpuImage`) is the **raw camera-sensor
buffer**, whose orientation is fixed to the hardware and never auto-rotated. ARKit
only rotates the *on-screen background* (display matrix); the raw buffer the model
reads stays sensor-native. So sensor frame ≠ display frame by a fixed rotation
(observed 90° CW here). `XRCpuImage.ConversionParams` has no rotate option (mirror
only), so it must be corrected in code.

**Fix:** `DualStreamHandProvider._sensorRotation` (`None/CW90/Rot180/CW270`,
default **CW90**). It rotates the 256² RGB+depth crop to upright before inference
(`SrcPatchIndex`) and rotates the depth anchor about the optical axis
(`RotateAnchor`), so the model sees an upright hand **and** outputs in the display
camera frame → spheres land correctly. **Requires a locked landscape orientation**
(Player Settings) so the offset stays a constant 90°; AutoRotation breaks it.

> Tune on device: if spheres are ~180° off → `CW270`; if only flipped → `Rot180`.

---

## 4. RGB-D recorder

`RgbdRecorder` saves synchronized frames to
`Application.persistentDataPath/rgbd_captures/` (UI: Capture / ● REC buttons).
Per capture:

- `cap_NNNN_rgb.png` — RGB, upright, sensor crop (full 1920×1440 by default).
- `cap_NNNN_depth.f32` — LiDAR depth, **little-endian float32, row-major H×W, metres** (0 = invalid).
- `cap_NNNN.json` — sizes, intrinsics (fx,fy,cx,cy,resolution), depth unit, screen orientation, timestamp.

Retrieve via Xcode → Devices → **Download Container**, or the Files app (needs
Player Settings → *Supports iTunes File Sharing* + *opening documents in place*).

**Captured set:** 280 frames, RGB 1920×1440, depth 256×192, fx=fy≈1418, screen
LandscapeLeft. Depth is dense (100% coverage), metric, and **perfectly aligned to
RGB** (verified).

---

## 5. Offline inference & model comparison

Script: **`infer_rgbd_captures.py`** — loads the captures, auto-rotates the sensor
frame upright (uprightness vote, see below), resizes 256×192 depth → RGB res, detects
the hand (MediaPipe), runs the model, writes `cap_NNNN_pred[_tag].png/json` +
`predictions_summary[_tag].csv` back into the folder.

```bash
python infer_rgbd_captures.py --arch dualstream     # ds_anchor_gate/best.pt
python infer_rgbd_captures.py --arch earlyfusion    # rgbd_real_0613/best.pt
python infer_rgbd_captures.py --arch earlyfusion --dir rgbd_captures_03 --rotate 0
```

> **Rotation-vote bug (fixed 2026-06-29).** `vote_rotation` used to pick the global
> rotation by counting MediaPipe **detections** — but MediaPipe is rotation-INVARIANT
> (detects a hand at every k·90°, so all rotations tie), and the tie-break forced
> **CW90**, rotating already-upright captures sideways and feeding the model a
> lying-down hand. That silently cost ~11 pts of pose accuracy (early-fusion
> pose-shape 54.9 % → **43.8 %**, collapse 46 % → 38 %). It now scores **uprightness**
> (fingertips above the wrist) instead, auto-picking k=0 for the LandscapeLeft
> captures. Override with `--rotate N` for a fingers-down session.

### Result (re-capture 2026-06-29, 280 frames, both detect 254/280; true hand ~33 cm)

| | dual-stream (ds_anchor_gate, learned root) | early-fusion (rgbd_real_0613, geometric root) |
|---|---|---|
| **root_z median** | 58 cm ❌ (~1.8× the real ~33 cm) | **33 cm ✅** (matches reality) |
| **root_z range** | 36–83 cm | 19–53 cm |
| **pose** | small blob at the palm (a *placement* artifact, not a collapsed mesh) | **spans the hand, correct structure/scale** |

**early-fusion clearly wins** on both localization and pose.

### Ablation — what actually drives the dual-stream collapse

The `--anchor {centroid,joint} --med {crop,hand}` flags isolate the cause
(median root_z; true hand ~33 cm, early-fusion ref 33 cm):

| anchor | depth_med | depth channel | root_z |
|---|---|---|---|
| centroid (orig) | crop median (orig) | full crop | 58 cm — collapse |
| **joint-pixel** | crop median | full crop | 58 cm — **fixing the anchor does nothing** |
| centroid | **hand median** | full crop | 41 cm — **−17 cm, the big mover** |
| joint-pixel | hand median | full crop | 42 cm |
| **hand z-band mask** (`--mask band`) | hand-only | **hand-only** | **28 cm ✅ — recovered, pose spans the hand** |

The last row reproduces the training `depth_mode="hand_masked"` at inference: a
MediaPipe-landmark **z-band hand mask** (keep depth within ±0.12 m of the joint-pixel
hand depth, inside the padded bbox; zero the rest *before* preprocess) so the depth
channel, depth_med **and** the anchor all see hand-only depth. Both contributors had
to be masked: depth_med (58→41) **and** the depth-channel background fed to the
DepthEncoder (41→28).

### Pose vs root — where the *remaining* error is (`diag_pose_vs_root.py`)

Once root is fixed, what's left? With no 3-D GT, we use MediaPipe's 21 2-D landmarks
as a pseudo-GT and similarity-Procrustes-align (remove translation/rotation/scale)
each model's projected joints; the residual is **pose-shape error**.

| model | pose-shape error (aligned resid / hand size) |
|---|---|
| dual-stream (mask-fixed) | **78.5 %** |
| early-fusion | **78.5 %** |
| dual-stream (orig, wrong root_z) | 86.7 % (inflated by perspective) |

- The two deploy-ready models have **identical** pose-shape error — expected, since
  they share a 4-channel (RGB+depth) DenseStack pose backbone; the only difference is
  the root method, which Procrustes removes.
- Overlays (RED = model, GREEN = MediaPipe) confirm the error is **real**: on free /
  open-hand poses the model consistently **cramps the fingers into a cluster near the
  palm** while MediaPipe spreads them to the real fingertips (cap_0000/0106/0150/0194/
  0279). Root_z is already ~right (28–33 cm).
- **The bottleneck is the shared POSE backbone, not the root.** Extended free-hand
  poses are under-represented in training (HanCo-synth + DexYCB/HO3D object-grasping).
  Retraining the dual-stream *root* head cannot beat early-fusion; improving the
  *pose* backbone with in-domain data helps **both** architectures equally — so the
  lever is data, not architecture.

**Why early-fusion looked good on ZED but not on the first iPhone set = PRESENTATION,
not the sensor.** The first iPhone captures were all one awkward OOD pose (relaxed
hand hanging down, palm-down, fingers pointing away, oblique top-down view); the ZED
test presented the hand frontally and in-range. A re-capture with a **frontal,
fingers-spread, palm-toward-camera** hand (still ~33 cm) dropped early-fusion's
pose-shape error **78.5 % → 51.4 % median** (best frame 13 %, p25 36 %), and overlays
show the model tracking the spread fingers again. Distance was held ~constant, so this
isolates **pose/viewpoint** as the cause — not distance, sensor, or model. Residual
51 % median / 80 % p75 is still imperfect (close-range foreshortening + real-iPhone
pose-domain gap + 2D-metric over-penalty on oblique poses). **Deploy guidance:**
present the hand frontally/spread for ZED-level results now; close the rest with
in-domain pose data later.

- The "small hand" is **not** a mesh/scale collapse — `scale = 1 + 0.2·tanh(·)` is
  hard-clamped to [0.8, 1.2], so the model *cannot* shrink the hand 2–3×. It's a
  **root-Z placement error**: a correct-size 3-D hand placed ~1.8× too far projects
  ~1.8× smaller in pixels.
- The confidence gate is **not** the problem either: `anchor_valid` median 0.976
  (96 % > 0.7). A flat background patch is coherent (low MAD) → gets *high* conf; the
  gate detects surface coherence, not hand-ness.
- **The driver is `depth_med`** (raw-depth median over the *whole crop*). With the
  hand close and `margin_frac=0.5`, the crop is background-heavy so depth_med is
  80–119 cm, and the learned `TranslationHead` over-relies on it as the absolute-
  distance cue → root ≈ 58 cm. Feeding a *correct* joint-pixel anchor (32.9 cm) is
  overridden: the learned residual adds +25 cm back. Even the full fix only reaches
  42 cm (vs early-fusion 33 cm) — the learned root head keeps an OOD bias.

---

## 6. Key findings

1. **iPhone LiDAR depth is good but low-res.** Metric accuracy and RGB alignment
   are excellent; the downgrade vs training depth (RealSense ~640×480) is
   **resolution (256×192)** and smoothed edges — fingers are only a few px. It is
   **low-res, not noisy** — ARKit smooths heavily (clean 0.16–0.4 m, 100 % coverage).
2. **The dual-stream collapse is a train/deploy DISTRIBUTION mismatch on the depth
   mask — not a depth-quality, scale, or gate bug — and it's fixable at inference
   with no retrain.** Training/eval loaders default to `depth_mode="hand_masked"`
   (dexycb seg==255 / ho3d z-band / HanCo hand-mesh-only depth), so depth_med *and*
   the depth channel are hand-only → great eval metric. Deploy had no mask → both
   were background-contaminated at close range → `TranslationHead` placed the root too
   far. Reproducing the mask at inference (`--mask band`) recovers root_z to 28 cm and
   a hand-spanning pose (§5), on par with early-fusion.
3. **The lidar-sim noise/degradation augmentation was effort in the wrong place.**
   `simulate_lidar_depth` (gaussian noise, dropout, holes, edge-drop) corrupts the
   depth *map* (DepthEncoder input), but the failure is the depth_med *scalar* + crop
   content — depth-map texture is irrelevant to it. And the premise was half-wrong:
   iPhone LiDAR isn't noisy. What was actually needed: train depth_med as a hand-
   masked / joint-pixel median (match deployment crop stats), or use the geometric
   root. No amount of depth-noise aug addresses the collapse.
4. **This reproduces the earlier ZED result** (early-fusion good, dual-stream
   collapse) — now confirmed and root-caused on iPhone LiDAR too.
5. **Once root is fixed, the remaining error is POSE, and it's shared.** Both models
   tie at 78.5 % pose-shape error (§5) and visibly cramp the fingers on free-hand
   poses. The bottleneck is the pose backbone / training-pose distribution, not the
   architecture — so retraining the dual-stream root head won't beat early-fusion.
6. Confound (now controlled): captures had the hand fairly close (~33 cm), nearer
   than training's ~40–70 cm — which is exactly what makes the crop background-heavy
   and depth_med wrong. A 40–70 cm recapture would shrink (but not design-fix) it.

---

## 7. POSE fix — iPhone pseudo-3D GT fine-tune (2026-06-30)

§5–§6 localised the remaining error to the **shared POSE backbone** (fingers cramp
on free-hand poses), and concluded the lever is **in-domain data, not architecture**.
We acted on that without any rig or manual labels.

### 7a. Pseudo-3D GT from captures (no rig) — `make_iphone_gt.py`

MediaPipe gives 21 2-D landmarks that are CORRECT on exactly the free-hand poses our
model cramps (§5, GREEN vs RED). We lift them to metric 3-D and use them as GT:

    cap_*_{rgb.png,depth.f32,json}  ->  uprightness rotation (vote/-—rotate)
      -> MediaPipe 21 px landmarks  ->  backproject via LiDAR depth + intrinsics
      -> per-joint VALIDITY mask (occluded / depth-outlier joints -> valid=0)
      -> surface->joint-centre offset along the ray
      => cap_*_gt.json { joints3d_cam[21,3] m, joint_valid[21], bbox, K, rot_k }
      +  cap_*_gtviz.png  (RED lifted-GT vs GREEN MediaPipe vs grey=masked)

Key idea: **sparse-but-trustworthy**, not dense-but-wrong. Occluded joints whose
LiDAR depth is unreliable are flagged `valid=0` and the masked training loss simply
skips them — no need to reconstruct them. Validated on the 5 capture folders
(rgbd_captures + _01.._04, ~1193 frames): ~244–257 hands/folder, **mean ~15–17 valid
joints/frame**, metric-sane (hand 3-D span median ~20 cm, z-span 7–12 cm — a real
spread hand, not a collapse). Reprojection matches MediaPipe tightly.

### 7b. Training loader — `datasets/iphone_captures.py`

`IPhoneCaptures_RGBD` emits the exact DexYCB_RGBD contract (4-ch 256 crop,
root-aligned keypoints3D/2D, root, cam, **joint_valid**), so it concatenates in
`train_mobrecon_rgbd.py --datasets ...,iphone --iphone_root <dir[,dir...]>` (multi-
folder, comma list). Deployment has no segmentation, so the depth channel is
**z-band hand-masked using the GT joint depth** (±0.12 m inside a padded bbox) —
reproducing the training `depth_mode="hand_masked"` distribution and avoiding the
background contamination §6 flagged.

### 7c. First fine-tune + held-out eval — IT WORKS

Run `ef_iphone_v1`: warm-start `rgbd_real_0613/best.pt`, train on 4 sessions
(rgbd_captures + _01.._03, 961 frames; **session _04 HELD OUT**), lr 1e-5, w2d 0.1,
batch 24, 25 ep, lidar_sim OFF. test_mpjpe 47 → 24 mm.

Held-out eval (`eval_iphone_heldout.py` on rgbd_captures_04, Procrustes pose-shape
vs the MediaPipe pseudo-GT over valid joints, 230 frames):

| model | pose-shape (median) | p25 | p75 | raw | scale s |
|---|---|---|---|---|---|
| baseline `rgbd_real_0613` | 50.6 % | 34 % | **80 %** | 69 % | 1.03 |
| **fine-tuned `ef_iphone_v1`** | **41.5 %** | 32 % | **63 %** | **50 %** | 1.14 |

Pose-shape error **−9 pts median (−18 % rel)**; the cramping tail **p75 80 → 63 %**
is the bigger win (exactly the target). Visible un-cramping on cap_0150 (fingers
extend vs baseline cluster). Minor regression: **scale s 1.03 → 1.14** (hand projects
slightly small) — an early forgetting/over-fit sign from the iphone-ONLY mix.

**Conclusion: the data lever is confirmed on UNSEEN data.** The pseudo-GT fine-tune
improves the shared pose backbone — no architecture change, no manual labels.

### 7d. base data is NOT obsolete

The base sets (HanCo/DexYCB/HO3D, already on disk) remain essential: they ARE the
warm-start foundation (`rgbd_real_0613`), and the iphone-only run's scale-s drift is
early catastrophic-forgetting. The production recipe is a **mixed** fine-tune
(`--datasets hanco,dexycb,ho3d,iphone`, iPhone over-sampled), not iphone-only. We do
NOT need BigHand/HANDS17 (ToF pseudo-RGB, domain-mismatched — stays skipped).

---

## 8. Next steps

- [ ] **Capture more iPhone sessions** (one folder each: rgbd_captures_05, _06 …) —
      target the v1 weak spots: spread/free-hand poses, oblique/top-down viewpoints,
      30/50/70 cm distances, multiple subjects + both hands. Labelling is free
      (pseudo-GT), so favour breadth; keep per-session folders for held-out eval.
- [ ] **Mixed fine-tune** `--datasets hanco,dexycb,ho3d,iphone` to keep base breadth
      and fix the scale-s drift; per-session held-out eval as the standard metric.
- [ ] Optional: add a **bone-length prior** loss to further suppress finger-cramping.
- [ ] **Deploy early-fusion to iPhone**: export `rgbd_real_0613` (or the fine-tuned
      checkpoint) to ONNX + geometric root recovery in Unity (sample depth at the 21
      joint pixels).
- [ ] **dual-stream remains viable offline** (recovered to 28 cm) via the §5 z-band
      mask if ever needed; early-fusion stays the deploy default.

---

## 9. File map

- Unity: `unity/DepthRefinement/Assets/Runtime/Recon/*.cs`, `Assets/Plugins/iOS/HandPoseVision.swift`
- Offline inference: `infer_rgbd_captures.py` (reuses `DualStreamHandPose` from
  `infer_zed_dualstream.py` and `RGBDHandPose` from `infer_rgbd_femtobolt.py`)
- **Pseudo-3D GT**: `make_iphone_gt.py` (captures → `cap_*_gt.json` + `_gtviz.png`)
- **Training loader**: `datasets/iphone_captures.py` (`IPhoneCaptures_RGBD`, multi-root,
  z-band hand-mask) wired in `train_mobrecon_rgbd.py --datasets iphone --iphone_root`
- **Held-out eval**: `eval_iphone_heldout.py` (Procrustes pose-shape vs pseudo-GT)
- Captures + predictions: `rgbd_captures/`, `rgbd_captures_01..04/` (5 sessions, ~1193 GT frames)
- Models: `mobrecon_ckpt/ds_anchor_gate/best.pt` (dual-stream),
  `mobrecon_ckpt/rgbd_real_0613/best.pt` (early-fusion baseline),
  `mobrecon_ckpt/ef_iphone_v1/best.pt` (early-fusion + iPhone pseudo-GT fine-tune)
- Related: `docs/RGBD_DUAL_STREAM_DESIGN.md`, `docs/DATASET_GENERALIZATION_EVAL.md`
