# Real-depth hand datasets — RGB-D adapter guide

How to feed **real sensor depth** into the existing RGB-D pipeline
(`train_mobrecon_rgbd.py` + `models.mobrecon_ds.LargeModel_Extra_RGBD`). Every
dataset is wrapped so it emits the *same* per-sample dict as the canonical loader
`datasets/hanco_ty.HanCo_ETRI_jitter`; the trainer then needs **zero** changes.

---

## 1. The contract every adapter must emit

`__getitem__` returns:

| key           | shape / dtype        | meaning |
|---------------|----------------------|---------|
| `image`       | `[4,256,256]` float  | ch0-2 RGB in `[0,1]`, ch3 **depth** in `[-1,1]`, pixel-aligned to the same 256 hand crop |
| `keypoints3D` | `[21,3]` float       | **root-aligned** camera-space joints in **metres**, rotated by the crop's in-plane rotation (joint 0 = wrist at origin) |
| `keypoints2D` | `[21,2]` float       | crop-pixel coords `/ 256` |
| `root`        | `[3]` float          | camera-space root joint (metres); used for the 2D reprojection loss |
| `cam`         | `[3,3]` float        | intrinsics adjusted to the 256 crop |

Invariants (must hold or the losses silently break):
- **Joint order = MANO/FreiHAND**: `0` wrist, then thumb→pinky, each MCP→TIP.
  Confirm a new dataset matches; if not, set a `JOINT_PERM`.
- **Metres**, camera frame, `y` down, `z` forward (`u=X·fx/z+cx`, `v=Y·fy/z+cy`).
- **Depth normalisation** = `normalize_depth` (per-frame **median centring**, bg=0,
  scale 0.1 → ±10 cm ↦ ±1). Median centring is inference-safe: it removes absolute
  camera distance, which a sensor also can't give reliably.
- **Depth distribution**: the model was trained on HanCo's *hand-only synthetic*
  depth (background = 0). Real scene depth (arm/table) is a distribution shift —
  **mask depth to the hand** (`depth_mode="hand_masked"`) to match, unless you are
  deliberately fine-tuning for full-scene deployment (`depth_mode="scene"`).

Reuse these shared pieces (don't re-implement): `datasets.dataset_utils.augmentation`
(crop+jitter+adjusted intrinsics), `datasets.depth_synth.{add_sensor_noise,
normalize_depth}`, `datasets.augmentation.color_jitter`.

---

## 2. DexYCB — implemented (`datasets/dexycb.py`)

**Status: ready.** Real RealSense aligned depth + MANO + 8-view, clean license.

Wire it into training by swapping the dataset in `train_mobrecon_rgbd.py`:
```python
from datasets.dexycb import DexYCB_RGBD
train_dataset = DexYCB_RGBD(root=DEX_YCB_DIR, mode="train", with_depth=True)
```
Smoke test (after download): `python -m datasets.dexycb --root <DexYCB> --check 8`
→ dumps `dexycb_check/*.png` (RGB+2D joints | depth colormap) to verify alignment.

Key choices baked in:
- joints already camera-space metres (auto mm→m guard); no extrinsics needed.
- `depth_mode="hand_masked"` uses `seg==255` so the depth matches HanCo training.
- right-hand-only by default (`hands="right"`); `flip_left=True` mirrors left
  hands (image+depth+joint x+cx) to reuse every frame under the right-hand contract.
- light sensor-noise aug only (depth is already real).

Splits: DexYCB's official `s0..s3` splits are subject/sequence based — pass
`subjects=[...]` to reproduce them rather than the random 95/5 the trainer does.

---

## 3. Other datasets — feasibility & adapter plan

Effort = new code on top of the shared contract. "Real depth" = sensor depth (not rendered).

| Dataset | Real depth | Sensor | Hand GT | Joint order | Adapter effort | Notes |
|---|---|---|---|---|---|---|
| **HO3D (v3)** | ✅ encoded depth PNG | RealSense (stereo) | MANO + 21 joints | MANO ✓ | **Done** (`ho3d.py`) | hand-object occlusion; OpenGL coords (coordChangeMat) + 2-ch depth decode; no seg → z-band depth mask |
| **H2O-3D** | ✅ encoded depth PNG | RealSense (stereo) | MANO + 21 joints ×2 hands | MANO ✓ | **Done** (`ho3d.py`) | two-hand+object, HOnnotate=HO3D format; `H2O3D_RGBD` reuses HO3D pipeline, picks right hand (left→right flip) |
| **MSRA15** | ✅ depth-only | **ToF** | 21 joints | perm→MANO | **Done** (`tof_depth.py`) | **best ToF fit**: 21 joints map cleanly, all valid; pseudo-RGB from depth |
| **ICVL** | ✅ depth-only | **ToF** | 16 joints (3/finger) | partial→MANO | **Done** (`tof_depth.py`) | ToF; 16 of 21 MANO slots valid (5 PIP absent) → masked loss; pseudo-RGB |
| **NYU** | ✅ RGB-D | structured-light | 36→14 joints | partial→MANO | **Done** (`tof_depth.py`) | NOT ToF; only fingertips+wrist (6 valid) mapped; needs scipy; pseudo-RGB |
| **BigHand2.2M** | ✅ depth-only | RealSense SR300 | 21 joints | perm→MANO | **Done** (`tof_depth.py`) | largest real-sensor depth; 21 joints all valid; pseudo-RGB; SR300 (coded-light, ~ToF). Full set ~2.2M lines → use `limit` |
| **HANDS17** | ✅ depth-only | SR300 | 21 joints | perm→MANO | **Done** (`tof_depth.py`) | same format as BigHand (`training/`); benchmark standard |
| **FPHA** | ✅ depth-only | SR300 (egocentric) | 21 joints | perm→MANO | **Done** (`tof_depth.py`) | world coords → `FPHA_CAM_EXTR` then SR300 depth intr; reorder_idx==BIGHAND_TO_MANO; 1st-person, matches Femto-Bolt use. Verify viz (colour/depth baseline) |
| **MVHand** | ✅ 4×RGB-D | RealSense D415 | 3D pose/mesh | check | Low-Med | multi-view RGB-D, smaller; good extra real-depth RGB pairs |
| **ContactPose** | ✅ Kinect-v2 depth | Kinect v2 | MANO + 21 (OpenPose) | MANO ✓ | **Done** (`oakink_contactpose.py`) | toolkit-wrapped; joints w.r.t object → camera via `object_pose`; real depth, z-band mask |
| **OakInk** | ⚠️ not via oikit | RealSense | MANO + 21 | MANO ✓ | **Done, RGB-only** (`oakink_contactpose.py`) | oikit exposes NO depth → depth is **rendered** from MANO (synthetic); use for diverse RGB+MANO, not real depth |
| **HOT3D** | ⚠️ point cloud / multi-view (not ToF map) | Aria/Quest3 | MANO+UmeTrack | MANO ✓ | High | egocentric; depth is not a single aligned ToF image → needs depth synthesis from point cloud; aligns with the AR/Android roadmap |
| **FreiHAND / HanCo / InterHand2.6M** | ❌ RGB-only | — | MANO / joints | MANO ✓ | (synthetic only) | no sensor depth → already handled via `depth_synth` render path |

### Practical recommendations
1. **DexYCB** (done, `dexycb.py`) → real RealSense depth + MANO, drop-in. Best first real-depth run.
2. **HO3D / H2O-3D** (done, `ho3d.py`) → second real-depth source; adds occlusion +
   object-in-scene robustness. Gotchas baked in: OpenGL→OpenCV `coordChangeMat`,
   2-channel depth decode (`dpt = R + G*256, ×0.000125 m`), train-split-only (eval has
   wrist only), z-band mask (no seg). `H2O3D_RGBD` subclasses HO3D, reusing the whole
   pipeline via `_finalize`; it reads per-hand `rightHandJoints3D`/`leftHandJoints3D`
   and folds the left hand into the right-hand convention. Concatenate via ConcatDataset.
3. **ToF domain match for Femto Bolt**: DexYCB/HO3D depth is **stereo**, Femto Bolt is
   **ToF** (different hole/edge noise). Two levers: (a) keep/strengthen
   `add_sensor_noise` to mimic ToF dropout; (b) optionally add **ICVL/MSRA (ToF,
   depth-only)** as a depth-branch pretrain. Don't expect stereo-trained depth to
   transfer to ToF without this.
4. **Depth-only sets (BigHand/HANDS17/ICVL/MSRA)** can't drive the 4-channel RGB-D
   stem directly (no RGB). Use them either for a separate depth-encoder pretrain, or
   stack the depth as a pseudo-RGB (repeat/colourise) for a depth-only ablation.

### ToF depth-only sets (MSRA / ICVL / NYU) — `datasets/tof_depth.py`
Implemented as `MSRA_RGBD`, `ICVL_RGBD`, `NYU_RGBD` on a shared `_ToFDepthBase`.
They have **no RGB** (pseudo-RGB = normalised depth ×3) and **partial joints**, so
they add `joint_valid [21]` and need a **masked 3D loss** in `train_mobrecon_rgbd.py`:
```python
out = model(img)
v = item_dict["joint_valid"].float().to(device)[..., None]          # [B,21,1]
loss_3d = (((out["keypoints"] - kps3d) * v) ** 2).sum() / (v.sum() * 3 + 1e-8)
# 2D loss: gate pred_xy/xy the same way (multiply both by v before MSE).
```
MSRA/BigHand/HANDS17/FPHA return `joint_valid = ones` (full 21, works unmasked);
ICVL has 16/21 valid (5 PIP absent), NYU 6/21 (fingertips+wrist). **Use these mainly for a ToF depth-branch
pre-train / fine-tune**, since the colour branch only sees depth. Each loader has
clearly-marked convention knobs (`MSRA.y_sign`, `NYU.kinect`, `*_MAP`, `*_TO_MANO`)
— run the `python -m datasets.tof_depth --dataset <d> --root <p> --check 8` viz
once on real data to confirm the 2D joints land on the hand before training.

### Toolkit-wrapped sets (ContactPose / OakInk) — `datasets/oakink_contactpose.py`
These are accessed through their own Python toolkits, so the loaders **wrap the
API** (lazy import, clear install hint) instead of reading raw files, and share an
`assemble_rgbd()` helper (same contract as ho3d `_finalize`).
- **ContactPose_RGBD** — real Kinect-v2 RGB-D; `hand_joints` are w.r.t. the object,
  mapped to the camera via `object_pose` (cTo). Index building instantiates each
  grasp once (one-time cost; scope with `p_nums`/`intents`/`limit`); grasps LRU-cached.
- **OakInk_RGBD** — oikit `OakInkImage` has **no depth getter**, so the depth channel
  is RENDERED from MANO joints (`depth_is_synthetic=True`). Diverse RGB+MANO, not a
  real-depth source. Real OakInk depth needs raw RealSense streams outside oikit.
Both assume OpenPose/MANO-21 == FreiHAND order (identity perm) — confirm with `--check`.

### Adapter template (reuse for HO3D, MVHand, …)
Copy `datasets/dexycb.py` and change only:
- `_build_index`: how frames/cameras/intrinsics are enumerated.
- label loading: field names for `joint_3d` (camera-frame metres) and the hand
  **segmentation/mask** used by `depth_mode="hand_masked"`.
- per-dataset `JOINT_PERM` if the 21-joint order differs from MANO.
Everything downstream (augmentation, depth warp+normalise, output dict) is shared.
