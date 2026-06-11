# RGB-D Hand-Pose Retraining

Early-fusion **RGB-D** variant of the MobRecon-style joint regressor, built on the
conclusion of the `depth_refine/` study: RGB-inferred monocular depth is a dead
end, but a real metric depth sensor has a large upper bound (root-rel MPJPE
31.1 → 14.2 mm, −54%), realisable only by **training** the network on paired
depth rather than post-processing. This package adds that training path.

HanCo (the training set) ships multi-view mocap GT (21 camera-space joints) but
**no sensor depth**, so we render a metric depth channel from the GT geometry and
corrupt it with a tunable sensor-noise model. This lets us validate the RGB-D
architecture end-to-end today and sweep sensor quality before committing to a
real paired-depth capture.

## What was added / changed

| File | Change |
|------|--------|
| `datasets/depth_synth.py` | **New.** numpy-only depth synthesis: z-buffer mesh rasteriser, joints→tapered-capsule mesh, sensor noise model, normalisation. Self-test: `python -m datasets.depth_synth`. |
| `datasets/gen_hanco_depth.py` | **New.** Pre-renders a metric-depth cache for HanCo from `xyz`+`calib` (8 cams) to `depth/<scene>/cam<c>/<frame>.png` (uint16 mm). Multiprocessing, resumable. |
| `datasets/hanco_ty.py` | `HanCo_ETRI_jitter.__init__(..., with_depth=False, depth_cfg=None, depth_source="render", depth_root=None)`. When `with_depth=True`, `__getitem__` returns a **4-channel** `[R,G,B,D]` image; `D` is rendered (or loaded from the cache when `depth_source="cache"`) from the camera-space GT joints, warped into the same crop as the RGB (`img2bb_trans`), corrupted, and median-centred to ~[-1, 1]. Default off → original behaviour unchanged. |
| `models/densestack.py` | `DenseStack_Backbone_like_prev(..., in_chans=3)`; the stem conv now accepts `in_chans` (3=RGB, 4=RGB-D). Default 3 keeps every existing model byte-identical. |
| `models/mobrecon_ds.py` | **New** `LargeModel_Extra_RGBD(cfg)` (+ `_onnx` twin). Reads `cfg.MODEL.IN_CHANS` (default 4); otherwise identical to `LargeModel_Extra`. |
| `train_mobrecon_rgbd.py` | **New.** Mirror of `train_mobrecon_jittering_3d.py` (same 3D L2 + 2D reprojection losses, optimiser, schedule, wandb logging) feeding 4-channel input, with a depth-channel wandb visual and an `--rgb_only` ablation. |
| `configs_rgbd.yaml` | **New.** `MODEL.IN_CHANS: 4` + `DEPTH:` block. |

## Architecture (early fusion)

```
HanCo RGB [3,256,256] ─┐
                       ├─ concat ─► [4,256,256] ─► conv_layer(4→64) ─► DenseStack ─► 21×3 joints
GT joints ─► capsule mesh ─► z-buffer depth ─► warp(crop) ─► +noise ─► median-centre ─► D[1,256,256] ─┘
```

- **Why early fusion:** minimal change, mobile-friendly (one extra input channel,
  no second encoder), reuses the entire backbone. A two-stream encoder is the
  natural next experiment if early fusion underperforms.
- **Depth normalisation is inference-safe:** depth is centred on its own per-frame
  *median* (a statistic a real sensor also provides at test time), so the network
  never sees absolute camera distance and cannot overfit to it. Background stays
  exactly 0, distinct from the centred hand.
- **Alignment:** depth is rendered at the original image resolution from the GT
  joints + original intrinsics, then warped by the **same** `img2bb_trans` affine
  used for the RGB crop. Crop rotation is about the optical axis, so camera-space
  z (the depth value) is preserved by the warp.

## Run

```bash
# self-test the renderer (no dataset needed)
python -m datasets.depth_synth

# (optional, one-off) pre-render the HanCo depth cache (~543k frames, ~2.3 GB,
# ~12 min on 22 cores). Skips frames already cached, so it is resumable.
python -m datasets.gen_hanco_depth                    # subset: --limit_scenes 2
# then enable cache loading in code: HanCo_ETRI_jitter(..., depth_source="cache")

# train RGB-D (depth rendered on-the-fly; pass depth_source="cache" in the loader
# to use the pre-rendered cache instead)
python train_mobrecon_rgbd.py --exp rgbd_v0 --cfg ./configs_rgbd.yaml --batch 32

# sensor-quality sweep (noisier depth)
python train_mobrecon_rgbd.py --exp rgbd_sigma10 --depth_sigma 0.010 --depth_dropout 0.10

# RGB-only ablation through the exact same pipeline (4th channel zeroed)
python train_mobrecon_rgbd.py --exp rgb_only --rgb_only
```

The key result to watch is `test_mpjpe`: **RGB-D should beat the `--rgb_only`
ablation**, and the gap should shrink as `--depth_sigma`/`--depth_dropout` rise —
quantifying how good a real sensor must be.

## Swapping in better geometry / real depth

The rasteriser is geometry-agnostic. Two upgrade paths, both via the same API:

1. **Real MANO mesh instead of the capsule proxy** (more faithful GT depth).
   Load HanCo MANO fits (the dataset's `shape/` dir: pose/shape/global_R/global_t)
   and run `models/manolayer.ManoLayer` to get `verts (778,3)` + `faces (1538,3)`
   (see `datasets/interhand.py:74-90` for the call pattern), then in
   `_render_depth_channel`:
   ```python
   from datasets.depth_synth import render_hand_depth
   depth_full = render_hand_depth(intr, H, W, verts=verts_cam, faces=faces)
   ```
   `render_hand_depth` already prefers `(verts, faces)` over `joints` when given.

2. **Real sensor depth** (the eventual production path). Load the sensor depth
   map (metres, camera frame), warp it with `img2bb_trans` exactly like the RGB,
   pass through `add_sensor_noise(..., sigma=0)` (or skip) and `normalize_depth`,
   and concat. The model and training script need **no** change — only the
   `_render_depth_channel` body.

## Real-time deployment — Femto Bolt (Orbbec) RGB-D camera

`infer_rgbd_femtobolt.py` is the **live** counterpart of `eval_rgbd.py`: it streams
RGB + metric depth from a Femto Bolt, finds the hand with MediaPipe, runs the
exact same 4-channel preprocessing the trainer used (it *reuses* the project's
`augmentation()` + `normalize_depth`), and recovers camera-space 3D joints. The
checkpoint loads as-is — the **only** runtime differences from training are the
depth *source* (real sensor instead of the capsule renderer — upgrade path #2
above) and the bbox *source* (MediaPipe at runtime instead of GT joints offline).

```
Femto Bolt ─► RGB ─► MediaPipe Hands ─► square bbox ─┐
          └─► depth(m, HW-aligned to colour) ─────────┤ augmentation('test') → img2bb_trans
   RGB  →warpAffine(LINEAR)→[3,256,256]/255 ──────────┤
   depth→warpAffine(NEAREST)→normalize_depth→[1,256,256]┴ concat →[1,4,256,256]→ model → 21×3 (root-rel m)
   root = backproject(wrist_px, depth@wrist, K)  →  absolute camera-space joints
```

```bash
# verify model + preprocessing wiring with NO hardware (synthetic 4-ch tensor)
python infer_rgbd_femtobolt.py --ckpt mobrecon_ckpt/rgbd_1hr/3.pt --source selftest

# live camera (needs pyorbbecsdk + Femto Bolt)
python infer_rgbd_femtobolt.py --ckpt mobrecon_ckpt/rgbd_1hr/3.pt --source femtobolt

# replay recorded rgb_*.png + depth_*.png(uint16 mm) pairs, save overlays headless
python infer_rgbd_femtobolt.py --ckpt mobrecon_ckpt/rgbd_1hr/3.pt \
       --source folder --path captures/ --no-gui --save out/femto_eval

# RGB webcam (depth zeroed == the --rgb_only ablation; for the bbox/loop only)
python infer_rgbd_femtobolt.py --ckpt mobrecon_ckpt/rgbd_1hr/3.pt --source webcam
```

**Dependencies** (beyond training):
- `mediapipe` (Tasks API; this build ships only `mediapipe.tasks`, not the legacy
  `mp.solutions`). The `hand_landmarker.task` model auto-downloads to
  `~/.cache/mediapipe/` on first run; pass `--hand_model <path>` if offline.
- `pyorbbecsdk` for the live `--source femtobolt` only (other sources need no
  SDK). **Do NOT `pip install pyorbbecsdk` from PyPI** — that 1.3.2 wheel is a
  mislabelled macOS build (ships `.dylib` + a darwin `.so`) and won't import on
  Linux. Install a prebuilt **Linux wheel from the GitHub Releases** instead
  (Femto Bolt needs Orbbec SDK v2):
  ```bash
  # cp310 / x86_64 example; pick the wheel matching your Python & arch
  pip install https://github.com/orbbec/pyorbbecsdk/releases/download/v2.0.15/pyorbbecsdk-2.0.15-cp310-cp310-linux_x86_64.whl
  # then install udev rules for non-root USB access (from the pyorbbecsdk repo):
  #   sudo bash scripts/install_udev_rules.sh && sudo udevadm control --reload
  ```
  Depth is **hardware-aligned to colour**: `FemtoBoltSource` pairs a D2C depth
  profile with the colour profile via `get_d2c_depth_profile_list(.., HW_MODE)`
  (per the SDK's `examples/hw_d2c_align.py`), so depth and RGB share the colour
  intrinsics (`get_camera_param().rgb_intrinsic`). Colour is decoded
  format-aware (RGB/BGR/MJPG/YUYV/NV12/...), so MJPG streams just work.

  **Verified without a camera (2026-06-11, pyorbbecsdk 2.0.15 Linux x86_64):**
  every symbol/method `FemtoBoltSource` calls exists in the installed SDK and
  matches the official examples; device enumeration runs (0 devices == none
  attached); `--source femtobolt` exits with a clear "no device" message instead
  of crashing. The live capture path itself is unrun pending hardware.

**Validation done** (no hardware): `selftest` passes; on real HanCo samples the
predicted **hand scale matches GT** (xy-span within ~5%, e.g. 13.9 vs 13.6 cm) at
the checkpoint's known ~33 mm MPJPE — confirming the deploy preprocessing is
identical to training. The model under-predicts z-span (~6 vs ~12 cm), a weak
3-epoch-checkpoint artefact that the 100-epoch path closes, not a pipeline bug.
Feeding *flat/arbitrary* depth collapses the prediction (the model leans hard on
an informative depth channel) — so a real, in-distribution sensor depth matters.

## Limitations / honesty notes

- **Synthetic depth has a sim-to-real gap.** Capsule-mesh depth is a coarse hand
  proxy (no soft tissue, fixed radii, no objects/occluders). Treat `test_mpjpe`
  here as an *optimistic* estimate; the capsule→MANO→real-sensor ladder above
  closes the gap.
- The depth noise model (Gaussian + speckle dropout + rectangular holes) is a
  first-order ToF/structured-light approximation, not a calibrated sensor model.
- Per the study, the realistic sensor gain is **conditional** on accurate 2D and
  surface-offset/occlusion handling — so a null or small RGB-D win at high noise
  is an expected, informative outcome, not necessarily a bug.
```
