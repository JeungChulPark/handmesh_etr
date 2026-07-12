# Dual-stream RGB-D hand model on iPhone (Unity Sentis)

`DualStreamHandProvider.cs` runs the **latest** dual-stream model (`ds_anchor_gate`)
end-to-end on iPhone: RGB (ARKit camera) + LiDAR depth → **absolute** camera-space
3D joints. It reproduces the Python preprocessing exactly
(`infer_zed_dualstream.py` + `datasets/depth_synth.py`), so on-device results match
the offline model.

Unlike `SentisHandJointProvider` (RGB-only, 1 input → relative joints + separate depth
refinement), this provider feeds the model's **4 inputs** and reads its learned
absolute output directly.

## Export the model

```
python convert_dualstream.py --ckpt mobrecon_ckpt/ds_anchor_gate/best.pt
# -> pretrain/ds_anchor_gate.onnx   (import into Unity as a ModelAsset)
```
Verified I/O: `image[1,4,256,256]`, `depth_med[1,1]`, `root_anchor[1,3]`,
`anchor_valid[1,1]` → `keypoints`, `root`, `scale`, `keypoints_abs[1,21,3]`.

## What it computes each frame (mirrors training)

| Model input | Built from |
|---|---|
| `image[0:3]` | ARKit camera CPU image → native crop (hand bbox) + resize to 256², RGB/255 |
| `image[3]`   | `normalize_depth(depth_crop, 0.1)` — median-centred LiDAR depth, [-1,1], bg 0 |
| `depth_med`  | median of the valid LiDAR depth crop (metres) |
| `root_anchor`, `anchor_valid` | `geometric_root_anchor(depth_crop, crop_cam)` — back-projected hand-depth centroid + MAD/hole confidence |

`crop_cam` is derived analytically from the camera intrinsics + the bbox (anisotropic
bbox→256, matching the test-mode augmentation).

## Scene setup (iPhone Pro, LiDAR)

1. AR rig: `ARSession`, `XR Origin` with `ARCameraManager` **and** `AROcclusionManager`
   (Environment Depth Mode = Best/Medium; temporal smoothing off for fast hands).
2. Add `DualStreamHandProvider`; assign:
   - **Camera Manager**, **Occlusion Manager**,
   - **Model Asset** = `ds_anchor_gate.onnx`,
   - leave preprocess params at defaults (they match training: normScale 0.1, win 8,
     minValid 50, madRef 0.02).
3. Each frame set `HandBboxNormalized` from your hand detector (sensor-image normalized
   coords, **pre-rotation** — same space as `ARCameraImageProvider.CropRectNormalized`).
   Without a detector it uses a centred box (fine for a first smoke test).
4. Read results: `provider.AbsJoints` (21 × `Vector3`, metres), `provider.Root`,
   `provider.Scale`, or subscribe to `OnPose`.

## Output coordinate frame

`AbsJoints` are in **CV camera space** (+X right, +Y down, +Z forward) in the camera's
**sensor** orientation — the frame the model was trained in. To draw in Unity world:

```csharp
// p = provider.AbsJoints[i] (CV camera space, metres)
Vector3 unityCam = new Vector3(p.x, -p.y, p.z);          // CV -> Unity (flip Y)
Vector3 world = arCamera.transform.TransformPoint(unityCam);
```

## Requirements / caveats

- **Hand bbox required.** Reuse your Android detector (MediaPipe etc.); feed its bbox
  into `HandBboxNormalized`. The model runs on the hand CROP, not the full frame.
- **iPhone Pro** (LiDAR). On non-LiDAR devices `AROcclusionManager` has no environment
  depth → provider returns no pose. (A monocular metric-depth model could fill in later.)
- **Sentis 2.x** API. If you pin a version without no-arg `Schedule()`, switch to
  `_worker.Schedule(image, med, ra, va)` (model input order).
- **Domain note:** the dual-stream model collapsed on ZED passive stereo, but iPhone
  LiDAR is ToF — the depth domain it was trained for — so it should behave far better
  here. This on-device run is the validation. See the repo memory
  `zed-dualstream-collapse-vs-earlyfusion`.
- For a known-good baseline, the **early-fusion** RGB model (`SentisHandJointProvider` +
  `rgbd_real_0613`) is the safe fallback if the dual-stream output looks off on device.
```
