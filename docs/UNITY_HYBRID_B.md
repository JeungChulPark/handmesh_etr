# Running model B (hybrid_B) in the Unity app

Goal: view **model B** (`hybrid_B`, the locked-2D + backbone-Z lifter, 27.01 mm
combined) in the `unity/DepthRefinement` app. Date: 2026-07-04.

## Why B is not a file-swap

The current app (`DualStreamHandProvider`) runs the **self-contained** dual-stream
model `ds_anchor_gate.onnx`: it takes a bbox, builds a crop, and outputs absolute
joints (`image`, `depth_med`, `root_anchor`, `anchor_valid` → `keypoints_abs`).

**B has a different contract.** Its 2D is *locked* to the detector's 21 landmarks
and only its Z (relative depth) is learned. Overwriting the ONNX would break Sentis
input binding. B needs: the 4-ch crop **plus** the 21 landmarks, the sensor wrist
depth, and the camera intrinsics — then a deterministic fusion assembles the pose.

## What was done (verified, non-destructive)

1. **`scripts/export/export_hybrid_b_onnx.py`** exports B's backbone (the only learned part) to
   `unity/DepthRefinement/Assets/Models/hybrid_B_backbone.onnx`
   (`image[1,4,256,256] → keypoints[1,21,3]`, opset 17).
   Parity-checked: ONNX runtime output matches `HybridLifter.backbone` to **5.6e-8**
   (bit-exact). Placed *alongside* `ds_anchor_gate.onnx`, which is untouched.
2. **`HybridBHandProvider.cs`** (new) implements the locked fusion and reuses the
   proven crop pipeline — see below.

The backbone's 4-ch crop (RGB 0..1 + median-centred depth) is **identical** to the
dual-stream `image` tensor, so B reuses `DualStreamHandProvider`'s crop/depth/
intrinsics code instead of duplicating it.

## The locked fusion (what B's provider computes)

```
kp    = backbone(image)            // [21,3]; only kp[:,2] (relative depth) is used
Z0    = sensor depth at wrist landmark (metres)
dz_j  = kp[j].z - kp[0].z          // backbone per-joint relative depth (wrist = 0)
Z_j   = Z0 + dz_j
X_j   = (u_j - cx) / fx * Z_j       // u_j,v_j = landmark j in full-image pixels
Y_j   = (v_j - cy) / fy * Z_j
```

In-plane crop rotation does not affect depth, so no output un-rotation is needed.

## Remaining wiring — 3 small additive hooks

`HybridBHandProvider.cs` reads three things the existing scripts already have but
don't yet expose. Add these (all additive, non-breaking):

**A. `HandBboxBridge.cs`** — retain the 21 landmarks (today only the bbox is kept):
```csharp
readonly List<Vector2> _lm = new List<Vector2>(21);
public IReadOnlyList<Vector2> LandmarksNormalized => _lm;
// inside SetLandmarksNormalized(pts):  _lm.Clear(); _lm.AddRange(pts);
```

**B. `DualStreamHandProvider.cs`** — crop-only mode + expose crop/intrinsics/depth.
Cache `fx,fy,cx,cy,imgW,imgH` into fields in `Update()`, allow a null `_modelAsset`
(skip `_worker`/`RunModel`, still build the crop), and add:
```csharp
public bool TryGetLastImageTensor(out float[] img) { img = _input; return _hasCrop; }
public bool TryGetLastIntrinsics(out float fx,out float fy,out float cx,out float cy,out int w,out int h){ /* return cached */ }
public bool TrySampleDepthMetres(float u01, float v01, out float z) {  // nearest sample of _depth
    z = 0f; if (!_depth.IsCreated) return false;
    int dx = Mathf.Clamp((int)(u01*_depthW),0,_depthW-1), dy = Mathf.Clamp((int)(v01*_depthH),0,_depthH-1);
    z = _depth[dy*_depthW+dx]; return z>0f; }
```

**C. Scene** — add a `HybridBHandProvider`, assign `hybrid_B_backbone.onnx`, point
`_crop` at the crop-only `DualStreamHandProvider` and `_landmarks` at the
`HandBboxBridge`. Drive the visualizer (`HandSphereDriver`) from
`HybridBHandProvider.AbsJoints` instead of the dual-stream provider.

## Caveats

* **Landmark order.** B was trained on **MediaPipe** 21-point order; the app's
  `HandVisionBridge` uses **Apple Vision**. Confirm the native `HandPoseVision.swift`
  emits MediaPipe order (the Python `landmarks_to_bbox` contract) — if not, remap
  before `SetLandmarksNormalized`. B's X,Y are the landmarks verbatim, so order/
  accuracy errors surface directly in the output.
* **Depth mode.** B anchors the root on sensor wrist depth. On the ZED, PERFORMANCE
  depth moved the root ~90 mm and the pose ~104 mm vs NEURAL — use the accurate
  depth mode on device (ARKit LiDAR is dense, so this is mainly a warning to keep
  the wrist sample robust; the provider already medians valid joints as a fallback).
* Cannot be compiled/tested outside Unity here — `HybridBHandProvider.cs` is a
  reference against the project's confirmed Sentis 2.x idioms; compile + wire in the
  editor.

## Files

| Path | Role |
|---|---|
| `scripts/export/export_hybrid_b_onnx.py` | **new** — export B backbone → ONNX (parity-checked). |
| `unity/.../Assets/Models/hybrid_B_backbone.onnx` | **new** — verified B backbone. |
| `unity/.../Assets/Runtime/Recon/HybridBHandProvider.cs` | **new** — locked-fusion provider. |
| `unity/.../Assets/Runtime/Recon/DualStreamHandProvider.cs` | reused crop pipeline (add hooks B). |
| `unity/.../Assets/Runtime/Recon/HandBboxBridge.cs` | landmark source (add hook A). |
