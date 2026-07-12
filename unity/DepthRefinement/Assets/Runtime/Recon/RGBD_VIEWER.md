# iPhone RGB-D viewer app (Unity)

Show the iPhone camera **RGB** + LiDAR **depth** on screen. Two pieces:

| Stream | How |
|---|---|
| **RGB** | AR Foundation **`ARCameraBackground`** draws the camera feed fullscreen automatically (handles orientation/aspect). No code. |
| **Depth** | `DepthImageVisualizer.cs` (this folder) colorizes LiDAR depth (near=red … far=blue) into a `Texture2D` and shows it on a UI `RawImage`. |

Result: live camera with a colorized depth overlay = an RGB-D viewer. Requires an
iPhone/iPad **Pro** (LiDAR).

## Scene setup

1. **AR rig**
   - `GameObject → XR → AR Session`
   - `GameObject → XR → XR Origin (AR)` — its child **Main Camera** gets:
     - `ARCameraManager`
     - `ARCameraBackground`   ← draws the RGB feed
     - `AROcclusionManager`   ← LiDAR depth. Set **Environment Depth Mode = Medium/Best**
       (Temporal Smoothing off for fast motion).
2. **Depth overlay UI**
   - `GameObject → UI → Canvas` (Render Mode = Screen Space - Overlay).
   - Under it `UI → Raw Image`; size/anchor it to a corner (e.g. 360×480 top-right).
3. **Visualizer**
   - Add `DepthImageVisualizer` (anywhere, e.g. on the Canvas).
   - Assign **Occlusion Manager** (the Main Camera's `AROcclusionManager`) and **Target**
     (the RawImage). Tune **Min/Max Depth** (default 0.2–3.0 m) for your colormap range.

Press Play on device → camera fills the screen, depth shows colorized in the corner.

## Optional: RGB also in a RawImage (side-by-side)

If you want RGB as its own image (not just the fullscreen background), add a second
`RawImage` and feed it from `ARCameraImageProvider` (set its `CropRectNormalized` to the
full frame and increase `ModelInputSize`, or use it as-is):

```csharp
[SerializeField] ARCameraImageProvider _rgb;
[SerializeField] RawImage _rgbImage;
void Update() { if (_rgb.HasFrame) _rgbImage.texture = _rgb.OutputTexture; }
```

## iOS build settings (Player Settings → iOS)

- **Camera Usage Description** (`NSCameraUsageDescription`) — required.
- **XR Plug-in Management → iOS → Apple ARKit** enabled (+ **ARKit XR Plugin** package).
- **Graphics API = Metal**, **Architecture = ARM64**, **Target minimum iOS = 14+**.
- **Requires ARKit** if the app is AR-only. Build needs a **Mac + Xcode** (iOS toolchain
  is macOS-only — you can author on Ubuntu but must build/sign on a Mac).

## Notes

- **LiDAR only.** On non-Pro devices `TryAcquireEnvironmentDepthCpuImage` returns nothing
  → `HasFrame` stays false (no depth overlay). RGB still shows.
- Depth is **low-res (~256×192)**; the RawImage scales it up (blocky is normal).
- `DepthImageVisualizer.Coverage` gives the valid-pixel fraction; `Texture` exposes the
  colorized texture if you want to save/stream it.
- If depth looks rotated/mirrored on device, toggle **Rotate 90 CW** on the visualizer.
- This viewer is a good **first on-device smoke test** before wiring the model
  (`DualStreamHandProvider`) — it confirms camera + LiDAR are flowing.
```
