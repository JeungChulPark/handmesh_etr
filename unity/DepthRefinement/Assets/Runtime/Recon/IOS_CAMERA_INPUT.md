# iPhone camera input for the hand model in Unity

How to extract the iPhone camera frame in Unity and feed it to the model — the
cross-platform replacement for the **Android-only ARCore** image extraction you
used before. Implemented by `ARCameraImageProvider.cs` (this folder).

## Why your Android code didn't carry over

On Android you grabbed RGB with ARCore directly (e.g. `TextureReader` /
`Frame.AcquireCameraImage`). That API is **Android-only**. The portable answer is
**AR Foundation**, which wraps **ARKit (iOS)** and **ARCore (Android)** behind one
API — so a single script feeds your model on **both** phones.

```
ARKit (iOS) ─┐
             ├─ AR Foundation  ──  ARCameraManager.TryAcquireLatestCpuImage()  ── model
ARCore (And.)┘                     (YUV→RGBA, downscale, rotate upright)
```

## Two ways to get the camera image

| Option | Use when |
|---|---|
| **AR Foundation `ARCameraManager`** (this implementation) | You also need LiDAR depth / AR tracking (the dual-stream RGB-D model). The proper ARKit path. |
| **`WebCamTexture`** (Unity built-in) | Quick RGB-only smoke test, no AR. `var cam = new WebCamTexture(); cam.Play();` then Blit/resize to 256. |

The dual-stream RGB-D model needs LiDAR depth, which comes from the **same AR rig**,
so AR Foundation is the right choice end-to-end.

## The flow (what `ARCameraImageProvider` does each frame)

1. `ARCameraManager.frameReceived` fires → `TryAcquireLatestCpuImage(out XRCpuImage image)`.
2. `image.Convert(...)` turns iOS **YUV 420** into **RGBA32**, cropping `CropRectNormalized`
   and downscaling to `ModelInputSize`² in native code (cheap, no GC).
3. CPU-rotate sensor-landscape → screen-upright (auto from `Screen.orientation`).
4. Expose the result as:
   - `OutputTexture` (a `Texture2D`) — for **Unity Sentis** (`TextureConverter.ToTensor`),
   - `GetInputTensorNHWC()` (a `float[]`, NHWC RGB, [0,1]) — for a **TFLite** interpreter.

## Scene setup (iOS)

1. AR rig: `ARSession`, `XR Origin` with an `ARCameraManager` (and `AROcclusionManager`
   if you want LiDAR depth — see `ARFoundationDepthProvider`).
2. Add `ARCameraImageProvider` to the AR camera object; assign the `ARCameraManager`,
   set **Model Input Size** = your model's input (e.g. 256).
3. Read the frame each `Update()` and run the model (examples below).

## Feeding TFLite (your existing Android model, same plugin on iOS)

The TFLite Unity plugin (e.g. `com.github.asus4.tflite`) builds for iOS too — same
`.tflite`, same C#. Input is `[1,256,256,3]` float, [0,1] (matches
`pretrain/saved_model/handmesh_etr_float32.tflite`).

```csharp
[SerializeField] ARCameraImageProvider _camera;
TensorFlowLite.Interpreter _interp; // your existing interpreter
readonly float[,,,] _in = new float[1, 256, 256, 3];
readonly float[,,] _out = new float[1, 21, 3];

void Update()
{
    if (!_camera.HasFrame) return;
    float[] nhwc = _camera.GetInputTensorNHWC();        // 256*256*3, [0,1]
    System.Buffer.BlockCopy(nhwc, 0, _in, 0, nhwc.Length * sizeof(float));
    _interp.SetInputTensorData(0, _in);
    _interp.Invoke();
    _interp.GetOutputTensorData(0, _out);               // (1,21,3) root-relative joints
    // ...draw / refine with LiDAR depth...
}
```

> On iOS, add the **GPU (Metal)** or **Core ML** TFLite delegate for speed; CPU works as a fallback.

## Feeding Unity Sentis (the .onnx path)

```csharp
[SerializeField] ARCameraImageProvider _camera;
[SerializeField] SentisHandJointProvider _recon;   // existing provider in this folder

void Update()
{
    if (_camera.HasFrame) _recon.InputTexture = _camera.OutputTexture;
}
```

## iOS build settings (Player Settings → iOS)

- **Camera Usage Description** (`NSCameraUsageDescription`): required, else the app is
  rejected/crashes on camera access. (`ios/HandPoseLiDAR/Info.plist.snippet` has an example.)
- **XR Plug-in Management → iOS → Apple ARKit** enabled; add **ARKit XR Plugin** package.
- **Graphics API = Metal**, **Architecture = ARM64**, **Target minimum iOS = 14+**
  (16+ if you also load the Core ML `.mlpackage`).
- For LiDAR depth: a **Pro** iPhone/iPad, `AROcclusionManager` with Environment Depth on.
- **Requires ARKit support** ON if the app is AR-only.

## Orientation notes

The camera sensor is landscape; we rotate to portrait via `Screen.orientation`
(auto). If the hand looks rotated/mirrored on device, toggle **Auto Rotate** off and
set **Rotation** / **Mirror Horizontal** in the Inspector (front camera usually needs
mirror; back-camera portrait is usually `CW90`).

## Cropping to the hand

The model was trained on hand **crops**, not full frames. Set
`ARCameraImageProvider.CropRectNormalized` (sensor space, pre-rotation) to your hand
bbox each frame — same idea as the Android pipeline / `SentisHandJointProvider._cropRect`.
Full-frame (default) is fine for a first smoke test.
