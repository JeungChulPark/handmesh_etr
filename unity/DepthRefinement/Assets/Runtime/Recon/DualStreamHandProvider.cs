#if SENTIS_PRESENT && ARFOUNDATION_PRESENT
using System;
using Unity.Collections;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;   // XRCpuImage, XRCameraIntrinsics

namespace HandMesh.DepthRefinement
{
    /// <summary>
    /// Runs the DUAL-STREAM RGB-D hand model (ds_anchor_gate.onnx) on iPhone via Unity Sentis,
    /// reproducing the exact training/inference preprocessing from the Python pipeline
    /// (infer_zed_dualstream.py + datasets/depth_synth.py). Unlike the RGB-only
    /// <see cref="SentisHandJointProvider"/>, this model predicts ABSOLUTE camera-space joints
    /// itself (learned root + scale), so it needs RGB + LiDAR depth + two depth-derived signals.
    ///
    /// Per frame, for a hand bbox (<see cref="HandBboxNormalized"/>, sensor-image space):
    ///   1. RGB crop  : ARKit camera CPU image -> native crop+resize to 256x256 RGBA (0..1).
    ///   2. depth crop: ARKit LiDAR depth (metres), sampled over the SAME bbox to 256x256.
    ///   3. depth_med : median of the valid depth crop (metres)  -- absolute-distance cue.
    ///   4. image[3]  : normalize_depth(depth_crop) = median-centred depth in [-1,1].
    ///   5. anchor    : geometric_root_anchor(depth_crop, crop_cam) -> root_anchor[3] + confidence.
    ///   6. model("image"[1,4,256,256], "depth_med", "root_anchor", "anchor_valid")
    ///        -> "keypoints_abs" [1,21,3] absolute camera-space metres.
    ///
    /// Output joints are in CV camera space (+X right, +Y down, +Z forward) in the camera's
    /// SENSOR orientation; convert to Unity world via the AR camera transform for display.
    ///
    /// REQUIRES a hand bbox each frame (set <see cref="HandBboxNormalized"/> from your detector,
    /// in sensor-image normalized coords). Export the model with `convert_dualstream.py`.
    /// </summary>
    public sealed class DualStreamHandProvider : MonoBehaviour
    {
        const int Size = 256;        // model input side
        const int JointCount = 21;

        /// <summary>Rotation applied to the sensor crop so the hand is UPRIGHT for the model.
        /// <c>Auto</c> derives it from the live <see cref="Screen.orientation"/> so ONE build works
        /// on iPhone AND iPad in any orientation (recommended); the fixed values force one rotation.</summary>
        public enum SensorRotation { None = 0, CW90 = 90, Rot180 = 180, CW270 = 270, Auto = -1 }

        [Header("AR sources")]
        [SerializeField] ARCameraManager _cameraManager;
        [SerializeField] AROcclusionManager _occlusionManager;

        [Header("Model")]
        [SerializeField] ModelAsset _modelAsset;   // ds_anchor_gate.onnx
        [SerializeField] BackendType _backend = BackendType.GPUCompute;

        [Header("Preprocess (must match training)")]
        [Tooltip("normalize_depth scale: metres mapped to 1.0 (depth_synth.py default 0.1).")]
        [SerializeField] float _normScale = 0.1f;
        [Tooltip("geometric_root_anchor window half-size (px in the 256 crop).")]
        [SerializeField] int _anchorWin = 8;
        [SerializeField] int _anchorMinValid = 50;
        [SerializeField] float _anchorMadRef = 0.02f;

        [Header("Sensor -> display orientation")]
        [Tooltip("Rotate the sensor crop so the hand is UPRIGHT for the model, and so output joints " +
                 "land in the DISPLAY camera frame. Auto = follow the device orientation (works on " +
                 "iPhone + iPad, portrait/landscape). Use a fixed value only to override; if the hand " +
                 "comes out rotated ~180°, the Auto map for that orientation needs the opposite value.")]
        [SerializeField] SensorRotation _sensorRotation = SensorRotation.Auto;

        /// <summary>Hand bounding box in SENSOR-image normalized coords (0..1), set each frame
        /// from your hand detector. Default = centred square. Pre-rotation (sensor) space.</summary>
        public Rect HandBboxNormalized { get; set; } = new Rect(0.30f, 0.30f, 0.40f, 0.40f);

        /// <summary>Absolute camera-space joints (metres, CV frame). Valid when <see cref="HasPose"/>.</summary>
        public Vector3[] AbsJoints { get; } = new Vector3[JointCount];
        public Vector3 Root { get; private set; }
        public float Scale { get; private set; }
        public bool HasPose { get; private set; }
        public event Action OnPose;

        Model _model;
        Worker _worker;

        // persistent scratch (GC-free steady state)
        NativeArray<byte> _rgbaCrop;     // 256*256*4
        NativeArray<float> _depth;       // depthW*depthH metres
        int _depthW, _depthH;
        readonly float[] _input = new float[4 * Size * Size]; // NCHW image tensor
        readonly float[] _depthCrop = new float[Size * Size]; // metres, 0 = invalid
        readonly float[] _sortBuf = new float[Size * Size];   // median scratch

        // per-frame crop + intrinsics cache, exposed for HybridBHandProvider (crop-only reuse)
        bool _hasCrop;
        float _fx, _fy, _cx, _cy;
        int _imgW, _imgH;

        void Awake()
        {
            if (_cameraManager == null) _cameraManager = FindObjectOfType<ARCameraManager>();
            if (_occlusionManager == null) _occlusionManager = FindObjectOfType<AROcclusionManager>();
            // ModelAsset optional: with none, run "crop-only" — still builds the 4-ch crop +
            // intrinsics + depth each frame for HybridBHandProvider, but produces no dual-stream pose.
            if (_modelAsset != null)
            {
                _model = ModelLoader.Load(_modelAsset);
                _worker = new Worker(_model, _backend);
            }
            else
            {
                Debug.LogWarning("[DualStream] No ModelAsset -> crop-only mode (feeds HybridBHandProvider).");
            }
        }

        void OnDestroy()
        {
            _worker?.Dispose();
            if (_rgbaCrop.IsCreated) _rgbaCrop.Dispose();
            if (_depth.IsCreated) _depth.Dispose();
        }

        // Mode switches disable this component; never leave a stale pose behind.
        void OnDisable() => HasPose = false;

        void Update()
        {
            HasPose = false;
            _hasCrop = false;
            if (_cameraManager == null) return;
            if (!TryGetIntrinsicsPx(out float fx, out float fy, out float cx, out float cy,
                                    out int imgW, out int imgH)) return;
            if (!AcquireDepth()) return;
            if (!AcquireRgbCrop(imgW, imgH)) return;

            // bbox in camera-image pixels
            Rect bb = HandBboxNormalized;
            float bx = bb.x * imgW, by = bb.y * imgH, bw = bb.width * imgW, bh = bb.height * imgH;
            if (bw < 1f || bh < 1f) return;

            // crop intrinsics: bbox -> 256, anisotropic (matches augmentation test mode)
            float sx = Size / bw, sy = Size / bh;
            float cfx = fx * sx, cfy = fy * sy;
            float ccx = (cx - bx) * sx, ccy = (cy - by) * sy;

            BuildDepthCrop(bb, imgW, imgH);          // fills _depthCrop (metres)
            float depthMed = MedianValid(_depthCrop, out int nValid);
            BuildImageTensor(depthMed, nValid);      // RGB(0..1) + normalize_depth into _input

            // crop + full-image intrinsics are ready this frame (used by HybridBHandProvider too)
            _fx = fx; _fy = fy; _cx = cx; _cy = cy; _imgW = imgW; _imgH = imgH;
            _hasCrop = true;

            if (_worker == null) return;   // crop-only mode: no dual-stream pose

            GeometricRootAnchor(cfx, cfy, ccx, ccy, nValid,
                                out Vector3 anchor, out float anchorValid);
            anchor = RotateAnchor(anchor);   // sensor-frame anchor -> display frame (matches output)

            RunModel(depthMed, anchor, anchorValid);
            HasPose = true;
            OnPose?.Invoke();
        }

        // ---- crop-only reuse API (consumed by HybridBHandProvider) ----------------------

        /// <summary>This frame's 4-ch model crop (RGB 0..1 + median-centred depth, NCHW 4*256*256).
        /// Returns false until a crop is built; works even in crop-only mode (no ModelAsset).</summary>
        public bool TryGetLastImageTensor(out float[] image) { image = _input; return _hasCrop; }

        /// <summary>This frame's FULL-IMAGE (sensor) intrinsics in pixels.</summary>
        public bool TryGetLastIntrinsics(out float fx, out float fy, out float cx, out float cy,
                                         out int imgW, out int imgH)
        {
            fx = _fx; fy = _fy; cx = _cx; cy = _cy; imgW = _imgW; imgH = _imgH;
            return _hasCrop;
        }

        /// <summary>Nearest sensor-depth sample (metres) at a normalized sensor-image point (0..1);
        /// false when there is no valid depth there.</summary>
        public bool TrySampleDepthMetres(float u01, float v01, out float z)
        {
            z = 0f;
            if (!_depth.IsCreated || _depthW <= 0 || _depthH <= 0) return false;
            int dx = Mathf.Clamp((int)(u01 * _depthW), 0, _depthW - 1);
            int dy = Mathf.Clamp((int)(v01 * _depthH), 0, _depthH - 1);
            z = _depth[dy * _depthW + dx];
            return z > 0f && !float.IsNaN(z) && !float.IsInfinity(z);
        }

        // ---- AR acquisition ------------------------------------------------------------

        bool TryGetIntrinsicsPx(out float fx, out float fy, out float cx, out float cy,
                                out int imgW, out int imgH)
        {
            fx = fy = cx = cy = 0; imgW = imgH = 0;
            if (!_cameraManager.TryGetIntrinsics(out XRCameraIntrinsics intr)) return false;
            if (intr.resolution.x <= 0 || intr.resolution.y <= 0) return false;
            // We crop in normalized coords, so express intrinsics at the intrinsics resolution
            // (bbox px below uses the same resolution via imgW/imgH).
            imgW = intr.resolution.x; imgH = intr.resolution.y;
            fx = intr.focalLength.x; fy = intr.focalLength.y;
            cx = intr.principalPoint.x; cy = intr.principalPoint.y;
            return true;
        }

        bool AcquireDepth()
        {
            if (_occlusionManager == null) return false;
            if (!_occlusionManager.TryAcquireEnvironmentDepthCpuImage(out XRCpuImage img)) return false;
            using (img)
            {
                _depthW = img.width; _depthH = img.height;
                EnsureNativeF(ref _depth, _depthW * _depthH);
                var raw = img.GetPlane(0).data;
                if (img.format == XRCpuImage.Format.DepthUint16)
                {
                    var src = raw.Reinterpret<ushort>(1);
                    int n = Mathf.Min(src.Length, _depth.Length);
                    for (int i = 0; i < n; i++) _depth[i] = src[i] * 0.001f; // mm -> m
                }
                else // DepthFloat32 (ARKit) or best-effort
                {
                    var src = raw.Reinterpret<float>(1);
                    NativeArray<float>.Copy(src, _depth, Mathf.Min(src.Length, _depth.Length));
                }
            }
            return true;
        }

        bool AcquireRgbCrop(int imgW, int imgH)
        {
            if (!_cameraManager.TryAcquireLatestCpuImage(out XRCpuImage img)) return false;
            using (img)
            {
                Rect bb = HandBboxNormalized;
                int px = Mathf.Clamp(Mathf.RoundToInt(bb.x * img.width), 0, img.width - 1);
                int py = Mathf.Clamp(Mathf.RoundToInt(bb.y * img.height), 0, img.height - 1);
                int pw = Mathf.Clamp(Mathf.RoundToInt(bb.width * img.width), 1, img.width - px);
                int ph = Mathf.Clamp(Mathf.RoundToInt(bb.height * img.height), 1, img.height - py);
                var p = new XRCpuImage.ConversionParams(img, TextureFormat.RGBA32, XRCpuImage.Transformation.None)
                {
                    inputRect = new RectInt(px, py, pw, ph),
                    outputDimensions = new Vector2Int(Size, Size),
                };
                int need = img.GetConvertedDataSize(p);
                EnsureNativeB(ref _rgbaCrop, need);
                // Convert(ConversionParams, NativeSlice<byte>) — no NativeArray overload in AR
                // Foundation 6.x. NativeSlice is safe, so no allowUnsafeCode needed.
                img.Convert(p, new NativeSlice<byte>(_rgbaCrop));
            }
            return true;
        }

        // ---- preprocessing (mirrors depth_synth.py) ------------------------------------

        // Sample LiDAR depth over the SAME bbox -> 256x256 (nearest, like cv2.INTER_NEAREST).
        void BuildDepthCrop(Rect bb, int imgW, int imgH)
        {
            for (int oy = 0; oy < Size; oy++)
            {
                float v = bb.y + (oy + 0.5f) / Size * bb.height; // normalized sensor-image v
                int dy = Mathf.Clamp((int)(v * _depthH), 0, _depthH - 1);
                int rowD = dy * _depthW;
                int rowO = oy * Size;
                for (int ox = 0; ox < Size; ox++)
                {
                    float u = bb.x + (ox + 0.5f) / Size * bb.width;
                    int dx = Mathf.Clamp((int)(u * _depthW), 0, _depthW - 1);
                    float m = _depth[rowD + dx];
                    _depthCrop[rowO + ox] = (m > 0f && !float.IsNaN(m) && !float.IsInfinity(m)) ? m : 0f;
                }
            }
        }

        // RGB(0..1) into channels 0-2 + normalize_depth into channel 3, NCHW.
        // The sensor crop is rotated into DISPLAY (upright) orientation here so the model sees an
        // upright hand: output pixel (ox,oy) reads sensor-patch source (ix,iy) via SrcPatchIndex.
        // center/nValid come from the same valid-median as depth_med (depth_synth.py reuses it).
        void BuildImageTensor(float center, int nValid)
        {
            const float inv = 1f / 255f;
            int plane = Size * Size;
            int dch = 3 * plane;
            bool noDepth = nValid == 0;
            for (int oy = 0; oy < Size; oy++)
            {
                for (int ox = 0; ox < Size; ox++)
                {
                    int o = oy * Size + ox;        // output (display) pixel
                    int si = SrcPatchIndex(ox, oy); // source (sensor) pixel for the rotation
                    int s = si * 4;
                    _input[0 * plane + o] = _rgbaCrop[s + 0] * inv; // R
                    _input[1 * plane + o] = _rgbaCrop[s + 1] * inv; // G
                    _input[2 * plane + o] = _rgbaCrop[s + 2] * inv; // B
                    // normalize_depth: centre on the valid median, scale, clip [-1,1], background 0
                    if (noDepth) { _input[dch + o] = 0f; }
                    else
                    {
                        float d = _depthCrop[si];
                        _input[dch + o] = d > 0f ? Mathf.Clamp((d - center) / _normScale, -1f, 1f) : 0f;
                    }
                }
            }
        }

        // Resolve the crop->display rotation. In Auto, follow the live device orientation so the
        // SAME build is upright on iPhone and iPad, portrait or landscape. The base case (Portrait ->
        // CW90) preserves the previous iPhone behaviour; other orientations rotate accordingly. If a
        // given orientation comes out 180° off on your device, swap that case for its opposite.
        SensorRotation EffectiveRotation()
        {
            if (_sensorRotation != SensorRotation.Auto) return _sensorRotation;
            switch (Screen.orientation)
            {
                case ScreenOrientation.Portrait:           return SensorRotation.CW90;
                case ScreenOrientation.PortraitUpsideDown: return SensorRotation.CW270;
                case ScreenOrientation.LandscapeLeft:      return SensorRotation.None;
                case ScreenOrientation.LandscapeRight:     return SensorRotation.Rot180;
                default:                                    return SensorRotation.CW90;
            }
        }

        // Sensor-patch source index for display output pixel (ox,oy) under the chosen rotation.
        int SrcPatchIndex(int ox, int oy)
        {
            int n = Size, ix, iy;
            switch (EffectiveRotation())
            {
                case SensorRotation.CW90:   ix = oy;         iy = n - 1 - ox; break; // image 90° CW
                case SensorRotation.Rot180: ix = n - 1 - ox; iy = n - 1 - oy; break;
                case SensorRotation.CW270:  ix = n - 1 - oy; iy = ox;         break; // image 90° CCW
                default:                    ix = ox;         iy = oy;         break;
            }
            return iy * n + ix;
        }

        // Rotate a sensor-frame 3D point about the optical (Z) axis to match the rotated image,
        // so the anchor is in the same DISPLAY camera frame as the model's output joints.
        Vector3 RotateAnchor(Vector3 a)
        {
            switch (EffectiveRotation())
            {
                case SensorRotation.CW90:   return new Vector3(-a.y,  a.x, a.z);
                case SensorRotation.Rot180: return new Vector3(-a.x, -a.y, a.z);
                case SensorRotation.CW270:  return new Vector3( a.y, -a.x, a.z);
                default:                    return a;
            }
        }

        // geometric_root_anchor(depth_crop, crop_cam): centroid -> robust local z -> back-project.
        void GeometricRootAnchor(float fx, float fy, float cx, float cy, int nValid,
                                 out Vector3 anchor, out float conf)
        {
            anchor = Vector3.zero; conf = 0f;
            if (nValid < _anchorMinValid) return;

            // hand-depth centroid (px in the 256 crop)
            double su = 0, sv = 0; int cnt = 0;
            for (int y = 0; y < Size; y++)
                for (int x = 0; x < Size; x++)
                    if (_depthCrop[y * Size + x] > 0f) { su += x; sv += y; cnt++; }
            if (cnt == 0) return;
            float uc = (float)(su / cnt), vc = (float)(sv / cnt);
            int iu = Mathf.RoundToInt(uc), iv = Mathf.RoundToInt(vc);

            // window = depth_crop[iv-win : iv+win+1, iu-win : iu+win+1], clipped (numpy slicing)
            int x0 = Mathf.Max(0, iu - _anchorWin), x1 = Mathf.Min(Size, iu + _anchorWin + 1);
            int y0 = Mathf.Max(0, iv - _anchorWin), y1 = Mathf.Min(Size, iv + _anchorWin + 1);
            int wsize = Mathf.Max(1, (y1 - y0) * (x1 - x0));
            int k = 0;
            for (int y = y0; y < y1; y++)
                for (int x = x0; x < x1; x++)
                {
                    float d = _depthCrop[y * Size + x];
                    if (d > 0f) _sortBuf[k++] = d;
                }

            float z0;
            if (k >= 5)
            {
                z0 = MedianInPlace(_sortBuf, k);
                // MAD = median(|wv - z0|): reuse window values (recollect, _sortBuf was sorted)
                int m = 0;
                for (int y = y0; y < y1; y++)
                    for (int x = x0; x < x1; x++)
                    {
                        float d = _depthCrop[y * Size + x];
                        if (d > 0f) _sortBuf[m++] = Mathf.Abs(d - z0);
                    }
                float mad = MedianInPlace(_sortBuf, m);
                float validFrac = (float)k / wsize;
                conf = validFrac * Mathf.Exp(-(mad / _anchorMadRef) * (mad / _anchorMadRef));
            }
            else
            {
                z0 = MedianValid(_depthCrop, out _);
                conf = 0.1f;
            }
            conf = Mathf.Clamp01(conf);
            float x0m = (uc - cx) / fx * z0;
            float y0m = (vc - cy) / fy * z0;
            anchor = new Vector3(x0m, y0m, z0);
        }

        // ---- inference -----------------------------------------------------------------

        void RunModel(float depthMed, Vector3 anchor, float anchorValid)
        {
            using var image = new Tensor<float>(new TensorShape(1, 4, Size, Size), _input);
            using var med = new Tensor<float>(new TensorShape(1, 1), new[] { depthMed });
            using var ra = new Tensor<float>(new TensorShape(1, 3), new[] { anchor.x, anchor.y, anchor.z });
            using var va = new Tensor<float>(new TensorShape(1, 1), new[] { anchorValid });

            // Sentis 2.x: set inputs by name, then Schedule(). If you pin a version without the
            // no-arg Schedule, use _worker.Schedule(image, med, ra, va) (model input order).
            _worker.SetInput("image", image);
            _worker.SetInput("depth_med", med);
            _worker.SetInput("root_anchor", ra);
            _worker.SetInput("anchor_valid", va);
            _worker.Schedule();

            if (_worker.PeekOutput("keypoints_abs") is not Tensor<float> absPeek) return;
            using Tensor<float> abs = absPeek.ReadbackAndClone();
            float[] a = abs.DownloadToArray();
            if (a.Length < JointCount * 3) return;
            for (int i = 0; i < JointCount; i++)
                AbsJoints[i] = new Vector3(a[i * 3 + 0], a[i * 3 + 1], a[i * 3 + 2]);

            if (_worker.PeekOutput("root") is Tensor<float> rootPeek)
            {
                using var r = rootPeek.ReadbackAndClone();
                float[] rr = r.DownloadToArray();
                if (rr.Length >= 3) Root = new Vector3(rr[0], rr[1], rr[2]);
            }
            if (_worker.PeekOutput("scale") is Tensor<float> scalePeek)
            {
                using var s = scalePeek.ReadbackAndClone();
                float[] ss = s.DownloadToArray();
                if (ss.Length >= 1) Scale = ss[0];
            }
        }

        // ---- helpers -------------------------------------------------------------------

        // Median of the strictly-positive entries of `buf` (length Size*Size). nValid = count.
        float MedianValid(float[] buf, out int nValid)
        {
            int k = 0;
            for (int i = 0; i < buf.Length; i++) if (buf[i] > 0f) _sortBuf[k++] = buf[i];
            nValid = k;
            return k == 0 ? 0f : MedianInPlace(_sortBuf, k);
        }

        // Median of the first `count` entries of `arr` (sorts them in place).
        static float MedianInPlace(float[] arr, int count)
        {
            if (count <= 0) return 0f;
            Array.Sort(arr, 0, count);
            int mid = count / 2;
            return (count & 1) == 1 ? arr[mid] : 0.5f * (arr[mid - 1] + arr[mid]);
        }

        static void EnsureNativeB(ref NativeArray<byte> b, int len)
        {
            if (b.IsCreated && b.Length >= len) return;
            if (b.IsCreated) b.Dispose();
            b = new NativeArray<byte>(len, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        }

        static void EnsureNativeF(ref NativeArray<float> b, int len)
        {
            if (b.IsCreated && b.Length >= len) return;
            if (b.IsCreated) b.Dispose();
            b = new NativeArray<float>(len, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        }
    }
}
#endif
