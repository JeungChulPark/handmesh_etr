#if ARFOUNDATION_PRESENT
using System;
using System.IO;
using System.Threading;
using System.Collections.Concurrent;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Experimental.Rendering;   // GraphicsFormat (off-thread PNG/JPG encode)
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;

namespace HandMesh.DepthRefinement
{
    /// <summary>
    /// Saves synchronized RGB-D frames from the iPhone to disk, for dataset capture / offline
    /// debugging. Each capture writes three files under
    /// <c>Application.persistentDataPath/&lt;subfolder&gt;/</c>:
    ///   cap_NNNN_rgb.png   — RGB, upright (origin top-left), sensor crop, optional downscale.
    ///   cap_NNNN_depth.f32 — LiDAR depth, raw little-endian float32, row-major HxW, METRES
    ///                        (0 = invalid). Load in Python: np.fromfile(...).reshape(H, W).
    ///   cap_NNNN.json      — metadata: sizes, camera intrinsics (fx,fy,cx,cy,resolution),
    ///                        depth unit, screen orientation, timestamp.
    ///
    /// The raw sensor frames are saved (same data the model reads), so RGB and depth are in the
    /// camera SENSOR orientation (no 90° rotation here — only a vertical flip); the JSON records
    /// orientation + intrinsics so the offline pipeline can rotate RGB+depth+K together.
    ///
    /// Performance (why fingers don't blur while recording): the expensive work — PNG/JPG
    /// encoding, the row-flip, and disk writes — runs on a BACKGROUND thread (①+②). The main
    /// thread only grabs the camera buffer, copies it, and measures focus, so the AR/camera loop
    /// never stalls (a stall drops the frame rate and lets the hand smear). A sharpness gate (④)
    /// optionally discards motion-blurred frames before they are queued.
    /// </summary>
    public sealed class RgbdRecorder : MonoBehaviour
    {
        [SerializeField] ARCameraManager _cameraManager;
        [SerializeField] AROcclusionManager _occlusionManager;

        [Header("Output")]
        [Tooltip("Subfolder under Application.persistentDataPath.")]
        [SerializeField] string _subfolder = "rgbd_captures";
        [Tooltip("Max RGB long side (px). 0 = full sensor resolution (e.g. 1920x1440).")]
        [SerializeField] int _rgbMaxSize = 0;
        [Tooltip("PNG (lossless, bigger) vs JPG (smaller, faster encode) for the RGB image.")]
        [SerializeField] bool _rgbAsJpg = false;
        [Range(1, 100)][SerializeField] int _jpgQuality = 90;

        [Header("Timed recording")]
        [Tooltip("Frames per second when recording (Start/StopRecording).")]
        [SerializeField] float _recordFps = 5f;

        [Header("Sharpness gate (④)")]
        [Tooltip("Skip frames whose focus measure (variance-of-Laplacian, computed on a coarse " +
                 "grid of the whole frame) is below this. 0 = save every frame. Watch LastSharpness " +
                 "live first, then set this just under the value you get for a still, in-focus hand.")]
        [SerializeField] float _sharpnessMin = 0f;

        /// <summary>Number of captures written (queued) this session.</summary>
        public int Count { get; private set; }
        /// <summary>Frames skipped by the sharpness gate this session.</summary>
        public int Skipped { get; private set; }
        /// <summary>Focus measure of the most recent captured frame (for tuning _sharpnessMin).</summary>
        public float LastSharpness { get; private set; }
        /// <summary>Frames still waiting to be encoded/written on the worker thread.</summary>
        public int Pending => _queue?.Count ?? 0;
        /// <summary>Folder captures are written to.</summary>
        public string OutputDir { get; private set; }
        public bool IsRecording { get; private set; }

        NativeArray<byte> _rgba;     // converted RGBA scratch (sensor), main thread only
        float[] _depthM;             // metres scratch, main thread only
        float _recTimer;

        // ---- background save pipeline (①) -------------------------------------------------
        BlockingCollection<SaveJob> _queue;
        Thread _worker;

        sealed class SaveJob
        {
            public string stem;
            public byte[] rgbaTopDown;        // sensor-order RGBA (row 0 = top), worker flips it
            public int w, h;
            public bool jpg; public int quality;
            public byte[] depthBytes;         // float32 LE, row-major, or null
            public string metaJson;
        }

        void Awake()
        {
            if (_cameraManager == null) _cameraManager = FindFirstObjectByType<ARCameraManager>();
            if (_occlusionManager == null) _occlusionManager = FindFirstObjectByType<AROcclusionManager>();
            OutputDir = Path.Combine(Application.persistentDataPath, _subfolder);
            Directory.CreateDirectory(OutputDir);

            _queue = new BlockingCollection<SaveJob>(boundedCapacity: 8);
            _worker = new Thread(WorkerLoop) { IsBackground = true, Name = "RgbdSaver" };
            _worker.Start();
        }

        void OnDestroy()
        {
            _queue?.CompleteAdding();
            try { _worker?.Join(2000); } catch { /* best effort */ }
            if (_rgba.IsCreated) _rgba.Dispose();
            _queue?.Dispose();
        }

        // ---- public API ----------------------------------------------------------------

        /// <summary>Void wrapper for a UI Button OnClick — UnityEvent's dropdown hides non-void
        /// methods, so bind buttons to this instead of <see cref="Capture"/>.</summary>
        public void CaptureOnce() => Capture();

        /// <summary>Toggle for a UI Button: starts if stopped, stops if recording.</summary>
        public void ToggleRecording() { if (IsRecording) StopRecording(); else StartRecording(); }

        public void StartRecording() { IsRecording = true; _recTimer = 0f; }
        public void StopRecording() { IsRecording = false; }

        /// <summary>Capture one RGB-D frame now. Returns true if it was queued for saving
        /// (false if no camera image, the sharpness gate rejected it, or the queue is full).
        /// Encoding + disk I/O happen on the worker thread, so this returns quickly.</summary>
        public bool Capture()
        {
            if (_cameraManager == null) { Debug.LogError("[RgbdRecorder] No ARCameraManager."); return false; }

            // RGB acquire (main thread: convert + copy out) -------------------------------
            if (!AcquireRgb(out byte[] rgba, out int rgbW, out int rgbH))
            { Debug.LogWarning("[RgbdRecorder] RGB capture failed (no camera image)."); return false; }

            // Sharpness gate (④) ---------------------------------------------------------
            LastSharpness = FocusMeasure(rgba, rgbW, rgbH);
            if (_sharpnessMin > 0f && LastSharpness < _sharpnessMin) { Skipped++; return false; }

            // intrinsics (best-effort)
            float fx = 0, fy = 0, cx = 0, cy = 0; int intrW = 0, intrH = 0;
            if (_cameraManager.TryGetIntrinsics(out XRCameraIntrinsics intr))
            {
                fx = intr.focalLength.x; fy = intr.focalLength.y;
                cx = intr.principalPoint.x; cy = intr.principalPoint.y;
                intrW = intr.resolution.x; intrH = intr.resolution.y;
            }

            int idx = Count;
            string stem = Path.Combine(OutputDir, $"cap_{idx:D4}");

            // Depth (optional — only LiDAR devices)
            byte[] depthBytes = null; int dW = 0, dH = 0; string depthFmt = "none", depthFile = null;
            if (AcquireDepth(out depthBytes, out dW, out dH, out depthFmt)) depthFile = $"cap_{idx:D4}_depth.f32";

            // Metadata (JsonUtility is main-thread only, so build the string here)
            var meta = new RgbdMeta
            {
                unixMs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(),
                rgbFile = $"cap_{idx:D4}" + (_rgbAsJpg ? "_rgb.jpg" : "_rgb.png"),
                rgbWidth = rgbW, rgbHeight = rgbH,
                depthFile = depthFile, depthWidth = dW, depthHeight = dH,
                depthFormat = depthFmt, depthUnit = "meter", depthLayout = "float32_le row-major HxW, 0=invalid",
                fx = fx, fy = fy, cx = cx, cy = cy, intrWidth = intrW, intrHeight = intrH,
                screenOrientation = Screen.orientation.ToString(),
                frame = "camera sensor (raw XRCpuImage); see SensorRotation in DualStreamHandProvider",
            };

            var job = new SaveJob
            {
                stem = stem, rgbaTopDown = rgba, w = rgbW, h = rgbH,
                jpg = _rgbAsJpg, quality = _jpgQuality,
                depthBytes = depthBytes, metaJson = JsonUtility.ToJson(meta, true),
            };
            if (!_queue.TryAdd(job))   // backpressure: worker can't keep up -> drop, don't stall
            { Debug.LogWarning("[RgbdRecorder] save queue full; dropped a frame."); return false; }

            Count++;
            return true;
        }

        void Update()
        {
            if (!IsRecording) return;
            _recTimer += Time.unscaledDeltaTime;
            float interval = 1f / Mathf.Max(0.5f, _recordFps);
            if (_recTimer < interval) return;
            _recTimer = 0f;
            Capture();
        }

        // ---- worker thread (① encode + ② flip + write) -----------------------------------

        void WorkerLoop()
        {
            byte[] flip = null;   // reused flip buffer, grows as needed
            foreach (var job in _queue.GetConsumingEnumerable())
            {
                try
                {
                    int stride = job.w * 4;
                    int n = stride * job.h;
                    if (flip == null || flip.Length < n) flip = new byte[n];
                    // ② fast row-flip (sensor top-down -> bottom-up that the PNG/JPG encoder
                    //    expects); per-row Buffer.BlockCopy instead of a per-byte loop.
                    for (int y = 0; y < job.h; y++)
                        Buffer.BlockCopy(job.rgbaTopDown, (job.h - 1 - y) * stride, flip, y * stride, stride);

                    // EncodeArrayTo* are CPU-only and safe off the main thread (unlike
                    // Texture2D.EncodeToPNG, which needs the texture on the main thread).
                    byte[] enc = job.jpg
                        ? ImageConversion.EncodeArrayToJPG(flip, GraphicsFormat.R8G8B8A8_SRGB,
                                                           (uint)job.w, (uint)job.h, 0, job.quality)
                        : ImageConversion.EncodeArrayToPNG(flip, GraphicsFormat.R8G8B8A8_SRGB,
                                                           (uint)job.w, (uint)job.h);
                    File.WriteAllBytes(job.stem + (job.jpg ? "_rgb.jpg" : "_rgb.png"), enc);
                    if (job.depthBytes != null) File.WriteAllBytes(job.stem + "_depth.f32", job.depthBytes);
                    File.WriteAllText(job.stem + ".json", job.metaJson);
                }
                catch (Exception e) { Debug.LogError($"[RgbdRecorder] save failed for {job.stem}: {e.Message}"); }
            }
        }

        // ---- RGB acquire (main thread) -------------------------------------------------

        bool AcquireRgb(out byte[] rgba, out int outW, out int outH)
        {
            rgba = null; outW = outH = 0;
            if (!_cameraManager.TryAcquireLatestCpuImage(out XRCpuImage img)) return false;
            using (img)
            {
                int srcW = img.width, srcH = img.height;
                int longSide = Mathf.Max(srcW, srcH);
                float s = (_rgbMaxSize > 0 && longSide > _rgbMaxSize) ? (float)_rgbMaxSize / longSide : 1f;
                int w = Mathf.Max(1, Mathf.RoundToInt(srcW * s));
                int h = Mathf.Max(1, Mathf.RoundToInt(srcH * s));

                var p = new XRCpuImage.ConversionParams(img, TextureFormat.RGBA32, XRCpuImage.Transformation.None)
                {
                    inputRect = new RectInt(0, 0, srcW, srcH),
                    outputDimensions = new Vector2Int(w, h),
                };
                int need = img.GetConvertedDataSize(p);
                if (!_rgba.IsCreated || _rgba.Length < need)
                {
                    if (_rgba.IsCreated) _rgba.Dispose();
                    _rgba = new NativeArray<byte>(need, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
                }
                img.Convert(p, new NativeSlice<byte>(_rgba));

                // Copy out of the reused scratch so the worker thread owns its data (the sensor
                // buffer stays top-down here; the worker flips it before encoding).
                rgba = new byte[need];
                NativeArray<byte>.Copy(_rgba, rgba, need);
                outW = w; outH = h;
            }
            return true;
        }

        // ---- Depth acquire (main thread) -----------------------------------------------

        bool AcquireDepth(out byte[] bytes, out int w, out int h, out string fmt)
        {
            bytes = null; w = h = 0; fmt = "none";
            if (_occlusionManager == null) return false;
            if (!_occlusionManager.TryAcquireEnvironmentDepthCpuImage(out XRCpuImage img)) return false;
            using (img)
            {
                w = img.width; h = img.height; fmt = img.format.ToString();
                int n = w * h;
                if (_depthM == null || _depthM.Length != n) _depthM = new float[n];

                var raw = img.GetPlane(0).data;
                if (img.format == XRCpuImage.Format.DepthUint16)
                {
                    var src = raw.Reinterpret<ushort>(1);
                    int m = Mathf.Min(src.Length, n);
                    for (int i = 0; i < m; i++) _depthM[i] = src[i] * 0.001f;   // mm -> m
                }
                else // DepthFloat32 (ARKit) or best-effort
                {
                    var src = raw.Reinterpret<float>(1);
                    int m = Mathf.Min(src.Length, n);
                    for (int i = 0; i < m; i++)
                    {
                        float v = src[i];
                        _depthM[i] = (v > 0f && !float.IsNaN(v) && !float.IsInfinity(v)) ? v : 0f;
                    }
                }

                bytes = new byte[n * 4];
                Buffer.BlockCopy(_depthM, 0, bytes, 0, n * 4);   // little-endian float32
            }
            return true;
        }

        // ---- Sharpness (④): variance of Laplacian on a coarse grayscale grid --------------

        static float FocusMeasure(byte[] rgbaTopDown, int w, int h)
        {
            int step = Mathf.Max(1, Mathf.Max(w, h) / 240);   // ~240px on the long side, for speed
            int gw = w / step, gh = h / step;
            if (gw < 3 || gh < 3) return 0f;
            var gray = new float[gw * gh];
            for (int y = 0; y < gh; y++)
                for (int x = 0; x < gw; x++)
                {
                    int i = ((y * step) * w + (x * step)) * 4;
                    gray[y * gw + x] = 0.299f * rgbaTopDown[i] + 0.587f * rgbaTopDown[i + 1] + 0.114f * rgbaTopDown[i + 2];
                }
            double sum = 0, sum2 = 0; int cnt = 0;
            for (int y = 1; y < gh - 1; y++)
                for (int x = 1; x < gw - 1; x++)
                {
                    float lap = 4f * gray[y * gw + x]
                              - gray[y * gw + x - 1] - gray[y * gw + x + 1]
                              - gray[(y - 1) * gw + x] - gray[(y + 1) * gw + x];
                    sum += lap; sum2 += lap * lap; cnt++;
                }
            if (cnt == 0) return 0f;
            double mean = sum / cnt;
            return (float)(sum2 / cnt - mean * mean);
        }

        [Serializable]
        class RgbdMeta
        {
            public long unixMs;
            public string rgbFile; public int rgbWidth, rgbHeight;
            public string depthFile; public int depthWidth, depthHeight;
            public string depthFormat; public string depthUnit; public string depthLayout;
            public float fx, fy, cx, cy; public int intrWidth, intrHeight;
            public string screenOrientation; public string frame;
        }
    }
}
#endif
