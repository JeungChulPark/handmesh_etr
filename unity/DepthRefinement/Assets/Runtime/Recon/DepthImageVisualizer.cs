#if ARFOUNDATION_PRESENT
using Unity.Collections;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;

namespace HandMesh.DepthRefinement
{
    /// <summary>
    /// Visualizes ARKit LiDAR environment depth as a colour image for on-screen display.
    /// Pairs with AR Foundation's <c>ARCameraBackground</c> (which draws the RGB feed) to make
    /// a simple RGB-D viewer: near = warm (red), far = cool (blue), no-data = black.
    ///
    /// Reads the CPU depth image each frame, maps metres -> colour, optionally rotates the
    /// sensor-landscape image to portrait, and writes a <see cref="Texture2D"/> that it assigns
    /// to a UI <see cref="RawImage"/> (and exposes via <see cref="Texture"/>). LiDAR depth is
    /// low-res (~256x192); the RawImage scales it up.
    ///
    /// Requires an iPhone/iPad **Pro** (LiDAR) and an <c>AROcclusionManager</c> with Environment
    /// Depth enabled. On non-LiDAR devices no depth is produced (<see cref="HasFrame"/> = false).
    /// </summary>
    public sealed class DepthImageVisualizer : MonoBehaviour
    {
        [SerializeField] AROcclusionManager _occlusionManager;

        [Tooltip("UI RawImage to display the colorized depth on (optional).")]
        [SerializeField] RawImage _target;

        [Tooltip("Depth (m) mapped to the warm end of the colormap.")]
        [SerializeField] float _minDepth = 0.2f;
        [Tooltip("Depth (m) mapped to the cool end of the colormap.")]
        [SerializeField] float _maxDepth = 3.0f;

        [Tooltip("Rotate the sensor-landscape depth to portrait. OFF = keep native landscape.")]
        [SerializeField] bool _rotate90CW = false;

        [Tooltip("Optional UI Text to show 'depth a-b m | cov c%'.")]
        [SerializeField] Text _statusLabel;

        /// <summary>Colorized depth texture (also usable as a model/debug input).</summary>
        public Texture2D Texture => _tex;
        /// <summary>True once a depth frame has been colorized this run.</summary>
        public bool HasFrame { get; private set; }
        /// <summary>Fraction of valid depth pixels in the last frame, [0,1].</summary>
        public float Coverage { get; private set; }
        /// <summary>Nearest / farthest valid depth in the last frame (metres).</summary>
        public float MinMeters { get; private set; }
        public float MaxMeters { get; private set; }

        Texture2D _tex;
        NativeArray<float> _depth;     // metres, 0 = invalid
        Color32[] _pixels;             // output, display orientation

        void Awake()
        {
            if (_occlusionManager == null) _occlusionManager = FindFirstObjectByType<AROcclusionManager>();
            if (_occlusionManager == null)
            {
                Debug.LogError("[DepthImageVisualizer] No AROcclusionManager (needs a LiDAR device + Environment Depth).");
                enabled = false;
            }
        }

        void OnDestroy() { if (_depth.IsCreated) _depth.Dispose(); }

        void Update()
        {
            HasFrame = false;
            if (_occlusionManager == null) return;
            if (!_occlusionManager.TryAcquireEnvironmentDepthCpuImage(out XRCpuImage img)) return;
            using (img)
            {
                int w = img.width, h = img.height;
                EnsureDepth(w * h);
                CopyDepth(img, _depth);
                Build(w, h);
            }
            if (_target != null) _target.texture = _tex;
            if (_statusLabel != null)
                _statusLabel.text = Coverage > 0f
                    ? $"depth {MinMeters:0.00}-{MaxMeters:0.00} m | cov {Coverage * 100f:0}%"
                    : "depth: no data";
            HasFrame = true;
        }

        // metres -> colour, with optional 90 CW rotation (sensor landscape -> portrait).
        void Build(int w, int h)
        {
            bool rot = _rotate90CW;
            int outW = rot ? h : w;
            int outH = rot ? w : h;
            if (_tex == null || _tex.width != outW || _tex.height != outH)
                _tex = new Texture2D(outW, outH, TextureFormat.RGBA32, false);
            if (_pixels == null || _pixels.Length != outW * outH)
                _pixels = new Color32[outW * outH];

            int valid = 0;
            float minV = float.MaxValue, maxV = 0f;
            float range = Mathf.Max(1e-4f, _maxDepth - _minDepth);
            for (int sy = 0; sy < h; sy++)
            {
                for (int sx = 0; sx < w; sx++)
                {
                    float m = _depth[sy * w + sx];
                    Color32 c;
                    if (m > 0f && !float.IsNaN(m) && !float.IsInfinity(m))
                    {
                        float t = Mathf.Clamp01((m - _minDepth) / range); // 0 near .. 1 far
                        c = Jet(t);
                        valid++;
                        if (m < minV) minV = m;
                        if (m > maxV) maxV = m;
                    }
                    else c = new Color32(0, 0, 0, 255); // no data

                    // Texture2D is bottom-up; flip Y so it isn't upside down.
                    int dx, dy;
                    if (rot) { dx = sy; dy = sx; }       // 90 CW: (sx,sy)->(dx=sy, dy=sx)
                    else     { dx = sx; dy = sy; }
                    int flippedY = outH - 1 - dy;
                    _pixels[flippedY * outW + dx] = c;
                }
            }
            Coverage = (float)valid / (w * h);
            MinMeters = valid > 0 ? minV : 0f;
            MaxMeters = valid > 0 ? maxV : 0f;
            _tex.SetPixels32(_pixels);
            _tex.Apply(false);
        }

        // 5-stop jet: near(red) -> yellow -> green -> cyan -> far(blue).
        static Color32 Jet(float t)
        {
            t = Mathf.Clamp01(t);
            float r, g, b;
            if (t < 0.25f)      { float u = t / 0.25f;          r = 1f;       g = u;        b = 0f; }
            else if (t < 0.5f)  { float u = (t - 0.25f) / 0.25f; r = 1f - u;   g = 1f;       b = 0f; }
            else if (t < 0.75f) { float u = (t - 0.5f) / 0.25f;  r = 0f;       g = 1f;       b = u; }
            else                { float u = (t - 0.75f) / 0.25f; r = 0f;       g = 1f - u;   b = 1f; }
            return new Color32((byte)(r * 255), (byte)(g * 255), (byte)(b * 255), 255);
        }

        void EnsureDepth(int len)
        {
            if (_depth.IsCreated && _depth.Length >= len) return;
            if (_depth.IsCreated) _depth.Dispose();
            _depth = new NativeArray<float>(len, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        }

        static void CopyDepth(XRCpuImage image, NativeArray<float> dst)
        {
            var raw = image.GetPlane(0).data;
            switch (image.format)
            {
                case XRCpuImage.Format.DepthFloat32:
                {
                    var src = raw.Reinterpret<float>(1);
                    NativeArray<float>.Copy(src, dst, Mathf.Min(src.Length, dst.Length));
                    break;
                }
                case XRCpuImage.Format.DepthUint16:
                {
                    var src = raw.Reinterpret<ushort>(1);
                    int n = Mathf.Min(src.Length, dst.Length);
                    for (int i = 0; i < n; i++) dst[i] = src[i] * 0.001f; // mm -> m
                    break;
                }
                default:
                {
                    var src = raw.Reinterpret<float>(1);
                    NativeArray<float>.Copy(src, dst, Mathf.Min(src.Length, dst.Length));
                    break;
                }
            }
        }
    }
}
#endif
