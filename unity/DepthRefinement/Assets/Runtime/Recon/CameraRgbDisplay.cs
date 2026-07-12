#if ARFOUNDATION_PRESENT
using Unity.Collections;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;

namespace HandMesh.DepthRefinement
{
    /// <summary>
    /// Shows the iPhone camera RGB feed in a UI <see cref="RawImage"/>, for a side-by-side
    /// RGB | Depth viewer. (AR Foundation's <c>ARCameraBackground</c> already draws RGB
    /// fullscreen; use this when you want RGB as an explicit panel next to the depth panel —
    /// you can then disable ARCameraBackground for a clean two-panel layout.)
    ///
    /// Each frame: acquire the camera CPU image, convert YUV->RGBA downscaled to a display
    /// size (aspect preserved), rotate sensor-landscape to portrait, and assign to the RawImage.
    /// Allocation-free in steady state. Mirrors <see cref="DepthImageVisualizer"/>'s orientation
    /// handling so the two panels line up.
    /// </summary>
    public sealed class CameraRgbDisplay : MonoBehaviour
    {
        [SerializeField] ARCameraManager _cameraManager;
        [SerializeField] RawImage _target;

        [Tooltip("Max output width/height (px) for the display texture; downscaled for perf.")]
        [SerializeField] int _maxSize = 640;
        [Tooltip("Rotate sensor-landscape to portrait. OFF = keep native landscape.")]
        [SerializeField] bool _rotate90 = false;
        [Tooltip("Mirror horizontally (front/selfie camera).")]
        [SerializeField] bool _mirrorHorizontal = false;

        public Texture2D Texture => _tex;
        public bool HasFrame { get; private set; }

        Texture2D _tex;
        NativeArray<byte> _rgba;   // converted RGBA, sensor orientation
        Color32[] _pixels;         // display orientation

        void OnEnable()
        {
            if (_cameraManager == null) _cameraManager = FindFirstObjectByType<ARCameraManager>();
            if (_cameraManager == null) { Debug.LogError("[CameraRgbDisplay] No ARCameraManager."); enabled = false; return; }
            _cameraManager.frameReceived += OnFrame;
        }

        void OnDisable()
        {
            if (_cameraManager != null) _cameraManager.frameReceived -= OnFrame;
            HasFrame = false;
            if (_rgba.IsCreated) _rgba.Dispose();
        }

        void OnFrame(ARCameraFrameEventArgs _)
        {
            if (!_cameraManager.TryAcquireLatestCpuImage(out XRCpuImage img)) return;
            using (img) Build(img);
            if (_target != null) _target.texture = _tex;
            HasFrame = true;
        }

        void Build(XRCpuImage img)
        {
            int srcW = img.width, srcH = img.height;
            int longSide = Mathf.Max(srcW, srcH);
            float s = longSide > _maxSize ? (float)_maxSize / longSide : 1f;
            int w = Mathf.Max(1, Mathf.RoundToInt(srcW * s));
            int h = Mathf.Max(1, Mathf.RoundToInt(srcH * s));

            var tf = _mirrorHorizontal ? XRCpuImage.Transformation.MirrorX : XRCpuImage.Transformation.None;
            var p = new XRCpuImage.ConversionParams(img, TextureFormat.RGBA32, tf)
            {
                inputRect = new RectInt(0, 0, srcW, srcH),
                outputDimensions = new Vector2Int(w, h),   // downscale (<= source) ok
            };
            int need = img.GetConvertedDataSize(p);
            EnsureNative(ref _rgba, need);
            img.Convert(p, new NativeSlice<byte>(_rgba));   // no NativeArray overload in AF6

            bool rot = _rotate90;
            int outW = rot ? h : w, outH = rot ? w : h;
            if (_tex == null || _tex.width != outW || _tex.height != outH)
                _tex = new Texture2D(outW, outH, TextureFormat.RGBA32, false);
            if (_pixels == null || _pixels.Length != outW * outH)
                _pixels = new Color32[outW * outH];

            for (int sy = 0; sy < h; sy++)
            {
                for (int sx = 0; sx < w; sx++)
                {
                    int si = (sy * w + sx) * 4;
                    var c = new Color32(_rgba[si], _rgba[si + 1], _rgba[si + 2], 255);
                    int dx, dy;
                    if (rot) { dx = sy; dy = sx; } else { dx = sx; dy = sy; }
                    int flippedY = outH - 1 - dy;            // Texture2D is bottom-up
                    _pixels[flippedY * outW + dx] = c;
                }
            }
            _tex.SetPixels32(_pixels);
            _tex.Apply(false);
        }

        static void EnsureNative(ref NativeArray<byte> b, int len)
        {
            if (b.IsCreated && b.Length >= len) return;
            if (b.IsCreated) b.Dispose();
            b = new NativeArray<byte>(len, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        }
    }
}
#endif
