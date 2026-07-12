#if SENTIS_PRESENT && ARFOUNDATION_PRESENT
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Unity.Collections;
using UnityEngine;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;

namespace HandMesh.DepthRefinement
{
    /// <summary>
    /// Feeds Apple Vision hand landmarks (the native iOS plugin <c>HandPoseVision.swift</c>)
    /// into <see cref="HandBboxBridge"/>, which turns them into the model crop box. This is the
    /// missing DETECTOR: without it the dual-stream provider crops the image CENTRE and the
    /// spheres collapse into a blob.
    ///
    /// Each frame (throttled by <see cref="_everyNFrames"/>) it grabs the AR camera CPU image,
    /// converts it to a downscaled BGRA buffer in SENSOR orientation, and submits it to Vision
    /// via <c>HandVision_Submit</c>. Every frame it polls <c>HandVision_Get</c> for the latest
    /// 21 landmarks and forwards the valid ones to <see cref="HandBboxBridge.SetLandmarksNormalized"/>.
    ///
    /// Landmarks are normalized [0,1], origin top-left, in the SENSOR (landscape) frame — so set
    /// <see cref="HandBboxBridge"/> Rotation = None. If the box is mirrored/rotated on device,
    /// fix it with that component's Mirror/Rotation knobs (don't edit the plugin).
    /// In the Editor the native calls are stubbed (no detection) — test on an iOS device.
    /// </summary>
    public sealed class HandVisionBridge : MonoBehaviour
    {
        [SerializeField] ARCameraManager _cameraManager;
        [SerializeField] HandBboxBridge _bridge;

        [Tooltip("Detection image long side (px). Smaller = faster, less accurate.")]
        [SerializeField] int _detectSize = 512;
        [Tooltip("Run Vision every N camera frames (1 = every frame). 2-3 is plenty.")]
        [SerializeField] int _everyNFrames = 2;
        [Tooltip("Minimum valid landmarks to accept a hand and update the box.")]
        [SerializeField] int _minPoints = 8;

#if UNITY_IOS && !UNITY_EDITOR
        [DllImport("__Internal")] static extern void HandVision_Submit(byte[] bgra, int w, int h, int bpr);
        [DllImport("__Internal")] static extern int HandVision_Get(float[] outXY, int maxPts);
#else
        static void HandVision_Submit(byte[] bgra, int w, int h, int bpr) { }
        static int HandVision_Get(float[] outXY, int maxPts) { return 0; }
#endif

        NativeArray<byte> _conv;          // XRCpuImage.Convert output (BGRA32)
        byte[] _managed;                  // marshalled copy for the native call
        readonly float[] _xy = new float[42];
        readonly List<Vector2> _pts = new List<Vector2>(21);
        int _frame;

        void Awake()
        {
            if (_cameraManager == null) _cameraManager = FindFirstObjectByType<ARCameraManager>();
            if (_bridge == null) _bridge = FindFirstObjectByType<HandBboxBridge>();
            if (_cameraManager == null) Debug.LogError("[HandVisionBridge] No ARCameraManager.");
            if (_bridge == null) Debug.LogError("[HandVisionBridge] No HandBboxBridge.");
        }

        void OnDestroy() { if (_conv.IsCreated) _conv.Dispose(); }

        void Update()
        {
            // 1) submit a frame to Vision every N frames (async on the native side)
            if (_cameraManager != null && (_frame++ % Mathf.Max(1, _everyNFrames) == 0))
                SubmitFrame();

            // 2) poll the latest landmarks every frame and feed the bbox bridge
            int n = HandVision_Get(_xy, 21);
            if (n <= 0 || _bridge == null) return;

            _pts.Clear();
            for (int i = 0; i < 21; i++)
            {
                float x = _xy[i * 2], y = _xy[i * 2 + 1];
                if (x >= 0f && y >= 0f) _pts.Add(new Vector2(x, y));  // -1 = missing joint
            }
            if (_pts.Count >= _minPoints) _bridge.SetLandmarksNormalized(_pts);
        }

        void SubmitFrame()
        {
            if (!_cameraManager.TryAcquireLatestCpuImage(out XRCpuImage img)) return;
            using (img)
            {
                int longSide = Mathf.Max(img.width, img.height);
                float s = longSide > _detectSize ? (float)_detectSize / longSide : 1f;
                int w = Mathf.Max(1, Mathf.RoundToInt(img.width * s));
                int h = Mathf.Max(1, Mathf.RoundToInt(img.height * s));

                // SENSOR orientation (Transformation.None) so landmarks share the crop frame.
                var p = new XRCpuImage.ConversionParams(img, TextureFormat.BGRA32, XRCpuImage.Transformation.None)
                {
                    inputRect = new RectInt(0, 0, img.width, img.height),
                    outputDimensions = new Vector2Int(w, h),
                };
                int need = img.GetConvertedDataSize(p);
                if (!_conv.IsCreated || _conv.Length < need)
                {
                    if (_conv.IsCreated) _conv.Dispose();
                    _conv = new NativeArray<byte>(need, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
                }
                img.Convert(p, new NativeSlice<byte>(_conv));

                if (_managed == null || _managed.Length != need) _managed = new byte[need];
                _conv.CopyTo(_managed);
                HandVision_Submit(_managed, w, h, w * 4);   // BGRA32 tightly packed: bpr = w*4
            }
        }
    }
}
#endif
