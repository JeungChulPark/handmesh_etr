#if ARFOUNDATION_PRESENT
using Unity.Collections;
using UnityEngine;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;

namespace HandMesh.DepthRefinement
{
    /// <summary>
    /// Steps 1-3: acquires metric environment depth + confidence from AR Foundation and
    /// exposes it as an <see cref="IDepthProvider"/>. On iOS this is LiDAR
    /// <c>sceneDepth</c> (256x192, meters); the AR Foundation "environmentDepth" abstraction
    /// maps to it directly. Each frame the CPU depth/confidence images are copied into
    /// persistent NativeArrays so the XRCpuImages can be disposed immediately and sampling
    /// stays allocation-free.
    ///
    /// NOTE: UV -> depth-pixel mapping here is the ARKit uniform-scale case (depth is
    /// registered to the camera image). ARCore returns a FOV CROP — for Android use
    /// XRCpuImage/displayMatrix-based mapping instead (see TryConvertUvToDepthPixel).
    /// </summary>
    [RequireComponent(typeof(AROcclusionManager))]
    public sealed class ARFoundationDepthProvider : MonoBehaviour, IDepthProvider, IDepthFrameUpdater
    {
        [SerializeField] AROcclusionManager _occlusionManager;
        [SerializeField] ARCameraManager _cameraManager;

        NativeArray<float> _depth;   // meters
        NativeArray<byte> _conf;     // 0/1/2 (low/med/high)
        int _w, _h;
        bool _available;
        bool _confValid;             // confidence image acquired this frame at matching resolution
        CameraIntrinsics _intrinsics;
        bool _hasIntrinsics;

        public bool IsDepthAvailable => _available;
        public int DepthWidth => _w;
        public int DepthHeight => _h;

        void Awake()
        {
            if (_occlusionManager == null) _occlusionManager = GetComponent<AROcclusionManager>();
            if (_cameraManager == null) _cameraManager = FindObjectOfType<ARCameraManager>();
        }

        void OnDisable()
        {
            _available = false;
            if (_depth.IsCreated) _depth.Dispose();
            if (_conf.IsCreated) _conf.Dispose();
        }

        /// <summary>Call once per frame (e.g. from the manager) before sampling joints.</summary>
        public void UpdateDepth()
        {
            _available = false;
            _confValid = false; // never trust a previous frame's confidence
            if (_occlusionManager == null) return;

            if (!_occlusionManager.TryAcquireEnvironmentDepthCpuImage(out XRCpuImage depthImage))
                return;

            using (depthImage)
            {
                _w = depthImage.width;
                _h = depthImage.height;
                EnsureBuffer(ref _depth, _w * _h);
                CopyDepth(depthImage, _depth);
            }

            if (_occlusionManager.TryAcquireEnvironmentDepthConfidenceCpuImage(out XRCpuImage confImage))
            {
                using (confImage)
                {
                    var plane = confImage.GetPlane(0).data;
                    // Only trust confidence when it matches the depth grid exactly.
                    if (confImage.width == _w && confImage.height == _h)
                    {
                        EnsureBuffer(ref _conf, _w * _h);
                        NativeArray<byte>.Copy(plane, _conf, _w * _h);
                        _confValid = true;
                    }
                }
            }

            UpdateIntrinsics();
            _available = _hasIntrinsics;
        }

        public bool TryGetIntrinsics(out CameraIntrinsics intrinsics)
        {
            intrinsics = _intrinsics;
            return _hasIntrinsics;
        }

        public bool TryConvertUvToDepthPixel(Vector2 uvNorm, out int px, out int py)
        {
            px = py = 0;
            if (!_available) return false;
            // ARKit: depth shares the camera FOV/aspect -> uniform scale. (Android: replace with crop mapping.)
            px = Mathf.RoundToInt(uvNorm.x * _w);
            py = Mathf.RoundToInt(uvNorm.y * _h);
            return px >= 0 && px < _w && py >= 0 && py < _h;
        }

        public bool TryGetDepthPixel(int px, int py, out float meters, out float confidence)
        {
            meters = 0f;
            confidence = 0f;
            if (!_available || px < 0 || px >= _w || py < 0 || py >= _h) return false;

            int idx = py * _w + px;
            meters = _depth[idx];
            if (meters <= 0f || float.IsNaN(meters) || float.IsInfinity(meters)) return false;

            if (_confValid && _conf.IsCreated && idx < _conf.Length)
                confidence = Mathf.Clamp01(_conf[idx] / 2f); // 0/1/2 -> 0/0.5/1
            else
                confidence = 1f; // no (valid) confidence channel this frame => trust

            return true;
        }

        void UpdateIntrinsics()
        {
            _hasIntrinsics = false;
            if (_cameraManager == null || !_cameraManager.TryGetIntrinsics(out XRCameraIntrinsics intr)) return;
            if (intr.resolution.x <= 0 || intr.resolution.y <= 0) return; // guard against Inf scale

            // intrinsics are reported for the camera-image resolution; scale to depth resolution.
            float sx = (float)_w / intr.resolution.x;
            float sy = (float)_h / intr.resolution.y;
            _intrinsics = new CameraIntrinsics
            {
                Fx = intr.focalLength.x * sx,
                Fy = intr.focalLength.y * sy,
                Cx = intr.principalPoint.x * sx,
                Cy = intr.principalPoint.y * sy,
                Width = _w,
                Height = _h
            };
            _hasIntrinsics = _intrinsics.IsValid;
        }

        static void EnsureBuffer<T>(ref NativeArray<T> buf, int len) where T : struct
        {
            if (buf.IsCreated && buf.Length >= len) return;
            if (buf.IsCreated) buf.Dispose();
            buf = new NativeArray<T>(len, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
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
                    // Unknown format: best-effort float reinterpret.
                    var src = raw.Reinterpret<float>(1);
                    NativeArray<float>.Copy(src, dst, Mathf.Min(src.Length, dst.Length));
                    break;
                }
            }
        }
    }
}
#endif
