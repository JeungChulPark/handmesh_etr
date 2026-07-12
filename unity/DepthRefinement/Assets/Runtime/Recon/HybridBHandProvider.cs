#if SENTIS_PRESENT && ARFOUNDATION_PRESENT
using System;
using System.Collections.Generic;
using Unity.InferenceEngine;
using UnityEngine;

namespace HandMesh.DepthRefinement
{
    /// <summary>
    /// Runs model B (hybrid_B, "locked" mode) — exported as hybrid_B_backbone.onnx by
    /// <c>export_hybrid_b_onnx.py</c> — on iPhone via Unity Sentis.
    ///
    /// B is NOT self-contained like <see cref="DualStreamHandProvider"/>: its 2D is LOCKED to
    /// the detector's 21 landmarks and only its Z (relative depth) comes from the backbone.
    /// So the delivered 3D pose is assembled DETERMINISTICALLY here:
    ///
    ///     kp   = backbone(image)              // [21,3], we use only kp[:,2] (relative depth)
    ///     Z0   = sensor depth at the wrist landmark (metres)
    ///     dz_j = kp[j].z - kp[0].z            // backbone per-joint relative depth (wrist=0)
    ///     Z_j  = Z0 + dz_j
    ///     X_j  = (u_j - cx)/fx * Z_j          // u_j,v_j = landmark j in full-image pixels
    ///     Y_j  = (v_j - cy)/fy * Z_j
    ///
    /// The backbone's 4-ch crop is IDENTICAL to the dual-stream `image` tensor, so this provider
    /// REUSES <see cref="DualStreamHandProvider"/>'s proven crop/depth/intrinsics pipeline rather
    /// than duplicating it. See docs/UNITY_HYBRID_B.md for the 3 small additive hooks the reused
    /// provider + HandBboxBridge need (LastImageTensor / LastIntrinsics / SampleDepthMetres /
    /// LandmarksNormalized), and for the Apple-Vision -> MediaPipe landmark-order caveat.
    ///
    /// Wire in the editor: put a DualStreamHandProvider in "crop-only" mode (leave its ModelAsset
    /// empty) to build the crop each frame, feed its landmarks via HandVisionBridge/HandBboxBridge,
    /// and assign hybrid_B_backbone.onnx below. Runs in LateUpdate so the crop is ready.
    ///
    /// NEURAL depth strongly recommended over PERFORMANCE: B anchors the root on the sensor wrist
    /// depth, and the frame-aligned ZED test showed PERFORMANCE moves the root ~90 mm and the pose
    /// ~104 mm vs NEURAL (docs/ZED_HYBRID_B_DEPLOY.md).
    /// </summary>
    public sealed class HybridBHandProvider : MonoBehaviour
    {
        const int Size = 256;
        const int JointCount = 21;
        const int Wrist = 0;

        [Header("Sources")]
        [Tooltip("DualStreamHandProvider in crop-only mode (no ModelAsset) — supplies the 4-ch crop, "
               + "full-image intrinsics, and a wrist-depth sampler each frame.")]
        [SerializeField] DualStreamHandProvider _crop;
        [Tooltip("Supplies the 21 detector landmarks (normalized sensor-image, MediaPipe order).")]
        [SerializeField] HandBboxBridge _landmarks;

        [Header("Model")]
        [SerializeField] ModelAsset _modelAsset;   // hybrid_B_backbone.onnx
        [SerializeField] BackendType _backend = BackendType.GPUCompute;

        /// <summary>Absolute camera-space joints (metres, CV frame +X right/+Y down/+Z fwd).</summary>
        public Vector3[] AbsJoints { get; } = new Vector3[JointCount];
        public bool HasPose { get; private set; }
        public event Action OnPose;

        Model _model;
        Worker _worker;
        readonly float[] _zbuf = new float[JointCount];

        void Awake()
        {
            if (_modelAsset == null) { Debug.LogError("[HybridB] ModelAsset (hybrid_B_backbone.onnx) not assigned."); enabled = false; return; }
            if (_crop == null) _crop = FindObjectOfType<DualStreamHandProvider>();
            if (_landmarks == null) _landmarks = FindObjectOfType<HandBboxBridge>();
            _model = ModelLoader.Load(_modelAsset);
            _worker = new Worker(_model, _backend);
        }

        void OnDestroy() => _worker?.Dispose();

        // Mode switches disable this component; never leave a stale pose behind.
        void OnDisable() => HasPose = false;

        void LateUpdate()   // after _crop.Update() has built this frame's crop
        {
            HasPose = false;
            if (_worker == null || _crop == null || _landmarks == null) return;
            if (!_crop.TryGetLastImageTensor(out float[] image)) return;               // [4*256*256]
            if (!_crop.TryGetLastIntrinsics(out float fx, out float fy, out float cx, out float cy,
                                            out int imgW, out int imgH)) return;
            IReadOnlyList<Vector2> lm = _landmarks.LandmarksNormalized;
            if (lm == null || lm.Count < JointCount) return;

            // backbone: 4-ch crop -> keypoints [1,21,3]. Only the Z column (relative depth) is used;
            // in-plane crop rotation does not affect depth, so no output un-rotation is needed.
            using var t = new Tensor<float>(new TensorShape(1, 4, Size, Size), image);
            _worker.SetInput("image", t);
            _worker.Schedule();
            if (_worker.PeekOutput("keypoints") is not Tensor<float> kpPeek) return;
            using Tensor<float> kp = kpPeek.ReadbackAndClone();
            float[] k = kp.DownloadToArray();
            if (k.Length < JointCount * 3) return;
            float zWristRel = k[Wrist * 3 + 2];

            // Z0 = sensor depth at the wrist landmark; fall back to the median of valid joint samples.
            float z0 = 0f; int nz = 0;
            for (int j = 0; j < JointCount; j++)
                if (_crop.TrySampleDepthMetres(lm[j].x, lm[j].y, out float zj) && zj > 0f) _zbuf[nz++] = zj;
            if (nz == 0) return;
            if (!_crop.TrySampleDepthMetres(lm[Wrist].x, lm[Wrist].y, out z0) || z0 <= 0f)
                z0 = Median(_zbuf, nz);

            for (int j = 0; j < JointCount; j++)
            {
                float u = lm[j].x * imgW, v = lm[j].y * imgH;
                float z = z0 + (k[j * 3 + 2] - zWristRel);              // Z0 + backbone relative depth
                AbsJoints[j] = new Vector3((u - cx) / fx * z, (v - cy) / fy * z, z);
            }
            HasPose = true;
            OnPose?.Invoke();
        }

        static float Median(float[] a, int n)
        {
            Array.Sort(a, 0, n);
            return (n & 1) == 1 ? a[n / 2] : 0.5f * (a[n / 2 - 1] + a[n / 2]);
        }
    }
}
#endif
