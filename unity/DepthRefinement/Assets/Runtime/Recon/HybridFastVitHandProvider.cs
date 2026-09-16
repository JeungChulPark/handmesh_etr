#if SENTIS_PRESENT && ARFOUNDATION_PRESENT
using System;
using System.Collections.Generic;
using Unity.InferenceEngine;
using UnityEngine;

namespace HandMesh.DepthRefinement
{
    /// <summary>
    /// Runs the FastViT hybrid lifter (deploy best, `hybrid_sa12_replay_ft_bb_uv2d`,
    /// combined 19.53 mm / iPhone 16.29 mm) — exported as hybrid_fastvit.onnx by
    /// <c>export_hybrid_fastvit_onnx.py</c> — on iPhone via Unity Sentis.
    ///
    /// Unlike <see cref="HybridBHandProvider"/> (locked mode, backbone-only graph + C# fusion),
    /// this model is BACKBONE-PRIMARY: the FastViT-SA12 backbone predicts 3D from the 4-ch crop
    /// and a learned MLP head refines it with the detector-2D + sensor-depth prior. Head and
    /// fusion are INSIDE the ONNX graph, so this provider only assembles the five inputs:
    ///
    ///     image [1,4,256,256]  4-ch crop (RGB 0..1 + median-centred depth) — the exact tensor
    ///                          DualStreamHandProvider already builds
    ///     feat  [1,21,4]       per joint (u*2-1, v*2-1, z_use - z_ref, valid)   (u,v norm 0..1)
    ///     uv    [1,21,2]       landmarks in full-image PIXELS
    ///     z_use [1,21]         per-joint sensor depth (metres; invalid -> z_ref)
    ///     K     [1,4]          fx, fy, cx, cy (full-image intrinsics)
    ///       ->  keypoints [1,21,3]  root-relative metric 3D (wrist = 0)
    ///
    /// The absolute pose is the output anchored at the wrist landmark back-projected to the
    /// sensor wrist depth (same anchoring as infer_hybrid.py / HybridBHandProvider).
    ///
    /// Wire in the editor exactly like HybridBHandProvider: a crop-only
    /// <see cref="DualStreamHandProvider"/> (ModelAsset left empty) supplies the crop /
    /// intrinsics / depth sampler; <see cref="HandBboxBridge"/> supplies the 21 landmarks
    /// (normalized sensor-image, MediaPipe order — see the Apple-Vision order caveat in
    /// docs/UNITY_HYBRID_FASTVIT.md). Assign hybrid_fastvit.onnx below. Runs in LateUpdate.
    /// </summary>
    public sealed class HybridFastVitHandProvider : MonoBehaviour
    {
        const int Size = 256;
        const int JointCount = 21;
        const int Wrist = 0;

        [Header("Sources")]
        [Tooltip("DualStreamHandProvider in crop-only mode (no ModelAsset) — supplies the 4-ch crop, "
               + "full-image intrinsics, and a per-joint depth sampler each frame.")]
        [SerializeField] DualStreamHandProvider _crop;
        [Tooltip("Supplies the 21 detector landmarks (normalized sensor-image, MediaPipe order).")]
        [SerializeField] HandBboxBridge _landmarks;

        [Header("Model")]
        [SerializeField] ModelAsset _modelAsset;   // hybrid_fastvit.onnx
        [SerializeField] BackendType _backend = BackendType.GPUCompute;

        /// <summary>Absolute camera-space joints (metres, CV frame +X right/+Y down/+Z fwd).</summary>
        public Vector3[] AbsJoints { get; } = new Vector3[JointCount];
        public bool HasPose { get; private set; }
        public event Action OnPose;

        Model _model;
        Worker _worker;
        readonly float[] _feat = new float[JointCount * 4];
        readonly float[] _uvPx = new float[JointCount * 2];
        readonly float[] _zUse = new float[JointCount];
        readonly float[] _kBuf = new float[4];
        readonly float[] _zRaw = new float[JointCount];   // sampled depth, 0 = invalid
        readonly float[] _sort = new float[JointCount];   // median scratch

        void Awake()
        {
            if (_modelAsset == null) { Debug.LogError("[HybridFastViT] ModelAsset (hybrid_fastvit.onnx) not assigned."); enabled = false; return; }
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

            // per-joint sensor depth; z_ref = median of the valid samples (train_zlifter contract)
            int nValid = 0;
            for (int j = 0; j < JointCount; j++)
            {
                _zRaw[j] = _crop.TrySampleDepthMetres(lm[j].x, lm[j].y, out float zj) && zj > 0f ? zj : 0f;
                if (_zRaw[j] > 0f) _sort[nValid++] = _zRaw[j];
            }
            if (nValid == 0) return;
            float zRef = Median(_sort, nValid);

            for (int j = 0; j < JointCount; j++)
            {
                bool valid = _zRaw[j] > 0f;
                _zUse[j] = valid ? _zRaw[j] : zRef;
                _uvPx[j * 2 + 0] = lm[j].x * imgW;
                _uvPx[j * 2 + 1] = lm[j].y * imgH;
                _feat[j * 4 + 0] = lm[j].x * 2f - 1f;          // u / W * 2 - 1
                _feat[j * 4 + 1] = lm[j].y * 2f - 1f;          // v / H * 2 - 1
                _feat[j * 4 + 2] = _zUse[j] - zRef;
                _feat[j * 4 + 3] = valid ? 1f : 0f;
            }
            _kBuf[0] = fx; _kBuf[1] = fy; _kBuf[2] = cx; _kBuf[3] = cy;

            using var tImage = new Tensor<float>(new TensorShape(1, 4, Size, Size), image);
            using var tFeat = new Tensor<float>(new TensorShape(1, JointCount, 4), _feat);
            using var tUv = new Tensor<float>(new TensorShape(1, JointCount, 2), _uvPx);
            using var tZ = new Tensor<float>(new TensorShape(1, JointCount), _zUse);
            using var tK = new Tensor<float>(new TensorShape(1, 4), _kBuf);
            _worker.SetInput("image", tImage);
            _worker.SetInput("feat", tFeat);
            _worker.SetInput("uv", tUv);
            _worker.SetInput("z_use", tZ);
            _worker.SetInput("K", tK);
            _worker.Schedule();
            if (_worker.PeekOutput("keypoints") is not Tensor<float> kpPeek) return;
            using Tensor<float> kp = kpPeek.ReadbackAndClone();
            float[] p = kp.DownloadToArray();                                          // [21*3] root-relative
            if (p.Length < JointCount * 3) return;

            // anchor the root at the wrist landmark back-projected to the sensor wrist depth
            float z0 = _zRaw[Wrist] > 0f ? _zRaw[Wrist] : zRef;
            float rx = (_uvPx[Wrist * 2 + 0] - cx) / fx * z0;
            float ry = (_uvPx[Wrist * 2 + 1] - cy) / fy * z0;
            for (int j = 0; j < JointCount; j++)
                AbsJoints[j] = new Vector3(rx + p[j * 3 + 0], ry + p[j * 3 + 1], z0 + p[j * 3 + 2]);
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
