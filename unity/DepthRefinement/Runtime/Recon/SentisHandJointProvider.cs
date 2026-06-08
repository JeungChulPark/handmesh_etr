#if SENTIS_PRESENT
using System;
using Unity.Sentis;
using UnityEngine;

namespace HandMesh.DepthRefinement
{
    /// <summary>
    /// Step 9: runs the Recon (MobRecon) ONNX model on-device via Unity Sentis and
    /// exposes its output as an <see cref="IHandJointProvider"/>.
    ///
    /// Model I/O (verified from the repo's ONNX export, convert_to_onnx.py /
    /// Extra_model_input256_onnx_convert.ipynb):
    ///   input  "input0"  : (1,3,256,256) NCHW, values [0,1] (ToTensor only, no mean/std)
    ///   output           : (1,21,3) sigmoid, root-relative joints, wrist = index 0
    ///
    /// Set <see cref="InputTexture"/> each frame to the cropped hand region (or full frame);
    /// TextureConverter resizes to 256 and normalizes to [0,1] automatically.
    ///
    /// API targets Sentis 2.x. Adjust Worker/Tensor calls if you pin a different version.
    /// </summary>
    public sealed class SentisHandJointProvider : MonoBehaviour, IHandJointProvider
    {
        [SerializeField] ModelAsset _modelAsset;
        [SerializeField] BackendType _backend = BackendType.GPUCompute;

        [Tooltip("Region of the full frame the model runs on, normalized [0,1]. Default = full frame.")]
        [SerializeField] Rect _cropRect = new Rect(0, 0, 1, 1);

        [SerializeField] Handedness _handedness = Handedness.Right;
        [SerializeField] int _frameWidth = 1920;
        [SerializeField] int _frameHeight = 1440;

        /// <summary>Cropped hand RGB to run inference on. Assign from your AR camera background each frame.</summary>
        public Texture InputTexture { get; set; }

        public event Action<HandPose> OnHandPoseUpdated;

        Model _model;
        Worker _worker;
        readonly HandPose _pose = new HandPose();
        readonly float[] _out = new float[HandSkeleton.JointCount * 3];

        void Awake()
        {
            if (_modelAsset == null)
            {
                Debug.LogError("[SentisHandJointProvider] ModelAsset not assigned.");
                enabled = false;
                return;
            }
            _model = ModelLoader.Load(_modelAsset);
            _worker = new Worker(_model, _backend);
        }

        void OnDestroy() => _worker?.Dispose();

        bool _warnedShape;

        void Update()
        {
            if (_worker == null) return;
            if (InputTexture == null)
            {
                _pose.IsValid = false; // no input => don't let a stale pose look current
                return;
            }
            RunInference(InputTexture);
            if (_pose.IsValid) OnHandPoseUpdated?.Invoke(_pose);
        }

        public bool TryGetLatest(out HandPose pose)
        {
            pose = _pose;
            return _pose.IsValid;
        }

        void RunInference(Texture tex)
        {
            using Tensor<float> input = TextureConverter.ToTensor(tex, 256, 256, 3);
            _worker.Schedule(input);

            if (!(_worker.PeekOutput() is Tensor<float> peek))
            {
                WarnShapeOnce("output is not a float tensor");
                _pose.IsValid = false;
                return;
            }

            using Tensor<float> output = peek.ReadbackAndClone();
            float[] arr = output.DownloadToArray();
            if (arr.Length < _out.Length) // expect (1,21,3) = 63
            {
                WarnShapeOnce($"output length {arr.Length} < expected {_out.Length} (1x21x3)");
                _pose.IsValid = false;
                return;
            }

            System.Array.Copy(arr, _out, _out.Length);
            BuildPose();
        }

        void WarnShapeOnce(string detail)
        {
            if (_warnedShape) return;
            _warnedShape = true;
            Debug.LogError($"[SentisHandJointProvider] Unexpected model output: {detail}. " +
                           "Expected a (1,21,3) float tensor. (further warnings suppressed)");
        }

        void BuildPose()
        {
            _pose.IsValid = true;
            _pose.Handedness = _handedness;
            _pose.Timestamp = Time.timeAsDouble;
            _pose.FrameWidth = _frameWidth;
            _pose.FrameHeight = _frameHeight;

            // Joint 0 (wrist) as the root for a guaranteed root-relative LocalPosition.
            float rx = _out[0], ry = _out[1], rz = _out[2];

            for (int i = 0; i < HandSkeleton.JointCount; i++)
            {
                float mx = _out[i * 3 + 0]; // [0,1] within the 256 crop
                float my = _out[i * 3 + 1];
                float mz = _out[i * 3 + 2];

                // Map crop-normalized (mx,my) back into FULL-FRAME normalized UV.
                _pose.Joints[i].Uv = new Vector2(
                    _cropRect.x + mx * _cropRect.width,
                    _cropRect.y + my * _cropRect.height);

                _pose.Joints[i].LocalPosition = new Vector3(mx - rx, my - ry, mz - rz);

                // Model has no per-joint confidence output (PRD Open Q8) -> assume high.
                _pose.Joints[i].Confidence = 1f;
                _pose.Joints[i].DepthConfidence = 0f;
                _pose.Joints[i].WasRefined = false;
            }
        }
    }
}
#endif
