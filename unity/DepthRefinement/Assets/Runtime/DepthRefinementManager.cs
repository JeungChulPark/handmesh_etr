using System;
using UnityEngine;

namespace HandMesh.DepthRefinement
{
    /// <summary>
    /// Step 1: the package entry point. Wires an <see cref="IHandJointProvider"/> (RGB joints)
    /// and an <see cref="IDepthProvider"/> (device depth) into the <see cref="DepthRefinementPipeline"/>,
    /// surfaces all tuning via the Inspector, and raises <see cref="OnRefined"/> each frame.
    ///
    /// Providers are assigned as plain MonoBehaviour fields and cast to their interfaces at
    /// runtime, so this manager has NO compile dependency on AR Foundation or Sentis — drop in
    /// any provider implementations.
    /// </summary>
    [DefaultExecutionOrder(100)]
    public sealed class DepthRefinementManager : MonoBehaviour
    {
        [Header("Providers (assign components implementing the interfaces)")]
        [Tooltip("Component implementing IHandJointProvider (e.g. SentisHandJointProvider or MockHandJointProvider).")]
        [SerializeField] MonoBehaviour _jointProviderSource;
        [Tooltip("Component implementing IDepthProvider (e.g. ARFoundationDepthProvider).")]
        [SerializeField] MonoBehaviour _depthProviderSource;

        [Header("Depth Sampling (FR-4)")]
        [SerializeField, Range(0, 5)] int _windowRadius = 2;
        [SerializeField, Range(0f, 1f)] float _minPixelConfidence = 0.5f;
        [SerializeField, Range(0f, 1f)] float _minValidRatio = 0.3f;
        [SerializeField] float _maxLocalSpreadMeters = 0.02f;

        [Header("Fusion / Alignment (FR-5, FR-6)")]
        [SerializeField] float _occlusionThresholdMeters = 0.03f;
        [SerializeField, Range(1, 21)] int _minInliers = 3;
        [Tooltip("Enable if the model's +z points toward the camera instead of away.")]
        [SerializeField] bool _flipDepthAxis = false;

        [Header("Bone-Length Constraint (FR-7)")]
        [SerializeField, Range(0f, 1f)] float _boneStrength = 0.5f;
        [SerializeField] bool _pinWrist = true;
        [Tooltip("Override reference bone lengths (20 entries, meters). Leave empty to use defaults.")]
        [SerializeField] float[] _referenceBoneLengths;

        [Header("Temporal Filter (FR-6)")]
        [SerializeField] bool _enableTemporalFilter = true;
        [SerializeField] float _filterMinCutoff = 1.0f;
        [SerializeField] float _filterBeta = 0.02f;
        [SerializeField] float _filterDCutoff = 1.0f;
        [SerializeField] float _filterZMinCutoff = 0.6f;
        [SerializeField] float _filterZBeta = 0.01f;

        [Header("Performance")]
        [Tooltip("Run depth acquisition every Nth frame (asymmetric rate). 1 = every frame.")]
        [SerializeField, Range(1, 4)] int _depthStride = 1;

        /// <summary>Raised after each refinement pass with the refined pose and the gate outcome.</summary>
        public event Action<HandPose, DepthRefinementPipeline.Outcome> OnRefined;

        public HandPose LatestPose { get; private set; }
        public DepthRefinementPipeline.Outcome LastOutcome { get; private set; }

        IHandJointProvider _jointProvider;
        IDepthProvider _depthProvider;
        IDepthFrameUpdater _depthUpdater;
        DepthRefinementPipeline _pipeline;
        readonly HandPose _working = new HandPose(); // owned copy — avoids provider single-instance aliasing
        int _frame;

        void Awake()
        {
            _jointProvider = _jointProviderSource as IHandJointProvider;
            _depthProvider = _depthProviderSource as IDepthProvider;
            _depthUpdater = _depthProviderSource as IDepthFrameUpdater;

            if (_jointProvider == null)
                Debug.LogError($"[DepthRefinementManager] Joint provider '{_jointProviderSource}' does not implement IHandJointProvider.");
            if (_depthProviderSource != null && _depthProvider == null)
                Debug.LogError($"[DepthRefinementManager] Depth provider '{_depthProviderSource}' does not implement IDepthProvider.");
            else if (_depthProvider == null)
                Debug.LogWarning("[DepthRefinementManager] No IDepthProvider assigned — running in RGB-only fallback.");
            else if (_depthUpdater == null)
                Debug.LogWarning("[DepthRefinementManager] Depth provider does not implement IDepthFrameUpdater — it must refresh its own depth each frame.");

            _pipeline = new DepthRefinementPipeline();
            if (_referenceBoneLengths != null && _referenceBoneLengths.Length == HandSkeleton.Bones.Length)
                _pipeline.SetReferenceBoneLengths(_referenceBoneLengths);
            _pipeline.SetConfig(BuildConfig());
        }

        void OnValidate()
        {
            _pipeline?.SetConfig(BuildConfig());
        }

        void Update()
        {
            if (_jointProvider == null) return;

            if (_depthUpdater != null && (_frame++ % Mathf.Max(1, _depthStride)) == 0)
                _depthUpdater.UpdateDepth();

            if (!_jointProvider.TryGetLatest(out HandPose pose) || pose == null || !pose.IsValid)
                return;

            // Copy into our owned instance: providers reuse a single HandPose, so refining
            // theirs in place would tear if the provider updates mid-frame or a consumer
            // reads LatestPose next frame.
            _working.CopyFrom(pose);
            LastOutcome = _pipeline.Process(_working, _depthProvider);
            LatestPose = _working;
            OnRefined?.Invoke(_working, LastOutcome);
        }

        DepthRefinementPipeline.Config BuildConfig() => new DepthRefinementPipeline.Config
        {
            Sampler = new JointDepthSampler.Config
            {
                WindowRadius = _windowRadius,
                MinPixelConfidence = _minPixelConfidence,
                MinValidRatio = _minValidRatio,
                MaxLocalSpreadMeters = _maxLocalSpreadMeters
            },
            Fusion = new ConfidenceFusion.Config
            {
                OcclusionThresholdMeters = _occlusionThresholdMeters,
                MinInliers = _minInliers,
                FlipDepthAxis = _flipDepthAxis
            },
            Bone = new BoneLengthConstraint.Config
            {
                Strength = _boneStrength,
                PinWrist = _pinWrist
            },
            EnableTemporalFilter = _enableTemporalFilter,
            FilterMinCutoff = _filterMinCutoff,
            FilterBeta = _filterBeta,
            FilterDCutoff = _filterDCutoff,
            FilterZMinCutoff = _filterZMinCutoff,
            FilterZBeta = _filterZBeta
        };
    }
}
