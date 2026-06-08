using System;
using UnityEngine;

namespace HandMesh.DepthRefinement
{
    /// <summary>
    /// Orchestrates FR-4..FR-8 on a single hand pose:
    ///   sample depth -> estimate scale -> shift-only root-depth align -> reconstruct
    ///   metric joints -> confidence-fuse -> bone-length constrain -> temporal filter,
    ///   all behind a runtime accept/reject GATE (FR-8): any failure falls back to
    ///   RGB-only output so refinement never makes a joint worse than the baseline.
    ///
    /// Pure C# (UnityEngine math only) so it runs in EditMode tests with a fake
    /// IDepthProvider. Allocation-free in steady state.
    /// </summary>
    public sealed class DepthRefinementPipeline
    {
        [Serializable]
        public struct Config
        {
            public JointDepthSampler.Config Sampler;
            public ConfidenceFusion.Config Fusion;
            public BoneLengthConstraint.Config Bone;
            public bool EnableTemporalFilter;
            public float FilterMinCutoff;
            public float FilterBeta;
            public float FilterDCutoff;
            public float FilterZMinCutoff;
            public float FilterZBeta;

            public static Config Default => new Config
            {
                Sampler = JointDepthSampler.Config.Default,
                Fusion = ConfidenceFusion.Config.Default,
                Bone = BoneLengthConstraint.Config.Default,
                EnableTemporalFilter = true,
                FilterMinCutoff = 1.0f,
                FilterBeta = 0.02f,
                FilterDCutoff = 1.0f,
                FilterZMinCutoff = 0.6f, // stronger smoothing on the noisy depth axis
                FilterZBeta = 0.01f
            };
        }

        public struct Outcome
        {
            public bool Refined;       // true if at least the global metric alignment succeeded
            public int RefinedJoints;  // count of joints that fused a sensor depth
            public string FallbackReason; // non-null when RGB-only fallback was taken
        }

        readonly JointDepthSampler _sampler = new JointDepthSampler();
        readonly ConfidenceFusion _fusion = new ConfidenceFusion();
        readonly BoneLengthConstraint _bone = new BoneLengthConstraint();
        readonly OneEuroFilterVector3[] _filters = new OneEuroFilterVector3[HandSkeleton.JointCount];

        readonly DepthSample[] _samples = new DepthSample[HandSkeleton.JointCount];
        readonly float[] _absDepth = new float[HandSkeleton.JointCount];
        readonly bool[] _refined = new bool[HandSkeleton.JointCount];

        float[] _refBoneLengths = (float[])HandSkeleton.DefaultBoneLengthsMeters.Clone();
        Config _cfg = Config.Default;
        bool _warnedFallback;

        public DepthRefinementPipeline()
        {
            for (int i = 0; i < _filters.Length; i++) _filters[i] = new OneEuroFilterVector3();
            ApplyConfig();
        }

        public void SetReferenceBoneLengths(float[] lengths)
        {
            if (lengths != null && lengths.Length == HandSkeleton.Bones.Length)
                _refBoneLengths = (float[])lengths.Clone();
        }

        public void SetConfig(Config cfg)
        {
            _cfg = cfg;
            ApplyConfig();
        }

        void ApplyConfig()
        {
            _sampler.SetConfig(_cfg.Sampler);
            _fusion.SetConfig(_cfg.Fusion);
            _bone.SetConfig(_cfg.Bone);
            foreach (var f in _filters)
                f.Configure(_cfg.FilterMinCutoff, _cfg.FilterBeta, _cfg.FilterDCutoff,
                            _cfg.FilterZMinCutoff, _cfg.FilterZBeta);
        }

        public void ResetTemporal()
        {
            foreach (var f in _filters) f.Reset();
        }

        /// <summary>
        /// Refines <paramref name="pose"/> in place (writes RefinedPosition / WasRefined / DepthConfidence).
        /// Returns an Outcome describing whether refinement applied or a fallback was taken.
        /// </summary>
        public Outcome Process(HandPose pose, IDepthProvider depth)
        {
            if (pose == null || !pose.IsValid)
                return new Outcome { Refined = false, FallbackReason = "no-hand" };

            // GATE: need depth + intrinsics, else RGB-only fallback (still emit a relative-metric shape).
            if (depth == null || !depth.IsDepthAvailable || !depth.TryGetIntrinsics(out var k) || !k.IsValid)
                return Fallback(pose, "no-depth-or-intrinsics");

            // FR-4: robust per-joint depth sampling.
            for (int i = 0; i < HandSkeleton.JointCount; i++)
                _samples[i] = _sampler.Sample(depth, pose.Joints[i].Uv);

            // FR-5/6: scale (bones) + shift-only root-depth alignment + confidence fusion.
            var fr = _fusion.Compute(pose, _samples, _refBoneLengths, _absDepth, _refined);
            if (!fr.Succeeded)
                return Fallback(pose, "alignment-failed");

            // FR-5: back-project each joint to metric camera space.
            for (int i = 0; i < HandSkeleton.JointCount; i++)
            {
                Vector2 px = JointReconstructor.UvToDepthPixel(pose.Joints[i].Uv, in k);
                pose.Joints[i].RefinedPosition = JointReconstructor.Unproject(px, _absDepth[i], in k);
                pose.Joints[i].WasRefined = _refined[i];
                pose.Joints[i].DepthConfidence = _samples[i].Valid ? _samples[i].Confidence : 0f;
            }

            // FR-7 (constraint): enforce bone lengths to repair residual depth errors.
            _bone.Apply(pose.Joints, _refBoneLengths);

            // FR-6: temporal smoothing (Z-focused), kills jitter without lagging fast motion.
            if (_cfg.EnableTemporalFilter)
            {
                for (int i = 0; i < HandSkeleton.JointCount; i++)
                    pose.Joints[i].RefinedPosition = _filters[i].Filter(pose.Joints[i].RefinedPosition, pose.Timestamp);
            }

            int refinedCount = 0;
            for (int i = 0; i < HandSkeleton.JointCount; i++) if (_refined[i]) refinedCount++;
            return new Outcome { Refined = true, RefinedJoints = refinedCount, FallbackReason = null };
        }

        /// <summary>
        /// RGB-only fallback: emit the raw root-relative joints (UNITLESS, not metric) with
        /// WasRefined = false. We deliberately do NOT multiply by a hand-derived scale — that
        /// would be the circular rescale the design forbids (PRD addendum §B-0). Downstream
        /// must treat these as non-metric when WasRefined is false. Warns once.
        /// </summary>
        Outcome Fallback(HandPose pose, string reason)
        {
            if (!_warnedFallback)
            {
                _warnedFallback = true;
                Debug.LogWarning($"[DepthRefinement] Refinement falling back to RGB-only ({reason}); " +
                                 "output joints are non-metric until depth + intrinsics are available. " +
                                 "(further fallbacks suppressed)");
            }
            for (int i = 0; i < HandSkeleton.JointCount; i++)
            {
                pose.Joints[i].RefinedPosition = pose.Joints[i].LocalPosition; // unitless, root-relative
                pose.Joints[i].WasRefined = false;
                pose.Joints[i].DepthConfidence = 0f;
            }
            return new Outcome { Refined = false, RefinedJoints = 0, FallbackReason = reason };
        }
    }
}
