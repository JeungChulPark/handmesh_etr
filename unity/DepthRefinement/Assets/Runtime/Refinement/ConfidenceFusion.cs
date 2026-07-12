using System;
using UnityEngine;

namespace HandMesh.DepthRefinement
{
    /// <summary>
    /// Step 6: turn the RGB root-relative pose + per-joint depth samples into a
    /// per-joint ABSOLUTE metric depth (Z along the optical axis).
    ///
    /// Design (PRD addendum B):
    ///   1. Fix metric scale `s` from bone lengths (NOT from the depth-vs-hand fit),
    ///      so depth contributes genuinely new absolute Z rather than a circular rescale.
    ///   2. Solve only the root-depth shift `t` (meters) by a robust RANSAC/MAD fit of
    ///      sampled depth vs predicted relative depth — avoids the s·t collinearity that
    ///      destabilises a full scale-and-shift fit on frontal hands.
    ///   3. Per inlier joint, blend sampled and predicted depth by confidence.
    ///   4. Occluded / outlier joints fall back to the predicted depth (no sensor trust).
    /// </summary>
    public sealed class ConfidenceFusion
    {
        [Serializable]
        public struct Config
        {
            public float OcclusionThresholdMeters; // |sampled - predicted| above this => outlier
            public int MinInliers;                 // need at least this many to trust alignment
            public bool FlipDepthAxis;             // set true if model's +z points toward camera

            public static Config Default => new Config
            {
                OcclusionThresholdMeters = 0.03f,
                MinInliers = 3,
                FlipDepthAxis = false
            };
        }

        public struct Result
        {
            public bool Succeeded;     // false => caller should RGB-only fallback
            public float Scale;        // metric scale (units -> meters)
            public float RootDepth;    // t (meters)
            public int InlierCount;
        }

        Config _cfg = Config.Default;
        readonly float[] _residuals = new float[HandSkeleton.JointCount];

        public void SetConfig(Config cfg) => _cfg = cfg;

        /// <summary>
        /// Metric scale from bone lengths: median over bones of (refLen / predictedLen).
        /// Robust to a few mis-predicted joints. Returns 0 if not estimable.
        /// </summary>
        public static float EstimateScale(HandPose pose, float[] refBoneLengths)
        {
            var bones = HandSkeleton.Bones;
            Span<float> ratios = stackalloc float[bones.Length];
            int n = 0;
            int count = Mathf.Min(bones.Length, refBoneLengths?.Length ?? 0);
            for (int b = 0; b < count; b++)
            {
                float predicted = Vector3.Distance(
                    pose.Joints[bones[b].parent].LocalPosition,
                    pose.Joints[bones[b].child].LocalPosition);
                if (predicted > 1e-6f && refBoneLengths[b] > 0f)
                    ratios[n++] = refBoneLengths[b] / predicted;
            }
            if (n == 0) return 0f;
            return MedianSpan(ratios, n);
        }

        /// <summary>
        /// Compute per-joint absolute depth. Fills <paramref name="absDepth"/> (meters) and
        /// <paramref name="refined"/> (true where a sensor depth was actually fused in).
        /// </summary>
        public Result Compute(
            HandPose pose,
            DepthSample[] samples,
            float[] refBoneLengths,
            float[] absDepth,
            bool[] refined)
        {
            float zSign = _cfg.FlipDepthAxis ? -1f : 1f;
            float s = EstimateScale(pose, refBoneLengths);
            if (s <= 0f)
                return new Result { Succeeded = false };

            // Collect residuals (sampled - s*z_rel) over joints with a valid sample.
            // Fingertips are excluded — sub-pixel under device depth (PRD SM-6), so they
            // must not drive the alignment nor be directly fused.
            int m = 0;
            for (int i = 0; i < HandSkeleton.JointCount; i++)
            {
                if (samples[i].Valid && !HandSkeleton.IsFingertip(i))
                    _residuals[m++] = samples[i].Meters - s * zSign * pose.Joints[i].LocalPosition.z;
            }
            if (m < _cfg.MinInliers)
                return new Result { Succeeded = false };

            // Robust shift: median residual, then MAD-gated inlier mean.
            float t = MedianCopy(_residuals, m);
            float mad = MedianAbsDeviation(_residuals, m, t);
            // Floor the band so an OcclusionThreshold of 0 / mad of 0 can't collapse it.
            float inlierBand = Mathf.Max(0.001f, Mathf.Max(_cfg.OcclusionThresholdMeters, 2.5f * mad));

            float sum = 0f;
            int inliers = 0;
            for (int i = 0; i < m; i++)
            {
                if (Mathf.Abs(_residuals[i] - t) <= inlierBand) { sum += _residuals[i]; inliers++; }
            }
            if (inliers < _cfg.MinInliers)
                return new Result { Succeeded = false };
            t = sum / inliers; // refined root-depth shift

            // Per-joint absolute depth.
            for (int i = 0; i < HandSkeleton.JointCount; i++)
            {
                float predicted = s * zSign * pose.Joints[i].LocalPosition.z + t;
                float w = Mathf.Clamp01(pose.Joints[i].Confidence) * Mathf.Clamp01(samples[i].Confidence);
                bool fuse = samples[i].Valid &&
                            !HandSkeleton.IsFingertip(i) &&        // fingertips: indirect only (SM-6)
                            w > 1e-3f &&                           // w~0 means depth not actually used
                            Mathf.Abs(samples[i].Meters - predicted) <= inlierBand;

                if (fuse)
                {
                    absDepth[i] = w * samples[i].Meters + (1f - w) * predicted;
                    refined[i] = true;
                }
                else
                {
                    absDepth[i] = predicted; // occluded / fingertip / no sample => RGB-metric prediction
                    refined[i] = false;
                }
            }

            return new Result { Succeeded = true, Scale = s, RootDepth = t, InlierCount = inliers };
        }

        // --- robust statistics helpers (allocation-free) ---

        static float MedianCopy(float[] src, int n)
        {
            Span<float> tmp = stackalloc float[n];
            for (int i = 0; i < n; i++) tmp[i] = src[i];
            return MedianSpan(tmp, n);
        }

        static float MedianAbsDeviation(float[] src, int n, float center)
        {
            Span<float> tmp = stackalloc float[n];
            for (int i = 0; i < n; i++) tmp[i] = Mathf.Abs(src[i] - center);
            return MedianSpan(tmp, n);
        }

        static float MedianSpan(Span<float> a, int n)
        {
            for (int i = 1; i < n; i++)
            {
                float key = a[i];
                int j = i - 1;
                while (j >= 0 && a[j] > key) { a[j + 1] = a[j]; j--; }
                a[j + 1] = key;
            }
            return (n & 1) == 1 ? a[n / 2] : 0.5f * (a[n / 2 - 1] + a[n / 2]);
        }
    }
}
