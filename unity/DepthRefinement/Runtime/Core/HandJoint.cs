using UnityEngine;

namespace HandMesh.DepthRefinement
{
    /// <summary>
    /// 21-joint hand layout. Index 0 is the wrist (root). Finger joints run
    /// MCP -> PIP -> DIP -> TIP. This matches the canonical re-ordered output of
    /// the Recon (MobRecon) model (models/manolayer.py new_order) and the
    /// MediaPipe hand topology used for the bone edges below.
    /// </summary>
    public enum HandJointId
    {
        Wrist = 0,
        ThumbCMC = 1, ThumbMCP = 2, ThumbIP = 3, ThumbTip = 4,
        IndexMCP = 5, IndexPIP = 6, IndexDIP = 7, IndexTip = 8,
        MiddleMCP = 9, MiddlePIP = 10, MiddleDIP = 11, MiddleTip = 12,
        RingMCP = 13, RingPIP = 14, RingDIP = 15, RingTip = 16,
        PinkyMCP = 17, PinkyPIP = 18, PinkyDIP = 19, PinkyTip = 20
    }

    /// <summary>Per-joint payload carried through the refinement pipeline.</summary>
    public struct HandJointData
    {
        /// <summary>2D joint position in FULL-FRAME normalized coords [0,1] (origin top-left).</summary>
        public Vector2 Uv;

        /// <summary>Root-relative 3D joint from the RGB model, in its arbitrary (non-metric) unit.</summary>
        public Vector3 LocalPosition;

        /// <summary>Refined metric joint in camera space (meters). Valid only after the pipeline runs.</summary>
        public Vector3 RefinedPosition;

        /// <summary>2D localization / visibility confidence from the RGB model heatmap, [0,1].</summary>
        public float Confidence;

        /// <summary>Depth-sample confidence assigned during sampling, [0,1]. 0 when no valid depth.</summary>
        public float DepthConfidence;

        /// <summary>True when this joint's depth was corrected from the sensor; false = RGB-only fallback.</summary>
        public bool WasRefined;
    }

    /// <summary>Static hand-skeleton metadata (bone edges + reference lengths).</summary>
    public static class HandSkeleton
    {
        public const int JointCount = 21;
        public const int WristIndex = 0;

        /// <summary>
        /// 20 directed bones as (parent, child) index pairs, ordered root -> tip per finger.
        /// Safe to iterate in order for forward kinematic correction (parent resolved first).
        /// </summary>
        public static readonly (int parent, int child)[] Bones =
        {
            (0, 1),  (1, 2),  (2, 3),  (3, 4),   // thumb
            (0, 5),  (5, 6),  (6, 7),  (7, 8),   // index
            (0, 9),  (9, 10), (10, 11),(11, 12), // middle
            (0, 13), (13, 14),(14, 15),(15, 16), // ring
            (0, 17), (17, 18),(18, 19),(19, 20)  // pinky
        };

        /// <summary>The 5 fingertip joints. Sub-pixel under device depth — corrected indirectly, not sampled directly.</summary>
        public static readonly int[] Fingertips = { 4, 8, 12, 16, 20 };

        static readonly bool[] FingertipMask = BuildFingertipMask();
        static bool[] BuildFingertipMask()
        {
            var m = new bool[JointCount];
            foreach (var f in Fingertips) m[f] = true;
            return m;
        }

        /// <summary>True for the 5 fingertips — excluded from direct depth fusion (PRD SM-6).</summary>
        public static bool IsFingertip(int jointIndex) => FingertipMask[jointIndex];

        /// <summary>Robust anchor joints (wrist + finger MCPs) used to seed shift-only root-depth alignment.</summary>
        public static readonly int[] RootAnchors = { 0, 5, 9, 13, 17 };

        /// <summary>
        /// Default reference bone lengths in meters (approximate adult right hand).
        /// Used to fix global metric scale (PRD: bone-length-fixed scale, shift-only alignment).
        /// Tunable via Inspector or first-frame calibration. Same order as <see cref="Bones"/>.
        /// </summary>
        public static readonly float[] DefaultBoneLengthsMeters =
        {
            0.040f, 0.035f, 0.030f, 0.025f, // thumb
            0.090f, 0.040f, 0.025f, 0.022f, // index (wrist->MCP is palm length)
            0.095f, 0.045f, 0.028f, 0.024f, // middle
            0.090f, 0.040f, 0.026f, 0.022f, // ring
            0.085f, 0.035f, 0.022f, 0.020f  // pinky
        };
    }
}
