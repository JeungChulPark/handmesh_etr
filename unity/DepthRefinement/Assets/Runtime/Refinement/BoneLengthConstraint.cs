using UnityEngine;

namespace HandMesh.DepthRefinement
{
    /// <summary>
    /// Step 7: enforce anatomical bone lengths on the reconstructed metric joints.
    /// Walks the skeleton root -> tip and pulls each child onto its reference bone
    /// length along the current bone direction. A wrong per-joint depth shows up as a
    /// violated bone length, so this corrects gross depth errors the sampler let through.
    /// Strength in [0,1] blends between "leave as-is" (0) and "snap to length" (1).
    /// </summary>
    public sealed class BoneLengthConstraint
    {
        [System.Serializable]
        public struct Config
        {
            public float Strength;        // 0 = off, 1 = hard snap
            public bool PinWrist;         // keep joint 0 fixed

            public static Config Default => new Config { Strength = 0.5f, PinWrist = true };
        }

        Config _cfg = Config.Default;
        public void SetConfig(Config cfg) => _cfg = cfg;

        /// <summary>
        /// Adjusts <c>RefinedPosition</c> of <paramref name="joints"/> in place.
        /// Bones are iterated in order so each parent is already corrected before its child.
        /// </summary>
        public void Apply(HandJointData[] joints, float[] refBoneLengths)
        {
            if (_cfg.Strength <= 0f) return;
            float strength = Mathf.Clamp01(_cfg.Strength);

            var bones = HandSkeleton.Bones;
            for (int b = 0; b < bones.Length; b++)
            {
                int p = bones[b].parent;
                int c = bones[b].child;
                float refLen = refBoneLengths[b];
                if (refLen <= 0f) continue;

                Vector3 dir = joints[c].RefinedPosition - joints[p].RefinedPosition;
                float len = dir.magnitude;
                if (len < 1e-6f) continue; // degenerate; leave child untouched

                Vector3 corrected = joints[p].RefinedPosition + dir * (refLen / len);
                joints[c].RefinedPosition = Vector3.Lerp(joints[c].RefinedPosition, corrected, strength);
            }
        }
    }
}
