using System;

namespace HandMesh.DepthRefinement
{
    /// <summary>
    /// Source of RGB-predicted hand joints. Implemented by SentisHandJointProvider
    /// (real Recon ONNX inference) and MockHandJointProvider (tests/editor).
    /// The refinement pipeline depends only on this interface, so the joint source
    /// is swappable without touching the refinement math.
    /// </summary>
    public interface IHandJointProvider
    {
        /// <summary>Raised once per frame a new hand pose is available.</summary>
        event Action<HandPose> OnHandPoseUpdated;

        /// <summary>
        /// Poll the most recent pose. Returns false when no hand has been detected yet.
        /// The returned instance is owned by the provider and reused; consume immediately.
        /// </summary>
        bool TryGetLatest(out HandPose pose);
    }
}
