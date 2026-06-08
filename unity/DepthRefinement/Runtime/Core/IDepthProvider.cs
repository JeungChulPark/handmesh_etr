using UnityEngine;

namespace HandMesh.DepthRefinement
{
    /// <summary>Pinhole intrinsics in the DEPTH image's pixel space.</summary>
    public struct CameraIntrinsics
    {
        public float Fx, Fy, Cx, Cy;
        public int Width, Height;

        public bool IsValid =>
            Fx > 0f && Fy > 0f && Width > 0 && Height > 0 &&
            float.IsFinite(Fx) && float.IsFinite(Fy) && float.IsFinite(Cx) && float.IsFinite(Cy);
    }

    /// <summary>
    /// Source of metric per-pixel depth (iOS LiDAR sceneDepth via AR Foundation
    /// environmentDepth, or a metric monocular depth model). Exposes pixel-level
    /// access so the sampler can do robust neighborhood sampling, plus the
    /// coordinate mapping from full-frame normalized UV to depth-image pixels
    /// (ARKit = uniform scale, ARCore = FOV crop, monocular = identity).
    /// </summary>
    public interface IDepthProvider
    {
        /// <summary>True when a depth frame + intrinsics are ready to be sampled this frame.</summary>
        bool IsDepthAvailable { get; }

        int DepthWidth { get; }
        int DepthHeight { get; }

        /// <summary>Depth-space intrinsics (already scaled to <see cref="DepthWidth"/> x <see cref="DepthHeight"/>).</summary>
        bool TryGetIntrinsics(out CameraIntrinsics intrinsics);

        /// <summary>
        /// Map a full-frame normalized UV [0,1] to integer depth-image pixel coords.
        /// Returns false if the UV falls outside the depth image's covered FOV.
        /// </summary>
        bool TryConvertUvToDepthPixel(Vector2 uvNorm, out int px, out int py);

        /// <summary>
        /// Read one depth pixel. <paramref name="meters"/> is metric depth along the optical axis,
        /// <paramref name="confidence"/> in [0,1] (low/medium/high mapped to 0/0.5/1 for ARKit).
        /// Returns false for out-of-range / invalid pixels.
        /// </summary>
        bool TryGetDepthPixel(int px, int py, out float meters, out float confidence);
    }

    /// <summary>
    /// Optional: a depth source that must pull a fresh frame each tick implements this so the
    /// manager can drive it without depending on the concrete provider type (keeps the manager
    /// free of AR Foundation / Sentis compile dependencies).
    /// </summary>
    public interface IDepthFrameUpdater
    {
        void UpdateDepth();
    }
}
