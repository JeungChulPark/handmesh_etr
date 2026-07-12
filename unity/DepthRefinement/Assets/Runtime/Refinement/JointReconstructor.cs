using UnityEngine;

namespace HandMesh.DepthRefinement
{
    /// <summary>
    /// Step 5: back-project a 2D joint + metric depth into a 3D camera-space point.
    /// Uses the classic pinhole model with DEPTH-space intrinsics. Output is in the
    /// computer-vision camera convention: +X right, +Y down, +Z forward (into scene).
    /// Convert to Unity world space by multiplying with the AR camera transform
    /// (and flipping Y) at the call site — kept out of here so the math stays testable.
    /// </summary>
    public static class JointReconstructor
    {
        /// <summary>Full-frame normalized UV [0,1] -> depth-image pixel coordinates.</summary>
        public static Vector2 UvToDepthPixel(Vector2 uvNorm, in CameraIntrinsics k)
            => new Vector2(uvNorm.x * k.Width, uvNorm.y * k.Height);

        /// <summary>(px, py in depth-image space) + metric depth (m) -> camera-space point (m).</summary>
        public static Vector3 Unproject(float px, float py, float depthMeters, in CameraIntrinsics k)
        {
            float x = (px - k.Cx) * depthMeters / k.Fx;
            float y = (py - k.Cy) * depthMeters / k.Fy;
            return new Vector3(x, y, depthMeters);
        }

        public static Vector3 Unproject(Vector2 pixel, float depthMeters, in CameraIntrinsics k)
            => Unproject(pixel.x, pixel.y, depthMeters, in k);
    }
}
