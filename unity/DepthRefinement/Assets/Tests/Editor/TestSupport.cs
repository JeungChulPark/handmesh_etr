using UnityEngine;

namespace HandMesh.DepthRefinement.Tests
{
    /// <summary>
    /// In-memory IDepthProvider for tests: a fixed-size depth image filled with a constant
    /// (or per-pixel) metric depth and confidence, with identity UV->pixel mapping.
    /// </summary>
    public sealed class FakeDepthProvider : IDepthProvider
    {
        readonly float[] _depth;
        readonly float[] _conf;
        readonly CameraIntrinsics _k;

        public bool IsDepthAvailable { get; set; } = true;
        public int DepthWidth { get; }
        public int DepthHeight { get; }

        public FakeDepthProvider(int w, int h, float depthMeters, float confidence)
        {
            DepthWidth = w; DepthHeight = h;
            _depth = new float[w * h];
            _conf = new float[w * h];
            for (int i = 0; i < _depth.Length; i++) { _depth[i] = depthMeters; _conf[i] = confidence; }
            _k = new CameraIntrinsics { Fx = 500f, Fy = 500f, Cx = w * 0.5f, Cy = h * 0.5f, Width = w, Height = h };
        }

        public void SetPixel(int x, int y, float meters, float conf)
        {
            _depth[y * DepthWidth + x] = meters;
            _conf[y * DepthWidth + x] = conf;
        }

        public bool TryGetIntrinsics(out CameraIntrinsics intrinsics) { intrinsics = _k; return true; }

        public bool TryConvertUvToDepthPixel(Vector2 uv, out int px, out int py)
        {
            px = Mathf.RoundToInt(uv.x * DepthWidth);
            py = Mathf.RoundToInt(uv.y * DepthHeight);
            return px >= 0 && px < DepthWidth && py >= 0 && py < DepthHeight;
        }

        public bool TryGetDepthPixel(int px, int py, out float meters, out float confidence)
        {
            meters = 0; confidence = 0;
            if (px < 0 || px >= DepthWidth || py < 0 || py >= DepthHeight) return false;
            int idx = py * DepthWidth + px;
            meters = _depth[idx]; confidence = _conf[idx];
            return meters > 0f;
        }
    }
}
