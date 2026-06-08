using System;
using UnityEngine;

namespace HandMesh.DepthRefinement
{
    /// <summary>One robust metric depth reading at a joint's 2D location.</summary>
    public struct DepthSample
    {
        public float Meters;
        public float Confidence; // [0,1]: avg pixel confidence * valid-ratio
        public bool Valid;

        public static DepthSample Invalid => new DepthSample { Valid = false };
    }

    /// <summary>
    /// Step 4: sample a metric depth map at a joint's 2D position robustly.
    /// Reads a neighborhood window, masks low-confidence / out-of-range pixels,
    /// rejects flying-pixel outliers around silhouettes via a median + local-spread
    /// test. Single-pixel and bilinear sampling are intentionally avoided
    /// (bilinear across a depth discontinuity invents fictional depth).
    /// Reuses an internal buffer to stay GC-free in the frame loop.
    /// </summary>
    public sealed class JointDepthSampler
    {
        [Serializable]
        public struct Config
        {
            public int WindowRadius;          // 2 => 5x5
            public float MinPixelConfidence;  // drop pixels below this
            public float MinValidRatio;       // need this fraction of window valid
            public float MaxLocalSpreadMeters;// reject samples far from window median (finger thickness)

            public static Config Default => new Config
            {
                WindowRadius = 2,
                MinPixelConfidence = 0.5f,
                MinValidRatio = 0.3f,
                MaxLocalSpreadMeters = 0.02f
            };
        }

        Config _cfg = Config.Default;
        float[] _buffer = new float[25];

        public void SetConfig(Config cfg)
        {
            _cfg = cfg;
            if (_cfg.WindowRadius < 0) _cfg.WindowRadius = 0;
            int side = 2 * _cfg.WindowRadius + 1;
            int cap = side * side;
            if (_buffer.Length < cap) _buffer = new float[cap];
        }

        public DepthSample Sample(IDepthProvider depth, Vector2 uvNorm)
        {
            if (depth == null || !depth.IsDepthAvailable) return DepthSample.Invalid;
            if (!depth.TryConvertUvToDepthPixel(uvNorm, out int cx, out int cy)) return DepthSample.Invalid;

            int r = Mathf.Max(0, _cfg.WindowRadius);
            int n = 0;
            int inBounds = 0; // denominator = pixels actually inside the depth image
            float confSum = 0f;

            for (int dy = -r; dy <= r; dy++)
            {
                for (int dx = -r; dx <= r; dx++)
                {
                    int x = cx + dx, y = cy + dy;
                    if (x < 0 || x >= depth.DepthWidth || y < 0 || y >= depth.DepthHeight) continue;
                    inBounds++;
                    if (depth.TryGetDepthPixel(x, y, out float m, out float c) &&
                        c >= _cfg.MinPixelConfidence && m > 0f)
                    {
                        _buffer[n++] = m;
                        confSum += c;
                    }
                }
            }

            if (inBounds == 0 || n == 0 || (float)n / inBounds < _cfg.MinValidRatio)
                return DepthSample.Invalid;

            float median = Median(_buffer, n);

            // Reject flying pixels: keep only samples within finger-thickness of the median.
            int kept = 0;
            float keptSum = 0f;
            for (int i = 0; i < n; i++)
            {
                if (Mathf.Abs(_buffer[i] - median) <= _cfg.MaxLocalSpreadMeters)
                {
                    keptSum += _buffer[i];
                    kept++;
                }
            }
            if (kept == 0) return DepthSample.Invalid;

            float meters = keptSum / kept;
            float validRatio = (float)n / inBounds;
            float avgConf = confSum / n;
            float keptRatio = (float)kept / n;

            return new DepthSample
            {
                Meters = meters,
                Confidence = Mathf.Clamp01(avgConf * validRatio * keptRatio),
                Valid = true
            };
        }

        /// <summary>In-place partial sort to find the median of the first <paramref name="n"/> entries.</summary>
        static float Median(float[] a, int n)
        {
            // insertion sort is fine for tiny windows (<=49) and allocation-free
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
