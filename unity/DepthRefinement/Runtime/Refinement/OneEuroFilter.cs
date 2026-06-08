using UnityEngine;

namespace HandMesh.DepthRefinement
{
    /// <summary>
    /// Scalar 1€ filter (Casiez et al., CHI'12). Speed-adaptive low-pass:
    /// smooths hard when slow (kills jitter), smooths little when fast (kills lag).
    /// </summary>
    public sealed class OneEuroFilter
    {
        public float MinCutoff = 1.0f;   // lower => more smoothing at rest
        public float Beta = 0.0f;        // higher => less lag on fast motion
        public float DCutoff = 1.0f;

        bool _init;
        float _xPrev;
        float _dxPrev;
        double _tPrev;

        static float Alpha(float cutoff, float dt)
        {
            float tau = 1f / (2f * Mathf.PI * cutoff);
            return 1f / (1f + tau / dt);
        }

        public float Filter(float x, double t)
        {
            if (!_init)
            {
                _init = true;
                _xPrev = x;
                _dxPrev = 0f;
                _tPrev = t;
                return x;
            }

            float dt = (float)(t - _tPrev);
            if (dt <= 0f) dt = 1f / 60f;       // stalled / duplicate / backward timestamp
            else if (dt > 0.1f) dt = 0.1f;     // cap huge gaps (pause/resume) so the step stays sane
            _tPrev = t;

            float dx = (x - _xPrev) / dt;
            float aD = Alpha(DCutoff, dt);
            float dxHat = Mathf.Lerp(_dxPrev, dx, aD);

            float cutoff = MinCutoff + Beta * Mathf.Abs(dxHat);
            float aX = Alpha(cutoff, dt);
            float xHat = Mathf.Lerp(_xPrev, x, aX);

            _xPrev = xHat;
            _dxPrev = dxHat;
            return xHat;
        }

        public void Reset() => _init = false;
    }

    /// <summary>
    /// Vector3 1€ filter. The Z (depth) axis can use its own, stronger settings
    /// because depth is the noisiest channel after fusion (PRD §FR-6).
    /// </summary>
    public sealed class OneEuroFilterVector3
    {
        readonly OneEuroFilter _x = new OneEuroFilter();
        readonly OneEuroFilter _y = new OneEuroFilter();
        readonly OneEuroFilter _z = new OneEuroFilter();

        public void Configure(float minCutoff, float beta, float dCutoff, float zMinCutoff, float zBeta)
        {
            _x.MinCutoff = _y.MinCutoff = minCutoff;
            _x.Beta = _y.Beta = beta;
            _x.DCutoff = _y.DCutoff = dCutoff;
            _z.MinCutoff = zMinCutoff;
            _z.Beta = zBeta;
            _z.DCutoff = dCutoff;
        }

        public Vector3 Filter(Vector3 v, double t)
            => new Vector3(_x.Filter(v.x, t), _y.Filter(v.y, t), _z.Filter(v.z, t));

        public void Reset()
        {
            _x.Reset();
            _y.Reset();
            _z.Reset();
        }
    }
}
