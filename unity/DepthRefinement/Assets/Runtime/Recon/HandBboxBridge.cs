#if SENTIS_PRESENT && ARFOUNDATION_PRESENT
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;

namespace HandMesh.DepthRefinement
{
    /// <summary>
    /// Bridges 2D hand landmarks (e.g. from the MediaPipe Unity Plugin, Apple Vision, or any
    /// detector) to the dual-stream model's hand crop box. Replaces
    /// <see cref="DualStreamHandProvider"/>'s default centred box with a real, tracked one.
    ///
    /// Computes the SAME box as the Python pipeline's <c>landmarks_to_bbox</c>: a SQUARE box
    /// (longer landmark extent) expanded by <see cref="_marginFrac"/> (0.50 = doubled, the
    /// value that matches the model's training crop ratio), centred on the hand, then assigns
    /// <see cref="DualStreamHandProvider.HandBboxNormalized"/> (sensor-image normalized).
    ///
    /// Call <see cref="SetLandmarksNormalized"/> from your detector's per-frame callback.
    /// Landmarks come in the DETECTOR image frame (usually the upright/portrait preview); the
    /// rotation/mirror fields map them into the camera SENSOR frame the provider crops in.
    /// </summary>
    public sealed class HandBboxBridge : MonoBehaviour
    {
        public enum Rotation { None = 0, CW90 = 90, Rot180 = 180, CW270 = 270 }

        [SerializeField] DualStreamHandProvider _provider;
        [Tooltip("For the sensor resolution (a square-in-pixels box needs the image aspect).")]
        [SerializeField] ARCameraManager _cameraManager;

        [Range(0f, 1f)]
        [Tooltip("landmarks_to_bbox margin (0.50 matches training).")]
        [SerializeField] float _marginFrac = 0.5f;

        [Header("Detector image -> camera sensor frame")]
        [Tooltip("Rotation applied to landmarks so they land in the SENSOR frame. None = the " +
                 "detector runs on the native landscape image. Try others if the crop is off.")]
        [SerializeField] Rotation _rotation = Rotation.None;
        [SerializeField] bool _mirrorX = false;
        [SerializeField] bool _mirrorY = false;

        /// <summary>True when a hand box was set this/last detection.</summary>
        public bool HasBox { get; private set; }
        /// <summary>The margin-expanded hand box in the DETECTOR display frame (normalized,
        /// origin top-left) — for a debug overlay over the upright RGB. (The provider gets the
        /// sensor-frame box.)</summary>
        public Rect BoxDisplayNormalized { get; private set; }

        readonly List<Vector2> _lmSensor = new List<Vector2>(21);
        /// <summary>The latest landmarks in SENSOR-normalized coords (same frame as the crop/depth),
        /// in the order supplied. Used by <see cref="HybridBHandProvider"/> (model B locks 2D to these);
        /// only complete 21-point hands are consumed there, so partial detections are safe.</summary>
        public IReadOnlyList<Vector2> LandmarksNormalized => _lmSensor;

        void Awake()
        {
            if (_provider == null) _provider = FindFirstObjectByType<DualStreamHandProvider>();
            if (_cameraManager == null) _cameraManager = FindFirstObjectByType<ARCameraManager>();
        }

        /// <summary>
        /// Feed the latest hand landmarks (normalized [0,1] in the detector image, origin
        /// top-left). N can be 21 (full hand) or any subset; only the extent is used.
        /// Call once per detection from your MediaPipe / Vision callback.
        /// </summary>
        public void SetLandmarksNormalized(IReadOnlyList<Vector2> pts)
        {
            if (_provider == null || pts == null || pts.Count == 0) return;

            // Sensor resolution — a square box must be square in PIXELS, so aspect matters.
            int w = 1920, h = 1440;
            if (_cameraManager != null && _cameraManager.TryGetIntrinsics(out XRCameraIntrinsics intr)
                && intr.resolution.x > 0 && intr.resolution.y > 0)
            { w = intr.resolution.x; h = intr.resolution.y; }

            float minX = 1f, minY = 1f, maxX = 0f, maxY = 0f;   // sensor frame
            float dMinX = 1f, dMinY = 1f, dMaxX = 0f, dMaxY = 0f; // detector/display frame
            _lmSensor.Clear();
            for (int i = 0; i < pts.Count; i++)
            {
                Vector2 d = pts[i];
                if (d.x < dMinX) dMinX = d.x;
                if (d.x > dMaxX) dMaxX = d.x;
                if (d.y < dMinY) dMinY = d.y;
                if (d.y > dMaxY) dMaxY = d.y;

                Vector2 s = ToSensor(d);
                _lmSensor.Add(s);                    // retained for HybridBHandProvider (2D-locked)
                if (s.x < minX) minX = s.x;
                if (s.x > maxX) maxX = s.x;
                if (s.y < minY) minY = s.y;
                if (s.y > maxY) maxY = s.y;
            }

            // Debug box in display space (square-ish in normalized units + same margin).
            float dside = Mathf.Max(dMaxX - dMinX, dMaxY - dMinY);
            dside += 2f * _marginFrac * dside;
            float dcx = 0.5f * (dMinX + dMaxX), dcy = 0.5f * (dMinY + dMaxY);
            BoxDisplayNormalized = ClampRect01(new Rect(dcx - dside * 0.5f, dcy - dside * 0.5f, dside, dside));
            HasBox = true;

            // landmarks_to_bbox in pixels: square (longer extent) + margin on each edge.
            float pxMinX = minX * w, pxMaxX = maxX * w, pxMinY = minY * h, pxMaxY = maxY * h;
            float side = Mathf.Max(pxMaxX - pxMinX, pxMaxY - pxMinY);
            side += 2f * _marginFrac * side;                 // 0.50 -> doubled
            float cx = 0.5f * (pxMinX + pxMaxX), cy = 0.5f * (pxMinY + pxMaxY);
            float x0 = cx - side * 0.5f, y0 = cy - side * 0.5f;

            // back to normalized + clamp to a valid in-image rect
            _provider.HandBboxNormalized = ClampRect01(new Rect(x0 / w, y0 / h, side / w, side / h));
        }

        // detector-normalized (u,v) -> sensor-normalized, with mirror then rotation.
        Vector2 ToSensor(Vector2 p)
        {
            float u = p.x, v = p.y;
            if (_mirrorX) u = 1f - u;
            if (_mirrorY) v = 1f - v;
            switch (_rotation)
            {
                case Rotation.CW90:   return new Vector2(1f - v, u);
                case Rotation.Rot180: return new Vector2(1f - u, 1f - v);
                case Rotation.CW270:  return new Vector2(v, 1f - u);
                default:              return new Vector2(u, v);
            }
        }

        static Rect ClampRect01(Rect r)
        {
            float x = Mathf.Clamp01(r.x), y = Mathf.Clamp01(r.y);
            float wd = Mathf.Clamp(r.width, 0.02f, 1f - x);
            float ht = Mathf.Clamp(r.height, 0.02f, 1f - y);
            return new Rect(x, y, wd, ht);
        }
    }
}
#endif
