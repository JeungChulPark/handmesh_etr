using UnityEngine;

namespace HandMesh.HandGrab
{
    /// <summary>
    /// Makes a grabbable see-through while the tracked hand is IN FRONT of it (closer to the
    /// camera) and overlaps it on screen. Virtual objects are always drawn over the video,
    /// so without this the object hid the real hand reaching in front of it and depth was
    /// hard to judge (recording 140407, 28.8 s / 33.9 s). The wireframe proximity box is
    /// a separate unlit mesh and stays solid, so the object's outline remains visible.
    /// </summary>
    [RequireComponent(typeof(HandStreamReceiver))]
    public class HandOcclusionFade : MonoBehaviour
    {
        [Tooltip("Opacity while the hand is in front of the object")]
        [Range(0f, 1f)] public float fadedAlpha = 0.35f;
        [Tooltip("The nearest hand joint must be at least this much closer than the object " +
                 "centre (m) to count as in front")]
        public float depthMargin = 0.01f;
        [Tooltip("Screen-space padding (px) around the hand when testing overlap")]
        public float screenPadding = 12f;
        [Tooltip("Opacity change per second (fade in / out speed)")]
        public float fadeSpeed = 4f;

        HandStreamReceiver _recv;
        static readonly Vector3[] Corners = new Vector3[8];

        void Start() => _recv = GetComponent<HandStreamReceiver>();

        void LateUpdate()
        {
            var cam = Camera.main;
            bool tracked = _recv.IsTracked && cam != null;
            Rect handRect = default;
            float handDepth = float.MaxValue;
            if (tracked)
            {
                Vector3 camPos = cam.transform.position, fwd = cam.transform.forward;
                float xMin = float.MaxValue, yMin = float.MaxValue, xMax = float.MinValue, yMax = float.MinValue;
                foreach (Vector3 j in _recv.Joints)
                {
                    handDepth = Mathf.Min(handDepth, Vector3.Dot(j - camPos, fwd));
                    Vector3 sp = cam.WorldToScreenPoint(j);
                    if (sp.z <= 0f) continue;
                    xMin = Mathf.Min(xMin, sp.x); xMax = Mathf.Max(xMax, sp.x);
                    yMin = Mathf.Min(yMin, sp.y); yMax = Mathf.Max(yMax, sp.y);
                }
                tracked = xMin <= xMax;
                handRect = Rect.MinMaxRect(xMin - screenPadding, yMin - screenPadding,
                                           xMax + screenPadding, yMax + screenPadding);
            }

            float step = fadeSpeed * Time.deltaTime;
            foreach (var g in FindObjectsOfType<Grabbable>())
            {
                float target = 1f;
                var mr = g.GetComponent<MeshRenderer>();
                if (tracked && mr != null)
                {
                    Bounds b = mr.bounds;
                    float objDepth = Vector3.Dot(b.center - cam.transform.position, cam.transform.forward);
                    if (handDepth < objDepth - depthMargin && handRect.Overlaps(ScreenRect(cam, b)))
                        target = fadedAlpha;
                }
                if (!Mathf.Approximately(g.Alpha, target))
                    g.SetAlpha(Mathf.MoveTowards(g.Alpha, target, step));
            }
        }

        static Rect ScreenRect(Camera cam, Bounds b)
        {
            Vector3 c = b.center, e = b.extents;
            int k = 0;
            for (int sx = -1; sx <= 1; sx += 2)
                for (int sy = -1; sy <= 1; sy += 2)
                    for (int sz = -1; sz <= 1; sz += 2)
                        Corners[k++] = c + Vector3.Scale(e, new Vector3(sx, sy, sz));
            float xMin = float.MaxValue, yMin = float.MaxValue, xMax = float.MinValue, yMax = float.MinValue;
            foreach (Vector3 w in Corners)
            {
                Vector3 sp = cam.WorldToScreenPoint(w);
                if (sp.z <= 0f) continue;
                xMin = Mathf.Min(xMin, sp.x); xMax = Mathf.Max(xMax, sp.x);
                yMin = Mathf.Min(yMin, sp.y); yMax = Mathf.Max(yMax, sp.y);
            }
            return xMin <= xMax ? Rect.MinMaxRect(xMin, yMin, xMax, yMax) : new Rect(-1e6f, -1e6f, 0f, 0f);
        }
    }
}
