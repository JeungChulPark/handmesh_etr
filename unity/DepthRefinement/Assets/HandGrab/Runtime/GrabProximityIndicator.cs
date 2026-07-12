using UnityEngine;

namespace HandMesh.HandGrab
{
    /// <summary>
    /// Shows how far the hand is from the nearest grabbable — the monitor gives no
    /// depth cue, so without this it is hard to tell WHERE an object sits relative
    /// to the hand. Draws a guide line from the pinch point (thumb-index mid) to the
    /// closest point on the nearest Grabbable, a small marker on that surface point,
    /// and a "12.3 cm" label at the line's midpoint. The line fades red (far) →
    /// yellow → green (within the PinchGrabber's grab radius = pinch now grabs it).
    /// Hidden while a grab is in progress or nothing is in range.
    /// </summary>
    [RequireComponent(typeof(HandStreamReceiver))]
    public class GrabProximityIndicator : MonoBehaviour
    {
        [Tooltip("Show the guide when the nearest grabbable is within this range (m)")]
        public float maxDistance = 1.5f;
        [Tooltip("Guide line width (m)")]
        public float lineWidth = 0.003f;
        [Tooltip("Show the distance text label (cm)")]
        public bool showLabel = true;

        HandStreamReceiver _recv;
        PinchGrabber _grabber;          // optional: supplies the "grabbable now" radius
        LineRenderer _line;
        Transform _marker;
        float _dist = -1f;              // <0 = nothing to show this frame
        Vector3 _labelWorld;
        GUIStyle _style;

        const int ThumbTip = 4, IndexTip = 8;

        void Start()
        {
            _recv = GetComponent<HandStreamReceiver>();
            _grabber = GetComponent<PinchGrabber>();
            var mat = new Material(Shader.Find("Sprites/Default"));

            var go = new GameObject("proximity_line");
            go.transform.SetParent(transform, false);
            _line = go.AddComponent<LineRenderer>();
            _line.positionCount = 2;
            _line.startWidth = _line.endWidth = lineWidth;
            _line.material = mat;
            _line.useWorldSpace = true;

            var m = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            m.name = "proximity_marker";
            Destroy(m.GetComponent<Collider>());        // visual only — no physics
            m.transform.SetParent(transform, false);
            m.transform.localScale = Vector3.one * 0.012f;
            m.GetComponent<MeshRenderer>().material = mat;
            _marker = m.transform;
        }

        void LateUpdate()
        {
            _dist = -1f;

            bool holding = false;
            foreach (var g in FindObjectsOfType<Grabbable>())
                if (g.IsHeld) { holding = true; break; }

            if (_recv.IsTracked && !holding)
            {
                Vector3 pinch = (_recv.Joints[ThumbTip] + _recv.Joints[IndexTip]) * 0.5f;

                Grabbable best = null;
                Vector3 bestPoint = Vector3.zero;
                float bestDist = maxDistance;
                foreach (var g in FindObjectsOfType<Grabbable>())
                {
                    Vector3 p = g.GetComponent<Collider>().ClosestPoint(pinch);
                    float d = Vector3.Distance(p, pinch);
                    if (d < bestDist) { bestDist = d; bestPoint = p; best = g; }
                }

                if (best != null)
                {
                    _dist = bestDist;
                    _labelWorld = (pinch + bestPoint) * 0.5f;

                    float near = _grabber != null ? _grabber.grabRadius : 0.10f;
                    Color c = bestDist <= near
                        ? new Color(0.25f, 0.95f, 0.35f)                       // grabbable now
                        : Color.Lerp(new Color(0.95f, 0.85f, 0.2f),            // yellow near
                                     new Color(0.9f, 0.25f, 0.2f),             // red far
                                     Mathf.InverseLerp(near, maxDistance, bestDist));

                    _line.SetPosition(0, pinch);
                    _line.SetPosition(1, bestPoint);
                    _line.startColor = _line.endColor = c;
                    _marker.position = bestPoint;
                    _marker.GetComponent<MeshRenderer>().material.color = c;
                }
            }

            bool show = _dist >= 0f;
            _line.gameObject.SetActive(show);
            _marker.gameObject.SetActive(show);
        }

        void OnGUI()
        {
            if (!showLabel || _dist < 0f || Camera.main == null) return;
            Vector3 sp = Camera.main.WorldToScreenPoint(_labelWorld);
            if (sp.z <= 0f) return;                     // behind the camera

            if (_style == null)
                _style = new GUIStyle(GUI.skin.label)
                {
                    fontSize = 18,
                    fontStyle = FontStyle.Bold,
                    alignment = TextAnchor.MiddleCenter,
                };

            string txt = $"{_dist * 100f:0.0} cm";
            var rect = new Rect(sp.x - 60, Screen.height - sp.y - 32, 120, 24);
            _style.normal.textColor = Color.black;      // cheap outline: shadow pass
            GUI.Label(new Rect(rect.x + 1, rect.y + 1, rect.width, rect.height), txt, _style);
            _style.normal.textColor = Color.white;
            GUI.Label(rect, txt, _style);
        }
    }
}
