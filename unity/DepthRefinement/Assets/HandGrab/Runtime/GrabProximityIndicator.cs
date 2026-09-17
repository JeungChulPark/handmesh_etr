using System.Collections.Generic;
using UnityEngine;

namespace HandMesh.HandGrab
{
    /// <summary>
    /// Proximity cue, split across two visuals (user-refined design):
    ///
    ///   * GUIDE LINE (distance measurement) — a solid NEUTRAL line from the pinch point
    ///     to the closest point on the nearest Grabbable, with a surface marker and a
    ///     "12.3 cm" label. Pure geometry; it no longer encodes distance in its colour
    ///     (that was hard to read against the video background).
    ///   * BOX (distance colour) — every Grabbable gets a solid wireframe box that
    ///     encodes the pinch distance: grey (far) → yellow (approaching) → GREEN
    ///     (within the PinchGrabber's grab radius = pinch now grabs it). Hidden while
    ///     that object is held.
    ///
    /// The box is 12 thin SOLID beams (scaled cube meshes) parented to the object, so it
    /// follows position/rotation/scale for free; only the colour is touched per frame.
    /// (A LineRenderer strip was tried first, but its camera-facing ribbon folds at 3D
    /// corners and never reads as clean solid edges.)
    /// </summary>
    [RequireComponent(typeof(HandStreamReceiver))]
    public class GrabProximityIndicator : MonoBehaviour
    {
        [Tooltip("Colour starts warming up (grey → yellow) inside this range (m)")]
        public float maxDistance = 1.5f;
        [Tooltip("Box edge width (m, world)")]
        public float lineWidth = 0.0025f;
        [Tooltip("Grow the box this much beyond the mesh so it doesn't touch the surface")]
        public float boxPadding = 1.08f;
        [Tooltip("Show the distance text label (cm) on the nearest grabbable")]
        public bool showLabel = true;
        [Tooltip("Sign the label by depth along the camera's view direction: + while the pinch " +
                 "is on the camera side of the object's centre, − once it has passed beyond it. " +
                 "A second line shows that depth offset itself.")]
        public bool signedByDepth = true;

        [Tooltip("Box colour when the object can be grabbed RIGHT NOW (pinch would take it)")]
        public Color grabbableColor = new Color(0.25f, 0.95f, 0.35f);
        [Tooltip("Box colour while approaching (blends from far colour)")]
        public Color nearColor = new Color(0.95f, 0.85f, 0.2f);
        [Tooltip("Box colour when far / hand not tracked (solid — boxes are always crisp)")]
        public Color farColor = new Color(0.7f, 0.7f, 0.7f);

        [Header("Guide line (distance measurement, neutral colour)")]
        [Tooltip("Draw the solid pinch→object guide line to the nearest grabbable")]
        public bool showGuideLine = true;
        [Tooltip("Guide line / marker colour — fixed; the BOX carries the distance colour")]
        public Color guideColor = Color.white;
        [Tooltip("Guide line width (m)")]
        public float guideWidth = 0.003f;

        sealed class BoxVis
        {
            public GameObject root;     // parented to the grabbable, holds the 12 beams
            public Material mat;        // per-box material instance — colour = proximity
        }

        HandStreamReceiver _recv;
        PinchGrabber _grabber;          // optional: supplies the "grabbable now" radius
        Material _mat;
        LineRenderer _guide;            // pinch → nearest-object solid line
        Transform _marker;              // surface point the line ends on
        readonly Dictionary<Grabbable, BoxVis> _boxes = new();
        readonly List<Grabbable> _stale = new();

        float _dist = -1f;              // nearest-object distance, <0 = no label this frame
        float _depth;                   // object-centre depth − pinch depth (m, camera forward)
        Vector3 _labelWorld;
        GUIStyle _style;

        const int ThumbTip = 4, IndexTip = 8;

        void Start()
        {
            _recv = GetComponent<HandStreamReceiver>();
            _grabber = GetComponent<PinchGrabber>();
            _mat = new Material(Shader.Find("Sprites/Default"));

            var go = new GameObject("proximity_line");
            go.transform.SetParent(transform, false);
            _guide = go.AddComponent<LineRenderer>();
            _guide.positionCount = 2;
            _guide.startWidth = _guide.endWidth = guideWidth;
            _guide.material = _mat;
            _guide.useWorldSpace = true;
            _guide.startColor = _guide.endColor = guideColor;

            var m = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            m.name = "proximity_marker";
            Destroy(m.GetComponent<Collider>());        // visual only — no physics
            m.transform.SetParent(transform, false);
            m.transform.localScale = Vector3.one * 0.012f;
            var mr = m.GetComponent<MeshRenderer>();
            mr.material = _mat;
            mr.material.color = guideColor;
            _marker = m.transform;
        }

        BoxVis MakeBox(Grabbable g)
        {
            // local mesh bounds (unit primitives: 0.5 extents) — the box inherits the
            // object's transform, so rotation/scale/motion need no per-frame work
            var mf = g.GetComponentInChildren<MeshFilter>();
            Bounds b = mf != null && mf.sharedMesh != null
                ? mf.sharedMesh.bounds
                : new Bounds(Vector3.zero, Vector3.one);
            Vector3 e = b.extents * boxPadding;
            Vector3 c = b.center;

            // beam thickness in the object's LOCAL units (world lineWidth / lossy scale)
            Vector3 s = g.transform.lossyScale;
            float t = lineWidth / Mathf.Max(1e-4f, (s.x + s.y + s.z) / 3f);

            // solid opaque unlit beams (built-in pipeline); Sprites/Default is kept for the
            // guide LINE only — it is a no-depth-write transparent shader, wrong for meshes
            Shader unlit = Shader.Find("Unlit/Color");
            var vis = new BoxVis
            {
                root = new GameObject("grab_box"),
                mat = unlit != null ? new Material(unlit) : new Material(_mat),
            };
            vis.root.transform.SetParent(g.transform, false);

            // 12 solid beams: 4 along each axis, on the box's edge lines
            for (int sy = -1; sy <= 1; sy += 2)
                for (int sz = -1; sz <= 1; sz += 2)
                    AddBeam(vis, c + new Vector3(0, sy * e.y, sz * e.z), new Vector3(2 * e.x + t, t, t));
            for (int sx = -1; sx <= 1; sx += 2)
                for (int sz = -1; sz <= 1; sz += 2)
                    AddBeam(vis, c + new Vector3(sx * e.x, 0, sz * e.z), new Vector3(t, 2 * e.y + t, t));
            for (int sx = -1; sx <= 1; sx += 2)
                for (int sy = -1; sy <= 1; sy += 2)
                    AddBeam(vis, c + new Vector3(sx * e.x, sy * e.y, 0), new Vector3(t, t, 2 * e.z + t));
            return vis;
        }

        void AddBeam(BoxVis vis, Vector3 localPos, Vector3 localScale)
        {
            var beam = GameObject.CreatePrimitive(PrimitiveType.Cube);
            beam.name = "edge";
            Destroy(beam.GetComponent<Collider>());     // visual only — must not block grabs
            beam.transform.SetParent(vis.root.transform, false);
            beam.transform.localPosition = localPos;
            beam.transform.localScale = localScale;
            var mr = beam.GetComponent<MeshRenderer>();
            mr.sharedMaterial = vis.mat;
            mr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            mr.receiveShadows = false;
        }

        static void SetBoxColor(BoxVis vis, Color c) => vis.mat.color = c;

        void LateUpdate()
        {
            _dist = -1f;
            bool tracked = _recv.IsTracked;
            Vector3 pinch = Vector3.zero;
            if (tracked)
                pinch = (_recv.Joints[ThumbTip] + _recv.Joints[IndexTip]) * 0.5f;
            float near = _grabber != null ? _grabber.grabRadius : 0.10f;

            Grabbable nearest = null;
            Vector3 nearestPoint = Vector3.zero;
            float nearestDist = float.MaxValue;
            bool holding = false;

            foreach (var g in FindObjectsOfType<Grabbable>())
            {
                if (!_boxes.TryGetValue(g, out BoxVis box) || box.root == null)
                    _boxes[g] = box = MakeBox(g);

                if (g.IsHeld) { holding = true; box.root.SetActive(false); continue; }
                box.root.SetActive(true);

                if (!tracked) { SetBoxColor(box, farColor); continue; }

                Vector3 p = g.GetComponent<Collider>().ClosestPoint(pinch);
                float d = Vector3.Distance(p, pinch);
                if (d < nearestDist) { nearestDist = d; nearestPoint = p; nearest = g; }

                // the BOX carries the distance colour: grey → yellow → green (grabbable)
                Color c = d <= near
                    ? grabbableColor
                    : Color.Lerp(nearColor, farColor, Mathf.InverseLerp(near, maxDistance, d));
                SetBoxColor(box, c);
            }

            // drop boxes whose grabbable was destroyed
            _stale.Clear();
            foreach (var kv in _boxes)
                if (kv.Key == null) _stale.Add(kv.Key);
            foreach (var k in _stale) _boxes.Remove(k);

            // the GUIDE LINE measures the distance: solid neutral pinch → surface segment
            bool guideOn = showGuideLine && tracked && !holding
                           && nearest != null && nearestDist <= maxDistance;
            if (guideOn)
            {
                _dist = nearestDist;
                Vector3 fwd = Camera.main != null ? Camera.main.transform.forward : Vector3.forward;
                _depth = Vector3.Dot(nearest.transform.position - pinch, fwd);
                _labelWorld = (pinch + nearestPoint) * 0.5f;
                _guide.SetPosition(0, pinch);
                _guide.SetPosition(1, nearestPoint);
                _marker.position = nearestPoint;
            }
            _guide.gameObject.SetActive(guideOn);
            _marker.gameObject.SetActive(guideOn);
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
            if (signedByDepth)
            {
                // + = hand still in front of the object (camera side), − = hand went past it
                char sign = _depth >= 0f ? '+' : '-';
                txt = $"{sign}{_dist * 100f:0.0} cm\ndepth {sign}{Mathf.Abs(_depth) * 100f:0.0} cm";
            }
            float sx = MirroredCameraView.ScreenX(sp.x);   // label follows the mirrored image
            var rect = new Rect(sx - 70, Screen.height - sp.y - 48, 140, signedByDepth ? 44 : 24);
            _style.normal.textColor = Color.black;      // cheap outline: shadow pass
            GUI.Label(new Rect(rect.x + 1, rect.y + 1, rect.width, rect.height), txt, _style);
            _style.normal.textColor = Color.white;
            GUI.Label(rect, txt, _style);
        }
    }
}
