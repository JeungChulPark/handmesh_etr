using UnityEngine;

namespace HandMesh.HandGrab
{
    /// <summary>
    /// Pinch-to-grab: when thumb tip (4) and index tip (8) close below
    /// grabDistance, the nearest Grabbable within grabRadius of the pinch point
    /// is held; it then follows the pinch point AND the hand's rotation (wrist +
    /// index/pinky MCPs define the hand frame), so twisting the hand rotates the
    /// object. Release (with hysteresis) restores physics and applies the hand's
    /// velocity so objects can be tossed.
    /// </summary>
    [RequireComponent(typeof(HandStreamReceiver))]
    public class PinchGrabber : MonoBehaviour
    {
        const int Wrist = 0, ThumbTip = 4, IndexTip = 8, IndexMcp = 5, PinkyMcp = 17;

        [Tooltip("Pinch closes below this thumb-index distance (m)")]
        public float grabDistance = 0.035f;
        [Tooltip("Pinch opens above this distance (m) — hysteresis")]
        public float releaseDistance = 0.055f;
        [Tooltip("Max distance from pinch point to a Grabbable's surface (m)")]
        public float grabRadius = 0.10f;

        HandStreamReceiver _recv;
        Grabbable _held;
        Quaternion _rotOffset;          // held-object rotation in the hand frame
        Vector3 _posOffset;             // held-object position in the hand frame
        bool _pinching;

        // velocity estimate for throw-on-release
        Vector3 _prevPinch;
        Quaternion _prevHandRot;
        Vector3 _velocity, _angVelocity;

        void Start() => _recv = GetComponent<HandStreamReceiver>();

        static Quaternion HandRotation(Vector3[] j)
        {
            // Palm frame: forward = wrist -> knuckle mid, up = palm normal.
            Vector3 fwd = ((j[IndexMcp] + j[PinkyMcp]) * 0.5f - j[Wrist]).normalized;
            Vector3 tangent = (j[IndexMcp] - j[PinkyMcp]).normalized;
            Vector3 up = Vector3.Cross(fwd, tangent).normalized;
            if (fwd.sqrMagnitude < 1e-8f || up.sqrMagnitude < 1e-8f)
                return Quaternion.identity;
            return Quaternion.LookRotation(fwd, up);
        }

        void Update()
        {
            if (!_recv.IsTracked)
            {
                if (_held != null) Release();
                _pinching = false;
                return;
            }

            var j = _recv.Joints;
            Vector3 pinch = (j[ThumbTip] + j[IndexTip]) * 0.5f;
            float pinchDist = Vector3.Distance(j[ThumbTip], j[IndexTip]);
            Quaternion handRot = HandRotation(j);

            float dt = Mathf.Max(Time.deltaTime, 1e-4f);
            if (_pinching)
            {
                _velocity = Vector3.Lerp(_velocity, (pinch - _prevPinch) / dt, 0.5f);
                var dq = handRot * Quaternion.Inverse(_prevHandRot);
                dq.ToAngleAxis(out float ang, out Vector3 axis);
                if (ang > 180f) ang -= 360f;
                _angVelocity = Vector3.Lerp(_angVelocity, axis * (ang * Mathf.Deg2Rad / dt), 0.5f);
            }
            _prevPinch = pinch;
            _prevHandRot = handRot;

            if (!_pinching && pinchDist < grabDistance)
            {
                _pinching = true;
                TryGrab(pinch, handRot);
            }
            else if (_pinching && pinchDist > releaseDistance)
            {
                _pinching = false;
                if (_held != null) Release();
            }

            if (_held != null)
                _held.MoveTo(pinch + handRot * _posOffset, handRot * _rotOffset);
        }

        void TryGrab(Vector3 pinch, Quaternion handRot)
        {
            Grabbable best = null;
            float bestDist = grabRadius;
            foreach (var g in FindObjectsOfType<Grabbable>())
            {
                if (g.IsHeld) continue;
                float d = Vector3.Distance(g.GetComponent<Collider>().ClosestPoint(pinch), pinch);
                if (d < bestDist) { bestDist = d; best = g; }
            }
            if (best == null) return;

            _held = best;
            var inv = Quaternion.Inverse(handRot);
            _rotOffset = inv * best.transform.rotation;
            _posOffset = inv * (best.transform.position - pinch);
            _velocity = _angVelocity = Vector3.zero;
            best.OnGrab();
        }

        void Release()
        {
            _held.OnRelease(_velocity, _angVelocity);
            _held = null;
        }
    }
}
