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
        public float releaseDistance = 0.06f;
        [Tooltip("Max distance from pinch point to a Grabbable's surface (m)")]
        public float grabRadius = 0.10f;

        [Header("Release robustness (joint noise must not drop the object)")]
        [Tooltip("The OPEN pose must persist this long before the object is released (s). "
               + "A single noisy frame over releaseDistance no longer drops the grab.")]
        public float releaseHoldSec = 0.12f;
        [Tooltip("Also require the index finger to be clearly EXTENDED (hand really open), "
               + "not just a thumb-index distance spike. Scale-invariant: "
               + "dist(wrist,indexTip) > openExtendRatio * dist(wrist,indexMcp).")]
        public bool requireOpenHand = true;
        [Tooltip("Index-extension ratio counting as 'open'. Pinch/curl ≈ 1.0-1.2, open ≈ 1.6+. "
               + "Lower = easier release; raise if noise drops the object again.")]
        public float openExtendRatio = 1.2f;
        [Tooltip("The held object follows the hand only while the pinch is this tight (m). As " +
                 "soon as the fingers start to open it stays put, and on release it is left " +
                 "exactly where the pinch was last firm — opening the hand no longer drags it.")]
        public float firmPinchDistance = 0.045f;

        HandStreamReceiver _recv;
        Grabbable _held;
        Quaternion _rotOffset;          // held-object rotation in the hand frame
        Vector3 _posOffset;             // held-object position in the hand frame
        bool _pinching;
        float _pinchDistSm = -1f;       // EMA of the thumb-index distance (spike filter)
        float _openSince = -1f;         // Time.time when the open pose began, -1 = not open
        Vector3 _firmPos;               // held-object pose at the last firm-pinch frame
        Quaternion _firmRot;
        bool _following;                // held object currently tracks the hand

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
                // tracking lost (0.3 s timeout upstream) → gentle drop, no throw:
                // the last velocity estimate is noise from the dying track, not a gesture.
                if (_held != null)
                {
                    _held.SnapTo(_firmPos, _firmRot);
                    _held.OnRelease(Vector3.zero, Vector3.zero);
                    _held = null;
                }
                _pinching = false;
                _pinchDistSm = -1f;
                _openSince = -1f;
                return;
            }

            var j = _recv.Joints;
            Vector3 pinch = (j[ThumbTip] + j[IndexTip]) * 0.5f;
            float pinchDist = Vector3.Distance(j[ThumbTip], j[IndexTip]);
            _pinchDistSm = _pinchDistSm < 0f ? pinchDist : Mathf.Lerp(_pinchDistSm, pinchDist, 0.5f);
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

            if (!_pinching && _pinchDistSm < grabDistance)
            {
                _pinching = true;
                _openSince = -1f;
                TryGrab(pinch, handRot);
            }
            else if (_pinching)
            {
                // release ONLY when the hand is clearly open AND stays open: smoothed distance
                // past the hysteresis threshold, index extended, held for releaseHoldSec.
                if (!IsHandOpen(j)) _openSince = -1f;
                else if (_openSince < 0f) _openSince = Time.time;
                else if (Time.time - _openSince >= releaseHoldSec)
                {
                    _pinching = false;
                    _openSince = -1f;
                    if (_held != null) Release();
                }
            }

            if (_held != null)
            {
                if (_pinchDistSm <= firmPinchDistance)
                {
                    if (!_following)
                    {
                        // fingers closed again without releasing: re-anchor on the frozen
                        // pose so the object doesn't jump to where the hand moved meanwhile
                        var inv = Quaternion.Inverse(handRot);
                        _rotOffset = inv * _firmRot;
                        _posOffset = inv * (_firmPos - pinch);
                        _following = true;
                    }
                    _firmPos = pinch + handRot * _posOffset;
                    _firmRot = handRot * _rotOffset;
                    _held.MoveTo(_firmPos, _firmRot);
                }
                else
                {
                    _following = false;   // opening: hold still until release is confirmed
                }
            }
        }

        bool IsHandOpen(Vector3[] j)
        {
            if (_pinchDistSm <= releaseDistance) return false;
            if (!requireOpenHand) return true;
            float mcp = Vector3.Distance(j[Wrist], j[IndexMcp]);
            float tip = Vector3.Distance(j[Wrist], j[IndexTip]);
            return mcp > 1e-5f && tip > openExtendRatio * mcp;
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
            _firmPos = best.transform.position;
            _firmRot = best.transform.rotation;
            _following = true;
            best.OnGrab();
        }

        void Release()
        {
            _held.SnapTo(_firmPos, _firmRot);   // no drift from the opening fingers / interpolation
            _held.OnRelease(_velocity, _angVelocity);
            _held = null;
        }
    }
}
