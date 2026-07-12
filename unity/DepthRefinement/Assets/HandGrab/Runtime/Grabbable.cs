using UnityEngine;

namespace HandMesh.HandGrab
{
    /// <summary>
    /// Marks an object as grabbable by the PinchGrabber. Needs a Collider;
    /// a Rigidbody is optional (made kinematic while held). On release the object
    /// either stays floating where the hand left it (<see cref="floatOnRelease"/>,
    /// default) or gets physics back with the hand's velocity so it can be tossed.
    /// </summary>
    [RequireComponent(typeof(Collider))]
    public class Grabbable : MonoBehaviour
    {
        [Tooltip("Tint applied while held (visual feedback)")]
        public Color heldTint = new Color(1f, 0.85f, 0.3f);

        [Tooltip("Keep the object hovering exactly where it is released instead of " +
                 "dropping (the rigidbody stays kinematic). Turn off to restore " +
                 "gravity + toss-on-release physics.")]
        public bool floatOnRelease = true;

        public bool IsHeld { get; private set; }

        Rigidbody _rb;
        bool _wasKinematic;
        Color _baseColor;
        MeshRenderer _mr;

        void Awake()
        {
            _rb = GetComponent<Rigidbody>();
            _mr = GetComponent<MeshRenderer>();
            if (_mr != null) _baseColor = _mr.material.color;
        }

        public void OnGrab()
        {
            IsHeld = true;
            if (_rb != null)
            {
                _wasKinematic = _rb.isKinematic;
                _rb.isKinematic = true;
            }
            if (_mr != null) _mr.material.color = heldTint;
        }

        public void OnRelease(Vector3 velocity, Vector3 angularVelocity)
        {
            IsHeld = false;
            if (_rb != null)
            {
                _rb.isKinematic = floatOnRelease || _wasKinematic;
                if (!_rb.isKinematic)
                {
#if UNITY_6000_0_OR_NEWER
                    _rb.linearVelocity = velocity;
#else
                    _rb.velocity = velocity;
#endif
                    _rb.angularVelocity = angularVelocity;
                }
            }
            if (_mr != null) _mr.material.color = _baseColor;
        }

        public void MoveTo(Vector3 pos, Quaternion rot)
        {
            if (_rb != null)
            {
                _rb.MovePosition(pos);
                _rb.MoveRotation(rot);
            }
            else
            {
                transform.SetPositionAndRotation(pos, rot);
            }
        }
    }
}
