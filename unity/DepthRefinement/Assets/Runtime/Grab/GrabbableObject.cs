using UnityEngine;

namespace HandMesh.DepthRefinement
{
    /// <summary>
    /// Marks an object as grabbable by <see cref="HandGrabController"/>. Needs a Collider;
    /// a Rigidbody is optional (made kinematic while held, hand velocity applied on release
    /// so objects can be tossed). Tint feedback while held works on Built-in and URP.
    /// </summary>
    [RequireComponent(typeof(Collider))]
    public sealed class GrabbableObject : MonoBehaviour
    {
        [Tooltip("Tint applied while held (visual feedback).")]
        public Color heldTint = new Color(1f, 0.85f, 0.3f);

        public bool IsHeld { get; private set; }

        Rigidbody _rb;
        Renderer _renderer;
        MaterialPropertyBlock _mpb;
        Color _baseColor = Color.white;
        bool _wasKinematic;

        void Awake()
        {
            _rb = GetComponent<Rigidbody>();
            _renderer = GetComponent<Renderer>();
            _mpb = new MaterialPropertyBlock();
            if (_renderer != null && _renderer.sharedMaterial != null)
                _baseColor = _renderer.sharedMaterial.HasProperty("_Color")
                    ? _renderer.sharedMaterial.color : Color.white;
        }

        public void OnGrab()
        {
            IsHeld = true;
            if (_rb != null)
            {
                _wasKinematic = _rb.isKinematic;
                _rb.isKinematic = true;
            }
            SetColor(heldTint);
        }

        public void OnRelease(Vector3 velocity, Vector3 angularVelocity)
        {
            IsHeld = false;
            if (_rb != null)
            {
                _rb.isKinematic = _wasKinematic;
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
            SetColor(_baseColor);
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

        void SetColor(Color c)
        {
            if (_renderer == null) return;
            _renderer.GetPropertyBlock(_mpb);
            _mpb.SetColor("_Color", c);
            if (_renderer.sharedMaterial != null && _renderer.sharedMaterial.HasProperty("_BaseColor"))
                _mpb.SetColor("_BaseColor", c);   // URP
            _renderer.SetPropertyBlock(_mpb);
        }
    }
}
