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
        /// <summary>Time.time of the last release, -999 = never released.</summary>
        public float ReleasedAt { get; private set; } = -999f;
        /// <summary>Current opacity (1 = opaque), driven by HandOcclusionFade.</summary>
        public float Alpha { get; private set; } = 1f;

        Rigidbody _rb;
        bool _wasKinematic;
        Color _baseColor;
        MeshRenderer _mr;
        bool _transparent;              // material currently in Standard "Fade" mode

        void Awake()
        {
            _rb = GetComponent<Rigidbody>();
            _mr = GetComponent<MeshRenderer>();
            if (_mr != null) _baseColor = _mr.material.color;
        }

        /// <summary>See-through when &lt; 1 (e.g. the real hand is in front of the object).
        /// Switches the Standard material between Opaque and Fade only when crossing 1, so
        /// fully visible objects keep normal depth sorting.</summary>
        public void SetAlpha(float a)
        {
            Alpha = Mathf.Clamp01(a);
            if (_mr == null) return;
            bool want = Alpha < 0.999f;
            if (want != _transparent) SetFadeMode(_mr.material, want);
            _transparent = want;
            ApplyColor();
        }

        void ApplyColor()
        {
            if (_mr == null) return;
            Color c = IsHeld ? heldTint : _baseColor;
            c.a = Alpha;
            _mr.material.color = c;
        }

        static void SetFadeMode(Material m, bool fade)
        {
            // Standard shader blend setup, as its inspector does for Opaque / Fade
            m.SetFloat("_Mode", fade ? 2f : 0f);
            m.SetInt("_SrcBlend", (int)(fade ? UnityEngine.Rendering.BlendMode.SrcAlpha
                                             : UnityEngine.Rendering.BlendMode.One));
            m.SetInt("_DstBlend", (int)(fade ? UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha
                                             : UnityEngine.Rendering.BlendMode.Zero));
            m.SetInt("_ZWrite", fade ? 0 : 1);
            m.DisableKeyword("_ALPHATEST_ON");
            m.DisableKeyword("_ALPHAPREMULTIPLY_ON");
            if (fade) m.EnableKeyword("_ALPHABLEND_ON"); else m.DisableKeyword("_ALPHABLEND_ON");
            m.renderQueue = fade ? (int)UnityEngine.Rendering.RenderQueue.Transparent : -1;
        }

        public void OnGrab()
        {
            IsHeld = true;
            if (_rb != null)
            {
                _wasKinematic = _rb.isKinematic;
                _rb.isKinematic = true;
            }
            ApplyColor();
        }

        public void OnRelease(Vector3 velocity, Vector3 angularVelocity)
        {
            IsHeld = false;
            ReleasedAt = Time.time;
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
            ApplyColor();
        }

        /// <summary>Teleport (no interpolation catch-up), e.g. to drop exactly where the
        /// pinch was last firm, or to return home.</summary>
        public void SnapTo(Vector3 pos, Quaternion rot)
        {
            if (_rb != null)
            {
                _rb.position = pos;
                _rb.rotation = rot;
                if (!_rb.isKinematic)
                {
#if UNITY_6000_0_OR_NEWER
                    _rb.linearVelocity = Vector3.zero;
#else
                    _rb.velocity = Vector3.zero;
#endif
                    _rb.angularVelocity = Vector3.zero;
                }
            }
            transform.SetPositionAndRotation(pos, rot);
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
