#if SENTIS_PRESENT && ARFOUNDATION_PRESENT
using UnityEngine;

namespace HandMesh.DepthRefinement
{
    /// <summary>
    /// Minimal proof-of-concept: pinch-to-move a virtual object with the estimated hand joints.
    /// Reads two joint spheres (thumb tip + index tip, already placed each frame by
    /// <see cref="HandSphereDriver"/>) and, when they close into a pinch, makes a target object
    /// follow the pinch point; releasing the pinch drops it.
    ///
    /// Setup (3 drags in the Inspector):
    ///   * Thumb Tip   = Canvas_sphere/thumb4
    ///   * Index Tip   = Canvas_sphere/index4
    ///   * Target      = the object to move (e.g. a small Cube)
    /// Optionally assign the provider to only act while a hand pose is valid this frame.
    /// </summary>
    public sealed class SimpleHandGrab : MonoBehaviour
    {
        [Header("Hand joints (from HandSphereDriver spheres)")]
        [SerializeField] Transform _thumbTip;     // Canvas_sphere/thumb4
        [SerializeField] Transform _indexTip;     // Canvas_sphere/index4

        [Header("Object to manipulate")]
        [SerializeField] Transform _target;
        [Tooltip("Optional: only grab while this provider reports a valid pose this frame.")]
        [SerializeField] HybridBHandProvider _provider;
        [Tooltip("Optional: SERVER-mode pose gate (joints inferred on the PC). Used while it " +
                 "is enabled — HandPoseModeController flips the enabled states.")]
        [SerializeField] ServerHandProvider _serverProvider;

        [Header("Pinch thresholds (metres)")]
        [Tooltip("Pinch is CLOSED (grab) when tip distance < this. ~3.5 cm.")]
        [SerializeField] float _grabDist = 0.035f;
        [Tooltip("Pinch is OPEN (release) when tip distance > this. Hysteresis avoids flicker.")]
        [SerializeField] float _releaseDist = 0.06f;
        [Tooltip("Smoothing for the object's follow motion (0 = snap, 0.9 = very smooth).")]
        [Range(0f, 0.95f)] [SerializeField] float _smoothing = 0.5f;

        [Header("Feedback (optional)")]
        [SerializeField] Color _idleColor = Color.white;
        [SerializeField] Color _grabColor = new Color(0.2f, 1f, 0.3f);

        bool _grabbed;
        Renderer _targetRenderer;
        MaterialPropertyBlock _mpb;

        void Start()
        {
            if (_target != null) _targetRenderer = _target.GetComponent<Renderer>();
            _mpb = new MaterialPropertyBlock();
            SetColor(_idleColor);
        }

        void Update()
        {
            if (_thumbTip == null || _indexTip == null || _target == null) return;
            // no hand this frame -> don't act (gate on whichever pose source is enabled)
            if (_serverProvider != null && _serverProvider.isActiveAndEnabled)
            { if (!_serverProvider.HasPose) return; }
            else if (_provider != null && _provider.isActiveAndEnabled && !_provider.HasPose) return;

            float d = Vector3.Distance(_thumbTip.position, _indexTip.position);
            Vector3 pinch = 0.5f * (_thumbTip.position + _indexTip.position);

            if (!_grabbed && d < _grabDist) { _grabbed = true; SetColor(_grabColor); }
            else if (_grabbed && d > _releaseDist) { _grabbed = false; SetColor(_idleColor); }

            if (_grabbed)
            {
                // move the object to the pinch point (smoothed)
                _target.position = Vector3.Lerp(pinch, _target.position, _smoothing);
            }
        }

        void SetColor(Color c)
        {
            if (_targetRenderer == null) return;
            _targetRenderer.GetPropertyBlock(_mpb);
            _mpb.SetColor("_Color", c);
            if (_targetRenderer.sharedMaterial != null && _targetRenderer.sharedMaterial.HasProperty("_BaseColor"))
                _mpb.SetColor("_BaseColor", c);   // URP
            _targetRenderer.SetPropertyBlock(_mpb);
        }
    }
}
#endif
