#if SENTIS_PRESENT && ARFOUNDATION_PRESENT
using UnityEngine;

namespace HandMesh.DepthRefinement
{
    /// <summary>
    /// Pinch-to-grab virtual objects with the live hand pose (iPhone/iPad, LiDAR).
    ///
    /// Reads absolute camera-space joints from <see cref="ServerHandProvider"/> (server-inference
    /// mode), <see cref="HybridBHandProvider"/> (on-device deploy model) or
    /// <see cref="DualStreamHandProvider"/> — whichever is enabled — converts them to world space
    /// exactly like <see cref="HandSphereDriver"/> (CV frame → Unity: flip Y, then the AR
    /// camera transform). When thumb tip (4) and index tip (8) close below the grab
    /// threshold, the nearest <see cref="GrabbableObject"/> within reach is held; it follows
    /// the pinch point AND the palm-frame rotation (wrist + index/little MCPs), so twisting
    /// the hand rotates the object. Release (with hysteresis) restores physics and applies
    /// the hand velocity, so objects can be tossed.
    ///
    /// Driven by the provider's <c>OnPose</c> event (fires in its LateUpdate), so no script
    /// execution order setup is needed. Everything auto-finds itself: drop this on any
    /// GameObject in the RGB-D viewer scene and press Play — with "Spawn Demo Objects" on,
    /// a cube/sphere/capsule float in front of the camera once the first pose arrives.
    /// </summary>
    public sealed class HandGrabController : MonoBehaviour
    {
        // model joint order (HandSphereDriver.JointNames / MediaPipe)
        const int Wrist = 0, ThumbTip = 4, IndexMcp = 5, IndexTip = 8, LittleMcp = 17;

        [Header("Pose source (auto-found when empty; server > hybrid B > dual-stream)")]
        [SerializeField] HybridBHandProvider _hybrid;
        [SerializeField] DualStreamHandProvider _dual;
        [Tooltip("SERVER-mode source (joints inferred on the PC by infer_ipad_stream.py and " +
                 "streamed back). Takes priority while enabled — HandPoseModeController flips " +
                 "the enabled states when switching modes.")]
        [SerializeField] ServerHandProvider _server;
        [Tooltip("AR Main Camera — joints are in ITS local frame. Defaults to Camera.main.")]
        [SerializeField] Transform _cameraTransform;

        [Header("Pinch thresholds (metres)")]
        [Tooltip("Pinch CLOSES (grab) when thumb–index tip distance < this. ~3.5 cm.")]
        [SerializeField] float _grabDist = 0.035f;
        [Tooltip("Pinch OPENS (release) when tip distance > this — hysteresis avoids flicker.")]
        [SerializeField] float _releaseDist = 0.055f;
        [Tooltip("Max distance from pinch point to a grabbable's surface (m).")]
        [SerializeField] float _grabRadius = 0.10f;

        [Header("Smoothing / robustness")]
        [Tooltip("Exponential smoothing of the pinch pose (0 = raw, 0.9 = very smooth).")]
        [Range(0f, 0.95f)] [SerializeField] float _smoothing = 0.4f;
        [Tooltip("Drop the held object (no throw) when no pose arrives for this long (s).")]
        [SerializeField] float _lostPoseReleaseSec = 0.4f;

        [Header("Demo objects")]
        [Tooltip("Spawn a cube/sphere/capsule in front of the camera on the first valid pose.")]
        [SerializeField] bool _spawnDemoObjects = true;
        [Tooltip("Spawn distance in front of the AR camera (m).")]
        [SerializeField] float _spawnDistance = 0.4f;
        [Tooltip("Object size (m). 0.06 = 6 cm.")]
        [SerializeField] float _objectSize = 0.06f;
        [Tooltip("Teleport a demo object back in front of the camera when it drifts this far (m).")]
        [SerializeField] float _respawnDistance = 2.5f;

        // pinch state
        bool _pinching;
        GrabbableObject _held;
        Vector3 _posOffset;             // held-object position in the hand frame
        Quaternion _rotOffset;          // held-object rotation in the hand frame

        // smoothed hand pose + velocity estimate (for throw-on-release)
        Vector3 _pinch, _prevPinch, _velocity, _angVelocity;
        Quaternion _handRot = Quaternion.identity, _prevHandRot = Quaternion.identity;
        bool _smoothInit;
        float _lastPoseTime = -1f;

        int _lastPoseFrame = -1;        // guard: several providers may fire in one frame
        GrabbableObject[] _demo;

        void Start()
        {
            if (_hybrid == null) _hybrid = FindFirstObjectByType<HybridBHandProvider>();
            if (_dual == null) _dual = FindFirstObjectByType<DualStreamHandProvider>();
            if (_server == null) _server = FindFirstObjectByType<ServerHandProvider>();
            if (_cameraTransform == null && Camera.main != null) _cameraTransform = Camera.main.transform;

            // Subscribe to every provider present; ActiveJoints() picks the enabled one per
            // frame, so mode switches (HandPoseModeController) need no re-subscribing.
            if (_hybrid != null) _hybrid.OnPose += HandlePose;
            if (_dual != null) _dual.OnPose += HandlePose;
            if (_server != null) _server.OnPose += HandlePose;
            if (_hybrid == null && _dual == null && _server == null)
                Debug.LogWarning("[HandGrab] no hand pose provider found in the scene.");
        }

        void OnDestroy()
        {
            if (_hybrid != null) _hybrid.OnPose -= HandlePose;
            if (_dual != null) _dual.OnPose -= HandlePose;
            if (_server != null) _server.OnPose -= HandlePose;
        }

        // Source priority: server (while enabled) > hybrid B > dual-stream.
        Vector3[] ActiveJoints()
        {
            if (_server != null && _server.isActiveAndEnabled && _server.HasPose) return _server.AbsJoints;
            if (_hybrid != null && _hybrid.isActiveAndEnabled && _hybrid.HasPose) return _hybrid.AbsJoints;
            if (_dual != null && _dual.isActiveAndEnabled && _dual.HasPose) return _dual.AbsJoints;
            return null;
        }

        void Update()
        {
            // tracking lost while holding → gentle drop, no throw
            if (_held != null && _lastPoseTime >= 0f && Time.time - _lastPoseTime > _lostPoseReleaseSec)
            {
                _held.OnRelease(Vector3.zero, Vector3.zero);
                _held = null;
                _pinching = false;
                _smoothInit = false;
            }
            RespawnStrays();
        }

        // Fires from a provider's update, only on frames with a valid pose.
        void HandlePose()
        {
            if (_cameraTransform == null) return;
            if (_lastPoseFrame == Time.frameCount) return;
            var j = ActiveJoints();
            if (j == null) return;
            _lastPoseFrame = Time.frameCount;
            float now = Time.time;
            if (_lastPoseTime >= 0f && now - _lastPoseTime > _lostPoseReleaseSec)
                _smoothInit = false;   // tracking gap — restart smoothing from the fresh pose
            _lastPoseTime = now;

            if (_spawnDemoObjects && _demo == null) SpawnDemoObjects();

            // CV camera frame (+X right, +Y down, +Z fwd) → world, same as HandSphereDriver
            Vector3 wrist = ToWorld(j[Wrist]);
            Vector3 thumb = ToWorld(j[ThumbTip]);
            Vector3 index = ToWorld(j[IndexTip]);
            Vector3 indexMcp = ToWorld(j[IndexMcp]);
            Vector3 littleMcp = ToWorld(j[LittleMcp]);

            float pinchDist = Vector3.Distance(thumb, index);
            Vector3 rawPinch = 0.5f * (thumb + index);
            Quaternion rawRot = PalmRotation(wrist, indexMcp, littleMcp);

            if (!_smoothInit)
            {
                _pinch = _prevPinch = rawPinch;
                _handRot = _prevHandRot = rawRot;
                _smoothInit = true;
            }
            else
            {
                _prevPinch = _pinch;
                _prevHandRot = _handRot;
                _pinch = Vector3.Lerp(rawPinch, _pinch, _smoothing);
                _handRot = Quaternion.Slerp(rawRot, _handRot, _smoothing);
            }

            // velocity estimate while pinching (for the throw)
            float dt = Mathf.Max(Time.deltaTime, 1e-4f);
            if (_pinching)
            {
                _velocity = Vector3.Lerp(_velocity, (_pinch - _prevPinch) / dt, 0.5f);
                var dq = _handRot * Quaternion.Inverse(_prevHandRot);
                dq.ToAngleAxis(out float ang, out Vector3 axis);
                if (ang > 180f) ang -= 360f;
                if (!float.IsNaN(axis.x))
                    _angVelocity = Vector3.Lerp(_angVelocity, axis * (ang * Mathf.Deg2Rad / dt), 0.5f);
            }

            if (!_pinching && pinchDist < _grabDist)
            {
                _pinching = true;
                TryGrab();
            }
            else if (_pinching && pinchDist > _releaseDist)
            {
                _pinching = false;
                if (_held != null)
                {
                    _held.OnRelease(_velocity, _angVelocity);
                    _held = null;
                }
            }

            if (_held != null)
                _held.MoveTo(_pinch + _handRot * _posOffset, _handRot * _rotOffset);
        }

        Vector3 ToWorld(Vector3 cv) => _cameraTransform.TransformPoint(new Vector3(cv.x, -cv.y, cv.z));

        // Palm frame: forward = wrist → knuckle mid, up = palm normal.
        static Quaternion PalmRotation(Vector3 wrist, Vector3 indexMcp, Vector3 littleMcp)
        {
            Vector3 fwd = (0.5f * (indexMcp + littleMcp) - wrist).normalized;
            Vector3 tangent = (indexMcp - littleMcp).normalized;
            Vector3 up = Vector3.Cross(fwd, tangent).normalized;
            if (fwd.sqrMagnitude < 1e-8f || up.sqrMagnitude < 1e-8f) return Quaternion.identity;
            return Quaternion.LookRotation(fwd, up);
        }

        void TryGrab()
        {
            GrabbableObject best = null;
            float bestDist = _grabRadius;
            foreach (var g in FindObjectsByType<GrabbableObject>(FindObjectsSortMode.None))
            {
                if (g.IsHeld) continue;
                var col = g.GetComponent<Collider>();
                float d = Vector3.Distance(col.ClosestPoint(_pinch), _pinch);
                if (d < bestDist) { bestDist = d; best = g; }
            }
            if (best == null) return;

            _held = best;
            var inv = Quaternion.Inverse(_handRot);
            _rotOffset = inv * best.transform.rotation;
            _posOffset = inv * (best.transform.position - _pinch);
            _velocity = _angVelocity = Vector3.zero;
            best.OnGrab();
        }

        // ---------- demo objects ----------

        void SpawnDemoObjects()
        {
            _demo = new[]
            {
                MakeDemo(PrimitiveType.Cube,    new Color(0.95f, 0.35f, 0.25f), -1f),
                MakeDemo(PrimitiveType.Sphere,  new Color(0.25f, 0.55f, 0.95f),  0f),
                MakeDemo(PrimitiveType.Capsule, new Color(0.35f, 0.85f, 0.40f), +1f),
            };
        }

        GrabbableObject MakeDemo(PrimitiveType type, Color color, float slot)
        {
            var go = GameObject.CreatePrimitive(type);
            go.name = $"Grabbable_{type}";
            float s = _objectSize;
            go.transform.localScale = type == PrimitiveType.Capsule
                ? new Vector3(s * 0.6f, s * 0.6f, s * 0.6f)   // capsule mesh is 2 units tall
                : new Vector3(s, s, s);
            go.transform.position = SpawnPoint(slot);
            go.transform.rotation = _cameraTransform.rotation;

            var mr = go.GetComponent<MeshRenderer>();
            mr.sharedMaterial = MakeLit(color);
            mr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            mr.receiveShadows = false;

            var rb = go.AddComponent<Rigidbody>();
            rb.useGravity = false;              // AR: no floor — objects float
#if UNITY_6000_0_OR_NEWER
            rb.linearDamping = 2f;
            rb.angularDamping = 2f;
#else
            rb.drag = 2f;
            rb.angularDrag = 2f;
#endif
            return go.AddComponent<GrabbableObject>();
        }

        Vector3 SpawnPoint(float slot)
        {
            Vector3 fwd = _cameraTransform.forward;
            Vector3 right = _cameraTransform.right;
            return _cameraTransform.position + fwd * _spawnDistance + right * (slot * 2.2f * _objectSize);
        }

        void RespawnStrays()
        {
            if (_demo == null || _cameraTransform == null) return;
            for (int i = 0; i < _demo.Length; i++)
            {
                var g = _demo[i];
                if (g == null || g.IsHeld) continue;
                if (Vector3.Distance(g.transform.position, _cameraTransform.position) < _respawnDistance) continue;
                var rb = g.GetComponent<Rigidbody>();
                if (rb != null)
                {
#if UNITY_6000_0_OR_NEWER
                    rb.linearVelocity = Vector3.zero;
#else
                    rb.velocity = Vector3.zero;
#endif
                    rb.angularVelocity = Vector3.zero;
                }
                g.transform.position = SpawnPoint(i - 1f);
                g.transform.rotation = _cameraTransform.rotation;
            }
        }

        // Lit material that works on Built-in (this project) with an URP fallback.
        static Material MakeLit(Color c)
        {
            Shader sh = Shader.Find("Standard");
            if (sh == null) sh = Shader.Find("Universal Render Pipeline/Lit");
            if (sh == null) sh = Shader.Find("Unlit/Color");
            var m = new Material(sh);
            m.color = c;
            if (m.HasProperty("_BaseColor")) m.SetColor("_BaseColor", c);
            if (m.HasProperty("_Glossiness")) m.SetFloat("_Glossiness", 0.35f);
            return m;
        }
    }
}
#endif
