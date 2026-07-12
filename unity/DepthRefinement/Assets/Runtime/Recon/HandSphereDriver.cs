#if SENTIS_PRESENT && ARFOUNDATION_PRESENT
using UnityEngine;

namespace HandMesh.DepthRefinement
{
    /// <summary>
    /// Drives EXISTING sphere GameObjects (one per joint) from the dual-stream model output,
    /// instead of creating them at runtime. Assign the parent that holds your 21 spheres
    /// (named <c>root, thumb1..4, index1..4, middle1..4, ring1..4, little1..4</c>) — they are
    /// matched by name in joint order — or fill the <see cref="_joints"/> array manually.
    ///
    /// Each frame, reads <see cref="DualStreamHandProvider.AbsJoints"/> (absolute camera-space
    /// metres, CV frame: +X right, +Y down, +Z forward), flips Y to Unity, and places each
    /// sphere in world space via the AR camera transform. Requires a wired
    /// <see cref="DualStreamHandProvider"/> (camera + LiDAR depth + ds_anchor_gate model).
    /// </summary>
    public sealed class HandSphereDriver : MonoBehaviour
    {
        /// <summary>Sphere names in model joint order (index 0..20). Matches HandJointId.</summary>
        public static readonly string[] JointNames =
        {
            "root",
            "thumb1", "thumb2", "thumb3", "thumb4",
            "index1", "index2", "index3", "index4",
            "middle1", "middle2", "middle3", "middle4",
            "ring1", "ring2", "ring3", "ring4",
            "little1", "little2", "little3", "little4",
        };

        [SerializeField] DualStreamHandProvider _provider;
        [Tooltip("Optional: if assigned, drive the spheres/bones from model B (HybridBHandProvider) " +
                 "instead of the dual-stream provider. Leave empty to use the dual-stream provider.")]
        [SerializeField] HybridBHandProvider _hybridProvider;
        [Tooltip("Optional: SERVER-mode source (joints inferred on the PC by infer_ipad_stream.py " +
                 "and streamed back). Takes priority while enabled — HandPoseModeController flips " +
                 "the enabled states when switching modes.")]
        [SerializeField] ServerHandProvider _serverProvider;
        [Tooltip("AR Main Camera — joints are in ITS local frame. Defaults to Camera.main.")]
        [SerializeField] Transform _cameraTransform;

        [Tooltip("Parent holding the 21 spheres, matched by name (e.g. Canvas_sphere).")]
        [SerializeField] Transform _jointsParent;
        [Tooltip("Optional: 21 transforms in joint order. Overrides name matching if filled.")]
        [SerializeField] Transform[] _joints;

        [Tooltip("Hide the spheres when there is no valid pose this frame.")]
        [SerializeField] bool _hideWhenNoPose = true;

        [Tooltip("Force each sphere's WORLD diameter (metres), regardless of parent/Canvas " +
                 "scale. 0.016 = 1.6 cm. Set 0 to leave the spheres' own scale alone.")]
        [SerializeField] float _sphereDiameter = 0.016f;

        [Header("Bones")]
        [Tooltip("Draw connecting lines between the joint spheres (HandSkeleton.Bones topology).")]
        [SerializeField] bool _drawBones = true;
        [Tooltip("Bone line width in metres (world space). 0.0008 = 0.8 mm.")]
        [SerializeField] float _boneWidth = 0.0008f;
        [SerializeField] Color _boneColor = new Color(1f, 1f, 1f, 0.85f);
        [Tooltip("Optional material override for the bones; if empty an unlit material is created.")]
        [SerializeField] Material _boneMaterial;

        bool _bound;
        GameObject _bonesRoot;
        LineRenderer[] _bones;
        Material _boneMat;

        void Start()
        {
            if (_provider == null) _provider = FindFirstObjectByType<DualStreamHandProvider>();
            if (_serverProvider == null) _serverProvider = FindFirstObjectByType<ServerHandProvider>();
            if (_cameraTransform == null && Camera.main != null) _cameraTransform = Camera.main.transform;
            EnsureBound();
        }

        void EnsureBound()
        {
            if (_bound) return;
            if (_joints == null || _joints.Length != HandSkeleton.JointCount)
            {
                if (_jointsParent == null) return;
                _joints = new Transform[HandSkeleton.JointCount];
                for (int i = 0; i < JointNames.Length; i++)
                {
                    _joints[i] = FindDeep(_jointsParent, JointNames[i]);
                    if (_joints[i] == null)
                        Debug.LogWarning($"[HandSphereDriver] sphere '{JointNames[i]}' not found under '{_jointsParent.name}'.");
                }
            }
            BuildBones();
            _bound = true;
        }

        // Create one world-space LineRenderer per skeleton bone, parented to an unscaled root so
        // the line width stays in true metres regardless of any Canvas/parent scaling.
        void BuildBones()
        {
            if (!_drawBones || _bones != null || _joints == null) return;
            _boneMat = _boneMaterial != null ? _boneMaterial : MakeUnlit(_boneColor);
            _bonesRoot = new GameObject("HandBones");   // scene root, identity scale
            _bones = new LineRenderer[HandSkeleton.Bones.Length];
            for (int b = 0; b < _bones.Length; b++)
            {
                var go = new GameObject("Bone" + b);
                go.transform.SetParent(_bonesRoot.transform, false);
                var lr = go.AddComponent<LineRenderer>();
                lr.positionCount = 2;
                lr.widthMultiplier = _boneWidth;
                lr.sharedMaterial = _boneMat;
                lr.numCapVertices = 2;
                lr.useWorldSpace = true;
                lr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
                lr.receiveShadows = false;
                _bones[b] = lr;
            }
        }

        // Pick this frame's joints. HandPoseModeController enables exactly one source, so the
        // isActiveAndEnabled checks make the same scene work in both inference modes.
        bool TryGetPose(out Vector3[] joints)
        {
            if (_serverProvider != null && _serverProvider.isActiveAndEnabled)
            { joints = _serverProvider.AbsJoints; return _serverProvider.HasPose; }
            if (_hybridProvider != null && _hybridProvider.isActiveAndEnabled)
            { joints = _hybridProvider.AbsJoints; return _hybridProvider.HasPose; }
            if (_provider != null && _provider.isActiveAndEnabled)
            { joints = _provider.AbsJoints; return _provider.HasPose; }
            joints = null;
            return false;
        }

        // After the provider's Update() produced this frame's pose.
        void LateUpdate()
        {
            EnsureBound();
            // Source priority: server (while enabled) > model B > dual-stream.
            bool hasPose = TryGetPose(out Vector3[] j);
            bool show = _cameraTransform != null && _joints != null && hasPose;
            if (!show)
            {
                if (_hideWhenNoPose) { SetActive(false); SetBonesActive(false); }
                return;
            }
            SetActive(true);

            for (int i = 0; i < _joints.Length && i < j.Length; i++)
            {
                if (_joints[i] == null) continue;
                Vector3 cv = j[i];
                _joints[i].position = _cameraTransform.TransformPoint(new Vector3(cv.x, -cv.y, cv.z));

                // Force world size independent of the parent (e.g. a scaled World-Space Canvas):
                // localScale = desiredWorldDiameter / parentLossyScale (sphere mesh = 1 unit dia).
                if (_sphereDiameter > 0f)
                {
                    Transform par = _joints[i].parent;
                    float ps = par != null ? Mathf.Abs(par.lossyScale.x) : 1f;
                    float s = ps > 1e-6f ? _sphereDiameter / ps : _sphereDiameter;
                    _joints[i].localScale = new Vector3(s, s, s);
                }
            }

            UpdateBones();
        }

        // Stretch each bone line between its two joint spheres' world positions.
        void UpdateBones()
        {
            if (!_drawBones || _bones == null) return;
            SetBonesActive(true);
            var bones = HandSkeleton.Bones;
            for (int b = 0; b < _bones.Length; b++)
            {
                Transform p = _joints[bones[b].parent], c = _joints[bones[b].child];
                bool ok = p != null && c != null;
                if (_bones[b].enabled != ok) _bones[b].enabled = ok;
                if (!ok) continue;
                _bones[b].SetPosition(0, p.position);
                _bones[b].SetPosition(1, c.position);
            }
        }

        void SetBonesActive(bool on)
        {
            if (_bonesRoot != null && _bonesRoot.activeSelf != on) _bonesRoot.SetActive(on);
        }

        // Unlit material that shows over the AR feed without scene lights (Built-in or URP).
        static Material MakeUnlit(Color c)
        {
            Shader sh = Shader.Find("Unlit/Color");
            if (sh == null) sh = Shader.Find("Universal Render Pipeline/Unlit");
            if (sh == null) sh = Shader.Find("Sprites/Default");
            if (sh == null) sh = Shader.Find("Standard");
            var m = new Material(sh);
            m.color = c;
            if (m.HasProperty("_BaseColor")) m.SetColor("_BaseColor", c); // URP
            return m;
        }

        void OnDestroy()
        {
            if (_boneMaterial == null && _boneMat != null) Destroy(_boneMat);
            if (_bonesRoot != null) Destroy(_bonesRoot);
        }

        void SetActive(bool on)
        {
            if (_joints == null) return;
            for (int i = 0; i < _joints.Length; i++)
                if (_joints[i] != null && _joints[i].gameObject.activeSelf != on)
                    _joints[i].gameObject.SetActive(on);
        }

        static Transform FindDeep(Transform parent, string name)
        {
            if (parent.name == name) return parent;
            for (int i = 0; i < parent.childCount; i++)
            {
                var r = FindDeep(parent.GetChild(i), name);
                if (r != null) return r;
            }
            return null;
        }
    }
}
#endif
