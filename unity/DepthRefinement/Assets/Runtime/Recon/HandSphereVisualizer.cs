#if SENTIS_PRESENT && ARFOUNDATION_PRESENT
using UnityEngine;

namespace HandMesh.DepthRefinement
{
    /// <summary>
    /// Draws the dual-stream model's 21 hand joints as spheres (plus optional bone lines) in the
    /// Unity world, reading <see cref="DualStreamHandProvider.AbsJoints"/> (absolute camera-space
    /// metres) each frame and converting them to world space via the AR camera transform.
    ///
    /// Model joints are CV camera space (+X right, +Y down, +Z forward); Unity camera space is
    /// +Y up, so we flip Y, then <c>camera.TransformPoint</c> places them in the world in front
    /// of the device. Spheres are radius-sized in METRES, so they sit at the real hand depth.
    /// </summary>
    public sealed class HandSphereVisualizer : MonoBehaviour
    {
        [SerializeField] DualStreamHandProvider _provider;
        [Tooltip("AR Main Camera — joints are in ITS local (camera) frame. Defaults to Camera.main.")]
        [SerializeField] Transform _cameraTransform;

        [Header("Spheres")]
        [SerializeField] float _sphereRadius = 0.008f;     // 8 mm
        [SerializeField] Color _jointColor = new Color(0.1f, 1f, 0.2f);
        [Tooltip("Optional material override; if empty an unlit material is created.")]
        [SerializeField] Material _jointMaterial;

        [Header("Bones")]
        [SerializeField] bool _drawBones = true;
        [SerializeField] float _boneWidth = 0.004f;
        [SerializeField] Color _boneColor = new Color(1f, 1f, 1f, 0.8f);

        GameObject _rootGo;
        Transform[] _spheres;
        LineRenderer[] _bones;
        Material _jointMat, _boneMat;

        void Start()
        {
            if (_provider == null) _provider = FindFirstObjectByType<DualStreamHandProvider>();
            if (_cameraTransform == null && Camera.main != null) _cameraTransform = Camera.main.transform;

            _jointMat = _jointMaterial != null ? _jointMaterial : MakeUnlit(_jointColor);
            _boneMat = MakeUnlit(_boneColor);

            _rootGo = new GameObject("HandJoints");
            _spheres = new Transform[HandSkeleton.JointCount];
            for (int i = 0; i < _spheres.Length; i++)
            {
                var go = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                go.name = ((HandJointId)i).ToString();
                var col = go.GetComponent<Collider>();
                if (col != null) Destroy(col);
                go.transform.SetParent(_rootGo.transform, false);
                go.transform.localScale = Vector3.one * (_sphereRadius * 2f); // diameter
                go.GetComponent<Renderer>().sharedMaterial = _jointMat;
                _spheres[i] = go.transform;
            }

            _bones = new LineRenderer[HandSkeleton.Bones.Length];
            for (int b = 0; b < _bones.Length; b++)
            {
                var go = new GameObject("Bone" + b);
                go.transform.SetParent(_rootGo.transform, false);
                var lr = go.AddComponent<LineRenderer>();
                lr.positionCount = 2;
                lr.widthMultiplier = _boneWidth;
                lr.material = _boneMat;
                lr.numCapVertices = 2;
                lr.useWorldSpace = true;
                lr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
                lr.receiveShadows = false;
                _bones[b] = lr;
            }
            SetVisible(false);
        }

        // After the provider's Update() has produced this frame's pose.
        void LateUpdate()
        {
            if (_provider == null || _cameraTransform == null || !_provider.HasPose)
            {
                SetVisible(false);
                return;
            }
            SetVisible(true);

            for (int i = 0; i < _spheres.Length; i++)
                _spheres[i].position = ToWorld(_provider.AbsJoints[i]);

            if (_drawBones)
            {
                var bones = HandSkeleton.Bones;
                for (int b = 0; b < bones.Length; b++)
                {
                    _bones[b].SetPosition(0, _spheres[bones[b].parent].position);
                    _bones[b].SetPosition(1, _spheres[bones[b].child].position);
                }
            }
        }

        // CV camera space (Y down, metres) -> Unity world via the AR camera transform.
        Vector3 ToWorld(Vector3 cv)
        {
            Vector3 local = new Vector3(cv.x, -cv.y, cv.z);
            return _cameraTransform.TransformPoint(local);
        }

        void SetVisible(bool on)
        {
            if (_rootGo != null && _rootGo.activeSelf != on) _rootGo.SetActive(on);
        }

        // Unlit material that shows up over the AR feed without scene lights (Built-in or URP).
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
            if (_jointMaterial == null && _jointMat != null) Destroy(_jointMat);
            if (_boneMat != null) Destroy(_boneMat);
        }
    }
}
#endif
