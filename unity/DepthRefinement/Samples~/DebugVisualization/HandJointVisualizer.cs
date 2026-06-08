using UnityEngine;

namespace HandMesh.DepthRefinement.Samples
{
    /// <summary>
    /// Step 10: draws the refined hand skeleton in the scene and (optionally) the
    /// raw RGB-only joints next to it, coloring each joint by whether device depth
    /// was actually fused in (green = refined, red = RGB-only fallback). Subscribe to
    /// <see cref="DepthRefinementManager.OnRefined"/>. Uses Gizmos (Scene view) and an
    /// optional pooled LineRenderer set for bones in the Game view.
    /// </summary>
    [RequireComponent(typeof(DepthRefinementManager))]
    public sealed class HandJointVisualizer : MonoBehaviour
    {
        [SerializeField] DepthRefinementManager _manager;
        [SerializeField] Transform _cameraSpaceRoot; // CV camera-space -> world anchor (e.g. AR camera)
        [SerializeField] float _jointRadius = 0.006f;
        [SerializeField] bool _showRgbOnly = true;
        [SerializeField] Color _refinedColor = Color.green;
        [SerializeField] Color _fallbackColor = Color.red;
        [SerializeField] Color _rgbColor = new Color(0.4f, 0.6f, 1f, 0.6f);

        HandPose _pose;

        void Reset() => _manager = GetComponent<DepthRefinementManager>();

        void OnEnable()
        {
            if (_manager == null) _manager = GetComponent<DepthRefinementManager>();
            if (_manager != null) _manager.OnRefined += OnRefined;
        }

        void OnDisable()
        {
            if (_manager != null) _manager.OnRefined -= OnRefined;
        }

        void OnRefined(HandPose pose, DepthRefinementPipeline.Outcome outcome) => _pose = pose;

        Vector3 ToWorld(Vector3 cameraSpace)
        {
            // CV convention (+Y down) -> Unity (+Y up); anchor under the AR camera if provided.
            var p = new Vector3(cameraSpace.x, -cameraSpace.y, cameraSpace.z);
            return _cameraSpaceRoot != null ? _cameraSpaceRoot.TransformPoint(p) : p;
        }

        void OnDrawGizmos()
        {
            if (_pose == null || !_pose.IsValid) return;

            // bones
            Gizmos.color = Color.white;
            foreach (var (p, c) in HandSkeleton.Bones)
                Gizmos.DrawLine(ToWorld(_pose.Joints[p].RefinedPosition), ToWorld(_pose.Joints[c].RefinedPosition));

            // joints colored by refinement state
            for (int i = 0; i < HandSkeleton.JointCount; i++)
            {
                Gizmos.color = _pose.Joints[i].WasRefined ? _refinedColor : _fallbackColor;
                Gizmos.DrawSphere(ToWorld(_pose.Joints[i].RefinedPosition), _jointRadius);
            }

            if (!_showRgbOnly) return;

            // raw RGB-only joints (LocalPosition) drawn faintly for comparison
            Gizmos.color = _rgbColor;
            for (int i = 0; i < HandSkeleton.JointCount; i++)
                Gizmos.DrawWireSphere(ToWorld(_pose.Joints[i].LocalPosition), _jointRadius);
        }
    }
}
