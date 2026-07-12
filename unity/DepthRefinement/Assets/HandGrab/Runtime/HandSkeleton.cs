using UnityEngine;

namespace HandMesh.HandGrab
{
    /// <summary>
    /// Renders the 21-joint hand as spheres + per-finger colored bone lines.
    /// Joint order = MediaPipe/FreiHAND: 0 wrist, 1-4 thumb, 5-8 index,
    /// 9-12 middle, 13-16 ring, 17-20 pinky (tips are 4/8/12/16/20).
    /// </summary>
    [RequireComponent(typeof(HandStreamReceiver))]
    public class HandSkeleton : MonoBehaviour
    {
        public float jointRadius = 0.008f;
        public float boneWidth = 0.004f;

        static readonly int[,] Bones =
        {
            {0,1},{1,2},{2,3},{3,4},          // thumb
            {0,5},{5,6},{6,7},{7,8},          // index
            {0,9},{9,10},{10,11},{11,12},     // middle
            {0,13},{13,14},{14,15},{15,16},   // ring
            {0,17},{17,18},{18,19},{19,20},   // pinky
        };

        static readonly Color[] FingerColors =
        {
            new Color(0.86f, 0.24f, 0.24f),   // thumb
            new Color(0.24f, 0.71f, 0.24f),   // index
            new Color(0.16f, 0.63f, 0.86f),   // middle
            new Color(0.78f, 0.24f, 0.78f),   // ring
            new Color(0.86f, 0.78f, 0.16f),   // pinky
        };

        HandStreamReceiver _recv;
        Transform[] _joints;
        LineRenderer[] _bones;

        void Start()
        {
            _recv = GetComponent<HandStreamReceiver>();
            var mat = new Material(Shader.Find("Sprites/Default"));

            _joints = new Transform[HandStreamReceiver.JointCount];
            for (int i = 0; i < _joints.Length; i++)
            {
                var s = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                s.name = $"joint_{i}";
                Destroy(s.GetComponent<Collider>());        // visual only — no physics
                s.transform.SetParent(transform, false);
                s.transform.localScale = Vector3.one * jointRadius * 2f;
                var mr = s.GetComponent<MeshRenderer>();
                mr.material = mat;
                mr.material.color = i == 0 ? Color.white : FingerColors[(i - 1) / 4];
                _joints[i] = s.transform;
            }

            _bones = new LineRenderer[Bones.GetLength(0)];
            for (int b = 0; b < _bones.Length; b++)
            {
                var go = new GameObject($"bone_{b}");
                go.transform.SetParent(transform, false);
                var lr = go.AddComponent<LineRenderer>();
                lr.positionCount = 2;
                lr.startWidth = lr.endWidth = boneWidth;
                lr.material = mat;
                lr.startColor = lr.endColor = FingerColors[b / 4];
                lr.useWorldSpace = true;
                _bones[b] = lr;
            }
        }

        void LateUpdate()
        {
            bool show = _recv.IsTracked;
            for (int i = 0; i < _joints.Length; i++)
            {
                _joints[i].gameObject.SetActive(show);
                if (show) _joints[i].position = _recv.Joints[i];
            }
            for (int b = 0; b < _bones.Length; b++)
            {
                _bones[b].gameObject.SetActive(show);
                if (!show) continue;
                _bones[b].SetPosition(0, _recv.Joints[Bones[b, 0]]);
                _bones[b].SetPosition(1, _recv.Joints[Bones[b, 1]]);
            }
        }
    }
}
