using System;
using UnityEngine;

namespace HandMesh.DepthRefinement
{
    /// <summary>
    /// Synthetic hand-joint source for tests and editor smoke checks. Produces a
    /// flat hand at a configurable distance with optional per-frame noise, so the
    /// pipeline and visualizer can run with no model or device. Implements
    /// IHandJointProvider so it drops into DepthRefinementManager unchanged.
    /// </summary>
    public sealed class MockHandJointProvider : MonoBehaviour, IHandJointProvider
    {
        [Tooltip("Nominal hand distance from camera, meters (only used by an accompanying mock depth source).")]
        public float HandDistanceMeters = 0.4f;

        [Tooltip("Uniform random jitter added to normalized UV each frame.")]
        public float UvNoise = 0.0f;

        public Handedness Handedness = Handedness.Right;

        public event Action<HandPose> OnHandPoseUpdated;

        readonly HandPose _pose = new HandPose();

        void Update()
        {
            BuildFlatHand(_pose, Time.timeAsDouble, UvNoise, Handedness);
            OnHandPoseUpdated?.Invoke(_pose);
        }

        public bool TryGetLatest(out HandPose pose)
        {
            pose = _pose;
            return _pose.IsValid;
        }

        /// <summary>Deterministic flat-hand layout in normalized UV + root-relative 3D (also used by tests).</summary>
        public static void BuildFlatHand(HandPose pose, double timestamp, float uvNoise, Handedness handed)
        {
            pose.IsValid = true;
            pose.Handedness = handed;
            pose.Timestamp = timestamp;
            pose.FrameWidth = 1920;
            pose.FrameHeight = 1440;

            // Canonical normalized layout: wrist at bottom-center, fingers fanning up.
            // x in [0,1], y in [0,1] (top-left origin), z root-relative (toward scene = +).
            Vector2[] uv = CanonicalUv;
            float n = uvNoise;
            for (int i = 0; i < HandSkeleton.JointCount; i++)
            {
                float jx = uv[i].x + (n > 0f ? (Mathf.PerlinNoise(i, (float)timestamp) - 0.5f) * n : 0f);
                float jy = uv[i].y + (n > 0f ? (Mathf.PerlinNoise(i + 50, (float)timestamp) - 0.5f) * n : 0f);
                pose.Joints[i].Uv = new Vector2(Mathf.Clamp01(jx), Mathf.Clamp01(jy));
                // root-relative 3D (arbitrary unit ~ normalized hand cube); slight z curl.
                pose.Joints[i].LocalPosition = new Vector3(uv[i].x - 0.5f, 0.5f - uv[i].y, CanonicalZ[i]);
                pose.Joints[i].Confidence = 0.9f;
                pose.Joints[i].DepthConfidence = 0f;
                pose.Joints[i].WasRefined = false;
            }
        }

        // Approximate normalized fingertip-up hand (not anatomically exact; enough for smoke tests).
        static readonly Vector2[] CanonicalUv =
        {
            new Vector2(0.50f, 0.90f), // wrist
            new Vector2(0.38f, 0.78f), new Vector2(0.32f, 0.70f), new Vector2(0.28f, 0.63f), new Vector2(0.25f, 0.57f), // thumb
            new Vector2(0.44f, 0.66f), new Vector2(0.43f, 0.54f), new Vector2(0.42f, 0.46f), new Vector2(0.41f, 0.40f), // index
            new Vector2(0.50f, 0.64f), new Vector2(0.50f, 0.50f), new Vector2(0.50f, 0.42f), new Vector2(0.50f, 0.36f), // middle
            new Vector2(0.56f, 0.66f), new Vector2(0.57f, 0.54f), new Vector2(0.58f, 0.46f), new Vector2(0.59f, 0.41f), // ring
            new Vector2(0.62f, 0.70f), new Vector2(0.64f, 0.60f), new Vector2(0.66f, 0.54f), new Vector2(0.67f, 0.49f)  // pinky
        };

        static readonly float[] CanonicalZ =
        {
            0.00f,
            0.01f, 0.02f, 0.03f, 0.04f,
            0.00f, 0.01f, 0.02f, 0.03f,
            0.00f, 0.01f, 0.02f, 0.03f,
            0.00f, 0.01f, 0.02f, 0.03f,
            0.00f, 0.01f, 0.02f, 0.03f
        };
    }
}
