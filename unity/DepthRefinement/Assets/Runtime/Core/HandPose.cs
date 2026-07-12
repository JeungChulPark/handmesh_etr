namespace HandMesh.DepthRefinement
{
    public enum Handedness
    {
        Unknown = 0,
        Left = 1,
        Right = 2
    }

    /// <summary>
    /// One frame of hand pose: 21 joints plus the frame metadata the refinement
    /// pipeline needs (handedness, capture timestamp, source frame dimensions).
    /// Joints array is pre-allocated and reused to avoid per-frame GC.
    /// </summary>
    public sealed class HandPose
    {
        public readonly HandJointData[] Joints = new HandJointData[HandSkeleton.JointCount];

        public Handedness Handedness = Handedness.Unknown;

        /// <summary>Capture timestamp in seconds (e.g. ARFrame timestamp). Used by temporal filtering.</summary>
        public double Timestamp;

        /// <summary>Source RGB frame dimensions, used to map normalized Uv back to pixels.</summary>
        public int FrameWidth;
        public int FrameHeight;

        /// <summary>False when no hand was detected this frame.</summary>
        public bool IsValid;

        public void CopyFrom(HandPose other)
        {
            Handedness = other.Handedness;
            Timestamp = other.Timestamp;
            FrameWidth = other.FrameWidth;
            FrameHeight = other.FrameHeight;
            IsValid = other.IsValid;
            for (int i = 0; i < HandSkeleton.JointCount; i++)
                Joints[i] = other.Joints[i];
        }
    }
}
