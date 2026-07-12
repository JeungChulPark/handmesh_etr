using NUnit.Framework;
using UnityEngine;

namespace HandMesh.DepthRefinement.Tests
{
    public class BoneLengthConstraintTests
    {
        [Test]
        public void HardSnap_MakesBoneLengthsMatchReference()
        {
            var joints = new HandJointData[HandSkeleton.JointCount];
            for (int i = 0; i < joints.Length; i++)
                joints[i].RefinedPosition = new Vector3(i * 0.013f + 0.001f, (i % 3) * 0.017f + 0.002f, (i % 5) * 0.011f + 0.003f);

            var bone = new BoneLengthConstraint();
            bone.SetConfig(new BoneLengthConstraint.Config { Strength = 1f, PinWrist = true });
            bone.Apply(joints, HandSkeleton.DefaultBoneLengthsMeters);

            var bones = HandSkeleton.Bones;
            for (int b = 0; b < bones.Length; b++)
            {
                float len = Vector3.Distance(joints[bones[b].parent].RefinedPosition, joints[bones[b].child].RefinedPosition);
                Assert.AreEqual(HandSkeleton.DefaultBoneLengthsMeters[b], len, 1e-3f,
                    $"Bone {b} length should match reference after hard snap.");
            }
        }

        [Test]
        public void PartialStrength_ReducesBoneLengthError()
        {
            var joints = new HandJointData[HandSkeleton.JointCount];
            for (int i = 0; i < joints.Length; i++)
                joints[i].RefinedPosition = new Vector3(i * 0.05f + 0.001f, (i % 3) * 0.02f, (i % 5) * 0.03f);

            float ErrorSum()
            {
                float e = 0f;
                var bones = HandSkeleton.Bones;
                for (int b = 0; b < bones.Length; b++)
                    e += Mathf.Abs(Vector3.Distance(joints[bones[b].parent].RefinedPosition,
                                                    joints[bones[b].child].RefinedPosition)
                                   - HandSkeleton.DefaultBoneLengthsMeters[b]);
                return e;
            }

            float before = ErrorSum();
            var bone = new BoneLengthConstraint();
            bone.SetConfig(new BoneLengthConstraint.Config { Strength = 0.5f, PinWrist = true });
            bone.Apply(joints, HandSkeleton.DefaultBoneLengthsMeters);
            Assert.Less(ErrorSum(), before, "Partial strength should reduce total bone-length error toward reference.");
        }

        [Test]
        public void ZeroStrength_LeavesJointsUntouched()
        {
            var joints = new HandJointData[HandSkeleton.JointCount];
            for (int i = 0; i < joints.Length; i++)
                joints[i].RefinedPosition = new Vector3(i, 0, 0);

            var bone = new BoneLengthConstraint();
            bone.SetConfig(new BoneLengthConstraint.Config { Strength = 0f, PinWrist = true });
            bone.Apply(joints, HandSkeleton.DefaultBoneLengthsMeters);

            for (int i = 0; i < joints.Length; i++)
                Assert.AreEqual(new Vector3(i, 0, 0), joints[i].RefinedPosition);
        }
    }
}
