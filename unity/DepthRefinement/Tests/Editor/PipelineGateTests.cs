using System.Text.RegularExpressions;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

namespace HandMesh.DepthRefinement.Tests
{
    public class PipelineGateTests
    {
        static HandPose FlatHand()
        {
            var pose = new HandPose();
            MockHandJointProvider.BuildFlatHand(pose, 0.0, 0f, Handedness.Right);
            return pose;
        }

        [Test]
        public void NoDepth_FallsBackToRgbOnly()
        {
            var pose = FlatHand();
            var pipeline = new DepthRefinementPipeline();

            var outcome = pipeline.Process(pose, null);

            Assert.IsFalse(outcome.Refined);
            Assert.IsNotNull(outcome.FallbackReason);
            for (int i = 0; i < HandSkeleton.JointCount; i++)
                Assert.IsFalse(pose.Joints[i].WasRefined, "All joints should be RGB-only on fallback.");
        }

        [Test]
        public void NoDepth_WarnsOnceThenSuppresses()
        {
            var pipeline = new DepthRefinementPipeline();
            LogAssert.Expect(LogType.Warning, new Regex("falling back to RGB-only"));
            pipeline.Process(FlatHand(), null); // logs the single expected warning
            pipeline.Process(FlatHand(), null); // must NOT log again (else LogAssert fails the test)
        }

        [Test]
        public void ConsistentDepth_AlignsAndRefines()
        {
            var pose = FlatHand();
            var refL = HandSkeleton.DefaultBoneLengthsMeters;
            float s = ConfidenceFusion.EstimateScale(pose, refL);
            Assert.Greater(s, 0f);

            const float rootDepth = 0.5f;
            var samples = new DepthSample[HandSkeleton.JointCount];
            for (int i = 0; i < samples.Length; i++)
                samples[i] = new DepthSample
                {
                    Meters = s * pose.Joints[i].LocalPosition.z + rootDepth,
                    Confidence = 1f,
                    Valid = true
                };

            var fusion = new ConfidenceFusion();
            fusion.SetConfig(ConfidenceFusion.Config.Default);
            var absDepth = new float[HandSkeleton.JointCount];
            var refined = new bool[HandSkeleton.JointCount];

            var r = fusion.Compute(pose, samples, refL, absDepth, refined);

            Assert.IsTrue(r.Succeeded);
            Assert.AreEqual(rootDepth, r.RootDepth, 1e-2f, "Shift-only alignment should recover the root depth.");
            Assert.GreaterOrEqual(r.InlierCount, 3);
        }

        [Test]
        public void OccludedJoint_IsRejectedFromFusion()
        {
            var pose = FlatHand();
            var refL = HandSkeleton.DefaultBoneLengthsMeters;
            float s = ConfidenceFusion.EstimateScale(pose, refL);

            const float rootDepth = 0.5f;
            var samples = new DepthSample[HandSkeleton.JointCount];
            for (int i = 0; i < samples.Length; i++)
                samples[i] = new DepthSample
                {
                    Meters = s * pose.Joints[i].LocalPosition.z + rootDepth,
                    Confidence = 1f,
                    Valid = true
                };
            // Joint 6 (IndexPIP, a non-fingertip) reads 0.2 m too near — an occluder surface.
            const int occluded = 6;
            samples[occluded].Meters += 0.2f;

            var fusion = new ConfidenceFusion();
            fusion.SetConfig(ConfidenceFusion.Config.Default);
            var absDepth = new float[HandSkeleton.JointCount];
            var refined = new bool[HandSkeleton.JointCount];

            var r = fusion.Compute(pose, samples, refL, absDepth, refined);

            Assert.IsTrue(r.Succeeded);
            Assert.IsFalse(refined[occluded], "Occluded joint must be rejected from fusion (RGB-only).");
            Assert.IsTrue(refined[0], "Consistent joints should still be refined.");
            Assert.IsFalse(refined[8], "Fingertips are never directly fused (SM-6).");
        }
    }
}
