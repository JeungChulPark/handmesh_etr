using NUnit.Framework;
using UnityEngine;

namespace HandMesh.DepthRefinement.Tests
{
    public class JointDepthSamplerTests
    {
        static JointDepthSampler MakeSampler()
        {
            var s = new JointDepthSampler();
            s.SetConfig(JointDepthSampler.Config.Default);
            return s;
        }

        [Test]
        public void ConstantDepth_SampleIsValidAndAccurate()
        {
            var depth = new FakeDepthProvider(64, 48, 0.5f, 1f);
            var sample = MakeSampler().Sample(depth, new Vector2(0.5f, 0.5f));
            Assert.IsTrue(sample.Valid);
            Assert.AreEqual(0.5f, sample.Meters, 1e-4f);
            Assert.Greater(sample.Confidence, 0.5f);
        }

        [Test]
        public void LowConfidence_SampleRejected()
        {
            var depth = new FakeDepthProvider(64, 48, 0.5f, 0.0f); // below MinPixelConfidence
            var sample = MakeSampler().Sample(depth, new Vector2(0.5f, 0.5f));
            Assert.IsFalse(sample.Valid);
        }

        [Test]
        public void FlyingPixel_RejectedByMedian()
        {
            var depth = new FakeDepthProvider(64, 48, 0.5f, 1f);
            // inject a single far outlier at the window center
            depth.SetPixel(32, 24, 0.95f, 1f);
            var sample = MakeSampler().Sample(depth, new Vector2(0.5f, 0.5f));
            Assert.IsTrue(sample.Valid);
            Assert.AreEqual(0.5f, sample.Meters, 1e-3f, "Median + spread test should reject the flying pixel.");
        }

        [Test]
        public void DepthUnavailable_ReturnsInvalid()
        {
            var depth = new FakeDepthProvider(64, 48, 0.5f, 1f) { IsDepthAvailable = false };
            var sample = MakeSampler().Sample(depth, new Vector2(0.5f, 0.5f));
            Assert.IsFalse(sample.Valid);
        }
    }
}
