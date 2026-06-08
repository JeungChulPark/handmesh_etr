using NUnit.Framework;
using UnityEngine;

namespace HandMesh.DepthRefinement.Tests
{
    public class OneEuroFilterTests
    {
        [Test]
        public void NoisyConstant_VarianceReduced()
        {
            var f = new OneEuroFilter { MinCutoff = 0.5f, Beta = 0.0f, DCutoff = 1.0f };
            const float baseValue = 1.0f;
            float dt = 1f / 60f;

            // pseudo-random but deterministic noise
            float inSum = 0, inSq = 0, outSum = 0, outSq = 0;
            int n = 0;
            for (int i = 0; i < 300; i++)
            {
                float noise = (Mathf.Sin(i * 12.9898f) * 43758.5453f % 1f) - 0.5f; // [-0.5,0.5]
                float x = baseValue + noise * 0.1f;
                float y = f.Filter(x, i * dt);
                if (i > 50) // skip warm-up
                {
                    inSum += x; inSq += x * x;
                    outSum += y; outSq += y * y;
                    n++;
                }
            }
            float inVar = inSq / n - (inSum / n) * (inSum / n);
            float outVar = outSq / n - (outSum / n) * (outSum / n);
            Assert.Less(outVar, inVar, "Filtered variance should be lower than input variance.");
        }

        [Test]
        public void StepInput_ConvergesWithoutExcessiveLag()
        {
            var f = new OneEuroFilter { MinCutoff = 1.0f, Beta = 0.1f, DCutoff = 1.0f };
            float dt = 1f / 60f;
            for (int i = 0; i < 30; i++) f.Filter(0f, i * dt);   // settle at 0
            float last = 0f;
            for (int i = 30; i < 90; i++) last = f.Filter(1f, i * dt); // step to 1
            Assert.Greater(last, 0.8f, "Filter should converge near the new value within ~1s.");
        }
    }
}
