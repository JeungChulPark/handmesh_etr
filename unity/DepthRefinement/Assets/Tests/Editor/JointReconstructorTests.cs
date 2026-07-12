using NUnit.Framework;
using UnityEngine;

namespace HandMesh.DepthRefinement.Tests
{
    public class JointReconstructorTests
    {
        static CameraIntrinsics K => new CameraIntrinsics { Fx = 500f, Fy = 500f, Cx = 128f, Cy = 96f, Width = 256, Height = 192 };

        [Test]
        public void Unproject_AtPrincipalPoint_GivesZeroXY()
        {
            var p = JointReconstructor.Unproject(128f, 96f, 2.0f, K);
            Assert.AreEqual(0f, p.x, 1e-4f);
            Assert.AreEqual(0f, p.y, 1e-4f);
            Assert.AreEqual(2.0f, p.z, 1e-4f);
        }

        [Test]
        public void Unproject_OffCenter_MatchesPinholeFormula()
        {
            // x = (u-cx)*z/fx = (178-128)*2/500 = 0.2 ; y = (146-96)*2/500 = 0.2
            var p = JointReconstructor.Unproject(178f, 146f, 2.0f, K);
            Assert.AreEqual(0.2f, p.x, 1e-4f);
            Assert.AreEqual(0.2f, p.y, 1e-4f);
            Assert.AreEqual(2.0f, p.z, 1e-4f);
        }

        [Test]
        public void UvToDepthPixel_ScalesByResolution()
        {
            var px = JointReconstructor.UvToDepthPixel(new Vector2(0.5f, 0.25f), K);
            Assert.AreEqual(128f, px.x, 1e-4f);
            Assert.AreEqual(48f, px.y, 1e-4f);
        }
    }
}
