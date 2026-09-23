using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using Shape = NumSharp.Shape;
using Box = Gym.Spaces.Box;
using NumSharp;

namespace Gym.Tests.Spaces
{
    [TestClass]
    public class BoxTest
    {
        [TestMethod]
        public void TestBoxBoundedTest()
        {
            Box box = new Box(-5.0f, 5.0f, np.float32);
            Assert.IsTrue(box.IsBounded(Gym.Spaces.BoundedMannerEnum.Both), "Box should be bounded at both boundaries.");
            
            box = new Box(-np.inf, 5.0f, np.float32);
            Assert.IsFalse(box.IsBounded(Gym.Spaces.BoundedMannerEnum.Below), "Box should be unbound at the low bound.");
            Assert.IsTrue(box.IsBounded(Gym.Spaces.BoundedMannerEnum.Above), "Box should be bounded at the high bound.");
            Assert.IsFalse(box.IsBounded(Gym.Spaces.BoundedMannerEnum.Both), "Box should not be bounded.");

            box = new Box(5.0f, np.inf, np.float32);
            Assert.IsFalse(box.IsBounded(Gym.Spaces.BoundedMannerEnum.Above), "Box should be unbound at the high bound.");
            Assert.IsTrue(box.IsBounded(Gym.Spaces.BoundedMannerEnum.Below), "Box should be bounded at the low bound.");
            Assert.IsFalse(box.IsBounded(Gym.Spaces.BoundedMannerEnum.Both), "Box should not be bounded.");

            box = new Box(-np.inf, np.inf, np.float32);
            Assert.IsFalse(box.IsBounded(Gym.Spaces.BoundedMannerEnum.Above), "Box should be unbound at the high bound.");
            Assert.IsFalse(box.IsBounded(Gym.Spaces.BoundedMannerEnum.Below), "Box should be unbound at the low bound.");
            Assert.IsFalse(box.IsBounded(Gym.Spaces.BoundedMannerEnum.Both), "Box should not be bounded.");
        }

        /// <summary>
        ///     Guards the scalar (0-d) sampling path: a box bounded on both sides must sample inside its bounds. This
        ///     is the only Box.Sample path the suite covers, so a regression in bounded uniform sampling surfaces here.
        /// </summary>
        [TestMethod]
        public void TestBoxBoundedSampling()
        {
            Box box = new Box(-5.0, 5.0);
            // NumSharp 0.70 made NDArray -> scalar conversions explicit and accepts them only for 0-d arrays; a scalar
            // box samples a 0-d array, so the cast reads the one value exactly.
            float sample = (float)box.Sample(null);
            Assert.IsTrue(sample >= -5.0 && sample <= 5.0, "Box sampling should be on the range [-5.0,5.0]");
        }
    }
}
