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

        [TestMethod]
        public void TestBoxBoundedSampling()
        {
            Box box = new Box(-5.0, 5.0);
            float sample = box.Sample(null);
            Assert.IsTrue(sample >= -5.0 && sample <= 5.0, "Box sampling should be on the range [-5.0,5.0]");
        }
    }
}
