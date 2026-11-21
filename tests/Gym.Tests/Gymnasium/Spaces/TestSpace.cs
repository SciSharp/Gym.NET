#nullable enable
using System;
using Gym.Spaces;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Gym.Tests.Gymnasium.Spaces {
    /// <summary>
    ///     Gymnasium's <c>Space</c> base tests (<c>tests/spaces/test_space.py</c>): a custom space that implements nothing
    ///     raises <c>NotImplementedError</c> from <c>sample</c>, <c>contains</c> and every <c>spaces.utils</c> function.
    /// </summary>
    /// <remarks>
    ///     C#'s abstract members make a custom space implement <c>Sample</c>, <c>Contains</c> and <c>IsNpFlattenable</c>;
    ///     <see cref="CustomSpace"/> implements them as Gymnasium's base class does, by raising.
    ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_space.py">tests/spaces/test_space.py</see>.
    /// </remarks>
    [TestClass]
    public class TestSpace {
        /// <summary>Each call on the custom space raises <c>NotImplementedError</c>.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_space.py#L9">test_not_implemented_errors</see>.</remarks>
        [TestMethod]
        public void TestNotImplementedErrors() {
            var calls = new (string Name, Action<CustomSpace> Call)[] {
                ("sample", space => space.Sample()),
                ("contains(None)", space => space.Contains(null)),
                ("flatdim", space => SpaceUtils.Flatdim(space)),
                ("flatten(None)", space => SpaceUtils.Flatten(space, null)),
                ("flatten_space", space => SpaceUtils.FlattenSpace(space)),
                ("unflatten(None)", space => SpaceUtils.Unflatten(space, null)),
            };
            foreach ((string name, Action<CustomSpace> call) in calls) {
                using var space = new CustomSpace();
                Assert.ThrowsExactly<NumSharp.NotImplementedError>(() => call(space), name);
            }
        }
    }
}
