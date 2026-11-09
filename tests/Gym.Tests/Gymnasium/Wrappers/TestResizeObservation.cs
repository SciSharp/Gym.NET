#nullable enable
using System.Linq;
using Gym.Envs;
using Gym.Exceptions;
using Gym.Spaces;
using Gym.Wrappers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Gymnasium.Wrappers {
    /// <summary>Gymnasium's <c>ResizeObservation</c> tests (<c>test_resize_observation.py</c>), run against <see cref="ResizeObservation{TObs, TAct}"/>.</summary>
    /// <remarks>
    ///     The cases on <c>CarRacing-v3</c> wait for P8 (Box2D) and are <c>[OpenBugs("P8")]</c>; the parity suite gates the
    ///     refusals they check over CartPole's frames instead.
    ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_resize_observation.py">tests/wrappers/test_resize_observation.py</see>.
    /// </remarks>
    [TestClass]
    public class TestResizeObservation {
        /// <summary>A random-image env of 3, 0 or 1 channels resized to 25 × 25: the wrapper's and the env's observations each in their own space only.</summary>
        /// <param name="imageShape"><c>60x60x3</c>, <c>60x60</c> or <c>60x60x1</c>.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_resize_observation.py#L39">tests/wrappers/test_resize_observation.py (test_resize_observation_wrapper)</see>.</remarks>
        [TestMethod]
        [DataRow("60x60x3")]
        [DataRow("60x60")]
        [DataRow("60x60x1")]
        public void TestResizeObservationWrapper(string imageShape) {
            long[] dimensions = imageShape.Split('x').Select(long.Parse).ToArray();
            var env = new GenericTestEnv(observationSpace: new Box(0, 255, new Shape(dimensions), np.uint8),
                resetFunc: WrapperUtils.RecordRandomObsReset, stepFunc: WrapperUtils.RecordRandomObsStep);

            using var wrappedEnv = new ResizeObservation<object?, object?>(env, new long[] { 25, 25 });
            Assert.IsInstanceOfType<Box>(ObjectSpace.Unview(wrappedEnv.ObservationSpace));
            CollectionAssert.AreEqual(new long[] { 25, 25 }, wrappedEnv.ObservationSpace.Shape!.Value.dimensions.Take(2).ToArray());

            var (obs, info) = wrappedEnv.Reset();
            WrapperUtils.CheckObs(env, wrappedEnv, obs, info["obs"]);

            (obs, _, _, _, info) = wrappedEnv.Step(null);
            WrapperUtils.CheckObs(env, wrappedEnv, obs, info["obs"]);
        }

        /// <summary>Resizing keeps the grayscale wrapper's choice of channel axis.</summary>
        /// <param name="keepDim">Whether the grayscale image keeps a channel axis.</param>
        /// <param name="height">The new height.</param>
        /// <param name="width">The new width.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_resize_observation.py#L55">tests/wrappers/test_resize_observation.py (test_resize_grayscale_observation)</see>.</remarks>
        [TestMethod]
        [DataRow(false, 3L, 4L)]
        [DataRow(false, 1L, 1L)]
        [DataRow(true, 3L, 4L)]
        [DataRow(true, 1L, 1L)]
        public void TestResizeGrayscaleObservation(bool keepDim, long height, long width) {
            var env = new GenericTestEnv(observationSpace: new Box(0, 255, new Shape(6, 8, 3), np.uint8),
                resetFunc: WrapperUtils.RecordRandomObsReset, stepFunc: WrapperUtils.RecordRandomObsStep);
            using var wrappedEnv = new ResizeObservation<NDArray, object?>(new GrayscaleObservation<object?, object?>(env, keepDim), new[] { height, width });
            long[] expectedShape = keepDim ? new[] { height, width, 1L } : new[] { height, width };
            CollectionAssert.AreEqual(expectedShape, wrappedEnv.ObservationSpace.Shape!.Value.dimensions);

            NDArray obs = wrappedEnv.Reset().Observation;
            CollectionAssert.AreEqual(expectedShape, obs.shape);
            Assert.AreEqual(np.uint8, obs.dtype);
            Assert.IsTrue(wrappedEnv.ObservationSpace.Contains(obs));

            obs = wrappedEnv.Step(null).Observation;
            CollectionAssert.AreEqual(expectedShape, obs.shape);
            Assert.AreEqual(np.uint8, obs.dtype);
            Assert.IsTrue(wrappedEnv.ObservationSpace.Contains(obs));
        }

        /// <summary>CarRacing's frames resized to four sizes stay in <c>Box(0, 255, shape + (3,), uint8)</c>.</summary>
        /// <param name="size">The new height and width.</param>
        /// <remarks>
        ///     CarRacing comes with P8.
        ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_resize_observation.py#L78">tests/wrappers/test_resize_observation.py (test_resize_shapes)</see>.
        /// </remarks>
        [TestMethod]
        [OpenBugs("P8")]
        [DataRow(10L)]
        [DataRow(20L)]
        [DataRow(60L)]
        [DataRow(100L)]
        public void TestResizeShapes(long size) {
            using var env = new ResizeObservation<NDArray, NDArray>((Env<NDArray, NDArray>)Make("CarRacing-v3"), new[] { size, size });
            using (var expected = new Box(0, 255, new Shape(size, size, 3), np.uint8))
                Assert.AreEqual(expected, env.ObservationSpace);

            NDArray obs = env.Reset().Observation;
            Assert.IsTrue(env.ObservationSpace.Contains(obs));
            obs = env.Step(env.ActionSpace.Sample()).Observation;
            Assert.IsTrue(env.ObservationSpace.Contains(obs));
        }

        /// <summary>Shapes of the wrong length or sign, a 1-D box and a <c>Tuple</c> space are refused.</summary>
        /// <remarks>
        ///     CarRacing comes with P8.
        ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_resize_observation.py#L90">tests/wrappers/test_resize_observation.py (test_invalid_input)</see>.
        /// </remarks>
        [TestMethod]
        [OpenBugs("P8")]
        public void TestInvalidInput() {
            using var env = (Env<NDArray, NDArray>)Make("CarRacing-v3");

            Assert.ThrowsExactly<ValueError>(() => new ResizeObservation<NDArray, NDArray>(env, new long[0]));
            Assert.ThrowsExactly<ValueError>(() => new ResizeObservation<NDArray, NDArray>(env, new long[] { 1 }));
            Assert.ThrowsExactly<ValueError>(() => new ResizeObservation<NDArray, NDArray>(env, new long[] { 1, 1, 1, 1 }));
            Assert.ThrowsExactly<ValueError>(() => new ResizeObservation<NDArray, NDArray>(env, new long[] { -1, 1 }));
            Assert.ThrowsExactly<ValueError>(() => new ResizeObservation<NDArray, NDArray>((Env<NDArray, NDArray>)Make("CartPole-v1"), new long[] { 1, 1 }));
            Assert.ThrowsExactly<TypeError>(() => new ResizeObservation<object?[], NDArray>((Env<object?[], NDArray>)Make("Blackjack-v1"), new long[] { 1, 1 }));
        }
    }
}
