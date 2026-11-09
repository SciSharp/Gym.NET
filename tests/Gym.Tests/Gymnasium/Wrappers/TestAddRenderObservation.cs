#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using Gym.Envs;
using Gym.Exceptions;
using Gym.Spaces;
using Gym.Utils;
using Gym.Wrappers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Gymnasium.Wrappers {
    /// <summary>Gymnasium's <c>AddRenderObservation</c> tests (<c>test_add_render_observation.py</c>), run against <see cref="AddRenderObservation{TObs, TAct}"/>.</summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_add_render_observation.py">tests/wrappers/test_add_render_observation.py</see>.</remarks>
    [TestClass]
    public class TestAddRenderObservation {
        /// <summary>A render function returning a black 32x32 RGB frame (<c>image_render_func</c>).</summary>
        /// <param name="self">The env.</param>
        /// <returns>The frame.</returns>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_add_render_observation.py#L13">tests/wrappers/test_add_render_observation.py (image_render_func)</see>.</remarks>
        public static RenderResult ImageRenderFunc(GenericTestEnv self) => new RenderResult.Rgb(np.zeros(new Shape(32, 32, 3), np.uint8));

        /// <summary>A <c>Dict</c> observation gains the frame under its key, or is replaced by it.</summary>
        /// <param name="pixelsOnly">Whether the frame replaces the observation.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_add_render_observation.py#L18">tests/wrappers/test_add_render_observation.py (test_dict_observation)</see>.</remarks>
        [TestMethod]
        [DataRow(true)]
        [DataRow(false)]
        public void TestDictObservation(bool pixelsOnly) {
            const string pixelKey = "rgb";
            var env = new GenericTestEnv(
                observationSpace: new Dict(new Dictionary<string, Space> { ["state"] = new Box(-1, 1, new Shape(2), np.float32) }),
                renderMode: "rgb_array", renderFunc: ImageRenderFunc);
            // Make sure we are testing the right environment for the test.
            Assert.IsInstanceOfType(ObjectSpace.Unview(env.ObservationSpace), typeof(Dict));

            using var wrappedEnv = new AddRenderObservation<object?, object?>(env, renderOnly: pixelsOnly, renderKey: pixelKey);
            var (obs, _) = wrappedEnv.Reset();
            NDArray renderedObs = RenderedObs(obs, wrappedEnv, pixelsOnly, pixelKey, new[] { pixelKey, "state" });
            Assert.AreEqual(3, renderedObs.ndim);
            Assert.AreEqual(np.uint8, renderedObs.dtype);
        }

        /// <summary>A <c>Box</c> observation sits beside the frame in a dict (<c>{"depth", "state"}</c>), or is replaced by it.</summary>
        /// <param name="pixelsOnly">Whether the frame replaces the observation.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_add_render_observation.py#L61">tests/wrappers/test_add_render_observation.py (test_single_array_observation)</see>.</remarks>
        [TestMethod]
        [DataRow(true)]
        [DataRow(false)]
        public void TestSingleArrayObservation(bool pixelsOnly) {
            const string pixelKey = "depth";
            var env = new GenericTestEnv(observationSpace: new Box(-1, 1, new Shape(2), np.float32), renderMode: "rgb_array", renderFunc: ImageRenderFunc);
            Assert.IsInstanceOfType(ObjectSpace.Unview(env.ObservationSpace), typeof(Box));

            using var wrappedEnv = new AddRenderObservation<object?, object?>(env, renderOnly: pixelsOnly, renderKey: pixelKey);
            var (obs, _) = wrappedEnv.Reset();
            NDArray renderedObs = RenderedObs(obs, wrappedEnv, pixelsOnly, pixelKey, new[] { pixelKey, "state" });
            Assert.AreEqual(3, renderedObs.ndim);
            Assert.AreEqual(np.uint8, renderedObs.dtype);
        }

        /// <summary>
        ///     The saved constructor arguments are the constructor's own parameter names, so re-creating the wrapper from them
        ///     (as <c>make</c> does from a spec) works.
        /// </summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_add_render_observation.py#L99">tests/wrappers/test_add_render_observation.py (test_record_constructor_args_roundtrip)</see>.</remarks>
        [TestMethod]
        public void TestRecordConstructorArgsRoundtrip() {
            static GenericTestEnv MakeEnv() => new(observationSpace: new Box(-1, 1, new Shape(2), np.float32), renderMode: "rgb_array", renderFunc: ImageRenderFunc);

            using var wrappedEnv = new AddRenderObservation<object?, object?>(MakeEnv(), renderOnly: true);
            Info savedKwargs = wrappedEnv.SavedKwargs;
            CollectionAssert.AreEquivalent(new[] { "render_only", "render_key", "obs_key" }, savedKwargs.Keys.ToArray());

            // Reconstruction (as performed from the env spec) must not raise.
            var kwargs = new Info(savedKwargs) { ["env"] = MakeEnv() };
            using var rebuilt = (IEnv)PyKwargs.Construct(typeof(AddRenderObservation<,>), kwargs);
        }

        /// <summary>The checks both tests share: the space kind and the frame's place in the observation.</summary>
        /// <param name="obs">The wrapper's observation.</param>
        /// <param name="wrappedEnv">The wrapper.</param>
        /// <param name="pixelsOnly">Whether the frame replaced the observation.</param>
        /// <param name="pixelKey">The frame's key.</param>
        /// <param name="expectedKeys">The dict space's keys when the frame did not replace the observation.</param>
        /// <returns>The frame.</returns>
        private static NDArray RenderedObs(object? obs, IEnv wrappedEnv, bool pixelsOnly, string pixelKey, string[] expectedKeys) {
            Space space = ObjectSpace.Unview(wrappedEnv.ObservationSpace)!;
            if (pixelsOnly) {
                Assert.IsInstanceOfType(space, typeof(Box));
                Assert.IsInstanceOfType(obs, typeof(NDArray));
                return (NDArray)obs!;
            }

            Assert.IsInstanceOfType(space, typeof(Dict));
            CollectionAssert.AreEqual(expectedKeys, ((Dict)space).Keys.ToArray());
            Assert.IsInstanceOfType(obs, typeof(Info));
            return (NDArray)((Info)obs!)[pixelKey]!;
        }
    }
}
