#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using Gym.Environments;
using Gym.Envs;
using Gym.Envs.Registration;
using Gym.Observations;
using Gym.Spaces;
using Gym.Vector;
using Gym.Wrappers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Envs.Registration {
    /// <summary>
    ///     Pins the facade's render-mode and enum overloads of <c>Make</c> and <c>MakeVec</c> (DR-84, DR-89): the mode reaches
    ///     the env as Gymnasium's string and the spec records it so, the viewer factory reaches whoever opens the human window
    ///     (the env, or the <c>HumanRendering</c> wrapper make applies) and is never recorded, a mode given twice is Python's
    ///     duplicate keyword, and the vector forms take the vectorization and autoreset modes as enums.
    /// </summary>
    [TestClass]
    public class MakeRenderModeTests {
        /// <summary>Both spellings of the mode build the same env, and the spec records Gymnasium's string either way.</summary>
        [TestMethod]
        public void Make_TakesTheModeAsEnumOrString() {
            using (IEnv fromEnum = Make("CartPole-v1", RenderMode.RgbArray)) {
                Assert.AreEqual("rgb_array", fromEnum.RenderMode);
                Assert.AreEqual("rgb_array", fromEnum.Spec!.Kwargs["render_mode"]);
            }

            using (IEnv fromString = Make("CartPole-v1", "rgb_array")) {
                Assert.AreEqual("rgb_array", fromString.RenderMode);
                Assert.AreEqual("rgb_array", fromString.Spec!.Kwargs["render_mode"]);
            }

            using (Env<NDArray, NDArray> typed = Make<NDArray, NDArray>("CartPole-v1", RenderMode.RgbArray))
                Assert.AreEqual("rgb_array", typed.RenderMode);

            using (IEnv none = Make("CartPole-v1", (string?)null))
                Assert.IsNull(none.RenderMode, "A null string gives no mode, as leaving it out does.");

            // A mode Gymnasium accepts but no enum member names keeps working through the string (DR-93).
            using (IEnv custom = Make("CartPole-v1", "no-render-mode"))
                Assert.AreEqual("no-render-mode", custom.RenderMode);
        }

        /// <summary>
        ///     The factory reaches the env's human window: a step shows a frame in the viewer it made, and the spec keeps only
        ///     Gymnasium's kwargs (no delegate), so <c>Make(env.Spec)</c> falls back to the app-wide factory.
        /// </summary>
        [TestMethod]
        public void Make_PassesTheFactoryToTheEnv_WithoutRecordingIt() {
            var viewer = new CountingViewer();
            using IEnv env = Make("CartPole-v1", RenderMode.Human, viewer.Factory);
            env.Metadata[Metadata.RenderFpsKey] = 1000L;
            (env.Reset(seed: 0).Observation as IDisposable)?.Dispose();
            Assert.AreEqual(1, viewer.Frames, "The reset's human render showed one frame in the factory's viewer.");
            Assert.AreEqual("cartpole-v1", viewer.Title);
            Assert.IsFalse(env.Spec!.Kwargs.ContainsKey("viewer_factory"), "The spec records Gymnasium's kwargs only.");
            Assert.IsFalse(env.Unwrapped.Spec!.Kwargs.ContainsKey("viewer_factory"));
        }

        /// <summary>
        ///     For an env that renders only images, make applies <c>HumanRendering</c> and hands it the factory; the env itself,
        ///     whose constructor takes no <c>viewer_factory</c>, is built without it.
        /// </summary>
        [TestMethod]
        public void Make_HandsTheFactoryToTheHumanRenderingItApplies() {
            Register("test.RgbOnly-v0", typeof(RgbOnlyEnv));
            try {
                var viewer = new CountingViewer();
                using IEnv env = Make("test.RgbOnly-v0", RenderMode.Human, viewer.Factory);
                Assert.IsTrue(env is HumanRendering<NDArray, NDArray>, "make applies HumanRendering for human mode on an rgb_array-only env.");
                Assert.AreEqual("rgb_array", env.Unwrapped.RenderMode);
                (env.Reset(seed: 0).Observation as IDisposable)?.Dispose();
                Assert.AreEqual(1, viewer.Frames, "HumanRendering showed the reset's frame in the factory's viewer.");
                Assert.AreEqual("HumanRendering", viewer.Title);
            } finally {
                Registry.Remove("test.RgbOnly-v0");
            }
        }

        /// <summary>A mode given both as the argument and in the kwargs is Python's duplicate keyword.</summary>
        [TestMethod]
        public void Make_ModeGivenTwice_IsADuplicateKeyword() {
            Assert.AreEqual("make() got multiple values for keyword argument 'render_mode'",
                Assert.ThrowsExactly<NumSharp.TypeError>(() => Make("CartPole-v1", RenderMode.Human, kwargs: new { render_mode = "rgb_array" })).Message);
            Assert.AreEqual("make() got multiple values for keyword argument 'render_mode'",
                Assert.ThrowsExactly<NumSharp.TypeError>(() => Make("CartPole-v1", "human", kwargs: new Dictionary<string, object?> { ["render_mode"] = "human" })).Message);
        }

        /// <summary>
        ///     The vector forms take the vectorization and autoreset modes as enums: the vector env runs in the asked autoreset
        ///     mode, and the spec records the vectorization mode's value and the autoreset member, as Gymnasium's does.
        /// </summary>
        [TestMethod]
        public void MakeVec_TakesTheModesAsEnums() {
            using IVectorEnv envs = MakeVec("CartPole-v1", 2, VectorizeMode.Sync, AutoresetMode.SameStep);
            Assert.IsInstanceOfType<SyncVectorEnv>(envs);
            Assert.AreEqual((object)AutoresetMode.SameStep, envs.Metadata["autoreset_mode"]);
            Assert.AreEqual("sync", envs.Spec!.Kwargs["vectorization_mode"]);
            var vectorKwargs = (Info)envs.Spec.Kwargs["vector_kwargs"]!;
            Assert.AreEqual((object)AutoresetMode.SameStep, vectorKwargs["autoreset_mode"]);

            Assert.AreEqual("make_vec() got multiple values for keyword argument 'autoreset_mode'",
                Assert.ThrowsExactly<NumSharp.TypeError>(() => MakeVec("CartPole-v1", 2, VectorizeMode.Sync, AutoresetMode.SameStep,
                    vectorKwargs: new { autoreset_mode = "NextStep" })).Message);
        }

        /// <summary>The vector form's render mode reaches every sub-env, and its factory every sub-env's window.</summary>
        [TestMethod]
        public void MakeVec_PassesTheModeAndFactoryToEverySubEnv() {
            using (IVectorEnv rgb = MakeVec("CartPole-v1", 2, RenderMode.RgbArray, vectorizationMode: VectorizeMode.Sync))
                Assert.AreEqual("rgb_array", rgb.RenderMode);

            var viewer = new CountingViewer();
            using IVectorEnv human = MakeVec("CartPole-v1", 2, RenderMode.Human, viewer.Factory, VectorizeMode.Sync);
            foreach (IEnv sub in ((SyncVectorEnv)human).Envs)
                sub.Metadata[Metadata.RenderFpsKey] = 1000L;
            human.Reset(seed: 0).Dispose();
            Assert.AreEqual(2, viewer.Frames, "Each sub-env's reset showed one frame in the factory's viewer.");
        }

        /// <summary>A viewer that counts the frames it is shown and records the title it was opened with.</summary>
        private sealed class CountingViewer : IEnvViewer {
            /// <summary>The frames rendered so far.</summary>
            public int Frames { get; private set; }

            /// <summary>The title the factory was last given.</summary>
            public string? Title { get; private set; }

            /// <summary>The factory handing out this viewer, recording the title.</summary>
            /// <param name="width">The width; not checked.</param>
            /// <param name="height">The height; not checked.</param>
            /// <param name="title">The title.</param>
            /// <returns>This viewer.</returns>
            public IEnvViewer Factory(int width, int height, string? title = null) {
                Title = title;
                return this;
            }

            /// <summary>Counts the frame.</summary>
            /// <param name="frame">The frame; not kept.</param>
            public void Render(NDArray frame) => Frames++;

            /// <summary>Nothing to close.</summary>
            public void Close() { }

            /// <summary>Nothing to release.</summary>
            public void Dispose() { }
        }

        /// <summary>
        ///     An env that renders only <c>rgb_array</c> frames (a black 4×4 frame), with no <c>viewer_factory</c> parameter: the
        ///     kind make wraps in <c>HumanRendering</c> for human mode.
        /// </summary>
        public sealed class RgbOnlyEnv : Env<NDArray, NDArray> {
            /// <summary>Creates the env; its spaces are one-dimensional boxes.</summary>
            /// <param name="renderMode">The render mode, <c>rgb_array</c> or <see langword="null"/>.</param>
            public RgbOnlyEnv(string? renderMode = null) {
                ObservationSpace = new Box(-1, 1, new Shape(1L));
                ActionSpace = new Box(-1, 1, new Shape(1L));
                Metadata = ClassMetadata;
                RenderMode = renderMode;
            }

            /// <summary><c>{"render_modes": ["rgb_array"], "render_fps": 1000}</c>: images only, paced fast for the tests.</summary>
            public static new Metadata ClassMetadata => new() { [Metadata.RenderModesKey] = new List<string> { "rgb_array" }, [Metadata.RenderFpsKey] = 1000L };

            /// <summary>A zero observation.</summary>
            /// <param name="seed">Seeds the generator, as every reset does.</param>
            /// <param name="options">Ignored.</param>
            /// <returns>The observation and an empty info.</returns>
            /// <exception cref="Gym.Exceptions.Error">Thrown for a negative seed.</exception>
            public override ResetResult<NDArray> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) {
                SeedNpRandom(seed);
                return new ResetResult<NDArray>(np.zeros(new Shape(1L), np.float32), new Info());
            }

            /// <summary>A zero observation, never ending.</summary>
            /// <param name="action">Ignored.</param>
            /// <returns>The transition.</returns>
            public override StepResult<NDArray> Step(NDArray action) => new StepResult<NDArray>(np.zeros(new Shape(1L), np.float32), 0.0, false, false, new Info());

            /// <summary>A black 4×4 frame in <c>rgb_array</c> mode, nothing otherwise.</summary>
            /// <returns>The frame, the caller's.</returns>
            public override RenderResult Render() =>
                RenderMode == "rgb_array" ? new RenderResult.Rgb(np.zeros(new Shape(4, 4, 3), np.uint8)) : RenderResult.None;
        }
    }
}
