#nullable enable
using System;
using System.Collections.Generic;
using Gym.Environments;
using Gym.Environments.Envs.Classic;
using Gym.Envs;
using Gym.Envs.Registration;
using Gym.Exceptions;
using Gym.Observations;
using Gym.Rendering;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using SixLabors.ImageSharp;
using Gym.Tests.Helpers;

namespace Gym.Tests.Envs.Classic {
    /// <summary>
    ///     Headless contract tests of the classic-control envs of P4 (<see cref="MountainCarEnv"/> in both modes,
    ///     <see cref="PendulumEnv"/>, <see cref="AcrobotEnv"/>) that need no Python: their registration and options records,
    ///     <c>repr</c>, the state attribute's lifecycle, determinism, render modes, the refusals Gym.NET documents, and the
    ///     public module functions. Byte-for-byte agreement with Gymnasium is the parity suite's job
    ///     (<c>tests/Gym.Tests.Parity/Tests/Envs</c>).
    /// </summary>
    [TestClass]
    public class ClassicControlP4Tests {
        /// <summary>
        ///     The four ids are registered with Gymnasium's step limits and reward thresholds, and <c>make</c> builds each over
        ///     the right class; the continuous id carries DR-32's <c>continuous=True</c>.
        /// </summary>
        [TestMethod]
        public void Registry_HoldsTheFourIds_WithGymnasiumsLimits() {
            GymEnvironments.Register();
            var expected = new (string Id, long? Steps, double? Threshold, Type Class)[] {
                ("MountainCar-v0", 200, -110.0, typeof(MountainCarEnv)),
                ("MountainCarContinuous-v0", 999, 90.0, typeof(MountainCarEnv)),
                ("Pendulum-v1", 200, null, typeof(PendulumEnv)),
                ("Acrobot-v1", 500, -100.0, typeof(AcrobotEnv)),
            };
            foreach ((string id, long? steps, double? threshold, Type type) in expected) {
                EnvSpec spec = Spec(id);
                Assert.AreEqual(steps, spec.MaxEpisodeSteps, $"{id} max_episode_steps");
                Assert.AreEqual(threshold, spec.RewardThreshold, $"{id} reward_threshold");
                using IEnv env = Make(id);
                Assert.IsInstanceOfType(env.Unwrapped, type, $"{id} builds a {type.Name}");
            }

            Assert.AreEqual(true, Spec("MountainCarContinuous-v0").Kwargs["continuous"]);
            Assert.IsTrue(((MountainCarEnv)Make("MountainCarContinuous-v0").Unwrapped).Continuous);
            Assert.IsFalse(((MountainCarEnv)Make("MountainCar-v0").Unwrapped).Continuous);
        }

        /// <summary>The generated options records carry each constructor's Python kwargs, and <c>make</c> takes them.</summary>
        [TestMethod]
        public void OptionsRecords_CarryThePythonKwargs() {
            // In constructor order: goal_velocity, then continuous.
            PyAssert.AreEqual(new Info { ["goal_velocity"] = 0.01, ["continuous"] = true }, new MountainCarEnvOptions { Continuous = true, GoalVelocity = 0.01 }.ToKwargs());
            PyAssert.AreEqual(new Info { ["g"] = 9.81 }, new PendulumEnvOptions { G = 9.81 }.ToKwargs());
            PyAssert.AreEqual(new Info { ["render_mode"] = "rgb_array" }, new AcrobotEnvOptions { RenderMode = "rgb_array" }.ToKwargs());

            using IEnv pendulum = Make("Pendulum-v1", new PendulumEnvOptions { G = 9.81 });
            Assert.AreEqual(9.81, ((PendulumEnv)pendulum.Unwrapped).Gravity);
        }

        /// <summary>
        ///     The merged MountainCar prints as Gymnasium's class of its mode, with and without a spec, inside <c>make</c>'s
        ///     stack too (DR-32).
        /// </summary>
        [TestMethod]
        public void MountainCar_PrintsAsGymnasiumsClassOfItsMode() {
            Assert.AreEqual("<MountainCarEnv instance>", new MountainCarEnv().ToString());
            Assert.AreEqual("<Continuous_MountainCarEnv instance>", new MountainCarEnv(continuous: true).ToString());
            using IEnv continuous = Make("MountainCarContinuous-v0");
            Assert.AreEqual("<TimeLimit<OrderEnforcing<PassiveEnvChecker<Continuous_MountainCarEnv<MountainCarContinuous-v0>>>>>", continuous.ToString());
            Assert.AreEqual("<PendulumEnv instance>", new PendulumEnv().ToString());
            Assert.AreEqual("<AcrobotEnv instance>", new AcrobotEnv().ToString());
        }

        /// <summary>
        ///     The state attribute's lifecycle in each env: absent before the first reset (Python's <c>AttributeError</c> on a
        ///     step; Acrobot's assert, as its <c>__init__</c> sets <c>None</c>), the values after it, and an assigned
        ///     <see langword="null"/> standing for Python's <c>None</c>.
        /// </summary>
        [TestMethod]
        public void State_LifecycleRaisesPythonsErrors() {
            var discrete = new MountainCarEnv();
            Assert.IsNull(discrete.State);
            Assert.AreEqual("'MountainCarEnv' object has no attribute 'state'", Assert.ThrowsExactly<AttributeError>(() => discrete.Step(NDArray.Scalar<long>(1))).Message);
            var continuous = new MountainCarEnv(continuous: true);
            Assert.AreEqual("'Continuous_MountainCarEnv' object has no attribute 'state'", Assert.ThrowsExactly<AttributeError>(() => continuous.Step(np.array(new[] { 0.5f }))).Message);

            discrete.Reset(seed: 0).Observation.Dispose();
            Assert.AreEqual(2, discrete.State!.Length);
            Assert.AreEqual(0.0, discrete.State[1]);
            discrete.State = null;
            Assert.AreEqual("cannot unpack non-iterable NoneType object", Assert.ThrowsExactly<TypeError>(() => discrete.Step(NDArray.Scalar<long>(1))).Message);

            var pendulum = new PendulumEnv();
            Assert.AreEqual("'PendulumEnv' object has no attribute 'state'", Assert.ThrowsExactly<AttributeError>(() => pendulum.Step(np.array(new[] { 0.5f }))).Message);
            pendulum.State = new[] { 0.1, 0.2, 0.3 };
            Assert.AreEqual("too many values to unpack (expected 2)", Assert.ThrowsExactly<ValueError>(() => pendulum.Step(np.array(new[] { 0.5f }))).Message);

            var acrobot = new AcrobotEnv();
            Assert.AreEqual("Call reset before using AcrobotEnv object.", Assert.ThrowsExactly<AssertionError>(() => acrobot.Step(NDArray.Scalar<long>(1))).Message);
            acrobot.Reset(seed: 0).Observation.Dispose();
            Assert.AreEqual(4, acrobot.State!.Length);
            // The reset state is float32 in Gymnasium (astype(np.float32)): its values are float32-exact.
            foreach (double value in acrobot.State)
                Assert.AreEqual(0.0, value - (float)value, $"{value:R} is not a float32 value.");
        }

        /// <summary>The same seed and actions give the same trajectory, observation for observation, in each env.</summary>
        [TestMethod]
        public void Episodes_AreDeterministicGivenTheSeed() {
            foreach (Func<IEnv> create in new Func<IEnv>[] { () => new MountainCarEnv(), () => new MountainCarEnv(continuous: true), () => new PendulumEnv(), () => new AcrobotEnv() }) {
                List<double> first = Trajectory(create(), 7), second = Trajectory(create(), 7);
                CollectionAssert.AreEqual(first, second, $"{create()} must replay its trajectory from the seed.");
            }
        }

        /// <summary>
        ///     <c>rgb_array</c> frames have each env's size; Pendulum returns its frame for any non-human mode (Gymnasium's
        ///     <c>else</c> branch) where MountainCar and Acrobot return nothing; Acrobot renders nothing before the first reset.
        /// </summary>
        [TestMethod]
        public void Render_RgbArray_ReturnsFramesOfEachEnvsSize() {
            AssertFrame(new MountainCarEnv(renderMode: "rgb_array"), 400, 600);
            AssertFrame(new MountainCarEnv(renderMode: "rgb_array", continuous: true), 400, 600);
            AssertFrame(new PendulumEnv(renderMode: "rgb_array"), 500, 500);
            AssertFrame(new AcrobotEnv(renderMode: "rgb_array"), 500, 500);

            var pendulum = new PendulumEnv(renderMode: "rgb_array_list");
            pendulum.Reset(seed: 0).Observation.Dispose();
            Assert.IsInstanceOfType(pendulum.Render(), typeof(RenderResult.Rgb), "Pendulum's render returns the frame for any mode but human.");
            var acrobot = new AcrobotEnv(renderMode: "rgb_array_list");
            acrobot.Reset(seed: 0).Observation.Dispose();
            Assert.IsTrue(acrobot.Render().IsNone, "Acrobot's render returns a frame only for rgb_array.");
            Assert.IsTrue(new AcrobotEnv(renderMode: "rgb_array").Render().IsNone, "Acrobot renders nothing before the first reset.");
        }

        /// <summary>
        ///     <c>human</c> mode renders every reset and step into the viewer from <c>viewer_factory</c>, paced by the clock, and
        ///     <c>Close</c> shuts it while keeping the screen, for each env.
        /// </summary>
        [TestMethod]
        public void Render_Human_RendersIntoTheFactorysViewer() {
            foreach (Func<EnvViewerFactory, IEnv> create in new Func<EnvViewerFactory, IEnv>[] {
                         factory => new MountainCarEnv(renderMode: "human", viewerFactory: factory),
                         factory => new PendulumEnv(renderMode: "human", viewerFactory: factory),
                         factory => new AcrobotEnv(renderMode: "human", viewerFactory: factory),
                     }) {
                var viewer = new CountingViewer();
                IEnv env = create((_, _, _) => viewer);
                // A high rate keeps the pacing short; Gymnasium's tests raise render_fps the same way.
                env.Metadata[Metadata.RenderFpsKey] = 1000L;
                (env.Reset(seed: 0).Observation as IDisposable)?.Dispose();
                using (var action = (NDArray)env.ActionSpace.Sample()!)
                    (env.Step(action).Observation as IDisposable)?.Dispose();
                Assert.IsTrue(env.Render().IsNone);
                Assert.AreEqual(3, viewer.Frames, $"{env}: reset, step and render each show a frame.");
                env.Close();
                Assert.IsTrue(viewer.Closed, $"{env}: close shuts the window.");
            }
        }

        /// <summary>A human render with no viewer factory anywhere raises Gymnasium's missing-pygame error (DR-20).</summary>
        [TestMethod]
        public void Render_HumanWithoutFactory_RaisesDependencyNotInstalled() {
            EnvViewerFactory? previous = HumanViewerFactory;
            HumanViewerFactory = null;
            try {
                Assert.ThrowsExactly<DependencyNotInstalled>(() => new MountainCarEnv(renderMode: "human").Reset(seed: 0));
                Assert.ThrowsExactly<DependencyNotInstalled>(() => new PendulumEnv(renderMode: "human").Reset(seed: 0));
                Assert.ThrowsExactly<DependencyNotInstalled>(() => new AcrobotEnv(renderMode: "human").Reset(seed: 0));
            } finally {
                HumanViewerFactory = previous;
            }
        }

        /// <summary>
        ///     The actions Gym.NET refuses rather than computing something else than Gymnasium: a continuous action of more than
        ///     one dimension, or of float16, for MountainCarContinuous and Pendulum.
        /// </summary>
        [TestMethod]
        public void Step_UnmodelledActions_AreRefused() {
            var continuous = new MountainCarEnv(continuous: true);
            continuous.Reset(seed: 0).Observation.Dispose();
            using (NDArray rows = np.array(new[,] { { 0.5f } }))
                Assert.ThrowsExactly<NotSupportedException>(() => continuous.Step(rows));
            using (NDArray half = np.array(new[] { (Half)0.5f }))
                Assert.ThrowsExactly<NotSupportedException>(() => continuous.Step(half));

            var pendulum = new PendulumEnv();
            pendulum.Reset(seed: 0).Observation.Dispose();
            using (NDArray rows = np.array(new[,] { { 0.5f } }))
                Assert.ThrowsExactly<NotSupportedException>(() => pendulum.Step(rows));
        }

        /// <summary>
        ///     An unsigned discrete action 0 wraps in NumPy's <c>action - 1</c>, so MountainCar pushes right at full speed (the
        ///     velocity clamps at 0.07), as Gymnasium's does.
        /// </summary>
        [TestMethod]
        public void MountainCar_UnsignedZeroAction_PushesRight() {
            var env = new MountainCarEnv();
            env.Reset(seed: 0).Observation.Dispose();
            using NDArray zero = NDArray.Scalar<byte>(0);
            env.Step(zero).Observation.Dispose();
            Assert.AreEqual(0.07, env.State![1]);
        }

        /// <summary>
        ///     The public module functions: Pendulum's <c>angle_normalize</c> into [-π, π), Acrobot's <c>wrap</c> by whole
        ///     periods and <c>bound</c> with NaN kept, and <c>rk4</c> on <c>dy/dt = y</c>, whose single step from 1 over 1 is
        ///     the classic 1 + 1 + 1/2 + 1/6 + 1/24.
        /// </summary>
        [TestMethod]
        public void ModuleFunctions_ComputeGymnasiumsResults() {
            Assert.AreEqual(-Math.PI, PendulumEnv.AngleNormalize(Math.PI));
            Assert.AreEqual(0.5, PendulumEnv.AngleNormalize(0.5), 1e-15);
            double below = 1.0 - 2 * Math.PI;
            // One period (M - m, exactly 2π) added once: the rounding of that one addition, as Python's loop rounds it.
            Assert.AreEqual(below + (Math.PI - -Math.PI), AcrobotEnv.Wrap(below, -Math.PI, Math.PI));
            Assert.AreEqual(4.0 - 2 * Math.PI, AcrobotEnv.Wrap(4.0, -Math.PI, Math.PI));
            Assert.AreEqual(2.0, AcrobotEnv.Bound(3.0, -2.0, 2.0));
            Assert.IsTrue(double.IsNaN(AcrobotEnv.Bound(double.NaN, -2.0, 2.0)));

            double[] e = AcrobotEnv.Rk4(y => (double[])y.Clone(), new[] { 1.0 }, new[] { 0.0, 1.0 });
            Assert.AreEqual(1.0 + 1.0 / 6.0 * (1.0 + 2 * 1.5 + 2 * 1.75 + 2.75), e[0]);
            Assert.AreEqual("operands could not be broadcast together with shapes (3,) (2,) ",
                Assert.ThrowsExactly<ValueError>(() => AcrobotEnv.Rk4(_ => new[] { 0.0, 0.0 }, new[] { 1.0, 2.0, 3.0 }, new[] { 0.0, 1.0 })).Message);
        }

        /// <summary>Pendulum's arrow is the embedded <c>clockwise.png</c> (DR-19): drawn after a step, with a NaN torque refused as pygame refuses its size.</summary>
        [TestMethod]
        public void Pendulum_Arrow_UsesTheEmbeddedImage() {
            var env = new PendulumEnv(renderMode: "rgb_array");
            env.Reset(seed: 0).Observation.Dispose();
            using (NDArray torque = np.array(new[] { 2.0f }))
                env.Step(torque).Observation.Dispose();
            Assert.AreEqual(2.0, env.LastU);
            AssertFrame(env, 500, 500, resetFirst: false);

            env.LastU = double.NaN;
            Assert.AreEqual("Cannot scale to negative size", Assert.ThrowsExactly<ValueError>(() => env.Render()).Message);
        }

        /// <summary>Runs 30 sampled steps from a seeded reset and lists every observation value and reward.</summary>
        /// <param name="env">The env; disposed here.</param>
        /// <param name="seed">The reset and action-space seed.</param>
        /// <returns>The values in order.</returns>
        private static List<double> Trajectory(IEnv env, int seed) {
            using (env) {
                var values = new List<double>();
                env.ActionSpace.Seed(seed);
                ResetResult<object?> reset = env.Reset(seed);
                using (var first = (NDArray)reset.Observation!)
                    values.AddRange(first.astype(np.float64).ToArray<double>());
                for (int i = 0; i < 30; i++) {
                    using var action = (NDArray)env.ActionSpace.Sample()!;
                    StepResult<object?> step = env.Step(action);
                    using (var observation = (NDArray)step.Observation!) {
                        using NDArray widened = observation.astype(np.float64);
                        values.AddRange(widened.ToArray<double>());
                    }

                    values.Add(step.Reward);
                }

                return values;
            }
        }

        /// <summary>Resets (unless told not to) and renders an <c>rgb_array</c> env, checking the frame's shape and dtype.</summary>
        /// <param name="env">The env, built with <c>render_mode="rgb_array"</c>.</param>
        /// <param name="height">The expected rows.</param>
        /// <param name="width">The expected columns.</param>
        /// <param name="resetFirst">Whether to reset with seed 0 first.</param>
        private static void AssertFrame(IEnv env, int height, int width, bool resetFirst = true) {
            if (resetFirst)
                (env.Reset(seed: 0).Observation as IDisposable)?.Dispose();
            var rgb = (RenderResult.Rgb)env.Render();
            using NDArray frame = rgb.Frame;
            CollectionAssert.AreEqual(new long[] { height, width, 3 }, frame.shape, $"{env}'s frame shape");
            Assert.AreEqual(NPTypeCode.Byte, frame.typecode);
        }

        /// <summary>A viewer that counts the frames it is shown and records being closed.</summary>
        private sealed class CountingViewer : IEnvViewer {
            /// <summary>The frames rendered so far.</summary>
            public int Frames { get; private set; }

            /// <summary>Whether <see cref="Close"/> ran.</summary>
            public bool Closed { get; private set; }

            /// <summary>Counts the frame.</summary>
            /// <param name="frame">The frame; not kept.</param>
            public void Render(NDArray frame) => Frames++;

            /// <summary>Records the close.</summary>
            public void Close() => Closed = true;

            /// <summary>Nothing to release.</summary>
            public void Dispose() { }
        }
    }
}
