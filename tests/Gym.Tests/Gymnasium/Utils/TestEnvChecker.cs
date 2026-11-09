#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using Gym.Environments;
using Gym.Envs;
using Gym.Observations;
using Gym.Spaces;
using Gym.Utils;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using Tuple = Gym.Spaces.Tuple;

namespace Gym.Tests.Gymnasium.Utils {
    /// <summary>
    ///     Gymnasium's env checker tests (<c>tests/utils/test_env_checker.py</c>): <c>check_env</c> passes the built-in and
    ///     generic envs without warnings, and each check catches the defect it is for, with Gymnasium's message.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <c>check_env</c> renders every declared mode, <c>human</c> included, so these tests run with
    ///     <see cref="NullEnvViewer.Factory"/> as the app-wide viewer (DR-20), where Gymnasium opens a pygame window.
    ///     </para>
    ///     <para>
    ///     Python-only parametrizations: reset functions without <c>seed</c> or <c>options</c> or with a seed default (C#
    ///     allows a <see cref="BigInteger"/>? default of <c>null</c> only); results that are lists or of the wrong length;
    ///     infos that are lists (the <see langword="null"/> info is tested instead); an object-dtype array in
    ///     <c>data_shares_objects</c> (NumSharp has no object dtype); a string where the env belongs (a
    ///     <see langword="null"/> env is tested instead). MountainCar-v0's case waits for P4.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_env_checker.py">tests/utils/test_env_checker.py</see>.</para>
    /// </remarks>
    [TestClass]
    public class TestEnvChecker {
        /// <summary>The warnings <c>check_env</c> may raise for the built-in envs (Gymnasium's <c>CHECK_ENV_IGNORE_WARNINGS</c>).</summary>
        private static readonly string[] CheckEnvIgnoreWarnings = new[] {
            "A Box observation space minimum value is -infinity. This is probably too low.",
            "A Box observation space maximum value is infinity. This is probably too high.",
            "For Box action spaces, we recommend using a symmetric and normalized space (range=[-1, 1] or [0, 1]). See https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html for more information.",
        }.Select(message => $"\u001b[33mWARN: {message}\u001b[0m").ToArray();

        /// <summary>CartPole and generic envs over composite spaces pass <c>check_env</c> with no warnings but the ignored ones.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_env_checker.py#L49">test_no_error_warnings</see>.</remarks>
        [TestMethod]
        public void TestNoErrorWarnings() {
            using var headless = new HeadlessHumanMode();
            var envs = new List<IEnv> { Make("CartPole-v1", disableEnvChecker: true).Unwrapped };
            envs.AddRange(GenericEnvsOverComposites());
            foreach (IEnv env in envs) {
                using (env) {
                    using WarningCatcher caught = Warnings.Catch();
                    EnvChecker.CheckEnv(env);
                    List<string> unexpected = caught.Records.Select(record => record.Message).Where(message => !CheckEnvIgnoreWarnings.Contains(message)).ToList();
                    Assert.AreEqual(0, unexpected.Count, $"{env}: {string.Join("\n", unexpected)}");
                }
            }
        }

        /// <summary>MountainCar-v0's case (P4).</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_env_checker.py#L49">test_no_error_warnings</see>.</remarks>
        [TestMethod]
        [OpenBugs("P4")]
        public void TestNoErrorWarnings_MountainCar() {
            using var headless = new HeadlessHumanMode();
            using IEnv env = Make("MountainCar-v0", disableEnvChecker: true).Unwrapped;
            using WarningCatcher caught = Warnings.Catch();
            EnvChecker.CheckEnv(env);
            Assert.AreEqual(0, caught.Records.Count(record => !CheckEnvIgnoreWarnings.Contains(record.Message)));
        }

        /// <summary>
        ///     The seed-determinism check catches a reset without <c>super().reset(seed=seed)</c> (no generator, or one that is
        ///     not reseeded) and a reset that reseeds with a fixed seed.
        /// </summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_env_checker.py#L148">test_check_reset_seed_determinism</see>.</remarks>
        [TestMethod]
        public void TestCheckResetSeedDeterminism() {
            using (var env = new GenericTestEnv(resetFunc: (self, seed, options) => new ResetResult<object?>(self.ObservationSpace.Sample(), new Info())))
                AssertRaises("Expects the random number generator to have been generated given a seed was passed to reset. Most likely the environment reset function does not call `super().reset(seed=seed)`.",
                    () => EnvChecker.CheckResetSeedDeterminism(env));
            using (var env = new GenericTestEnv(resetFunc: NoSuperReset))
                AssertRaises("Most likely the environment reset function does not call `super().reset(seed=seed)` as the random generates are not same when the same seeds are passed to `env.reset`.",
                    () => EnvChecker.CheckResetSeedDeterminism(env));
            using (var env = new GenericTestEnv(resetFunc: SuperResetFixed))
                AssertRaises("Most likely the environment reset function does not call `super().reset(seed=seed)` as the random number generators are not different when different seeds are passed to `env.reset`.",
                    () => EnvChecker.CheckResetSeedDeterminism(env));
        }

        /// <summary>A reset that reuses its buffer with a new value each time is caught, because the first observation is snapshotted.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_env_checker.py#L163">test_check_reset_seed_determinism_snapshots_reused_observations</see>.</remarks>
        [TestMethod]
        public void TestCheckResetSeedDeterminismSnapshotsReusedObservations() {
            var buffers = new ReusedBuffer();
            using var env = new GenericTestEnv(observationSpace: new Box(0, 100, new Shape(1L), dtype: np.float32), resetFunc: buffers.NondeterministicReset);
            StringAssert.Contains(Assert.ThrowsExactly<AssertionError>(() => EnvChecker.CheckResetSeedDeterminism(env)).Message,
                "Using `env.reset(seed=123)` is non-deterministic as the observations are not equivalent.");
        }

        /// <summary>The reset return check: an observation outside the space, or a missing info.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_env_checker.py#L236">test_check_reset_return_type</see>.</remarks>
        [TestMethod]
        public void TestCheckResetReturnType() {
            using (var env = new GenericTestEnv(resetFunc: (self, seed, options) => {
                       // sample + high: every entry beyond the bound.
                       using var sample = (NDArray)self.ObservationSpace.Sample()!;
                       return new ResetResult<object?>(sample + 1f, new Info());
                   }))
                AssertRaises("The first element returned by `env.reset()` is not within the observation space.", () => EnvChecker.CheckResetReturnType(env));
            using (var env = new GenericTestEnv(resetFunc: (self, seed, options) => new ResetResult<object?>(self.ObservationSpace.Sample(), null!)))
                AssertRaises("The second element returned by `env.reset()` was not a dictionary, actual type: <class 'NoneType'>", () => EnvChecker.CheckResetReturnType(env));
        }

        /// <summary>A reset that still takes <c>return_info</c> warns.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_env_checker.py#L258">test_check_reset_return_info_deprecation</see>.</remarks>
        [TestMethod]
        public void TestCheckResetReturnInfoDeprecation() {
            using var env = new ReturnInfoEnv();
            using WarningCatcher caught = Warnings.Catch();
            EnvChecker.CheckResetReturnInfoDeprecation(env);
            CollectionAssert.Contains(caught.Records.Select(record => record.Message).ToList(),
                "\u001b[33mWARN: `return_info` is deprecated as an optional argument to `reset`. `reset`should now always return `obs, info` where `obs` is an observation, and `info` is a dictionarycontaining additional information.\u001b[0m");
        }

        /// <summary>A <c>seed</c> function warns; a <c>seed</c> attribute that is not a function, or none, does not.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_env_checker.py#L273">test_check_seed_deprecation</see>.</remarks>
        [TestMethod]
        public void TestCheckSeedDeprecation() {
            const string message = "Official support for the `seed` function is dropped. Standard practice is to reset gymnasium environments using `env.reset(seed=<desired seed>)`";
            using var env = new GenericTestEnv();

            using (WarningCatcher caught = Warnings.Catch()) {
                PyAttributes.SetAttr(env, "seed", (Action<object?>)(_ => { }));
                EnvChecker.CheckSeedDeprecation(env);
                CollectionAssert.Contains(caught.Records.Select(record => record.Message).ToList(), $"\u001b[33mWARN: {message}\u001b[0m");
            }

            using (WarningCatcher caught = Warnings.Catch()) {
                PyAttributes.SetAttr(env, "seed", new List<object?>());
                EnvChecker.CheckSeedDeprecation(env);
                PyAttributes.SetAttr(env, "seed", 123L);
                EnvChecker.CheckSeedDeprecation(env);
                ((IDynamicAttributes)env).DynamicAttributes.Remove("seed");
                EnvChecker.CheckSeedDeprecation(env);
                Assert.AreEqual(0, caught.Records.Count, string.Join("\n", caught.Records));
            }
        }

        /// <summary>A step whose observation, reward or info is random despite the seed is caught.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_env_checker.py#L313">test_check_step_determinism</see>.</remarks>
        [TestMethod]
        public void TestCheckStepDeterminism() {
            // np.random.normal(): the global, unseeded stream.
            (Func<GenericTestEnv, object?, StepResult<object?>> Step, string Message)[] cases = {
                ((self, action) => new StepResult<object?>(Random.Shared.NextDouble(), 0, false, false, new Info()),
                    "Deterministic step observations are not equivalent for the same seed and action"),
                ((self, action) => new StepResult<object?>(0L, Random.Shared.NextDouble(), false, false, new Info()),
                    "Deterministic step rewards are not equivalent for the same seed and action"),
                ((self, action) => new StepResult<object?>(0L, 0, false, false, new Info { ["value"] = Random.Shared.NextDouble() }),
                    "Deterministic step info are not equivalent for the same seed and action"),
            };
            foreach ((Func<GenericTestEnv, object?, StepResult<object?>> step, string message) in cases) {
                using var env = new GenericTestEnv(stepFunc: step);
                AssertRaises(message, () => EnvChecker.CheckStepDeterminism(env));
            }
        }

        /// <summary>
        ///     The generator state after the step is compared even when both transitions are identical: a step that draws a
        ///     different amount the second time is caught unless the spec declares the env nondeterministic.
        /// </summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_env_checker.py#L316">test_check_step_determinism_rng_state</see>.</remarks>
        [TestMethod]
        public void TestCheckStepDeterminismRngState() {
            foreach (int secondDrawCount in new[] { 1, 2 }) {
                foreach (bool declaredNondeterministic in new[] { false, true }) {
                    var drawCounts = new Queue<int>(new[] { 1, secondDrawCount });
                    using var env = new GenericTestEnv(stepFunc: (self, action) => {
                        using (self.NpRandom.random(drawCounts.Dequeue())) { }
                        return new StepResult<object?>(np.zeros(new Shape(1L), np.float32), 0.0, false, false, new Info());
                    });
                    env.Spec!.Nondeterministic = declaredNondeterministic;
                    EnvChecker.CheckResetSeedDeterminism(env);

                    if (secondDrawCount != 1 && !declaredNondeterministic)
                        StringAssert.Contains(Assert.ThrowsExactly<AssertionError>(() => EnvChecker.CheckStepDeterminism(env)).Message, "not properly been updated after step");
                    else
                        EnvChecker.CheckStepDeterminism(env);
                }
            }
        }

        /// <summary>A step that changes a reused buffer is caught, because the first transition is snapshotted.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_env_checker.py#L341">test_check_step_determinism_snapshots_reused_observations</see>.</remarks>
        [TestMethod]
        public void TestCheckStepDeterminismSnapshotsReusedObservations() {
            var buffers = new ReusedBuffer();
            using var env = new GenericTestEnv(observationSpace: new Box(0, 100, new Shape(1L), dtype: np.float32), resetFunc: buffers.DeterministicReset,
                stepFunc: buffers.NondeterministicStep);
            AssertRaises("Deterministic step observations are not equivalent for the same seed and action", () => EnvChecker.CheckStepDeterminism(env));
        }

        /// <summary>A new view of a reused buffer is caught too: a different object over the same memory.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_env_checker.py#L396">test_check_returned_data_not_reused_buffer_view</see>.</remarks>
        [TestMethod]
        public void TestCheckReturnedDataNotReusedBufferView() {
            var buffers = new ReusedBuffer();
            using var env = new GenericTestEnv(observationSpace: new Box(0, 100, new Shape(1L), dtype: np.float32), resetFunc: buffers.ViewingReset, stepFunc: buffers.ViewingStep);
            StringAssert.Contains(Assert.ThrowsExactly<AssertionError>(() => EnvChecker.CheckReturnedDataNotReused(env)).Message, "share an object");
        }

        /// <summary>Distinct data is never reported as shared: equal arrays, numbers, strings, dicts, differing types and keys.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_env_checker.py#L420">test_data_shares_objects_distinct</see>.</remarks>
        [TestMethod]
        public void TestDataSharesObjectsDistinct() {
            using NDArray a = np.zeros(new Shape(3L)), b = np.zeros(new Shape(3L)), c = np.zeros(new Shape(3L)), d = np.zeros(new Shape(3L)), e = np.zeros(new Shape(3L));
            Assert.IsFalse(PassiveChecker.DataSharesObjects(a, b), "arrays");
            Assert.IsFalse(PassiveChecker.DataSharesObjects(0L, 0L), "ints");
            Assert.IsFalse(PassiveChecker.DataSharesObjects("abc", "abc"), "strs");
            Assert.IsFalse(PassiveChecker.DataSharesObjects(new Info { ["a"] = c }, new Info { ["a"] = d }), "dicts");
            // Differing types and keys mean there is nothing that can be shared.
            Assert.IsFalse(PassiveChecker.DataSharesObjects(e, 0L), "differing types");
            Assert.IsFalse(PassiveChecker.DataSharesObjects(new Info { ["a"] = a }, new Info { ["b"] = a }), "differing keys");
        }

        /// <summary>Every mutable container and a view of an array is detected as shared.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_env_checker.py#L426">test_data_shares_objects_shared</see>.</remarks>
        [TestMethod]
        public void TestDataSharesObjectsShared() {
            using NDArray array = np.zeros(new Shape(3L)), inner = np.zeros(new Shape(3L));
            using NDArray view = array[":"];
            var dictionary = new Info { ["a"] = inner };

            Assert.IsTrue(PassiveChecker.DataSharesObjects(array, array));
            Assert.IsTrue(PassiveChecker.DataSharesObjects(array, view));
            Assert.IsTrue(PassiveChecker.DataSharesObjects(dictionary, dictionary));
            Assert.IsTrue(PassiveChecker.DataSharesObjects(new Info { ["a"] = array }, new Info { ["a"] = array }));
            Assert.IsTrue(PassiveChecker.DataSharesObjects(new object?[] { array }, new object?[] { array }));
        }

        /// <summary>A reused observation buffer is rejected, naming the two calls.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_env_checker.py#L440">test_check_returned_data_not_reused</see>.</remarks>
        [TestMethod]
        public void TestCheckReturnedDataNotReused() {
            var buffers = new ReusedBuffer();
            using var env = new GenericTestEnv(observationSpace: new Box(0, 100, new Shape(1L), dtype: np.float32), resetFunc: buffers.DeterministicReset,
                stepFunc: buffers.NondeterministicStep);
            StringAssert.Contains(Assert.ThrowsExactly<AssertionError>(() => EnvChecker.CheckReturnedDataNotReused(env)).Message,
                "The observations returned by `env.reset() (call 1)` and `env.step() (call 2)` share an object");
        }

        /// <summary>A reused info dict is rejected the same way.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_env_checker.py#L458">test_check_returned_data_not_reused_info</see>.</remarks>
        [TestMethod]
        public void TestCheckReturnedDataNotReusedInfo() {
            var info = new Info();
            using var env = new GenericTestEnv(
                resetFunc: (self, seed, options) => {
                    self.BaseReset(seed);
                    info["count"] = 0L;
                    return new ResetResult<object?>(self.ObservationSpace.Sample(), info);
                },
                stepFunc: (self, action) => {
                    info["count"] = (long)info["count"]! + 1;
                    return new StepResult<object?>(self.ObservationSpace.Sample(), 0.0, false, false, info);
                });
            StringAssert.Contains(Assert.ThrowsExactly<AssertionError>(() => EnvChecker.CheckReturnedDataNotReused(env)).Message,
                "The infos returned by `env.reset() (call 1)` and `env.step() (call 2)` share an object");
        }

        /// <summary>The same immutable info value on every call is safe.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_env_checker.py#L474">test_check_returned_data_not_reused_immutable_info_value</see>.</remarks>
        [TestMethod]
        public void TestCheckReturnedDataNotReusedImmutableInfoValue() {
            object?[] constant = { "unchanging", 3L };
            using var env = new GenericTestEnv(
                resetFunc: (self, seed, options) => {
                    self.BaseReset(seed);
                    return new ResetResult<object?>(self.ObservationSpace.Sample(), new Info { ["constant"] = constant });
                },
                stepFunc: (self, action) => new StepResult<object?>(self.ObservationSpace.Sample(), 0, false, false, new Info { ["constant"] = constant }));
            EnvChecker.CheckReturnedDataNotReused(env);
        }

        /// <summary><c>check_env</c> refuses a missing env and envs without spaces, with Gymnasium's messages.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_env_checker.py#L490">test_check_env</see>.</remarks>
        [TestMethod]
        public void TestCheckEnv() {
            Assert.AreEqual("The environment must inherit from the gymnasium.Env class, actual class: <class 'NoneType'>. See https://gymnasium.farama.org/introduction/create_custom_env/ for more info.",
                Assert.ThrowsExactly<TypeError>(() => EnvChecker.CheckEnv(null!)).Message);
            using (var env = new GenericTestEnv(true, null, GenericTestEnv.DefaultSpace, TestingEnv.BasicResetFunc, TestingEnv.BasicStepFunc, TestingEnv.BasicRenderFunc))
                Assert.AreEqual("The environment must specify an action space. See https://gymnasium.farama.org/introduction/create_custom_env/ for more info.",
                    Assert.ThrowsExactly<AttributeError>(() => EnvChecker.CheckEnv(env)).Message);
            using (var env = new GenericTestEnv(true, GenericTestEnv.DefaultSpace, null, TestingEnv.BasicResetFunc, TestingEnv.BasicStepFunc, TestingEnv.BasicRenderFunc))
                Assert.AreEqual("The environment must specify an observation space. See https://gymnasium.farama.org/introduction/create_custom_env/ for more info.",
                    Assert.ThrowsExactly<AttributeError>(() => EnvChecker.CheckEnv(env)).Message);
        }

        /// <summary>Gymnasium's generic envs over a dict, a tuple and a nested dict of a discrete and a box space.</summary>
        /// <returns>The envs; the caller owns them.</returns>
        private static IEnumerable<IEnv> GenericEnvsOverComposites() {
            yield return new GenericTestEnv(observationSpace: new Dict(new Dictionary<string, Space> { ["a"] = new Discrete(10), ["b"] = UnitBox() }));
            yield return new GenericTestEnv(observationSpace: new Tuple(new Discrete(10), UnitBox()));
            yield return new GenericTestEnv(observationSpace: new Dict(new Dictionary<string, Space> {
                ["a"] = new Tuple(new Discrete(10), UnitBox()),
                ["b"] = UnitBox(),
            }));
        }

        /// <summary>A float32 box of 2 dimensions in <c>[0, 1]</c> (Gymnasium's <c>Box(np.zeros(2), np.ones(2))</c>).</summary>
        /// <returns>The box.</returns>
        private static Box UnitBox() {
            using NDArray low = np.zeros(new Shape(2L), np.float32);
            using NDArray high = np.ones(new Shape(2L), np.float32);
            return new Box(low, high);
        }

        /// <summary>Gymnasium's <c>_no_super_reset</c>: draws from a generator it never seeds, and returns a fixed-seed sample.</summary>
        /// <param name="self">The env.</param>
        /// <param name="seed">Ignored: never passed to the base reset.</param>
        /// <param name="options">Ignored.</param>
        /// <returns>The reset result.</returns>
        private static ResetResult<object?> NoSuperReset(GenericTestEnv self, BigInteger? seed, IReadOnlyDictionary<string, object?>? options) {
            // Generates a new generator on first use.
            using (self.NpRandom.random()) { }
            // A seed-deterministic result.
            self.ObservationSpace.Seed(0);
            return new ResetResult<object?>(self.ObservationSpace.Sample(), new Info());
        }

        /// <summary>Gymnasium's <c>_super_reset_fixed</c>: calls the base reset with a fixed seed, ignoring the one passed.</summary>
        /// <param name="self">The env.</param>
        /// <param name="seed">Ignored: seed 1 is always used.</param>
        /// <param name="options">Ignored.</param>
        /// <returns>The reset result.</returns>
        private static ResetResult<object?> SuperResetFixed(GenericTestEnv self, BigInteger? seed, IReadOnlyDictionary<string, object?>? options) {
            self.BaseReset(1);
            // A deterministic output: the space shares the env's generator.
            self.ObservationSpace.NpRandom = self.NpRandom;
            return new ResetResult<object?>(self.ObservationSpace.Sample(), new Info());
        }

        /// <summary>Asserts that the call raises <see cref="AssertionError"/> with exactly <paramref name="message"/>.</summary>
        /// <param name="message">The message.</param>
        /// <param name="call">The call.</param>
        private static void AssertRaises(string message, Action call) => Assert.AreEqual(message, Assert.ThrowsExactly<AssertionError>(call).Message);

        /// <summary>
        ///     The buffer-reusing reset and step functions of Gymnasium's tests: one float32 array of one element, returned
        ///     itself (or as a new view) and changed in place on every call.
        /// </summary>
        private sealed class ReusedBuffer {
            /// <summary>The reused buffer, created on the first reset.</summary>
            private NDArray? _buffer;

            /// <summary>Resets done, Gymnasium's <c>_reset_count</c>.</summary>
            private long _resetCount;

            /// <summary>Steps done, Gymnasium's <c>_step_count</c>.</summary>
            private long _stepCount;

            /// <summary>The buffer, created on first use.</summary>
            private NDArray Buffer => _buffer ??= np.zeros(new Shape(1L), np.float32);

            /// <summary>Gymnasium's <c>_buffer_reusing_nondeterministic_reset</c>: the buffer, set to the reset count.</summary>
            /// <param name="self">The env.</param>
            /// <param name="seed">The seed, for the base reset.</param>
            /// <param name="options">Ignored.</param>
            /// <returns>The buffer and an empty info.</returns>
            public ResetResult<object?> NondeterministicReset(GenericTestEnv self, BigInteger? seed, IReadOnlyDictionary<string, object?>? options) {
                self.BaseReset(seed);
                _resetCount++;
                Buffer.SetAtIndex((float)_resetCount, 0);
                return new ResetResult<object?>(Buffer, new Info());
            }

            /// <summary>Gymnasium's <c>_buffer_reusing_deterministic_reset</c>: the buffer, set to 0.</summary>
            /// <param name="self">The env.</param>
            /// <param name="seed">The seed, for the base reset.</param>
            /// <param name="options">Ignored.</param>
            /// <returns>The buffer and an empty info.</returns>
            public ResetResult<object?> DeterministicReset(GenericTestEnv self, BigInteger? seed, IReadOnlyDictionary<string, object?>? options) {
                self.BaseReset(seed);
                Buffer.SetAtIndex(0f, 0);
                return new ResetResult<object?>(Buffer, new Info());
            }

            /// <summary>Gymnasium's <c>_buffer_reusing_nondeterministic_step</c>: the buffer, set to the step count.</summary>
            /// <param name="self">The env.</param>
            /// <param name="action">Ignored.</param>
            /// <returns>The buffer, reward 0, no signals, an empty info.</returns>
            public StepResult<object?> NondeterministicStep(GenericTestEnv self, object? action) {
                _stepCount++;
                Buffer.SetAtIndex((float)_stepCount, 0);
                return new StepResult<object?>(Buffer, 0.0, false, false, new Info());
            }

            /// <summary>Gymnasium's <c>_buffer_viewing_reset</c>: a new view of the buffer, set to 0.</summary>
            /// <param name="self">The env.</param>
            /// <param name="seed">The seed, for the base reset.</param>
            /// <param name="options">Ignored.</param>
            /// <returns>The view and an empty info.</returns>
            public ResetResult<object?> ViewingReset(GenericTestEnv self, BigInteger? seed, IReadOnlyDictionary<string, object?>? options) {
                self.BaseReset(seed);
                Buffer.SetAtIndex(0f, 0);
                return new ResetResult<object?>(Buffer[":"], new Info());
            }

            /// <summary>Gymnasium's <c>_buffer_viewing_step</c>: the buffer incremented, as a new view.</summary>
            /// <param name="self">The env.</param>
            /// <param name="action">Ignored.</param>
            /// <returns>The view, reward 0, no signals, an empty info.</returns>
            public StepResult<object?> ViewingStep(GenericTestEnv self, object? action) {
                Buffer.SetAtIndex(Buffer.GetSingle(0) + 1f, 0);
                return new StepResult<object?>(Buffer[":"], 0.0, false, false, new Info());
            }
        }

        /// <summary>
        ///     Makes <see cref="NullEnvViewer.Factory"/> the app-wide human viewer for a test (DR-20: headless tests pass it
        ///     explicitly), restoring the previous one when disposed.
        /// </summary>
        private sealed class HeadlessHumanMode : IDisposable {
            /// <summary>The factory before the test.</summary>
            private readonly IEnvironmentViewerFactoryDelegate? _previous;

            /// <summary>Installs the headless viewer.</summary>
            public HeadlessHumanMode() {
                _previous = HumanViewerFactory;
                HumanViewerFactory = NullEnvViewer.Factory;
            }

            /// <summary>Restores the previous factory.</summary>
            public void Dispose() => HumanViewerFactory = _previous;
        }

        /// <summary>An env whose reset still has the pre-0.26 <c>return_info</c> overload (Gymnasium's <c>_deprecated_return_info</c>).</summary>
        private sealed class ReturnInfoEnv : Env<object?, object?> {
            /// <summary>Unit box spaces.</summary>
            public ReturnInfoEnv() {
                ((IEnv)this).ActionSpace = UnitBox();
                ((IEnv)this).ObservationSpace = UnitBox();
            }

            /// <summary>The deprecated reset: an observation, with the info only when asked.</summary>
            /// <param name="returnInfo">Gymnasium's <c>return_info</c>.</param>
            /// <returns>The observation, or the observation and info.</returns>
            public object? Reset(bool returnInfo) => returnInfo ? new object?[] { ObservationSpace.Sample(), new Info() } : ObservationSpace.Sample();

            /// <summary>The current API's reset.</summary>
            /// <param name="seed">The seed.</param>
            /// <param name="options">Ignored.</param>
            /// <returns>A sample and an empty info.</returns>
            public override ResetResult<object?> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) {
                SeedNpRandom(seed);
                return new ResetResult<object?>(ObservationSpace.Sample(), new Info());
            }

            /// <summary>A sample, reward 0, no signals.</summary>
            /// <param name="action">Ignored.</param>
            /// <returns>The transition.</returns>
            public override StepResult<object?> Step(object? action) => new(ObservationSpace.Sample(), 0, false, false, new Info());
        }
    }
}
