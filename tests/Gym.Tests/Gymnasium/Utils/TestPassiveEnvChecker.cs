#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Gym.Envs;
using Gym.Observations;
using Gym.Spaces;
using Gym.Utils;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using Tuple = Gym.Spaces.Tuple;

namespace Gym.Tests.Gymnasium.Utils {
    /// <summary>
    ///     Gymnasium's passive checker tests (<c>tests/utils/test_passive_env_checker.py</c>): the space checks, the
    ///     observation checks, the reset, step and render checks, and the data-reuse check.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Gymnasium mutates spaces to make them invalid (<c>space.n = -1</c>); Gym.NET's spaces are immutable, so
    ///     <see cref="ModifySpace"/> writes the property's backing field instead, the same effect for the checks.
    ///     </para>
    ///     <para>
    ///     Python-only parametrizations, which C#'s types make impossible: a string where a space belongs; reset functions
    ///     without <c>seed</c> or <c>options</c>, or with a seed default; results that are not tuples, of the wrong length, of
    ///     the old 4-tuple API, or with non-boolean signals or non-numeric rewards; infos that are not dicts (the reachable
    ///     counterpart, a <see langword="null"/> info, is tested instead).
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_passive_env_checker.py">tests/utils/test_passive_env_checker.py</see>.</para>
    /// </remarks>
    [TestClass]
    public class TestPassiveEnvChecker {
        /// <summary>The space checks of an observation space: equal box bounds warn, mismatched bound shapes and invalid spaces assert.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_passive_env_checker.py#L92">test_check_observation_space</see>.</remarks>
        [TestMethod]
        public void TestCheckObservationSpace() {
            AssertWarns("A Box observation space maximum and minimum values are equal.", () => PassiveChecker.CheckObservationSpace(ZeroBox(5)));
            AssertRaises<AssertionError>("The Box observation space shape and low shape have different shapes, low shape: (3,), box shape: (2,)",
                () => PassiveChecker.CheckObservationSpace(ModifySpace(UnitBox(), "Low", np.zeros(new Shape(3L), np.float32))));
            AssertRaises<AssertionError>("The Box observation space shape and high shape have have different shapes, high shape: (3,), box shape: (2,)",
                () => PassiveChecker.CheckObservationSpace(ModifySpace(UnitBox(), "High", np.ones(new Shape(3L), np.float32))));
            AssertRaises<AssertionError>("Discrete observation space's number of elements must be positive, actual number of elements: -1",
                () => PassiveChecker.CheckObservationSpace(ModifySpace(new Discrete(5), "N", -1L)));
            AssertRaises<AssertionError>("Multi-discrete observation space's all nvec elements must be greater than 0, actual nvec: [ 2 -1]",
                () => PassiveChecker.CheckObservationSpace(ModifySpace(new MultiDiscrete(new long[] { 2, 2 }), "Nvec", np.array(new long[] { 2, -1 }))));
            AssertRaises<AssertionError>("Multi-discrete observation space's shape must be equal to the nvec shape, space shape: (2, 1, 2), nvec shape: (2,)",
                () => PassiveChecker.CheckObservationSpace(ModifySpace(new MultiDiscrete(new long[] { 2, 2 }), "Shape", (Shape?)new Shape(2L, 1L, 2L))));
            AssertRaises<AssertionError>("Multi-binary observation space's all shape elements must be greater than 0, actual shape: (2, -1)",
                () => PassiveChecker.CheckObservationSpace(ModifySpace(new MultiBinary(new long[] { 2, 2 }), "Shape", (Shape?)UncheckedShape(2, -1))));
            AssertRaises<AssertionError>("An empty Tuple observation space is not allowed.", () => PassiveChecker.CheckObservationSpace(new Tuple(Array.Empty<Space>())));
            AssertRaises<AssertionError>("An empty Dict observation space is not allowed.", () => PassiveChecker.CheckObservationSpace(new Dict(new Dictionary<string, Space>())));
            // Gymnasium passes a string; a missing space is C#'s counterpart.
            AssertRaises<TypeError>("observation space does not inherit from `gymnasium.spaces.Space`, actual type: <class 'NoneType'>",
                () => PassiveChecker.CheckObservationSpace(null));
        }

        /// <summary>The space checks of an action space, with the action messages (whose "have have" sits on the low bound).</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_passive_env_checker.py#L162">test_check_action_space</see>.</remarks>
        [TestMethod]
        public void TestCheckActionSpace() {
            AssertWarns("A Box action space maximum and minimum values are equal.", () => PassiveChecker.CheckActionSpace(ZeroBox(5)));
            AssertRaises<AssertionError>("The Box action space shape and low shape have have different shapes, low shape: (3,), box shape: (2,)",
                () => PassiveChecker.CheckActionSpace(ModifySpace(UnitBox(), "Low", np.zeros(new Shape(3L), np.float32))));
            AssertRaises<AssertionError>("The Box action space shape and high shape have different shapes, high shape: (3,), box shape: (2,)",
                () => PassiveChecker.CheckActionSpace(ModifySpace(UnitBox(), "High", np.ones(new Shape(3L), np.float32))));
            AssertRaises<AssertionError>("Discrete action space's number of elements must be positive, actual number of elements: -1",
                () => PassiveChecker.CheckActionSpace(ModifySpace(new Discrete(5), "N", -1L)));
            AssertRaises<AssertionError>("Multi-discrete action space's shape must be equal to the nvec shape, space shape: (2, -1), nvec shape: (2,)",
                () => PassiveChecker.CheckActionSpace(ModifySpace(new MultiDiscrete(new long[] { 2, 2 }), "Shape", (Shape?)UncheckedShape(2, -1))));
            AssertRaises<AssertionError>("Multi-binary action space's all shape elements must be greater than 0, actual shape: (2, -1)",
                () => PassiveChecker.CheckActionSpace(ModifySpace(new MultiBinary(new long[] { 2, 2 }), "Shape", (Shape?)UncheckedShape(2, -1))));
            AssertRaises<AssertionError>("An empty Tuple action space is not allowed.", () => PassiveChecker.CheckActionSpace(new Tuple(Array.Empty<Space>())));
            AssertRaises<AssertionError>("An empty Dict action space is not allowed.", () => PassiveChecker.CheckActionSpace(new Dict(new Dictionary<string, Space>())));
            AssertRaises<TypeError>("action space does not inherit from `gymnasium.spaces.Space`, actual type: <class 'NoneType'>", () => PassiveChecker.CheckActionSpace(null));
        }

        /// <summary>The observation checks: a value outside the space or of another type warns; malformed tuples and dicts assert.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_passive_env_checker.py#L219">test_check_obs</see>.</remarks>
        [TestMethod]
        public void TestCheckObs() {
            using (var space = new Discrete(2))
                AssertWarns("The obs returned by the `testing()` method is not within the observation space.", () => PassiveChecker.CheckObs(3L, space, "testing"));
            using (var space = new Discrete(1))
            using (NDArray uint8Zero = NDArray.Scalar((byte)0))
                AssertWarns("The obs returned by the `testing()` method should be an int or np.int64, actual type: <class 'numpy.uint8'>", () => PassiveChecker.CheckObs(uint8Zero, space, "testing"));
            using (var space = new Tuple(new Discrete(1), new Discrete(2)))
                AssertWarns("The obs returned by the `testing()` method was expecting a tuple, actual type: <class 'list'>",
                    () => PassiveChecker.CheckObs(new List<object?> { 0L, 1L }, space, "testing"));
            using (var space = new Tuple(new Discrete(1), new Discrete(2)))
                AssertRaises<AssertionError>("The obs returned by the `testing()` method length is not same as the observation space length, obs length: 3, space length: 2",
                    () => PassiveChecker.CheckObs(new object?[] { 1L, 2L, 3L }, space, "testing"));
            using (var space = new Dict(new Dictionary<string, Space> { ["a"] = new Discrete(1), ["b"] = new Discrete(2) }))
                AssertRaises<AssertionError>("The obs returned by the `testing()` method must be a dict, actual type: <class 'set'>",
                    () => PassiveChecker.CheckObs(new HashSet<long> { 1, 2, 3 }, space, "testing"));
            using (var space = new Dict(new Dictionary<string, Space> { ["a"] = new Discrete(1), ["b"] = new Discrete(2) }))
                AssertRaises<AssertionError>("The obs returned by the `testing()` method observation keys is not same as the observation space keys, obs keys: ['a', 'c'], space keys: ['a', 'b']",
                    () => PassiveChecker.CheckObs(new Info { ["a"] = 1L, ["c"] = 2L }, space, "testing"));
        }

        /// <summary>The reset check: a <see langword="null"/> info asserts (Gymnasium's case returns a set as the info).</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_passive_env_checker.py#L287">test_passive_env_reset_checker</see>.</remarks>
        [TestMethod]
        public void TestPassiveEnvResetChecker() {
            using var env = new GenericTestEnv(resetFunc: (self, seed, options) => new ResetResult<object?>(np.array(new[] { 0f }), null!));
            AssertRaises<AssertionError>("The second element returned by `env.reset()` was not a dictionary, actual type: <class 'NoneType'>",
                () => PassiveChecker.EnvResetPassiveChecker(env));
        }

        /// <summary>The step check: a NaN or infinite reward warns, a <see langword="null"/> info asserts.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_passive_env_checker.py#L371">test_passive_env_step_checker</see>.</remarks>
        [TestMethod]
        public void TestPassiveEnvStepChecker() {
            using (var env = new GenericTestEnv(stepFunc: (self, action) => ModifiedStep(self, reward: double.NaN)))
                AssertWarns("The reward is a NaN value.", () => PassiveChecker.EnvStepPassiveChecker(env, (object?)0L));
            using (var env = new GenericTestEnv(stepFunc: (self, action) => ModifiedStep(self, reward: double.PositiveInfinity)))
                AssertWarns("The reward is an inf value.", () => PassiveChecker.EnvStepPassiveChecker(env, (object?)0L));
            using (var env = new GenericTestEnv(stepFunc: (self, action) => ModifiedStep(self, missingInfo: true)))
                AssertRaises<AssertionError>("The `info` returned by `step()` must be a python dictionary, actual type: <class 'NoneType'>",
                    () => PassiveChecker.EnvStepPassiveChecker(env, (object?)0L));
        }

        /// <summary>The render check's metadata warnings and render-mode assertions, with Gymnasium's messages.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_passive_env_checker.py#L445">test_passive_render_checker</see>.</remarks>
        [TestMethod]
        public void TestPassiveRenderChecker() {
            using (var env = new GenericTestEnv(metadata: new Metadata { ["render_modes"] = null }))
                AssertWarns("No render modes was declared in the environment (env.metadata['render_modes'] is None or not defined), you may have trouble when calling `.render()`.",
                    () => PassiveChecker.EnvRenderPassiveChecker(env));
            using (var env = new GenericTestEnv(metadata: new Metadata { ["render_modes"] = "Testing mode", ["render_fps"] = 1L }))
                AssertWarns("Expects the render_modes to be a sequence (i.e. list, tuple), actual type: <class 'str'>", () => PassiveChecker.EnvRenderPassiveChecker(env));
            using (var env = new GenericTestEnv(metadata: new Metadata { ["render_modes"] = new List<object?> { "Testing mode", 1L }, ["render_fps"] = 1L }))
                AssertWarns("Expects all render modes to be strings, actual types: [<class 'str'>, <class 'int'>]", () => PassiveChecker.EnvRenderPassiveChecker(env));
            using (var env = new GenericTestEnv(metadata: new Metadata { ["render_modes"] = new List<string> { "Testing mode" }, ["render_fps"] = null },
                       renderMode: "Testing mode", renderFunc: self => RenderResult.None))
                AssertWarns("No render fps was declared in the environment (env.metadata['render_fps'] is None or not defined), rendering may occur at inconsistent fps.",
                    () => PassiveChecker.EnvRenderPassiveChecker(env));
            using (var env = new GenericTestEnv(metadata: new Metadata { ["render_modes"] = new List<string> { "Testing mode" }, ["render_fps"] = "fps" }))
                AssertWarns("Expects the `env.metadata['render_fps']` to be an integer or a float, actual type: <class 'str'>", () => PassiveChecker.EnvRenderPassiveChecker(env));
            using (var env = new GenericTestEnv(metadata: new Metadata { ["render_modes"] = new List<string>(), ["render_fps"] = 30L }, renderMode: "Test"))
                AssertRaises<AssertionError>("With no render_modes, expects the Env.render_mode to be None, actual value: Test", () => PassiveChecker.EnvRenderPassiveChecker(env));
            using (var env = new GenericTestEnv(metadata: new Metadata { ["render_modes"] = new List<string> { "Testing mode" }, ["render_fps"] = 30L }, renderMode: "Non mode"))
                AssertRaises<AssertionError>("The environment was initialized successfully however with an unsupported render mode. Render mode: Non mode, modes: ['Testing mode']",
                    () => PassiveChecker.EnvRenderPassiveChecker(env));
        }

        /// <summary>Sharing an observation or info object between two calls warns: the same array, a new view of it, a shared dict value.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_passive_env_checker.py#L459">test_env_data_reuse_passive_checker</see>.</remarks>
        [TestMethod]
        public void TestEnvDataReusePassiveChecker() {
            using NDArray buffer = np.zeros(new Shape(3L));
            AssertWarnsContaining("The observations returned by `reset` and the following `step` share an object",
                () => PassiveChecker.EnvDataReusePassiveChecker((buffer, new Info()), (buffer, new Info()), "reset", "step"));

            // A new view of a reused buffer is a different object, but modifying it modifies the other.
            using (NDArray view1 = buffer[":"])
            using (NDArray view2 = buffer[":"])
                AssertWarnsContaining("The observations returned by `step` and the following `step` share an object",
                    () => PassiveChecker.EnvDataReusePassiveChecker((view1, new Info()), (view2, new Info()), "step", "step"));

            using NDArray shared = np.zeros(new Shape(3L)), fresh = np.zeros(new Shape(3L)), other = np.zeros(new Shape(3L));
            using NDArray fresh1 = np.zeros(new Shape(3L)), fresh2 = np.zeros(new Shape(3L));
            AssertWarnsContaining("The infos returned by `reset` and the following `step` share an object",
                () => PassiveChecker.EnvDataReusePassiveChecker((fresh, new Info { ["shared"] = shared, ["fresh"] = fresh1 }),
                    (other, new Info { ["shared"] = shared, ["fresh"] = fresh2 }), "reset", "step"));
        }

        /// <summary>Distinct observation and info data never warns.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_passive_env_checker.py#L493">test_env_data_reuse_passive_checker_distinct_data</see>.</remarks>
        [TestMethod]
        public void TestEnvDataReusePassiveCheckerDistinctData() {
            using NDArray obs1 = np.zeros(new Shape(3L)), obs2 = np.zeros(new Shape(3L)), a1 = np.zeros(new Shape(3L)), a2 = np.zeros(new Shape(3L));
            using WarningCatcher caught = Warnings.Catch();
            PassiveChecker.EnvDataReusePassiveChecker((obs1, new Info { ["a"] = a1, ["b"] = 0L }), (obs2, new Info { ["a"] = a2, ["b"] = 0L }), "reset", "step");
            Assert.AreEqual(0, caught.Records.Count, string.Join("\n", caught.Records));
        }

        /// <summary>
        ///     Python's <c>setattr(space, attribute, value)</c> on a Gym.NET space, whose attributes are read-only: writes the
        ///     auto-property's backing field, found on the space's class or a base, so the checks see an invalid space.
        /// </summary>
        /// <typeparam name="TSpace">The space's type.</typeparam>
        /// <param name="space">The space, modified in place.</param>
        /// <param name="property">The C# property (<c>N</c>, <c>Nvec</c>, <c>Low</c>, <c>Shape</c>).</param>
        /// <param name="value">The new value.</param>
        /// <returns>The space.</returns>
        /// <exception cref="InvalidOperationException">Thrown when no class declares a backing field for the property.</exception>
        private static TSpace ModifySpace<TSpace>(TSpace space, string property, object? value) where TSpace : Space {
            for (Type? level = space.GetType(); level != null; level = level.BaseType) {
                FieldInfo? field = level.GetField($"<{property}>k__BackingField", BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.DeclaredOnly);
                if (field is null)
                    continue;
                field.SetValue(space, value);
                return space;
            }

            throw new InvalidOperationException($"{space.GetType().Name} has no backing field for {property}");
        }

        /// <summary>
        ///     A shape holding dimensions as given, negative ones included, as Python's unchecked tuple <c>(2, -1)</c>: built
        ///     with valid dimensions, then its dimension array (shared, not copied, by the struct) overwritten.
        /// </summary>
        /// <param name="dimensions">The dimensions.</param>
        /// <returns>The shape; only its dimensions are meaningful.</returns>
        private static Shape UncheckedShape(params long[] dimensions) {
            var shape = new Shape(dimensions.Select(_ => 1L).ToArray());
            Array.Copy(dimensions, shape.dimensions, dimensions.Length);
            return shape;
        }

        /// <summary>A float32 box of <paramref name="size"/> dimensions whose bounds are all 0 (Gymnasium's <c>Box(zeros, zeros)</c>).</summary>
        /// <param name="size">The dimension count.</param>
        /// <returns>The box.</returns>
        private static Box ZeroBox(long size) {
            using NDArray zeros = np.zeros(new Shape(size), np.float32);
            return new Box(zeros, zeros);
        }

        /// <summary>A float32 box of 2 dimensions in <c>[0, 1]</c>.</summary>
        /// <returns>The box.</returns>
        private static Box UnitBox() {
            using NDArray low = np.zeros(new Shape(2L), np.float32);
            using NDArray high = np.ones(new Shape(2L), np.float32);
            return new Box(low, high);
        }

        /// <summary>
        ///     Gymnasium's <c>_modified_step</c>: a sample of the observation space with the given reward and an empty info, or
        ///     none at all, the reachable counterpart of Gymnasium's info that is not a dict.
        /// </summary>
        /// <param name="self">The env.</param>
        /// <param name="reward">The reward.</param>
        /// <param name="missingInfo">Return a <see langword="null"/> info instead of <c>{}</c>.</param>
        /// <returns>The transition.</returns>
        private static StepResult<object?> ModifiedStep(GenericTestEnv self, double reward = 0, bool missingInfo = false) =>
            new(self.ObservationSpace.Sample(), reward, false, false, missingInfo ? null! : new Info());

        /// <summary>Asserts that the call issues exactly the given user warning (Logger's colouring included), as <c>pytest.warns(match="^...$")</c>.</summary>
        /// <param name="message">The warning text without colouring.</param>
        /// <param name="call">The call.</param>
        private static void AssertWarns(string message, Action call) {
            using WarningCatcher caught = Warnings.Catch();
            call();
            CollectionAssert.Contains(caught.Records.Select(record => record.Message).ToList(), $"\u001b[33mWARN: {message}\u001b[0m");
        }

        /// <summary>Asserts that the call issues a user warning containing <paramref name="text"/>.</summary>
        /// <param name="text">The text.</param>
        /// <param name="call">The call.</param>
        private static void AssertWarnsContaining(string text, Action call) {
            using WarningCatcher caught = Warnings.Catch();
            call();
            Assert.IsTrue(caught.Records.Any(record => record.Message.Contains(text, StringComparison.Ordinal)), string.Join("\n", caught.Records));
        }

        /// <summary>
        ///     Asserts that the call throws exactly <typeparamref name="TException"/> with exactly <paramref name="message"/> and
        ///     issues no warning, as Gymnasium's <c>pytest.raises(match="^...$")</c> inside <c>catch_warnings</c>.
        /// </summary>
        /// <typeparam name="TException">The exception type.</typeparam>
        /// <param name="message">The message.</param>
        /// <param name="call">The call.</param>
        private static void AssertRaises<TException>(string message, Action call) where TException : Exception {
            using WarningCatcher caught = Warnings.Catch();
            Assert.AreEqual(message, Assert.ThrowsExactly<TException>(call).Message);
            Assert.AreEqual(0, caught.Records.Count, string.Join("\n", caught.Records));
        }
    }
}
