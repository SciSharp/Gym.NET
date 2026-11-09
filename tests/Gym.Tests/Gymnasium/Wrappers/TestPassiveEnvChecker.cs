#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using Gym.Envs;
using Gym.Observations;
using Gym.Spaces;
using Gym.Tests.Gymnasium.Envs;
using Gym.Utils;
using Gym.Wrappers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Gymnasium.Wrappers {
    /// <summary>
    ///     Gymnasium's <c>PassiveEnvChecker</c> wrapper tests (<c>tests/wrappers/test_passive_env_checker.py</c>; the class
    ///     carries a suffix because <c>tests/utils</c> has a file of the same name): no warnings for the built-in envs, the
    ///     construction failures, the checks running once each, and the data-reuse warnings of the first two steps.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Python-only: spaces that are strings (<c>action_space="error"</c>) and a step returning a string, whose
    ///     <c>Expects step result to be a tuple</c> assertion C#'s result type rules out; the API test checks the step instead.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_passive_env_checker.py">tests/wrappers/test_passive_env_checker.py</see>.</para>
    /// </remarks>
    [TestClass]
    public class TestPassiveEnvCheckerWrapper {
        /// <summary>The warnings the built-in envs may raise under the checker: an out-of-date version (Gymnasium's <c>PASSIVE_CHECK_IGNORE_WARNING</c>).</summary>
        private static readonly Regex PassiveCheckIgnoreWarning = new(@"\u001b\[33mWARN: The environment (.*?) is out of date\. You should consider upgrading to version `v(\d)`\.\u001b\[0m");

        /// <summary>Every built-in env passes a reset, a step and a close under the checker without unexpected warnings.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_passive_env_checker.py#L16">test_passive_checker_wrapper_warnings</see>.</remarks>
        [TestMethod]
        public void TestPassiveCheckerWrapperWarnings() {
            List<IEnv> envs = EnvTestingUtils.AllTestingInitialisedEnvs();
            Assert.IsTrue(envs.Count > 0);
            foreach (IEnv env in envs) {
                if (env.Spec is { DisableEnvChecker: true }) {
                    env.Dispose();
                    continue;
                }

                using WarningCatcher caught = Warnings.Catch();
                using (IEnv checkerEnv = Wrap(env)) {
                    checkerEnv.Reset();
                    checkerEnv.Step(checkerEnv.ActionSpace.Sample());
                    checkerEnv.Close();
                }

                foreach (WarningMessage warning in caught.Records) {
                    if (!PassiveCheckIgnoreWarning.IsMatch(warning.Message))
                        Assert.Fail($"Unexpected warning: {warning.Message}");
                }
            }
        }

        /// <summary>A missing action or observation space fails construction with Gymnasium's messages.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_passive_env_checker.py#L36">test_initialise_failures</see>.</remarks>
        [TestMethod]
        public void TestInitialiseFailures() {
            using (var env = new GenericTestEnv(true, null, GenericTestEnv.DefaultSpace, TestingEnv.BasicResetFunc, TestingEnv.BasicStepFunc, TestingEnv.BasicRenderFunc))
                Assert.AreEqual("The environment must specify an action space. https://gymnasium.farama.org/introduction/create_custom_env/",
                    Assert.ThrowsExactly<AttributeError>(() => new PassiveEnvChecker<object?, object?>(env)).Message);
            using (var env = new GenericTestEnv(true, GenericTestEnv.DefaultSpace, null, TestingEnv.BasicResetFunc, TestingEnv.BasicStepFunc, TestingEnv.BasicRenderFunc))
                Assert.AreEqual("The environment must specify an observation space. https://gymnasium.farama.org/introduction/create_custom_env/",
                    Assert.ThrowsExactly<AttributeError>(() => new PassiveEnvChecker<object?, object?>(env)).Message);
        }

        /// <summary>Each of reset, step and render is checked on its first call: a reset outside the space and render modes that are a string warn.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_passive_env_checker.py#L72">test_api_failures</see>.</remarks>
        [TestMethod]
        public void TestApiFailures() {
            var genericEnv = new GenericTestEnv(resetFunc: (self, seed, options) => new ResetResult<object?>(np.array(new[] { -1f }), new Info()),
                metadata: new Metadata { ["render_modes"] = "error" });
            using var env = new PassiveEnvChecker<object?, object?>(genericEnv);
            Assert.IsFalse(env.CheckedReset);
            Assert.IsFalse(env.CheckedStep);
            Assert.IsFalse(env.CheckedRender);

            using (WarningCatcher caught = Warnings.Catch()) {
                env.Reset();
                Assert.IsTrue(caught.Records.Any(record => record.Message.Contains("The obs returned by the `reset()` method is not within the observation space", StringComparison.Ordinal)));
            }

            Assert.IsTrue(env.CheckedReset);

            env.Step(env.ActionSpace.Sample());
            Assert.IsTrue(env.CheckedStep);

            using (WarningCatcher caught = Warnings.Catch()) {
                using RenderResult _ = env.Render();
                Assert.IsTrue(caught.Records.Any(record => record.Message.Contains("Expects the render_modes to be a sequence (i.e. list, tuple), actual type: <class 'str'>", StringComparison.Ordinal)));
            }

            Assert.IsTrue(env.CheckedRender);
        }

        /// <summary>
        ///     An env that reuses one buffer and one info dict warns for the reset and first step, then for the first and second
        ///     steps, and never again; the wrapper then lets go of the data.
        /// </summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_passive_env_checker.py#L135">test_passive_checker_wrapper_data_reuse</see>.</remarks>
        [TestMethod]
        public void TestPassiveCheckerWrapperDataReuse() {
            using NDArray buffer = np.zeros(new Shape(1L), np.float32);
            var info = new Info();
            long count = 0;
            var genericEnv = new GenericTestEnv(
                actionSpace: new Discrete(1),
                observationSpace: new Box(0, 100, new Shape(1L), dtype: np.float32),
                resetFunc: (self, seed, options) => {
                    self.BaseReset(seed);
                    count = 0;
                    buffer.SetAtIndex(0f, 0);
                    info["count"] = 0L;
                    return new ResetResult<object?>(buffer, info);
                },
                stepFunc: (self, action) => {
                    // The buffer and the info are mutated in place: the very objects of the previous call.
                    count++;
                    buffer.SetAtIndex((float)count, 0);
                    info["count"] = count;
                    return new StepResult<object?>(buffer, 0.0, false, false, info);
                });
            using var checkerEnv = new PassiveEnvChecker<object?, object?>(genericEnv);
            using NDArray action = NDArray.Scalar(0L);

            using (WarningCatcher caught = Warnings.Catch()) {
                checkerEnv.Reset();
                for (int i = 0; i < 4; i++)
                    checkerEnv.Step(action);

                string[] expectedWarnings = {
                    "The observations returned by `reset` and the following `step` share an object",
                    "The infos returned by `reset` and the following `step` share an object",
                    "The observations returned by `step` and the following `step` share an object",
                    "The infos returned by `step` and the following `step` share an object",
                };
                Assert.AreEqual(4, caught.Records.Count, string.Join("\n", caught.Records));
                for (int i = 0; i < expectedWarnings.Length; i++)
                    StringAssert.Contains(caught.Records[i].Message, expectedWarnings[i]);
            }

            // The checks are not repeated and the wrapper stops holding a reference to the user's data.
            Assert.IsTrue(checkerEnv.CheckedDataReuse);
            Assert.IsNull(checkerEnv.PreviousData);
            Assert.IsNull(PyAttributes.GetAttr(checkerEnv, "_previous_data"));
        }

        /// <summary>Wraps an untyped env in the checker, closing the generic wrapper over the env's own types.</summary>
        /// <param name="env">The env.</param>
        /// <returns>The checker; it owns the env.</returns>
        private static IEnv Wrap(IEnv env) =>
            (IEnv)PyKwargs.Construct(typeof(PassiveEnvChecker<,>), new Info { ["env"] = env });
    }
}
