#nullable enable
#pragma warning disable CS0618 // The legacy API is what these tests pin.
using System;
using System.Linq;
using System.Threading.Tasks;
using Gym.Environments.Envs.Classic;
using Gym.Envs;
using Gym.Exceptions;
using Gym.Legacy;
using Gym.Wrappers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using SixLabors.ImageSharp;

namespace Gym.Tests.Legacy {
    /// <summary>
    ///     Pins <see cref="LegacyEnvAdapter"/>, the pre-0.26 API over a current env (port plan §16): <c>Seed(s)</c> then
    ///     <c>Reset()</c> is <c>Reset(seed: s)</c>, a step is the done API of <c>step_api_compatibility</c>, and rendering
    ///     takes only the construction-time mode.
    /// </summary>
    [TestClass]
    public class LegacyEnvAdapterTests {
        /// <summary><c>Seed(s)</c> then <c>Reset()</c> starts exactly where <c>Reset(seed: s)</c> does; the seed is used once.</summary>
        [TestMethod]
        public void SeedThenReset_IsASeededReset() {
            var adapter = new LegacyEnvAdapter(new CartPoleEnv());
            adapter.Seed(42);
            using NDArray legacy = adapter.Reset();
            using NDArray modern = new CartPoleEnv().Reset(seed: 42).Observation;

            Assert.IsTrue(np.array_equal(modern, legacy));
            Assert.AreEqual(42, (int)adapter.Env.NpRandomSeed);
            using NDArray unseeded = adapter.Reset();
            Assert.AreEqual(42, (int)adapter.Env.NpRandomSeed, "An unseeded reset keeps the generator.");
            Assert.AreEqual("Seed must be greater or equal to zero, actual value: -1", Assert.ThrowsExactly<Error>(() => adapter.Seed(-1)).Message);
        }

        /// <summary>
        ///     A step reports <c>terminated or truncated</c> as done and marks the ending in
        ///     <c>Information["TimeLimit.truncated"]</c>: false for a termination, true for a <c>TimeLimit</c> truncation.
        /// </summary>
        [TestMethod]
        public void Step_ReportsTheDoneApi() {
            var terminating = new LegacyEnvAdapter(new CartPoleEnv());
            terminating.Seed(0);
            terminating.Reset().Dispose();
            Step step;
            do {
                step = terminating.Step(1);
            } while (!step.Done);
            Assert.AreEqual(false, step.Information!["TimeLimit.truncated"]);
            Assert.AreEqual(1.0, step.Reward, "The reward stays float64.");

            var truncating = new LegacyEnvAdapter(new TimeLimit<NDArray, long>(new CartPoleEnv(), 2));
            truncating.Seed(0);
            truncating.Reset().Dispose();
            step = truncating.Step(0);
            Assert.IsFalse(step.Done);
            Assert.IsFalse(step.Information!.ContainsKey("TimeLimit.truncated"), "The key is only added when the episode ends.");
            step = truncating.Step(1);
            Assert.IsTrue(step.Done);
            Assert.AreEqual(true, step.Information!["TimeLimit.truncated"]);
        }

        /// <summary>
        ///     <see cref="LegacyEnvAdapter.StepAsync"/> awaits the env's own <c>StepAsync</c>: the same transition as
        ///     <see cref="LegacyEnvAdapter.Step"/>, completed on the calling thread; a step error faults the task.
        /// </summary>
        /// <returns>The test's task.</returns>
        [TestMethod]
        public async Task StepAsync_MatchesStep() {
            var synchronous = new LegacyEnvAdapter(new CartPoleEnv());
            var asynchronous = new LegacyEnvAdapter(new CartPoleEnv());
            synchronous.Seed(7);
            asynchronous.Seed(7);
            synchronous.Reset().Dispose();
            asynchronous.Reset().Dispose();

            for (int i = 0; i < 20; i++) {
                ValueTask<Step> pending = asynchronous.StepAsync(i % 2);
                Assert.IsTrue(pending.IsCompletedSuccessfully);
                Step expected = synchronous.Step(i % 2), actual = await pending;
                // Field by field: the legacy Step compares its Dict by reference.
                Assert.IsTrue(np.array_equal(expected.Observation, actual.Observation), $"step {i}");
                Assert.AreEqual(expected.Reward, actual.Reward);
                Assert.AreEqual(expected.Done, actual.Done);
                CollectionAssert.AreEquivalent(expected.Information!.ToList(), actual.Information!.ToList());
            }

            ValueTask<Step> invalid = asynchronous.StepAsync(5);
            Assert.IsTrue(invalid.IsFaulted, "CartPole's assertion faults the task.");
            await Assert.ThrowsExactlyAsync<AssertionError>(async () => await invalid);
        }

        /// <summary>Rendering takes only the env's own render mode; an <c>rgb_array</c> frame comes back as an image.</summary>
        [TestMethod]
        public void Render_OnlyTheConstructionTimeMode() {
            var adapter = new LegacyEnvAdapter(new CartPoleEnv(renderMode: "rgb_array"));
            adapter.Reset().Dispose();

            using Image? image = adapter.Render("rgb_array");
            Assert.IsNotNull(image);
            Assert.AreEqual(600, image!.Width);
            Assert.AreEqual(400, image.Height);
            Assert.ThrowsExactly<UnsupportedMode>(() => adapter.Render("human"));
        }

        /// <summary>The legacy metadata is a snapshot of the env's; assigning one replaces the env's.</summary>
        [TestMethod]
        public void Metadata_IsASnapshot() {
            var env = new CartPoleEnv();
            var adapter = new LegacyEnvAdapter(env);

            var snapshot = adapter.Metadata;
            CollectionAssert.AreEqual(new[] { "render_modes", "render_fps" }, snapshot.Keys.ToArray());
            snapshot["render_fps"] = 1;
            Assert.AreEqual(50L, env.Metadata["render_fps"]);
            adapter.Metadata = snapshot;
            Assert.AreEqual(1, env.Metadata["render_fps"]);
        }
    }
}
